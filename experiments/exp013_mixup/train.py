"""
Exp013: Dual View + Pseudo-Label + Mixup
Mixup: 두 이미지를 섞어서 학습 → 부드러운 확률 예측 → LogLoss 개선
"""
import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

# ── 설정 ──
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4
IMG_SIZE = 224
LABEL_SMOOTHING = 0.1
MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
GRAD_ACCUM = 4
N_FOLDS = 5
PSEUDO_THRESHOLD = 0.90
MIXUP_ALPHA = 0.4  # ★ Mixup 강도. 클수록 더 많이 섞음 (0.2~0.4 권장)


# ── Mixup 함수 ──
# 두 이미지를 비율(lam)에 따라 섞음
# 예: lam=0.7이면 이미지A 70% + 이미지B 30%
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # Beta 분포에서 섞는 비율 샘플링
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)  # 랜덤으로 짝 지을 인덱스

    mixed_x = lam * x + (1 - lam) * x[index]  # 이미지 섞기
    y_a, y_b = y, y[index]  # 두 이미지의 라벨
    return mixed_x, y_a, y_b, lam


# Mixup용 손실 함수: 두 라벨에 대해 가중 평균
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Dataset ──
class DualViewDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.label_map = {"unstable": 0, "stable": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        img_dir = row.get("img_dir", self.data_dir)

        front = Image.open(os.path.join(img_dir, sample_id, "front.png")).convert("RGB")
        top = Image.open(os.path.join(img_dir, sample_id, "top.png")).convert("RGB")
        front = front.resize((IMG_SIZE, IMG_SIZE))
        top = top.resize((IMG_SIZE, IMG_SIZE))

        combined = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))
        combined.paste(front, (0, 0))
        combined.paste(top, (IMG_SIZE, 0))

        if self.transform:
            combined = self.transform(combined)

        if self.is_test:
            return combined, sample_id

        label = row["label"]
        if isinstance(label, str):
            label = self.label_map[label]
        else:
            label = int(label)
        return combined, label


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DualViewClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true * np.log(pred), axis=1))


# ★ Mixup이 적용된 학습 함수
def train_one_epoch(model, loader, criterion, optimizer, scaler, use_mixup=True):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 50% 확률로 Mixup 적용 (항상 하면 순수한 샘플을 못 봄)
        if use_mixup and np.random.random() > 0.5:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels)
            with torch.amp.autocast("cuda"):
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam) / GRAD_ACCUM
        else:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels) / GRAD_ACCUM

        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM * images.size(0)
        # Mixup 시에는 원본 라벨 기준 정확도 측정
        with torch.no_grad():
            if use_mixup:
                _, predicted = model(images).max(1)
            else:
                _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    true_onehot = np.zeros_like(all_probs)
    true_onehot[np.arange(len(all_labels)), all_labels] = 1
    score = logloss(true_onehot, all_probs)
    acc = (all_probs.argmax(1) == all_labels).mean()
    return score, acc


def train_fold(fold, train_df, val_df, train_dir, dev_dir):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*50}")

    train_ds = DualViewDataset(train_df, train_dir, transform=get_transforms(True))
    val_ds = DualViewDataset(val_df, dev_dir, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    n_real = len(train_df[train_df["id"].str.startswith("TRAIN") | train_df["id"].str.startswith("DEV")])
    n_pseudo = len(train_df[train_df["id"].str.startswith("TEST")])
    print(f"Train: {len(train_ds)} (real: {n_real}, pseudo: {n_pseudo}), Val: {len(val_ds)}")
    print(f"Mixup alpha: {MIXUP_ALPHA}")

    model = DualViewClassifier(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda")
    best_score = float("inf")
    patience = 0
    max_patience = 10

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, use_mixup=True)
        val_score, val_acc = evaluate(model, val_loader)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

        if val_score < best_score:
            best_score = val_score
            patience = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth"))
            print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

    print(f"Fold {fold+1} complete. Best Val LogLoss: {best_score:.4f}")
    return best_score


def main():
    print(f"Device: {DEVICE}")
    print(f"Mixup alpha: {MIXUP_ALPHA}")

    # 원본 데이터
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    train_csv["img_dir"] = os.path.join(DATA_ROOT, "train")
    dev_csv["img_dir"] = os.path.join(DATA_ROOT, "dev")

    test_dir = os.path.join(DATA_ROOT, "test")

    # Pseudo-label (exp012에서 생성된 것 사용)
    exp012_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp012_calibration_pseudo")
    test_logits = np.load(os.path.join(exp012_dir, "test_logits.npy"))
    test_ids = np.load(os.path.join(exp012_dir, "test_ids.npy"))
    test_probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
    mask = test_probs.max(axis=1) >= PSEUDO_THRESHOLD

    pseudo_df = pd.DataFrame({
        "id": test_ids[mask],
        "label": test_probs[mask].argmax(axis=1),
        "img_dir": test_dir,
    })

    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    print(f"Real: {len(all_real)}, Pseudo: {len(pseudo_df)}, Total: {len(all_real) + len(pseudo_df)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, real_labels)):
        val_fold_df = all_real.iloc[val_idx]
        train_fold_df = pd.concat([all_real.iloc[train_idx], pseudo_df], ignore_index=True)
        score = train_fold(fold, train_fold_df, val_fold_df,
                           os.path.join(DATA_ROOT, "train"), os.path.join(DATA_ROOT, "dev"))
        fold_scores.append(score)

    print(f"\n{'='*50}")
    print(f"All folds complete!")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean: {np.mean(fold_scores):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
