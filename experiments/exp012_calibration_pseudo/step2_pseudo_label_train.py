"""
Step 2: Pseudo-Labeling으로 재학습
1. exp011 모델의 테스트 예측 중 확신도 높은 것(>threshold)을 가짜 라벨로 사용
2. train(1000) + dev(100) + pseudo(~800?) = ~1900개로 재학습
3. 테스트 도메인 데이터가 학습에 포함되어 도메인 갭 감소!
"""
import os
import sys
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
PSEUDO_THRESHOLD = 0.90  # 90% 이상 확신하는 예측만 사용


# ── Dataset: front + top 듀얼 뷰 (pseudo-label 지원) ──
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

        # pseudo-label인 경우 숫자로 저장되어 있음
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


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels) / GRAD_ACCUM
        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * GRAD_ACCUM * images.size(0)
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

    # pseudo-label 포함 데이터 수 표시
    n_real = len(train_df[train_df["id"].str.startswith("TRAIN") | train_df["id"].str.startswith("DEV")])
    n_pseudo = len(train_df[train_df["id"].str.startswith("TEST")])
    print(f"Train: {len(train_ds)} (real: {n_real}, pseudo: {n_pseudo}), Val: {len(val_ds)}")

    model = DualViewClassifier(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda")
    best_score = float("inf")
    patience = 0
    max_patience = 10

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_score, val_acc = evaluate(model, val_loader)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

        if val_score < best_score:
            best_score = val_score
            patience = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_pseudo_fold{fold}.pth"))
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
    print(f"Pseudo-label threshold: {PSEUDO_THRESHOLD}")

    # 1. 원본 데이터
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")

    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    # 2. Pseudo-label 생성 (exp011 모델 예측 기반)
    test_logits = np.load(os.path.join(SAVE_DIR, "test_logits.npy"))
    test_ids = np.load(os.path.join(SAVE_DIR, "test_ids.npy"))

    # softmax로 확률 변환
    test_probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
    max_probs = test_probs.max(axis=1)
    pred_labels = test_probs.argmax(axis=1)

    # 확신도 높은 것만 선택
    confident_mask = max_probs >= PSEUDO_THRESHOLD
    pseudo_ids = test_ids[confident_mask]
    pseudo_labels = pred_labels[confident_mask]
    pseudo_probs = max_probs[confident_mask]

    print(f"\nPseudo-label 통계:")
    print(f"  전체 test: {len(test_ids)}")
    print(f"  확신도 >= {PSEUDO_THRESHOLD}: {confident_mask.sum()} ({confident_mask.mean()*100:.1f}%)")
    print(f"  - unstable: {(pseudo_labels==0).sum()}")
    print(f"  - stable: {(pseudo_labels==1).sum()}")
    print(f"  평균 확신도: {pseudo_probs.mean():.4f}")

    # pseudo-label DataFrame 생성
    pseudo_df = pd.DataFrame({
        "id": pseudo_ids,
        "label": pseudo_labels,
        "img_dir": test_dir,
    })

    # 3. 원본 + pseudo 합치기
    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    all_with_pseudo = pd.concat([all_real, pseudo_df], ignore_index=True)
    print(f"\n총 학습 데이터: {len(all_with_pseudo)} (원본: {len(all_real)}, pseudo: {len(pseudo_df)})")

    # 4. K-Fold (원본 데이터만으로 fold 분할, pseudo는 항상 학습에 포함)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, real_labels)):
        # val은 원본 데이터만 (공정한 평가)
        val_fold_df = all_real.iloc[val_idx]

        # train은 원본 학습분 + 전체 pseudo
        train_real = all_real.iloc[train_idx]
        train_fold_df = pd.concat([train_real, pseudo_df], ignore_index=True)

        score = train_fold(fold, train_fold_df, val_fold_df, train_dir, dev_dir)
        fold_scores.append(score)

    print(f"\n{'='*50}")
    print(f"All folds complete!")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean: {np.mean(fold_scores):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
