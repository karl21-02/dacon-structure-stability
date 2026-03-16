"""
Step 2: Soft Label + Dual View + Pseudo-Label로 학습

핵심 변경: CrossEntropyLoss 대신 soft target을 사용하는 KLDivLoss
- hard label: [1, 0] 또는 [0, 1]
- soft label: [0.85, 0.15] 또는 [0.03, 0.97] — 영상 분석 기반
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
GRAD_ACCUM = 4
N_FOLDS = 5
PSEUDO_THRESHOLD = 0.90


# ── Dataset: Soft Label 지원 ──
class DualViewSoftDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test

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

        # soft_label 컬럼이 있으면 사용, 없으면 hard label
        if "soft_label" in row.index and not pd.isna(row["soft_label"]):
            unstable_prob = float(row["soft_label"])
        else:
            # pseudo-label이나 soft_label 없는 경우
            label = row["label"]
            if isinstance(label, str):
                unstable_prob = 1.0 if label == "unstable" else 0.0
            else:
                unstable_prob = 1.0 if int(label) == 0 else 0.0

        # [unstable_prob, stable_prob] soft target
        soft_target = torch.tensor([unstable_prob, 1.0 - unstable_prob], dtype=torch.float32)
        # hard label (정확도 계산용)
        hard_label = 0 if unstable_prob >= 0.5 else 1

        return combined, soft_target, hard_label


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


# ★ Soft Label 학습: KLDivLoss 사용
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for i, (images, soft_targets, hard_labels) in enumerate(loader):
        images = images.to(DEVICE)
        soft_targets = soft_targets.to(DEVICE)
        hard_labels = hard_labels.to(DEVICE)

        with torch.amp.autocast("cuda"):
            logits = model(images)
            # KL Divergence Loss: 모델 출력과 soft target 간의 차이
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, soft_targets, reduction="batchmean") / GRAD_ACCUM

        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(hard_labels).sum().item()
        total += hard_labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for images, soft_targets, hard_labels in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(hard_labels.numpy())
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

    train_ds = DualViewSoftDataset(train_df, train_dir, transform=get_transforms(True))
    val_ds = DualViewSoftDataset(val_df, dev_dir, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    n_real = len(train_df[train_df["id"].str.startswith("TRAIN") | train_df["id"].str.startswith("DEV")])
    n_pseudo = len(train_df[train_df["id"].str.startswith("TEST")])
    has_soft = train_df["soft_label"].notna().sum() if "soft_label" in train_df.columns else 0
    print(f"Train: {len(train_ds)} (real: {n_real}, pseudo: {n_pseudo}, soft_label: {has_soft})")

    model = DualViewClassifier(pretrained=True).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda")
    best_score = float("inf")
    patience = 0
    max_patience = 10

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler)
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
    print(f"Loss: KLDivLoss (soft label)")

    # 1. Soft label 읽기
    soft_label_path = os.path.join(SAVE_DIR, "soft_labels.csv")
    if not os.path.exists(soft_label_path):
        print("ERROR: soft_labels.csv가 없습니다. step1_analyze_video.py를 먼저 실행하세요.")
        return

    soft_df = pd.read_csv(soft_label_path)
    print(f"Soft labels loaded: {len(soft_df)} samples")

    # 2. img_dir 추가
    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    soft_df["img_dir"] = soft_df["id"].apply(
        lambda x: train_dir if x.startswith("TRAIN") else dev_dir
    )

    # 3. Pseudo-label 추가 (exp012에서)
    exp012_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp012_calibration_pseudo")
    test_logits = np.load(os.path.join(exp012_dir, "test_logits.npy"))
    test_ids = np.load(os.path.join(exp012_dir, "test_ids.npy"))
    test_probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
    mask = test_probs.max(axis=1) >= PSEUDO_THRESHOLD

    pseudo_df = pd.DataFrame({
        "id": test_ids[mask],
        "label": ["unstable" if p == 0 else "stable" for p in test_probs[mask].argmax(axis=1)],
        "soft_label": test_probs[mask, 0],  # unstable 확률을 soft label로
        "img_dir": test_dir,
        "max_diff": np.nan,
        "last_diff": np.nan,
        "mean_diff": np.nan,
    })

    all_with_pseudo = pd.concat([soft_df, pseudo_df], ignore_index=True)
    print(f"Total: {len(all_with_pseudo)} (real: {len(soft_df)}, pseudo: {len(pseudo_df)})")

    # 4. K-Fold (원본만으로 분할)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    real_labels = soft_df["label"].map({"unstable": 0, "stable": 1}).values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(soft_df, real_labels)):
        val_fold_df = soft_df.iloc[val_idx]
        train_fold_df = pd.concat([soft_df.iloc[train_idx], pseudo_df], ignore_index=True)
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
