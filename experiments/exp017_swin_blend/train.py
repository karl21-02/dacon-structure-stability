"""
Exp017: Swin Transformer + Dual View + Pseudo-Label + Multi-Frame

ConvNeXt와 완전히 다른 아키텍처로 학습 → 나중에 블렌딩.
Swin Transformer: 윈도우 기반 attention → 지역적 패턴을 더 잘 포착.
ConvNeXt는 CNN 계열, Swin은 Transformer 계열 → 서로 보완.
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

EPOCHS = 40
BATCH_SIZE = 8
LR = 2e-4              # Swin은 ConvNeXt보다 약간 낮은 LR이 좋음
IMG_SIZE = 224
LABEL_SMOOTHING = 0.1
MODEL_NAME = "swin_base_patch4_window7_224"  # ★ Swin-Base: 88M params
GRAD_ACCUM = 4
N_FOLDS = 5
PSEUDO_THRESHOLD = 0.90
EXTRA_FRAMES = [1, 10, 20, 30]


# ── Dataset ──
class MultiFrameDualViewDataset(Dataset):
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

        front_file = row.get("front_file", "front.png")
        front_path = os.path.join(img_dir, sample_id, front_file)
        if not os.path.exists(front_path):
            front_path = os.path.join(img_dir, sample_id, "front.png")

        front = Image.open(front_path).convert("RGB")
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


# ★ Swin Transformer 듀얼뷰 분류기
# Swin은 고정 입력 크기(224x224)만 지원 → side-by-side 불가
# 대신: front와 top을 각각 backbone에 통과시키고 특징을 결합
class SwinClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        # front + top 특징을 결합 → 2배 크기
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (B, 3, 224, 448) → front(왼쪽 224), top(오른쪽 224)으로 분리
        front = x[:, :, :, :224]   # (B, 3, 224, 224)
        top = x[:, :, :, 224:]     # (B, 3, 224, 224)

        front_feat = self.backbone(front)  # (B, feat_dim)
        top_feat = self.backbone(top)      # (B, feat_dim)

        combined = torch.cat([front_feat, top_feat], dim=1)  # (B, feat_dim*2)
        return self.head(combined)


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

    train_ds = MultiFrameDualViewDataset(train_df, train_dir, transform=get_transforms(True))
    val_ds = MultiFrameDualViewDataset(val_df, dev_dir, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"Model: {MODEL_NAME}")

    model = SwinClassifier(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda")
    best_score = float("inf")
    patience = 0
    max_patience = 12

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
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth"))
            print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

    print(f"Fold {fold+1} complete. Best Val LogLoss: {best_score:.4f}")
    return best_score


def expand_with_frames(df, data_dir):
    rows = []
    for _, row in df.iterrows():
        new_row = row.copy()
        new_row["front_file"] = "front.png"
        rows.append(new_row)
        if row["id"].startswith("TRAIN"):
            for frame_num in EXTRA_FRAMES:
                frame_file = f"front_frame{frame_num}.png"
                frame_path = os.path.join(data_dir, row["id"], frame_file)
                if os.path.exists(frame_path):
                    new_row = row.copy()
                    new_row["front_file"] = frame_file
                    rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")

    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    # Pseudo-label
    exp012_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp012_calibration_pseudo")
    test_logits = np.load(os.path.join(exp012_dir, "test_logits.npy"))
    test_ids = np.load(os.path.join(exp012_dir, "test_ids.npy"))
    test_probs = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
    mask = test_probs.max(axis=1) >= PSEUDO_THRESHOLD

    pseudo_df = pd.DataFrame({
        "id": test_ids[mask],
        "label": ["unstable" if p == 0 else "stable" for p in test_probs[mask].argmax(axis=1)],
        "img_dir": test_dir,
        "front_file": "front.png",
    })

    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    print(f"Real: {len(all_real)}, Pseudo: {len(pseudo_df)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, real_labels)):
        val_fold_df = all_real.iloc[val_idx].copy()
        val_fold_df["front_file"] = "front.png"

        train_real = all_real.iloc[train_idx]
        train_expanded = expand_with_frames(train_real, train_dir)
        train_fold_df = pd.concat([train_expanded, pseudo_df], ignore_index=True)

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
