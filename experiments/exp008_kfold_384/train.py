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
BATCH_SIZE = 4         # 384 해상도는 메모리를 ~3배 더 먹음 → 배치 절반으로 줄임
LR = 3e-4
IMG_SIZE = 384          # ★ 핵심 변경: 224 → 384. 이미지를 더 크게 넣어서 구조물의 세밀한 부분까지 볼 수 있음
LABEL_SMOOTHING = 0.1
MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
GRAD_ACCUM = 8          # 4→8로 늘림. 실효 배치: 4x8=32 (exp006과 동일한 실효 배치 유지)
N_FOLDS = 5


# ── Dataset ──
class StructureDataset(Dataset):
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
        front_path = os.path.join(img_dir, sample_id, "front.png")
        img = Image.open(front_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, sample_id
        label = self.label_map[row["label"]]
        return img, label


# ── Augmentation ──
# 384 해상도에 맞게 크롭 크기도 조정
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # 416x416으로 키우고
            transforms.RandomCrop(IMG_SIZE),                     # 384x384로 랜덤 크롭
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Model ──
class ConvNeXtClassifier(nn.Module):
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


# ── Utils ──
def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    loss = -np.sum(true * np.log(pred), axis=1)
    return np.mean(loss)


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

    train_ds = StructureDataset(train_df, train_dir, transform=get_transforms(True))
    val_ds = StructureDataset(val_df, dev_dir, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"IMG_SIZE: {IMG_SIZE}, Batch: {BATCH_SIZE}, GradAccum: {GRAD_ACCUM}, Effective Batch: {BATCH_SIZE * GRAD_ACCUM}")

    model = ConvNeXtClassifier(pretrained=True).to(DEVICE)
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
    print(f"Model: {MODEL_NAME}")
    print(f"IMG_SIZE: {IMG_SIZE} (vs exp006: 224)")
    print(f"Folds: {N_FOLDS}")

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")

    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")

    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    all_df = pd.concat([train_csv, dev_csv], ignore_index=True)
    print(f"Total samples: {len(all_df)} (train: {len(train_csv)}, dev: {len(dev_csv)})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    labels = all_df["label"].values

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, labels)):
        train_fold_df = all_df.iloc[train_idx]
        val_fold_df = all_df.iloc[val_idx]
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
