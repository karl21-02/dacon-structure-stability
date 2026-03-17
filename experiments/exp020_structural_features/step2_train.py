"""
Step 2: 배경 제거 이미지 + 구조 피처 결합 모델 학습
- 입력 1: 배경 제거된 front+top 이미지
- 입력 2: 구조 피처 벡터 (20차원)
- ConvNeXt backbone + struct features → concat → head
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

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 16
GRAD_ACCUM = 2
LR = 3e-4
LABEL_SMOOTHING = 0.1
MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
EXTRA_FRAMES = [1, 10, 20, 30]

# 구조 피처 컬럼
FEAT_COLS = [
    "front_pixels", "front_h", "front_w", "front_hw_ratio", "front_lean",
    "front_top_base_ratio", "front_cy_ratio", "front_cx_ratio",
    "front_fill_ratio", "front_symmetry",
    "top_pixels", "top_h", "top_w", "top_hw_ratio", "top_lean",
    "top_top_base_ratio", "top_cy_ratio", "top_cx_ratio",
    "top_fill_ratio", "top_symmetry",
]
N_FEATS = len(FEAT_COLS)


class StructuralDataset(Dataset):
    def __init__(self, df, masked_dir, orig_dir, feat_mean, feat_std, transform=None):
        self.df = df.reset_index(drop=True)
        self.masked_dir = masked_dir
        self.orig_dir = orig_dir
        self.transform = transform
        self.label_map = {"unstable": 0, "stable": 1}
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]

        # 배경 제거 이미지 사용 (있으면), 없으면 원본
        front_file = row.get("front_file", "front.png")

        # masked 이미지는 front.png만 있음 (멀티프레임은 원본 사용)
        if front_file == "front.png":
            img_dir = row.get("masked_dir", self.masked_dir)
        else:
            img_dir = row.get("img_dir", self.orig_dir)

        front_path = os.path.join(img_dir, sample_id, front_file)
        if not os.path.exists(front_path):
            front_path = os.path.join(img_dir, sample_id, "front.png")
        if not os.path.exists(front_path):
            # fallback to original
            orig_dir = row.get("img_dir", self.orig_dir)
            front_path = os.path.join(orig_dir, sample_id, "front.png")

        # top은 항상 masked 시도
        top_masked = os.path.join(row.get("masked_dir", self.masked_dir), sample_id, "top.png")
        if os.path.exists(top_masked):
            top_path = top_masked
        else:
            orig_dir = row.get("img_dir", self.orig_dir)
            top_path = os.path.join(orig_dir, sample_id, "top.png")

        front = Image.open(front_path).convert("RGB")
        top = Image.open(top_path).convert("RGB")

        front = front.resize((IMG_SIZE, IMG_SIZE))
        top = top.resize((IMG_SIZE, IMG_SIZE))

        combined = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))
        combined.paste(front, (0, 0))
        combined.paste(top, (IMG_SIZE, 0))

        if self.transform:
            combined = self.transform(combined)

        # 구조 피처
        feats = np.array([row.get(c, 0) for c in FEAT_COLS], dtype=np.float32)
        feats = np.nan_to_num(feats, 0)
        feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)
        feats = torch.tensor(feats, dtype=torch.float32)

        label = row["label"]
        if isinstance(label, str):
            label = self.label_map[label]
        else:
            label = int(label)

        return combined, feats, label


class StructuralClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True, n_feats=N_FEATS, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        # 구조 피처 처리
        self.feat_net = nn.Sequential(
            nn.Linear(n_feats, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # image feat (x2 for dual view) + struct feat
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, struct_feats):
        front = x[:, :, :, :IMG_SIZE]
        top = x[:, :, :, IMG_SIZE:]
        front_feat = self.backbone(front)
        top_feat = self.backbone(top)

        struct_feat = self.feat_net(struct_feats)

        combined = torch.cat([front_feat, top_feat, struct_feat], dim=1)
        return self.head(combined)


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


def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true * np.log(pred), axis=1))


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for i, (images, feats, labels) in enumerate(loader):
        images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images, feats)
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
    for images, feats, labels in loader:
        images, feats = images.to(DEVICE), feats.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images, feats)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    true_onehot = np.zeros_like(all_probs)
    true_onehot[np.arange(len(all_labels)), all_labels] = 1
    score = logloss(true_onehot, all_probs)
    acc = (all_probs.argmax(1) == all_labels).mean()
    return score, acc, all_probs


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
    print(f"Structural features: {N_FEATS}")

    # 데이터 로드
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    # 구조 피처 로드
    struct_feats = pd.read_csv(os.path.join(SAVE_DIR, "structural_features.csv"))
    feat_map = {}
    for _, row in struct_feats.iterrows():
        feat_map[row["id"]] = {c: row.get(c, 0) for c in FEAT_COLS}

    # train/dev에 피처 추가
    for df in [train_csv, dev_csv]:
        for col in FEAT_COLS:
            df[col] = df["id"].map(lambda x: feat_map.get(x, {}).get(col, 0))

    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)

    # masked image 디렉토리
    masked_train = os.path.join(SAVE_DIR, "masked_images", "train")
    masked_dev = os.path.join(SAVE_DIR, "masked_images", "dev")
    masked_test = os.path.join(SAVE_DIR, "masked_images", "test")
    all_real["masked_dir"] = all_real["id"].apply(
        lambda x: masked_train if x.startswith("TRAIN") else masked_dev)

    # Pseudo-label
    pseudo_path = os.path.join(SAVE_DIR, "..", "exp018_triple_stack", "pseudo_labels_round2.csv")
    pseudo_df = pd.read_csv(pseudo_path)
    pseudo_df["img_dir"] = test_dir
    pseudo_df["masked_dir"] = masked_test
    pseudo_df["front_file"] = "front.png"
    if "confidence" in pseudo_df.columns:
        pseudo_df = pseudo_df.drop(columns=["confidence"])
    # pseudo에도 피처 추가
    for col in FEAT_COLS:
        pseudo_df[col] = pseudo_df["id"].map(lambda x: feat_map.get(x, {}).get(col, 0))

    print(f"Real: {len(all_real)}, Pseudo: {len(pseudo_df)}")

    # 피처 정규화 통계 (train+dev 기준)
    feat_values = all_real[FEAT_COLS].values.astype(np.float32)
    feat_mean = np.nanmean(feat_values, axis=0)
    feat_std = np.nanstd(feat_values, axis=0)
    np.save(os.path.join(SAVE_DIR, "feat_mean.npy"), feat_mean)
    np.save(os.path.join(SAVE_DIR, "feat_std.npy"), feat_std)

    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(all_real), 2))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, real_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*50}")

        val_fold_df = all_real.iloc[val_idx].copy()
        val_fold_df["front_file"] = "front.png"

        train_real = all_real.iloc[train_idx]
        train_expanded = expand_with_frames(train_real, train_dir)
        train_fold_df = pd.concat([train_expanded, pseudo_df], ignore_index=True)

        train_ds = StructuralDataset(train_fold_df, masked_train, train_dir,
                                     feat_mean, feat_std, transform=get_transforms(True))
        val_ds = StructuralDataset(val_fold_df, masked_dev, dev_dir,
                                   feat_mean, feat_std, transform=get_transforms(False))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True)

        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        model = StructuralClassifier(pretrained=True).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda")

        best_score = float("inf")
        patience = 0
        max_patience = 12

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            val_score, val_acc, val_probs = evaluate(model, val_loader)
            scheduler.step()

            print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

            if val_score < best_score:
                best_score = val_score
                patience = 0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_struct_fold{fold}.pth"))
                best_val_probs = val_probs.copy()
                print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

        oof_preds[val_idx] = best_val_probs
        fold_scores.append(best_score)
        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1} complete. Best: {best_score:.4f}")

    mean_score = np.mean(fold_scores)
    print(f"\n{'='*50}")
    print(f"Structural Model Complete!")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean: {mean_score:.4f}")
    print(f"{'='*50}")

    np.save(os.path.join(SAVE_DIR, "oof_structural.npy"), oof_preds)


if __name__ == "__main__":
    main()
