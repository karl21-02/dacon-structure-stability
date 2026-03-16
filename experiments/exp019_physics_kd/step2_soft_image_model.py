"""
Step 2: Video soft label로 이미지 모델 학습
- Train 데이터: hard label 대신 Video 모델의 soft prediction 사용
- Pseudo-label (test): 기존 exp018에서 생성한 것 그대로 사용
- KL Divergence loss로 soft label 학습
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

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
N_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 16
GRAD_ACCUM = 2
LR = 3e-4
MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
EXTRA_FRAMES = [1, 10, 20, 30]
ALPHA = 0.7  # soft label 비중
PSEUDO_THRESHOLD = 0.85


class SoftLabelDualViewDataset(Dataset):
    """soft label을 지원하는 듀얼뷰 데이터셋"""
    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
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

        # Soft label 처리
        if "soft_unstable" in row.index and not np.isnan(row["soft_unstable"]):
            soft_label = torch.tensor([row["soft_unstable"], row["soft_stable"]], dtype=torch.float)
            has_soft = 1.0
        else:
            # Hard label (pseudo-label 또는 soft label이 없는 경우)
            label = row["label"]
            if isinstance(label, str):
                label = self.label_map[label]
            soft_label = torch.zeros(2, dtype=torch.float)
            soft_label[int(label)] = 1.0
            has_soft = 0.0

        return combined, soft_label, has_soft


class DualViewClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        front = x[:, :, :, :IMG_SIZE]
        top = x[:, :, :, IMG_SIZE:]
        front_feat = self.backbone(front)
        top_feat = self.backbone(top)
        combined = torch.cat([front_feat, top_feat], dim=1)
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


def soft_cross_entropy(logits, soft_targets):
    """Soft label용 cross entropy (KL divergence 기반)"""
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1).mean()
    return loss


def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true * np.log(pred), axis=1))


def train_one_epoch(model, loader, optimizer, scaler, alpha):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()

    for i, (images, soft_labels, has_soft) in enumerate(loader):
        images = images.to(DEVICE)
        soft_labels = soft_labels.to(DEVICE)
        has_soft = has_soft.to(DEVICE)

        with torch.amp.autocast("cuda"):
            logits = model(images)

            # Soft label이 있는 샘플: soft CE
            # Soft label이 없는 샘플 (pseudo): hard CE
            soft_loss = soft_cross_entropy(logits, soft_labels)

            # has_soft 비율에 따라 alpha 조절
            # soft label 있는 샘플은 alpha 비중으로, 없는 샘플은 hard CE로
            hard_targets = soft_labels.argmax(dim=1)
            hard_loss = F.cross_entropy(logits, hard_targets)

            # 혼합 loss
            soft_mask = has_soft.bool()
            if soft_mask.any() and (~soft_mask).any():
                loss = (alpha * soft_cross_entropy(logits[soft_mask], soft_labels[soft_mask]) +
                        (1 - alpha) * F.cross_entropy(logits[~soft_mask], hard_targets[~soft_mask]))
            elif soft_mask.any():
                loss = soft_cross_entropy(logits[soft_mask], soft_labels[soft_mask])
            else:
                loss = F.cross_entropy(logits, hard_targets)

            loss = loss / GRAD_ACCUM

        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(hard_targets).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for images, soft_labels, _ in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(soft_labels.argmax(dim=1).numpy())
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
    print(f"Alpha (soft label weight): {ALPHA}")

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    # Video soft labels 로드
    oof_video = pd.read_csv(os.path.join(SAVE_DIR, "oof_video_soft_labels.csv"))
    video_soft_map = {}
    for _, row in oof_video.iterrows():
        video_soft_map[row["id"]] = (row["video_unstable_prob"], row["video_stable_prob"])

    # Train/Dev에 soft label 추가
    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    all_real["soft_unstable"] = all_real["id"].map(lambda x: video_soft_map.get(x, (np.nan, np.nan))[0])
    all_real["soft_stable"] = all_real["id"].map(lambda x: video_soft_map.get(x, (np.nan, np.nan))[1])

    # Pseudo-label (exp018에서 생성)
    pseudo_path = os.path.join(SAVE_DIR, "..", "exp018_triple_stack", "pseudo_labels_round2.csv")
    pseudo_df = pd.read_csv(pseudo_path)
    pseudo_df["img_dir"] = test_dir
    pseudo_df["front_file"] = "front.png"
    pseudo_df["soft_unstable"] = np.nan  # pseudo는 soft label 없음
    pseudo_df["soft_stable"] = np.nan
    if "confidence" in pseudo_df.columns:
        pseudo_df = pseudo_df.drop(columns=["confidence"])

    print(f"Real: {len(all_real)}, Pseudo: {len(pseudo_df)}")

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

        train_ds = SoftLabelDualViewDataset(train_fold_df, train_dir, transform=get_transforms(True))
        val_ds = SoftLabelDualViewDataset(val_fold_df, dev_dir, transform=get_transforms(False))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        model = DualViewClassifier(pretrained=True).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda")

        best_score = float("inf")
        patience = 0
        max_patience = 12

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, ALPHA)
            val_score, val_acc, val_probs = evaluate(model, val_loader)
            scheduler.step()

            print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

            if val_score < best_score:
                best_score = val_score
                patience = 0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_soft_fold{fold}.pth"))
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
    print(f"Soft-Label Image Model Complete!")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean: {mean_score:.4f}")
    print(f"{'='*50}")

    np.save(os.path.join(SAVE_DIR, "oof_soft_image.npy"), oof_preds)


if __name__ == "__main__":
    main()
