"""
Step 1: Video Swin Transformer로 영상 직접 학습
- 10초 영상(300프레임)에서 32프레임 균등 샘플링
- 구조물의 동적 거동을 직접 관찰 → 이미지보다 압도적 정확도
"""
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.video import swin3d_s, Swin3D_S_Weights
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N_FRAMES = 32
IMG_SIZE = 224
N_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 4
GRAD_ACCUM = 8
LR = 1e-4
LABEL_SMOOTHING = 0.05


class VideoDataset(Dataset):
    def __init__(self, df, data_dir, is_train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.is_train = is_train
        self.label_map = {"unstable": 0, "stable": 1}

        self.spatial_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def _sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.is_train:
            # 랜덤 시작점 + 균등 샘플링 (temporal augmentation)
            max_start = max(0, total_frames - N_FRAMES * (total_frames // N_FRAMES))
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            indices = np.linspace(start, total_frames - 1, N_FRAMES, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, N_FRAMES, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        cap.release()
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        img_dir = row.get("img_dir", self.data_dir)
        video_path = os.path.join(img_dir, sample_id, "simulation.mp4")

        frames = self._sample_frames(video_path)

        # Transform each frame
        transformed = []
        for f in frames:
            t = self.spatial_transform(f)
            transformed.append(t)

        # (T, C, H, W) → (C, T, H, W)
        video = torch.stack(transformed, dim=1)

        if self.is_train:
            # Random horizontal flip (all frames together)
            if np.random.random() > 0.5:
                video = video.flip(-1)

        label = row["label"]
        if isinstance(label, str):
            label = self.label_map[label]
        return video, label


class VideoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = swin3d_s(weights=Swin3D_S_Weights.DEFAULT)
        feat_dim = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true * np.log(pred), axis=1))


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for i, (videos, labels) in enumerate(loader):
        videos, labels = videos.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(videos)
            loss = criterion(outputs, labels) / GRAD_ACCUM
        scaler.scale(loss).backward()
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * GRAD_ACCUM * videos.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for videos, labels in loader:
        videos = videos.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(videos)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    true_onehot = np.zeros_like(all_probs)
    true_onehot[np.arange(len(all_labels)), all_labels] = 1
    score = logloss(true_onehot, all_probs)
    acc = (all_probs.argmax(1) == all_labels).mean()
    return score, acc, all_probs, all_labels


def main():
    print(f"Device: {DEVICE}")

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")

    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")

    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    print(f"Total samples: {len(all_real)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(all_real), 2))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, real_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*50}")

        train_df = all_real.iloc[train_idx]
        val_df = all_real.iloc[val_idx]

        train_ds = VideoDataset(train_df, train_dir, is_train=True)
        val_ds = VideoDataset(val_df, dev_dir, is_train=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True)

        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        model = VideoClassifier().to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda")

        best_score = float("inf")
        patience = 0
        max_patience = 10

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            val_score, val_acc, val_probs, val_labels = evaluate(model, val_loader)
            scheduler.step()

            print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

            if val_score < best_score:
                best_score = val_score
                patience = 0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_video_fold{fold}.pth"))
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
    print(f"Video Model Complete!")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean: {mean_score:.4f}")
    print(f"{'='*50}")

    # OOF soft predictions 저장
    np.save(os.path.join(SAVE_DIR, "oof_video_preds.npy"), oof_preds)
    np.save(os.path.join(SAVE_DIR, "oof_video_labels.npy"), real_labels)

    # CSV로도 저장
    oof_df = pd.DataFrame({
        "id": all_real["id"].values,
        "label": all_real["label"].values,
        "video_unstable_prob": oof_preds[:, 0],
        "video_stable_prob": oof_preds[:, 1],
    })
    oof_df.to_csv(os.path.join(SAVE_DIR, "oof_video_soft_labels.csv"), index=False)
    print(f"Saved OOF soft labels to oof_video_soft_labels.csv")


if __name__ == "__main__":
    main()
