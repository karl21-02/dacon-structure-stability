"""
Step 2: 3개 아키텍처 순차 학습 (Round 2 Pseudo-Label 사용)
1. ConvNeXt-Small (CNN)
2. Swin-Base (Transformer)
3. EVA-02-Small (Hybrid)

각 모델의 OOF prediction도 저장 → Step 3에서 stacking에 사용
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
LABEL_SMOOTHING = 0.1
EXTRA_FRAMES = [1, 10, 20, 30]

MODELS = {
    "convnext": {
        "name": "convnext_small.fb_in22k_ft_in1k",
        "lr": 3e-4,
        "epochs": 40,
        "batch_size": 16,
        "grad_accum": 2,
        "patience": 12,
    },
    "swin": {
        "name": "swin_base_patch4_window7_224",
        "lr": 2e-4,
        "epochs": 40,
        "batch_size": 8,
        "grad_accum": 4,
        "patience": 12,
    },
    "eva02": {
        "name": "eva02_small_patch14_336.mim_in22k_ft_in1k",
        "lr": 2e-4,
        "epochs": 40,
        "batch_size": 8,
        "grad_accum": 4,
        "patience": 12,
        "img_size": 336,
    },
}


class MultiFrameDualViewDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, is_test=False, img_size=IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.img_size = img_size
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

        front = front.resize((self.img_size, self.img_size))
        top = top.resize((self.img_size, self.img_size))

        combined = Image.new("RGB", (self.img_size * 2, self.img_size))
        combined.paste(front, (0, 0))
        combined.paste(top, (self.img_size, 0))

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


class DualViewClassifier(nn.Module):
    """front/top을 각각 backbone에 통과시키고 특징 결합"""
    def __init__(self, model_name, pretrained=True, num_classes=2, img_size=IMG_SIZE):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
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
        front = x[:, :, :, :self.img_size]
        top = x[:, :, :, self.img_size:]
        front_feat = self.backbone(front)
        top_feat = self.backbone(top)
        combined = torch.cat([front_feat, top_feat], dim=1)
        return self.head(combined)


def get_transforms(is_train=True, img_size=IMG_SIZE):
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


def train_one_epoch(model, loader, criterion, optimizer, scaler, grad_accum):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels) / grad_accum
        scaler.scale(loss).backward()
        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * grad_accum * images.size(0)
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
    return score, acc, all_probs, all_labels


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


def train_model(arch_key, all_real, pseudo_df, train_dir, dev_dir, test_dir, skf_splits):
    cfg = MODELS[arch_key]
    model_name = cfg["name"]
    img_size = cfg.get("img_size", IMG_SIZE)

    print(f"\n{'='*60}")
    print(f"Training: {arch_key} ({model_name})")
    print(f"{'='*60}")

    oof_preds = np.zeros((len(all_real), 2))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf_splits):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        val_fold_df = all_real.iloc[val_idx].copy()
        val_fold_df["front_file"] = "front.png"

        train_real = all_real.iloc[train_idx]
        train_expanded = expand_with_frames(train_real, train_dir)
        train_fold_df = pd.concat([train_expanded, pseudo_df], ignore_index=True)

        train_ds = MultiFrameDualViewDataset(
            train_fold_df, train_dir, transform=get_transforms(True, img_size), img_size=img_size)
        val_ds = MultiFrameDualViewDataset(
            val_fold_df, dev_dir, transform=get_transforms(False, img_size), img_size=img_size)

        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        model = DualViewClassifier(model_name, pretrained=True, img_size=img_size).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda")

        best_score = float("inf")
        patience = 0
        ckpt_path = os.path.join(SAVE_DIR, f"best_{arch_key}_fold{fold}.pth")

        for epoch in range(cfg["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, cfg["grad_accum"])
            val_score, val_acc, val_probs, val_labels = evaluate(model, val_loader)
            scheduler.step()

            print(f"[Epoch {epoch+1:02d}/{cfg['epochs']}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

            if val_score < best_score:
                best_score = val_score
                patience = 0
                torch.save(model.state_dict(), ckpt_path)
                best_val_probs = val_probs.copy()
                print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

        # OOF predictions 저장
        oof_preds[val_idx] = best_val_probs
        fold_scores.append(best_score)
        del model; torch.cuda.empty_cache()

        print(f"Fold {fold+1} complete. Best: {best_score:.4f}")

    mean_score = np.mean(fold_scores)
    print(f"\n{arch_key} complete! Folds: {fold_scores}, Mean: {mean_score:.4f}")

    # OOF 저장
    np.save(os.path.join(SAVE_DIR, f"oof_{arch_key}.npy"), oof_preds)
    return fold_scores


def main():
    print(f"Device: {DEVICE}")

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")

    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    # Round 2 pseudo labels
    pseudo_path = os.path.join(SAVE_DIR, "pseudo_labels_round2.csv")
    pseudo_df = pd.read_csv(pseudo_path)
    pseudo_df["img_dir"] = test_dir
    pseudo_df["front_file"] = "front.png"
    if "confidence" in pseudo_df.columns:
        pseudo_df = pseudo_df.drop(columns=["confidence"])

    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    print(f"Real: {len(all_real)}, Pseudo R2: {len(pseudo_df)}")

    # 동일한 fold split 사용 (모든 모델에서)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    skf_splits = list(skf.split(all_real, real_labels))

    results = {}
    for arch_key in ["convnext", "swin", "eva02"]:
        scores = train_model(arch_key, all_real, pseudo_df, train_dir, dev_dir, test_dir, skf_splits)
        results[arch_key] = scores

    print(f"\n{'='*60}")
    print("All models trained!")
    for k, v in results.items():
        print(f"  {k}: {v} → Mean: {np.mean(v):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
