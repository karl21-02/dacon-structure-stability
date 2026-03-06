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

# ── 설정 ──
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4
IMG_SIZE = 224
LABEL_SMOOTHING = 0.1
MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
GRAD_ACCUM = 4  # effective batch size = 8 * 4 = 32


# ── Dataset ──
class StructureDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, is_test=False):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test
        self.label_map = {"unstable": 0, "stable": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        front_path = os.path.join(self.data_dir, sample_id, "front.png")
        img = Image.open(front_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            return img, sample_id
        label = self.label_map[row["label"]]
        return img, label


# ── Augmentation ──
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
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


# ── Main ──
def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")

    train_ds = StructureDataset(
        csv_path=os.path.join(DATA_ROOT, "train.csv"),
        data_dir=os.path.join(DATA_ROOT, "train"),
        transform=get_transforms(is_train=True),
    )
    dev_ds = StructureDataset(
        csv_path=os.path.join(DATA_ROOT, "dev.csv"),
        data_dir=os.path.join(DATA_ROOT, "dev"),
        transform=get_transforms(is_train=False),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

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
        dev_score, dev_acc = evaluate(model, dev_loader)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Dev LogLoss: {dev_score:.4f} Acc: {dev_acc:.4f}")

        if dev_score < best_score:
            best_score = dev_score
            patience = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

    print(f"\nTraining complete. Best Dev LogLoss: {best_score:.4f}")


if __name__ == "__main__":
    main()
