import os
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

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4
IMG_SIZE = 224


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
        top_path = os.path.join(self.data_dir, sample_id, "top.png")
        front_img = Image.open(front_path).convert("RGB")
        top_img = Image.open(top_path).convert("RGB")
        if self.transform:
            front_img = self.transform(front_img)
            top_img = self.transform(top_img)
        if self.is_test:
            return front_img, top_img, sample_id
        label = self.label_map[row["label"]]
        return front_img, top_img, label


# ── Model ──
class MultiViewResNet(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, front, top):
        front_feat = self.backbone(front)
        top_feat = self.backbone(top)
        combined = torch.cat([front_feat, top_feat], dim=1)
        return self.head(combined)


# ── Augmentation ──
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Utils ──
def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    loss = -np.sum(true * np.log(pred), axis=1)
    return np.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for front, top, labels in loader:
        front, top, labels = front.to(DEVICE), top.to(DEVICE), labels.to(DEVICE)
        outputs = model(front, top)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * front.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for front, top, labels in loader:
        front, top = front.to(DEVICE), top.to(DEVICE)
        outputs = model(front, top)
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

    print(f"Train: {len(train_ds)} samples, Dev: {len(dev_ds)} samples")

    model = MultiViewResNet(model_name="resnet50", pretrained=True, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = float("inf")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        dev_score, dev_acc = evaluate(model, dev_loader)
        scheduler.step()

        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Dev LogLoss: {dev_score:.4f} Acc: {dev_acc:.4f}")

        if dev_score < best_score:
            best_score = dev_score
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")

    print(f"\nTraining complete. Best Dev LogLoss: {best_score:.4f}")


if __name__ == "__main__":
    main()
