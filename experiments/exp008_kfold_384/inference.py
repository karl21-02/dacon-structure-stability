import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# exp008에서 모델 구조 가져오기
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import ConvNeXtClassifier, DATA_ROOT, IMG_SIZE, N_FOLDS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        img = Image.open(os.path.join(self.data_dir, sample_id, "front.png")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample_id


# TTA 5종 (384 해상도 기준)
TTA_TRANSFORMS = [
    # 원본
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 좌우 반전
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 상하 반전
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 크게 → 중앙 크롭
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 밝기 변경
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_ids, all_probs = [], []
    for images, sample_ids in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_ids.extend(sample_ids)
        all_probs.append(probs)
    return all_ids, np.concatenate(all_probs)


def main():
    print(f"Device: {DEVICE}")
    print(f"IMG_SIZE: {IMG_SIZE}")
    print(f"Folds: {N_FOLDS}, TTA: {len(TTA_TRANSFORMS)}")
    print(f"Total predictions: {N_FOLDS} x {len(TTA_TRANSFORMS)} = {N_FOLDS * len(TTA_TRANSFORMS)}")

    all_probs_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth")
        model = ConvNeXtClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        for t, tfm in enumerate(TTA_TRANSFORMS):
            ds = TestDataset(
                csv_path=os.path.join(DATA_ROOT, "sample_submission.csv"),
                data_dir=os.path.join(DATA_ROOT, "test"),
                transform=tfm,
            )
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            ids, probs = predict(model, loader)
            all_probs_list.append(probs)
        print(f"Fold {fold+1}/{N_FOLDS} done (5 TTA each)")

    avg_probs = np.mean(all_probs_list, axis=0)

    submission = pd.DataFrame({
        "id": ids,
        "unstable_prob": avg_probs[:, 0],
        "stable_prob": avg_probs[:, 1],
    })

    save_path = os.path.join(SAVE_DIR, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"\nSubmission saved to {save_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
