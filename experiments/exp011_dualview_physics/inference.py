import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import DualViewClassifier, DATA_ROOT, IMG_SIZE, N_FOLDS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDualViewDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]

        front = Image.open(os.path.join(self.data_dir, sample_id, "front.png")).convert("RGB")
        top = Image.open(os.path.join(self.data_dir, sample_id, "top.png")).convert("RGB")

        front = front.resize((IMG_SIZE, IMG_SIZE))
        top = top.resize((IMG_SIZE, IMG_SIZE))

        combined = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))
        combined.paste(front, (0, 0))
        combined.paste(top, (IMG_SIZE, 0))

        if self.transform:
            combined = self.transform(combined)
        return combined, sample_id


# TTA: 듀얼 뷰용 (좌우 반전은 front/top 관계를 깨뜨리므로 제외)
TTA_TRANSFORMS = [
    # 원본
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 상하 반전
    transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 밝기 변경
    transforms.Compose([
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
    print(f"Folds: {N_FOLDS}, TTA: {len(TTA_TRANSFORMS)}")

    all_probs_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth")
        model = DualViewClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        for tfm in TTA_TRANSFORMS:
            ds = TestDualViewDataset(
                csv_path=os.path.join(DATA_ROOT, "sample_submission.csv"),
                data_dir=os.path.join(DATA_ROOT, "test"),
                transform=tfm,
            )
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            ids, probs = predict(model, loader)
            all_probs_list.append(probs)
        print(f"Fold {fold+1}/{N_FOLDS} done ({len(TTA_TRANSFORMS)} TTA)")

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
