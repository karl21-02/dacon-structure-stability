"""
Step 3: Soft Label 모델로 추론 + Temperature Scaling
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step2_train import DualViewClassifier, DATA_ROOT, IMG_SIZE, N_FOLDS

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


TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.ColorJitter(brightness=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


@torch.no_grad()
def get_logits(model, loader):
    model.eval()
    all_ids, all_logits = [], []
    for images, sample_ids in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        all_ids.extend(sample_ids)
        all_logits.append(outputs.cpu().numpy())
    return all_ids, np.concatenate(all_logits)


def main():
    print(f"Device: {DEVICE}")
    print(f"Folds: {N_FOLDS}, TTA: {len(TTA_TRANSFORMS)}")

    all_logits_list = []
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
            ids, logits = get_logits(model, loader)
            all_logits_list.append(logits)

        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1}/{N_FOLDS} done")

    avg_logits = np.mean(all_logits_list, axis=0)

    for t in [0.5, 0.7, 0.8, 1.0]:
        scaled = avg_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)

        submission = pd.DataFrame({
            "id": ids,
            "unstable_prob": probs[:, 0],
            "stable_prob": probs[:, 1],
        })

        save_path = os.path.join(SAVE_DIR, f"submission_T{t:.1f}.csv")
        submission.to_csv(save_path, index=False)
        print(f"T={t:.1f}: range [{probs[:,0].min():.4f}, {probs[:,0].max():.4f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
