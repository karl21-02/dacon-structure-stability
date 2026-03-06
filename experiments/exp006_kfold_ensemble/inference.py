import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = TestDataset(
        csv_path=os.path.join(DATA_ROOT, "sample_submission.csv"),
        data_dir=os.path.join(DATA_ROOT, "test"),
        transform=transform,
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # 5개 fold 모델의 예측 평균
    all_probs_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth")
        model = ConvNeXtClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        ids, probs = predict(model, test_loader)
        all_probs_list.append(probs)
        print(f"Fold {fold+1}/{N_FOLDS} done")

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
