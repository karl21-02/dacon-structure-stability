import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp003_convnext_front"))
from train import ConvNeXtClassifier, MODEL_NAME

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224


class StructureDataset(Dataset):
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


# TTA: 여러 변환을 적용한 뒤 예측 평균
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
    # 살짝 크게 → 중앙 크롭
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    # 밝기 약간 증가
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
    # 모델 로드 (exp003 체크포인트)
    model = ConvNeXtClassifier(pretrained=False).to(DEVICE)
    ckpt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "exp003_convnext_front", "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"TTA transforms: {len(TTA_TRANSFORMS)}")

    # TTA: 각 변환별로 예측 후 평균
    all_probs_list = []
    for i, tfm in enumerate(TTA_TRANSFORMS):
        ds = StructureDataset(
            csv_path=os.path.join(DATA_ROOT, "sample_submission.csv"),
            data_dir=os.path.join(DATA_ROOT, "test"),
            transform=tfm,
        )
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        ids, probs = predict(model, loader)
        all_probs_list.append(probs)
        print(f"  TTA {i+1}/{len(TTA_TRANSFORMS)} done")

    # 평균
    avg_probs = np.mean(all_probs_list, axis=0)

    submission = pd.DataFrame({
        "id": ids,
        "unstable_prob": avg_probs[:, 0],
        "stable_prob": avg_probs[:, 1],
    })

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"\nSubmission saved to {save_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
