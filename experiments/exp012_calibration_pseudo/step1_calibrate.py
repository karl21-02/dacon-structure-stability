"""
Step 1: Temperature Scaling으로 확률 보정
exp011 모델의 logits에 온도를 적용해서 여러 버전의 submission 생성
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp011_dualview_physics"))
from train import DualViewClassifier, DATA_ROOT, IMG_SIZE, N_FOLDS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp011_dualview_physics")


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


@torch.no_grad()
def get_logits(model, loader):
    """softmax 전의 raw logits를 수집 (Temperature Scaling에 필요)"""
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

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = TestDualViewDataset(
        csv_path=os.path.join(DATA_ROOT, "sample_submission.csv"),
        data_dir=os.path.join(DATA_ROOT, "test"),
        transform=tfm,
    )
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    # 5개 모델의 logits 평균
    all_logits = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(MODEL_DIR, f"best_model_fold{fold}.pth")
        model = DualViewClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        ids, logits = get_logits(model, loader)
        all_logits.append(logits)
        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1} done")

    avg_logits = np.mean(all_logits, axis=0)

    # 여러 Temperature로 submission 생성
    temperatures = [0.5, 0.7, 0.8, 1.0, 1.5]
    for t in temperatures:
        # Temperature Scaling: softmax(logits / T)
        scaled = avg_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)

        submission = pd.DataFrame({
            "id": ids,
            "unstable_prob": probs[:, 0],
            "stable_prob": probs[:, 1],
        })

        save_path = os.path.join(SAVE_DIR, f"submission_T{t:.1f}.csv")
        submission.to_csv(save_path, index=False)
        print(f"T={t:.1f}: unstable range [{probs[:,0].min():.4f}, {probs[:,0].max():.4f}] → {save_path}")

    # logits도 저장 (pseudo-labeling에 사용)
    np.save(os.path.join(SAVE_DIR, "test_logits.npy"), avg_logits)
    np.save(os.path.join(SAVE_DIR, "test_ids.npy"), np.array(ids))
    print("\nLogits saved for pseudo-labeling!")


if __name__ == "__main__":
    main()
