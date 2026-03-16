"""
Step 1: Pseudo-Label Round 2
exp017(Swin) + exp015(ConvNeXt) 블렌딩 모델로 더 정확한 pseudo label 생성
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# exp017의 모델 클래스 재사용
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp017_swin_blend"))
from train import SwinClassifier, IMG_SIZE, N_FOLDS, MODEL_NAME

# exp015의 모델 클래스
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp015_multiframe"))
from step2_train import DualViewClassifier

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
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
    print("=== Pseudo-Label Round 2 ===\n")

    test_csv = os.path.join(DATA_ROOT, "sample_submission.csv")
    test_dir = os.path.join(DATA_ROOT, "test")

    # === Swin 추론 ===
    print("--- Swin Transformer (exp017) ---")
    swin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp017_swin_blend")
    swin_logits_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(swin_dir, f"best_model_fold{fold}.pth")
        model = SwinClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        for tfm in TTA_TRANSFORMS:
            ds = TestDualViewDataset(test_csv, test_dir, transform=tfm)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            ids, logits = get_logits(model, loader)
            swin_logits_list.append(logits)
        del model; torch.cuda.empty_cache()
        print(f"  Swin fold {fold+1}/{N_FOLDS} done")
    swin_logits = np.mean(swin_logits_list, axis=0)

    # === ConvNeXt 추론 ===
    print("--- ConvNeXt (exp015) ---")
    convnext_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp015_multiframe")
    convnext_logits_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(convnext_dir, f"best_model_fold{fold}.pth")
        model = DualViewClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        for tfm in TTA_TRANSFORMS:
            ds = TestDualViewDataset(test_csv, test_dir, transform=tfm)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            _, logits = get_logits(model, loader)
            convnext_logits_list.append(logits)
        del model; torch.cuda.empty_cache()
        print(f"  ConvNeXt fold {fold+1}/{N_FOLDS} done")
    convnext_logits = np.mean(convnext_logits_list, axis=0)

    # === 블렌딩하여 pseudo label 생성 ===
    # Swin probs + ConvNeXt probs (60:40)
    swin_probs = np.exp(swin_logits) / np.exp(swin_logits).sum(axis=1, keepdims=True)
    convnext_probs = np.exp(convnext_logits) / np.exp(convnext_logits).sum(axis=1, keepdims=True)
    blended_probs = 0.4 * swin_probs + 0.6 * convnext_probs

    # 높은 신뢰도만 pseudo label로 사용
    max_probs = blended_probs.max(axis=1)
    pred_labels = blended_probs.argmax(axis=1)

    for threshold in [0.85, 0.90, 0.95]:
        mask = max_probs >= threshold
        n_selected = mask.sum()
        n_unstable = (pred_labels[mask] == 0).sum()
        n_stable = (pred_labels[mask] == 1).sum()
        print(f"\nThreshold {threshold}: {n_selected}/{len(ids)} selected "
              f"(unstable: {n_unstable}, stable: {n_stable})")

    # 0.85 threshold로 저장 (더 많은 데이터 활용)
    THRESHOLD = 0.85
    mask = max_probs >= THRESHOLD

    pseudo_df = pd.DataFrame({
        "id": np.array(ids)[mask],
        "label": ["unstable" if p == 0 else "stable" for p in pred_labels[mask]],
        "confidence": max_probs[mask],
    })

    save_path = os.path.join(SAVE_DIR, "pseudo_labels_round2.csv")
    pseudo_df.to_csv(save_path, index=False)
    print(f"\nSaved {len(pseudo_df)} pseudo labels to {save_path}")

    # logits도 저장 (나중에 사용)
    np.save(os.path.join(SAVE_DIR, "blend_logits_r2.npy"), blended_probs)
    np.save(os.path.join(SAVE_DIR, "test_ids_r2.npy"), np.array(ids))


if __name__ == "__main__":
    main()
