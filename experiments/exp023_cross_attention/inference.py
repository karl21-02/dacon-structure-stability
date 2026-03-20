"""
Exp023 추론: Spatial Cross-Attention 모델 + Temperature Scaling
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
from train import CrossAttentionClassifier, FEAT_COLS, IMG_SIZE, N_FOLDS, MODEL_NAME, N_FEATS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
EXP020_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp020_structural_features")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestStructuralDataset(Dataset):
    def __init__(self, df, masked_dir, orig_dir, feat_mean, feat_std, transform=None):
        self.df = df.reset_index(drop=True)
        self.masked_dir = masked_dir
        self.orig_dir = orig_dir
        self.transform = transform
        self.feat_mean = feat_mean
        self.feat_std = feat_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]

        front_masked = os.path.join(self.masked_dir, sample_id, "front.png")
        top_masked = os.path.join(self.masked_dir, sample_id, "top.png")

        if os.path.exists(front_masked):
            front = Image.open(front_masked).convert("RGB")
        else:
            front = Image.open(os.path.join(self.orig_dir, sample_id, "front.png")).convert("RGB")

        if os.path.exists(top_masked):
            top = Image.open(top_masked).convert("RGB")
        else:
            top = Image.open(os.path.join(self.orig_dir, sample_id, "top.png")).convert("RGB")

        front = front.resize((IMG_SIZE, IMG_SIZE))
        top = top.resize((IMG_SIZE, IMG_SIZE))

        combined = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))
        combined.paste(front, (0, 0))
        combined.paste(top, (IMG_SIZE, 0))

        if self.transform:
            combined = self.transform(combined)

        feats = np.array([row.get(c, 0) for c in FEAT_COLS], dtype=np.float32)
        feats = np.nan_to_num(feats, 0)
        feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)
        feats = torch.tensor(feats, dtype=torch.float32)

        return combined, feats, sample_id


@torch.no_grad()
def get_logits(model, loader):
    model.eval()
    all_ids, all_logits = [], []
    for images, feats, sample_ids in loader:
        images, feats = images.to(DEVICE), feats.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images, feats)
        all_ids.extend(sample_ids)
        all_logits.append(outputs.cpu().numpy())
    return all_ids, np.concatenate(all_logits)


def main():
    print(f"Device: {DEVICE}")

    # 피처 로드
    feat_df = pd.read_csv(os.path.join(EXP020_DIR, "structural_features.csv"))
    feat_mean = np.load(os.path.join(SAVE_DIR, "feat_mean.npy"))
    feat_std = np.load(os.path.join(SAVE_DIR, "feat_std.npy"))

    test_feat = feat_df[feat_df.id.str.startswith("TEST")].copy()
    sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"), encoding="utf-8-sig")
    test_feat = sample_sub[["id"]].merge(test_feat, on="id", how="left")

    masked_test = os.path.join(EXP020_DIR, "masked_images", "test")
    test_dir = os.path.join(DATA_ROOT, "test")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = TestStructuralDataset(test_feat, masked_test, test_dir, feat_mean, feat_std, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # 5 fold 평균 logits
    all_logits = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth")
        model = CrossAttentionClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        ids, logits = get_logits(model, test_loader)
        all_logits.append(logits)
        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1}/{N_FOLDS} done")

    avg_logits = np.mean(all_logits, axis=0)

    # Temperature Scaling submissions
    for t in [0.4, 0.47, 0.5, 0.7, 1.0]:
        scaled = avg_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
        sub.to_csv(os.path.join(SAVE_DIR, f"submission_T{t:.2f}.csv"), index=False)
        print(f"T={t:.2f}: range [{probs[:,0].min():.6f}, {probs[:,0].max():.6f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
