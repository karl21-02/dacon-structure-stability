"""
Step 3: Soft-label 이미지 모델 추론 + 기존 모델과 최종 블렌딩
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
from step2_soft_image_model import DualViewClassifier, IMG_SIZE, N_FOLDS, MODEL_NAME

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

    test_csv = os.path.join(DATA_ROOT, "sample_submission.csv")
    test_dir = os.path.join(DATA_ROOT, "test")

    # === Soft-label 이미지 모델 추론 ===
    print("\n=== Soft-Label Image Model Inference ===")
    logits_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_soft_fold{fold}.pth")
        model = DualViewClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        for tfm in TTA_TRANSFORMS:
            ds = TestDualViewDataset(test_csv, test_dir, transform=tfm)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            ids, logits = get_logits(model, loader)
            logits_list.append(logits)

        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1}/{N_FOLDS} done")

    soft_logits = np.mean(logits_list, axis=0)
    soft_probs = np.exp(soft_logits) / np.exp(soft_logits).sum(axis=1, keepdims=True)

    # Soft-label 모델 단독 submission
    for t in [0.7, 0.8, 1.0]:
        scaled = soft_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
        sub.to_csv(os.path.join(SAVE_DIR, f"submission_soft_T{t:.1f}.csv"), index=False)
        print(f"Soft T={t:.1f}: range [{probs[:,0].min():.4f}, {probs[:,0].max():.4f}]")

    # === 기존 모델과 블렌딩 ===
    print("\n=== Blending with exp018 models ===")

    exp018_dir = os.path.join(SAVE_DIR, "..", "exp018_triple_stack")

    # exp018 모델들의 submission 로드 (있으면)
    blend_models = {}

    # exp017(swin) + exp015(convnext) 블렌딩 결과
    exp017_dir = os.path.join(SAVE_DIR, "..", "exp017_swin_blend")
    blend_sub = os.path.join(exp017_dir, "submission_blend_conv60_swin40_T0.7.csv")
    if os.path.exists(blend_sub):
        df = pd.read_csv(blend_sub)
        blend_models["exp017_blend"] = df[["unstable_prob", "stable_prob"]].values
        print(f"Loaded exp017 blend")

    # exp019 soft model
    blend_models["exp019_soft"] = soft_probs

    # 블렌딩
    if len(blend_models) >= 2:
        keys = list(blend_models.keys())
        for w in [0.5, 0.6, 0.7]:
            blended = w * blend_models[keys[0]] + (1-w) * blend_models[keys[1]]
            for t in [0.7]:
                log_b = np.log(np.clip(blended, 1e-15, 1))
                scaled = log_b / t
                final = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
                sub = pd.DataFrame({"id": ids, "unstable_prob": final[:, 0], "stable_prob": final[:, 1]})
                fname = f"submission_blend_{keys[0]}_{int(w*100)}_{keys[1]}_{int((1-w)*100)}_T{t:.1f}.csv"
                sub.to_csv(os.path.join(SAVE_DIR, fname), index=False)
                print(f"Blend {keys[0]} {int(w*100)}% + {keys[1]} {int((1-w)*100)}% T={t:.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
