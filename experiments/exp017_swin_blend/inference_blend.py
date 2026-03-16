"""
Swin 추론 + ConvNeXt 블렌딩

1. Swin 단독 추론
2. ConvNeXt(exp015) + Swin(exp017) 블렌딩
3. Temperature Scaling 적용
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
from train import SwinClassifier, DATA_ROOT, IMG_SIZE, N_FOLDS, MODEL_NAME

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

    # === Swin 추론 ===
    print("\n=== Swin Transformer inference ===")
    swin_logits_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth")
        model = SwinClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        for tfm in TTA_TRANSFORMS:
            ds = TestDualViewDataset(
                csv_path=os.path.join(DATA_ROOT, "sample_submission.csv"),
                data_dir=os.path.join(DATA_ROOT, "test"),
                transform=tfm,
            )
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            ids, logits = get_logits(model, loader)
            swin_logits_list.append(logits)

        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1}/{N_FOLDS} done")

    swin_logits = np.mean(swin_logits_list, axis=0)

    # Swin 단독 submission
    for t in [0.7, 0.8, 1.0]:
        scaled = swin_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
        sub.to_csv(os.path.join(SAVE_DIR, f"submission_swin_T{t:.1f}.csv"), index=False)
        print(f"Swin T={t:.1f}: range [{probs[:,0].min():.4f}, {probs[:,0].max():.4f}]")

    # === ConvNeXt + Swin 블렌딩 ===
    print("\n=== Blending with ConvNeXt (exp015) ===")

    # exp015 logits 로드 (없으면 submission에서 역산)
    exp015_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp015_multiframe")

    # exp015 submission에서 probs 가져오기
    convnext_sub = pd.read_csv(os.path.join(exp015_dir, "submission_T1.0.csv"))
    convnext_probs = convnext_sub[["unstable_prob", "stable_prob"]].values

    swin_probs_t1 = np.exp(swin_logits) / np.exp(swin_logits).sum(axis=1, keepdims=True)

    # 다양한 블렌딩 비율
    for conv_w in [0.5, 0.6, 0.7]:
        swin_w = 1 - conv_w
        blended = conv_w * convnext_probs + swin_w * swin_probs_t1

        for t in [0.7, 0.8, 1.0]:
            # 블렌딩 후 temperature (logit 변환 → scaling → softmax)
            log_blended = np.log(np.clip(blended, 1e-15, 1))
            scaled = log_blended / t
            final_probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)

            sub = pd.DataFrame({"id": ids, "unstable_prob": final_probs[:, 0], "stable_prob": final_probs[:, 1]})
            fname = f"submission_blend_conv{int(conv_w*100)}_swin{int(swin_w*100)}_T{t:.1f}.csv"
            sub.to_csv(os.path.join(SAVE_DIR, fname), index=False)

        print(f"ConvNeXt {int(conv_w*100)}% + Swin {int(swin_w*100)}% done")

    print("\nDone! 추천 제출: blend_conv60_swin40_T0.7.csv")


if __name__ == "__main__":
    main()
