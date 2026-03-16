"""
Step 3: 3개 모델 추론 + Stacking
1. ConvNeXt, Swin, EVA-02 각각 TTA 추론
2. OOF predictions으로 Logistic Regression meta-learner 학습
3. 최적 가중치로 test 예측 블렌딩
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step2_train_all import DualViewClassifier, MODELS, IMG_SIZE, N_FOLDS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDualViewDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, img_size=IMG_SIZE):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        front = Image.open(os.path.join(self.data_dir, sample_id, "front.png")).convert("RGB")
        top = Image.open(os.path.join(self.data_dir, sample_id, "top.png")).convert("RGB")
        front = front.resize((self.img_size, self.img_size))
        top = top.resize((self.img_size, self.img_size))
        combined = Image.new("RGB", (self.img_size * 2, self.img_size))
        combined.paste(front, (0, 0))
        combined.paste(top, (self.img_size, 0))
        if self.transform:
            combined = self.transform(combined)
        return combined, sample_id


def get_tta_transforms(img_size=IMG_SIZE):
    return [
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


def infer_model(arch_key):
    cfg = MODELS[arch_key]
    model_name = cfg["name"]
    img_size = cfg.get("img_size", IMG_SIZE)

    print(f"\n--- Inference: {arch_key} ({model_name}) ---")

    test_csv = os.path.join(DATA_ROOT, "sample_submission.csv")
    test_dir = os.path.join(DATA_ROOT, "test")

    logits_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_{arch_key}_fold{fold}.pth")
        model = DualViewClassifier(model_name, pretrained=False, img_size=img_size).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        for tfm in get_tta_transforms(img_size):
            ds = TestDualViewDataset(test_csv, test_dir, transform=tfm, img_size=img_size)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
            ids, logits = get_logits(model, loader)
            logits_list.append(logits)

        del model; torch.cuda.empty_cache()
        print(f"  Fold {fold+1}/{N_FOLDS} done")

    mean_logits = np.mean(logits_list, axis=0)
    probs = np.exp(mean_logits) / np.exp(mean_logits).sum(axis=1, keepdims=True)
    return ids, probs


def main():
    print(f"Device: {DEVICE}")

    # === 1. 각 모델 test 추론 ===
    all_probs = {}
    for arch_key in ["convnext", "swin", "eva02"]:
        ids, probs = infer_model(arch_key)
        all_probs[arch_key] = probs

    # === 2. OOF로 Stacking ===
    print("\n=== Stacking with OOF predictions ===")

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    # OOF predictions 로드
    oof_convnext = np.load(os.path.join(SAVE_DIR, "oof_convnext.npy"))
    oof_swin = np.load(os.path.join(SAVE_DIR, "oof_swin.npy"))
    oof_eva02 = np.load(os.path.join(SAVE_DIR, "oof_eva02.npy"))

    # Stacking features: 각 모델의 unstable_prob
    X_stack = np.column_stack([
        oof_convnext[:, 0],
        oof_swin[:, 0],
        oof_eva02[:, 0],
    ])

    # Logistic Regression meta-learner
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(X_stack, real_labels)
    print(f"Meta-learner coefficients: {meta.coef_[0]}")
    print(f"Meta-learner intercept: {meta.intercept_[0]:.4f}")

    # OOF score
    oof_meta_probs = meta.predict_proba(X_stack)
    true_onehot = np.zeros((len(real_labels), 2))
    true_onehot[np.arange(len(real_labels)), real_labels] = 1
    from step2_train_all import logloss
    oof_score = logloss(true_onehot, oof_meta_probs)
    print(f"Stacking OOF LogLoss: {oof_score:.4f}")

    # === 3. Test prediction with stacking ===
    X_test_stack = np.column_stack([
        all_probs["convnext"][:, 0],
        all_probs["swin"][:, 0],
        all_probs["eva02"][:, 0],
    ])
    test_meta_probs = meta.predict_proba(X_test_stack)

    # Stacking submission
    for t in [0.7, 0.8, 1.0]:
        log_probs = np.log(np.clip(test_meta_probs, 1e-15, 1))
        scaled = log_probs / t
        final = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"id": ids, "unstable_prob": final[:, 0], "stable_prob": final[:, 1]})
        sub.to_csv(os.path.join(SAVE_DIR, f"submission_stack_T{t:.1f}.csv"), index=False)
        print(f"Stacking T={t:.1f}: range [{final[:,0].min():.4f}, {final[:,0].max():.4f}]")

    # === 4. 단순 가중평균도 생성 (비교용) ===
    print("\n=== Simple weighted average ===")
    weights_list = [
        (0.4, 0.3, 0.3, "w433"),
        (0.35, 0.35, 0.3, "w353530"),
        (0.5, 0.25, 0.25, "w502525"),
    ]
    for w1, w2, w3, name in weights_list:
        blended = w1 * all_probs["convnext"] + w2 * all_probs["swin"] + w3 * all_probs["eva02"]
        for t in [0.7]:
            log_b = np.log(np.clip(blended, 1e-15, 1))
            scaled = log_b / t
            final = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            sub = pd.DataFrame({"id": ids, "unstable_prob": final[:, 0], "stable_prob": final[:, 1]})
            sub.to_csv(os.path.join(SAVE_DIR, f"submission_{name}_T{t:.1f}.csv"), index=False)
            print(f"{name} T={t:.1f}: range [{final[:,0].min():.4f}, {final[:,0].max():.4f}]")

    print(f"\nDone! 추천 제출: submission_stack_T0.7.csv")


if __name__ == "__main__":
    main()
