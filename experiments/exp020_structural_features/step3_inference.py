"""
Step 3: 추론 + exp018과 최종 stacking
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from step2_train import (StructuralClassifier, FEAT_COLS, IMG_SIZE, N_FOLDS,
                         MODEL_NAME, N_FEATS)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
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

        # masked 이미지
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
    for images, feats, sample_ids in loader:
        images, feats = images.to(DEVICE), feats.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images, feats)
        all_ids.extend(sample_ids)
        all_logits.append(outputs.cpu().numpy())
    return all_ids, np.concatenate(all_logits)


def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true * np.log(pred), axis=1))


def main():
    print(f"Device: {DEVICE}")

    # 피처 정규화 통계
    feat_mean = np.load(os.path.join(SAVE_DIR, "feat_mean.npy"))
    feat_std = np.load(os.path.join(SAVE_DIR, "feat_std.npy"))

    # 구조 피처 로드
    struct_feats = pd.read_csv(os.path.join(SAVE_DIR, "structural_features.csv"))
    feat_map = {}
    for _, row in struct_feats.iterrows():
        feat_map[row["id"]] = {c: row.get(c, 0) for c in FEAT_COLS}

    # Test 데이터
    test_csv = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"), encoding="utf-8-sig")
    for col in FEAT_COLS:
        test_csv[col] = test_csv["id"].map(lambda x: feat_map.get(x, {}).get(col, 0))

    test_dir = os.path.join(DATA_ROOT, "test")
    masked_test = os.path.join(SAVE_DIR, "masked_images", "test")

    # === exp020 추론 ===
    print("\n=== Structural Model Inference ===")
    logits_list = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(SAVE_DIR, f"best_struct_fold{fold}.pth")
        model = StructuralClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        for tfm in TTA_TRANSFORMS:
            ds = TestStructuralDataset(test_csv, masked_test, test_dir,
                                       feat_mean, feat_std, transform=tfm)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
            ids, logits = get_logits(model, loader)
            logits_list.append(logits)

        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1}/{N_FOLDS} done")

    struct_logits = np.mean(logits_list, axis=0)
    struct_probs = np.exp(struct_logits) / np.exp(struct_logits).sum(axis=1, keepdims=True)

    # 단독 submission
    for t in [0.7, 0.8, 1.0]:
        scaled = struct_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
        sub.to_csv(os.path.join(SAVE_DIR, f"submission_struct_T{t:.1f}.csv"), index=False)
        print(f"Structural T={t:.1f}: range [{probs[:,0].min():.4f}, {probs[:,0].max():.4f}]")

    # === exp018 + exp020 Stacking ===
    print("\n=== Final Stacking (exp018 + exp020) ===")

    exp018_dir = os.path.join(SAVE_DIR, "..", "exp018_triple_stack")

    # OOF 로드
    oof_convnext = np.load(os.path.join(exp018_dir, "oof_convnext.npy"))
    oof_swin = np.load(os.path.join(exp018_dir, "oof_swin.npy"))
    oof_eva02 = np.load(os.path.join(exp018_dir, "oof_eva02.npy"))
    oof_struct = np.load(os.path.join(SAVE_DIR, "oof_structural.npy"))

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    # 4종 stacking
    X_stack = np.column_stack([
        oof_convnext[:, 0],
        oof_swin[:, 0],
        oof_eva02[:, 0],
        oof_struct[:, 0],
    ])

    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(X_stack, real_labels)
    print(f"Meta-learner coefficients: {meta.coef_[0]}")

    # OOF score
    oof_meta = meta.predict_proba(X_stack)
    true_onehot = np.zeros((len(real_labels), 2))
    true_onehot[np.arange(len(real_labels)), real_labels] = 1
    oof_score = logloss(true_onehot, oof_meta)
    print(f"4-model Stacking OOF LogLoss: {oof_score:.4f}")

    # exp018 test predictions 로드
    exp018_stack_sub = pd.read_csv(os.path.join(exp018_dir, "submission_stack_T1.0.csv"))
    # 개별 모델 probs가 필요 — 없으면 단순 블렌딩
    # 여기서는 struct_probs와 exp018_stack을 블렌딩

    exp018_probs = exp018_stack_sub[["unstable_prob", "stable_prob"]].values

    for w in [0.3, 0.4, 0.5]:
        blended = w * struct_probs + (1 - w) * exp018_probs
        for t in [0.7, 0.8]:
            log_b = np.log(np.clip(blended, 1e-15, 1))
            scaled = log_b / t
            final = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            sub = pd.DataFrame({"id": ids, "unstable_prob": final[:, 0], "stable_prob": final[:, 1]})
            sub.to_csv(os.path.join(SAVE_DIR,
                       f"submission_final_struct{int(w*100)}_exp018{int((1-w)*100)}_T{t:.1f}.csv"),
                       index=False)
            print(f"struct {int(w*100)}% + exp018 {int((1-w)*100)}% T={t:.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
