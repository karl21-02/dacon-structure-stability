"""
Exp021: Platt Scaling 확률 보정

Temperature Scaling: softmax(logit / T) → 파라미터 1개
Platt Scaling: sigmoid(a * logit + b) → 파라미터 2개
→ 확률의 기울기(a)와 편향(b)을 모두 조절 가능

추가로 Isotonic Regression도 시도 (비모수 보정)
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp020_structural_features"))
from step2_train import (StructuralClassifier, StructuralDataset, FEAT_COLS,
                         IMG_SIZE, N_FOLDS, MODEL_NAME, N_FEATS,
                         get_transforms, expand_with_frames)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP020_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp020_structural_features")


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

        feats = np.array([row[c] for c in FEAT_COLS], dtype=np.float32)
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


@torch.no_grad()
def get_val_logits(model, loader):
    """validation용: labels 반환"""
    model.eval()
    all_logits, all_labels = [], []
    for images, feats, labels in loader:
        images, feats = images.to(DEVICE), feats.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images, feats)
        all_logits.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true * np.log(pred), axis=1))


def main():
    print(f"Device: {DEVICE}")

    # 데이터 준비
    feat_df = pd.read_csv(os.path.join(EXP020_DIR, "structural_features.csv"))
    feat_mean = np.load(os.path.join(EXP020_DIR, "feat_mean.npy"))
    feat_std = np.load(os.path.join(EXP020_DIR, "feat_std.npy"))

    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    train_csv["img_dir"] = os.path.join(DATA_ROOT, "train")
    dev_csv["img_dir"] = os.path.join(DATA_ROOT, "dev")

    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)
    all_real = all_real.merge(feat_df, on="id", how="left", suffixes=("", "_feat"))
    if "label_feat" in all_real.columns:
        all_real.drop(columns=["label_feat"], inplace=True)

    masked_train = os.path.join(EXP020_DIR, "masked_images", "train")
    masked_dev = os.path.join(EXP020_DIR, "masked_images", "dev")
    masked_test = os.path.join(EXP020_DIR, "masked_images", "test")

    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    tfm_val = get_transforms(False)

    # Step 1: 전체 OOF logits 수집
    print("\n=== Step 1: OOF logits 수집 ===")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    oof_logits = np.zeros((len(all_real), 2))
    oof_labels = np.zeros(len(all_real))

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, real_labels)):
        val_df = all_real.iloc[val_idx].copy()
        val_df["front_file"] = "front.png"

        # val용 dataset (label 반환)
        val_ds = StructuralDataset(val_df, masked_dev if val_df.iloc[0]["id"].startswith("DEV") else masked_train,
                                   dev_dir if val_df.iloc[0]["id"].startswith("DEV") else train_dir,
                                   feat_mean, feat_std, transform=tfm_val)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

        ckpt = os.path.join(EXP020_DIR, f"best_struct_fold{fold}.pth")
        model = StructuralClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))

        logits, labels = get_val_logits(model, val_loader)
        oof_logits[val_idx] = logits
        oof_labels[val_idx] = labels
        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1} OOF collected")

    # Step 2: 보정 방법 비교
    print("\n=== Step 2: 보정 방법 비교 ===")
    true_onehot = np.zeros_like(oof_logits)
    true_onehot[np.arange(len(oof_labels)), oof_labels.astype(int)] = 1

    # 원본 (T=1.0)
    probs_t1 = np.exp(oof_logits) / np.exp(oof_logits).sum(axis=1, keepdims=True)
    print(f"Temperature T=1.0: {logloss(true_onehot, probs_t1):.6f}")

    # Temperature Scaling
    best_t, best_t_score = 1.0, float("inf")
    for t in np.arange(0.1, 2.0, 0.01):
        scaled = oof_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        score = logloss(true_onehot, probs)
        if score < best_t_score:
            best_t, best_t_score = t, score
    print(f"Best Temperature T={best_t:.2f}: {best_t_score:.6f}")

    # Platt Scaling: logistic regression on logits
    # unstable logit만 사용 (binary)
    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    platt.fit(oof_logits[:, 0:1], oof_labels)
    platt_probs_raw = platt.predict_proba(oof_logits[:, 0:1])
    # platt은 [stable_prob, unstable_prob] 순서일 수 있음 → 확인
    if platt.classes_[0] == 0:  # 0=unstable
        platt_probs = platt_probs_raw
    else:
        platt_probs = platt_probs_raw[:, ::-1]
    print(f"Platt Scaling: {logloss(true_onehot, platt_probs):.6f}")

    # Isotonic Regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs_t1[:, 0], (oof_labels == 0).astype(float))  # unstable prob 보정
    iso_unstable = iso.predict(probs_t1[:, 0])
    iso_probs = np.stack([iso_unstable, 1 - iso_unstable], axis=1)
    print(f"Isotonic Regression: {logloss(true_onehot, iso_probs):.6f}")

    # Step 3: 최고 보정 방법으로 test 추론
    print("\n=== Step 3: Test 추론 ===")
    test_feat = feat_df[feat_df.id.str.startswith("TEST")].copy()
    sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"), encoding="utf-8-sig")
    test_feat = sample_sub[["id"]].merge(test_feat, on="id", how="left")

    test_ds = TestStructuralDataset(test_feat, masked_test, test_dir, feat_mean, feat_std, transform=tfm_val)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # 5 fold 평균 logits
    all_test_logits = []
    for fold in range(N_FOLDS):
        ckpt = os.path.join(EXP020_DIR, f"best_struct_fold{fold}.pth")
        model = StructuralClassifier(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        ids, logits = get_logits(model, test_loader)
        all_test_logits.append(logits)
        del model; torch.cuda.empty_cache()
        print(f"Fold {fold+1} test inference done")

    avg_test_logits = np.mean(all_test_logits, axis=0)

    # Temperature Scaling submission
    for t in [best_t, 0.3, 0.4, 0.5, 0.7]:
        scaled = avg_test_logits / t
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
        sub.to_csv(os.path.join(SAVE_DIR, f"submission_temp_T{t:.2f}.csv"), index=False)
        print(f"Temp T={t:.2f}: range [{probs[:,0].min():.6f}, {probs[:,0].max():.6f}]")

    # Platt Scaling submission
    platt_test_probs_raw = platt.predict_proba(avg_test_logits[:, 0:1])
    if platt.classes_[0] == 0:
        platt_test_probs = platt_test_probs_raw
    else:
        platt_test_probs = platt_test_probs_raw[:, ::-1]
    sub = pd.DataFrame({"id": ids, "unstable_prob": platt_test_probs[:, 0], "stable_prob": platt_test_probs[:, 1]})
    sub.to_csv(os.path.join(SAVE_DIR, "submission_platt.csv"), index=False)
    print(f"Platt: range [{platt_test_probs[:,0].min():.6f}, {platt_test_probs[:,0].max():.6f}]")

    # Isotonic submission
    test_probs_t1 = np.exp(avg_test_logits) / np.exp(avg_test_logits).sum(axis=1, keepdims=True)
    iso_test_unstable = iso.predict(test_probs_t1[:, 0])
    iso_test_probs = np.stack([iso_test_unstable, 1 - iso_test_unstable], axis=1)
    sub = pd.DataFrame({"id": ids, "unstable_prob": iso_test_probs[:, 0], "stable_prob": iso_test_probs[:, 1]})
    sub.to_csv(os.path.join(SAVE_DIR, "submission_isotonic.csv"), index=False)
    print(f"Isotonic: range [{iso_test_probs[:,0].min():.6f}, {iso_test_probs[:,0].max():.6f}]")

    # exp018과 블렌딩 + Platt
    print("\n=== Step 4: exp018 블렌딩 + Platt ===")
    exp018_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exp018_triple_stack")
    exp018_sub = os.path.join(exp018_dir, "submission_stack_T0.7.csv")
    if os.path.exists(exp018_sub):
        e18 = pd.read_csv(exp018_sub)
        for w in [0.3, 0.4, 0.5]:
            blended_unstable = w * platt_test_probs[:, 0] + (1-w) * e18["unstable_prob"].values
            blended_stable = w * platt_test_probs[:, 1] + (1-w) * e18["stable_prob"].values
            sub = pd.DataFrame({"id": ids, "unstable_prob": blended_unstable, "stable_prob": blended_stable})
            sub.to_csv(os.path.join(SAVE_DIR, f"submission_platt{int(w*100)}_exp018{int((1-w)*100)}.csv"), index=False)
            print(f"Platt {int(w*100)}% + exp018 {int((1-w)*100)}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
