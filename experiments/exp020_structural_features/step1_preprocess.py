"""
Step 1: 배경 제거 + 구조 피처 추출
- 체커보드 배경 제거 → 구조물만 남긴 이미지 저장
- 기하학적 피처 추출 → CSV 저장
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def remove_background(img):
    """체커보드 배경을 검정으로 마스킹"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 체커보드: 연한 파랑 or 흰색
    mask_blue = (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130) & (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 150)
    mask_white = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 200)
    bg_mask = mask_blue | mask_white

    # 배경 → 검정
    result = img.copy()
    result[bg_mask] = 0
    return result, ~bg_mask


def extract_features(img, struct_mask, view):
    """구조물 마스크에서 기하학적 피처 추출"""
    ys, xs = np.where(struct_mask)

    if len(ys) < 10:
        return {f"{view}_pixels": 0, f"{view}_h": 0, f"{view}_w": 0,
                f"{view}_hw_ratio": 0, f"{view}_lean": 0,
                f"{view}_top_base_ratio": 0, f"{view}_cy_ratio": 0.5,
                f"{view}_cx_ratio": 0.5, f"{view}_fill_ratio": 0,
                f"{view}_symmetry": 0}

    h = ys.max() - ys.min()
    w = xs.max() - xs.min()
    cy = ys.mean()
    cx = xs.mean()
    img_h, img_w = struct_mask.shape

    # lean: 무게중심이 베이스 중심에서 얼마나 벗어났는지
    base_ys_thresh = ys.max() - max(h * 0.2, 5)
    base_xs = xs[ys > base_ys_thresh]
    base_cx = base_xs.mean() if len(base_xs) > 0 else cx
    lean = abs(cx - base_cx)

    # top vs base 너비 비율
    top_ys_thresh = ys.min() + max(h * 0.2, 5)
    top_xs = xs[ys < top_ys_thresh]
    top_w = (top_xs.max() - top_xs.min()) if len(top_xs) > 1 else 0
    base_w = (base_xs.max() - base_xs.min()) if len(base_xs) > 1 else 1

    # bounding box 내 fill ratio
    fill_ratio = len(ys) / max(h * w, 1)

    # 좌우 대칭성
    left_pixels = (xs < cx).sum()
    right_pixels = (xs >= cx).sum()
    symmetry = min(left_pixels, right_pixels) / max(left_pixels, right_pixels, 1)

    return {
        f"{view}_pixels": len(ys),
        f"{view}_h": h,
        f"{view}_w": w,
        f"{view}_hw_ratio": h / max(w, 1),
        f"{view}_lean": lean,
        f"{view}_top_base_ratio": top_w / max(base_w, 1),
        f"{view}_cy_ratio": cy / img_h,
        f"{view}_cx_ratio": cx / img_w,
        f"{view}_fill_ratio": fill_ratio,
        f"{view}_symmetry": symmetry,
    }


def process_sample(sample_id, data_dir, save_masked_dir):
    """한 샘플의 front/top 처리"""
    features = {"id": sample_id}

    for view in ["front", "top"]:
        img_path = os.path.join(data_dir, sample_id, f"{view}.png")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        masked_img, struct_mask = remove_background(img)

        # 배경 제거 이미지 저장
        out_dir = os.path.join(save_masked_dir, sample_id)
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{view}.png"), masked_img)

        # 피처 추출
        feat = extract_features(img, struct_mask, view)
        features.update(feat)

    return features


def main():
    masked_dir = os.path.join(SAVE_DIR, "masked_images")

    all_features = []

    for split in ["train", "dev", "test"]:
        csv_name = f"{split}.csv" if split != "test" else "sample_submission.csv"
        csv_path = os.path.join(DATA_ROOT, csv_name)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        data_dir = os.path.join(DATA_ROOT, split)
        split_masked_dir = os.path.join(masked_dir, split)

        print(f"Processing {split} ({len(df)} samples)...")

        for i, row in df.iterrows():
            sample_id = row["id"]
            features = process_sample(sample_id, data_dir, split_masked_dir)

            if "label" in row:
                features["label"] = row["label"]

            all_features.append(features)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(df)} done")

        print(f"  {split} complete!")

    # 저장
    feat_df = pd.DataFrame(all_features)
    save_path = os.path.join(SAVE_DIR, "structural_features.csv")
    feat_df.to_csv(save_path, index=False)
    print(f"\nSaved {len(feat_df)} samples to {save_path}")

    # 통계
    labeled = feat_df[feat_df["label"].notna()]
    print("\n=== Feature comparison (front) ===")
    for col in [c for c in feat_df.columns if c.startswith("front_")]:
        u = labeled[labeled["label"] == "unstable"][col]
        s = labeled[labeled["label"] == "stable"][col]
        print(f"{col:25s}: unstable={u.mean():.2f}±{u.std():.2f}  stable={s.mean():.2f}±{s.std():.2f}")


if __name__ == "__main__":
    main()
