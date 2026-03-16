"""
Step 1: 전체 train/dev 영상에서 물리 특징 추출
- Optical flow 기반 변위 측정
- 첫 프레임 대비 누적 변화
- FFT 주파수 분석
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    ret, prev = cap.read()
    if not ret:
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    first_gray = prev_gray.copy()

    displacements = []
    cumulative_diffs = []

    for _ in range(299):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 프레임 간 optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        displacements.append(mag.mean())

        # 첫 프레임 대비 누적 변화
        diff = np.abs(gray.astype(float) - first_gray.astype(float)).mean()
        cumulative_diffs.append(diff)

        prev_gray = gray

    cap.release()

    d = np.array(displacements)
    c = np.array(cumulative_diffs)

    # FFT 주파수 분석
    fft = np.fft.fft(d)
    freqs = np.fft.fftfreq(len(d), 1 / 30)
    power = np.abs(fft[: len(fft) // 2]) ** 2

    # 지배 주파수 (DC 제외)
    dominant_freq = freqs[1 : len(fft) // 2][np.argmax(power[1:])]

    # 상위 3개 주파수의 에너지 비율
    sorted_power = np.sort(power[1:])[::-1]
    top3_ratio = sorted_power[:3].sum() / (power[1:].sum() + 1e-10)

    return {
        # 프레임간 변위 통계
        "disp_mean": d.mean(),
        "disp_max": d.max(),
        "disp_std": d.std(),
        "disp_q75": np.percentile(d, 75),
        "disp_q25": np.percentile(d, 25),
        # 시간 구간별 변위
        "disp_first30": d[:30].mean(),
        "disp_last30": d[-30:].mean(),
        "disp_mid": d[100:200].mean(),
        # 누적 변화
        "cum_diff_max": c.max(),
        "cum_diff_last": c[-1],
        "cum_diff_slope": np.polyfit(np.arange(len(c)), c, 1)[0],
        "cum_diff_std": c.std(),
        # 주파수
        "dominant_freq": dominant_freq,
        "total_energy": power.sum(),
        "top3_freq_ratio": top3_ratio,
        # 변위 변화율 (가속도 느낌)
        "disp_accel_mean": np.abs(np.diff(d)).mean(),
        "disp_accel_max": np.abs(np.diff(d)).max(),
    }


def main():
    results = []

    for split in ["train", "dev"]:
        csv_path = os.path.join(DATA_ROOT, f"{split}.csv")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        split_dir = os.path.join(DATA_ROOT, split)

        print(f"Processing {split} ({len(df)} samples)...")

        for i, row in df.iterrows():
            sample_id = row["id"]
            video_path = os.path.join(split_dir, sample_id, "simulation.mp4")

            if not os.path.exists(video_path):
                print(f"  WARNING: {video_path} not found, skipping")
                continue

            features = extract_features(video_path)
            if features is None:
                print(f"  WARNING: failed to process {sample_id}")
                continue

            features["id"] = sample_id
            features["label"] = row["label"]
            results.append(features)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(df)} done")

        print(f"  {split} complete!")

    # 저장
    feat_df = pd.DataFrame(results)
    cols = ["id", "label"] + [c for c in feat_df.columns if c not in ["id", "label"]]
    feat_df = feat_df[cols]

    save_path = os.path.join(SAVE_DIR, "physics_features.csv")
    feat_df.to_csv(save_path, index=False)
    print(f"\nSaved {len(feat_df)} samples to {save_path}")

    # 통계 비교
    print("\n=== Feature comparison ===")
    for col in feat_df.columns:
        if col in ["id", "label"]:
            continue
        unstable = feat_df[feat_df["label"] == "unstable"][col]
        stable = feat_df[feat_df["label"] == "stable"][col]
        print(f"{col:20s}: unstable={unstable.mean():.4f}±{unstable.std():.4f}  "
              f"stable={stable.mean():.4f}±{stable.std():.4f}")


if __name__ == "__main__":
    main()
