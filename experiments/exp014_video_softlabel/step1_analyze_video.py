"""
Step 1: 영상 분석 → 움직임 측정 → Soft Label 생성

영상(simulation.mp4)에서 구조물이 얼마나 무너졌는지 측정해서
hard label(0/1) 대신 soft label(0.0~1.0)을 만든다.

예:
  - 크게 무너진 구조물 → unstable_soft = 0.99
  - 살짝 흔들린 구조물 → unstable_soft = 0.65
  - 완전히 안정된 구조물 → unstable_soft = 0.02
"""
import os
import cv2
import numpy as np
import pandas as pd

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def measure_motion(video_path):
    """영상에서 구조물의 움직임 정도를 측정"""
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        return None

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY).astype(float)

    max_diff = 0.0
    frame_diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        diff = np.abs(first_gray - gray).mean()
        frame_diffs.append(diff)
        max_diff = max(max_diff, diff)

    cap.release()

    last_diff = frame_diffs[-1] if frame_diffs else 0
    mean_diff = np.mean(frame_diffs) if frame_diffs else 0

    return {
        "max_diff": max_diff,
        "last_diff": last_diff,
        "mean_diff": mean_diff,
    }


def create_soft_labels(df):
    """
    움직임 정도를 0~1 soft label로 변환

    unstable 샘플: max_diff가 클수록 → 1.0에 가까움
    stable 샘플: max_diff가 작을수록 → 0.0에 가까움
    """
    # unstable/stable 각각의 max_diff 분포로 정규화
    unstable_mask = df["label"] == "unstable"
    stable_mask = df["label"] == "stable"

    # unstable: max_diff 기준으로 0.55~0.99 사이로 매핑
    unstable_diffs = df.loc[unstable_mask, "max_diff"]
    if len(unstable_diffs) > 0:
        u_min, u_max = unstable_diffs.min(), unstable_diffs.max()
        if u_max > u_min:
            normalized = (unstable_diffs - u_min) / (u_max - u_min)
        else:
            normalized = pd.Series(0.5, index=unstable_diffs.index)
        df.loc[unstable_mask, "soft_label"] = 0.55 + normalized * 0.44  # 0.55 ~ 0.99

    # stable: max_diff 기준으로 0.01~0.15 사이로 매핑
    stable_diffs = df.loc[stable_mask, "max_diff"]
    if len(stable_diffs) > 0:
        s_min, s_max = stable_diffs.min(), stable_diffs.max()
        if s_max > s_min:
            normalized = (stable_diffs - s_min) / (s_max - s_min)
        else:
            normalized = pd.Series(0.5, index=stable_diffs.index)
        df.loc[stable_mask, "soft_label"] = 0.01 + normalized * 0.14  # 0.01 ~ 0.15

    return df


def main():
    # Train 데이터 분석
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")

    results = []
    for i, row in train_csv.iterrows():
        sid = row["id"]
        video_path = os.path.join(DATA_ROOT, "train", sid, "simulation.mp4")
        if os.path.exists(video_path):
            motion = measure_motion(video_path)
            if motion:
                motion["id"] = sid
                motion["label"] = row["label"]
                results.append(motion)
        if (i + 1) % 100 == 0:
            print(f"Train: {i+1}/1000 done")

    # Dev 데이터 분석
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    for i, row in dev_csv.iterrows():
        sid = row["id"]
        video_path = os.path.join(DATA_ROOT, "dev", sid, "simulation.mp4")
        if os.path.exists(video_path):
            motion = measure_motion(video_path)
            if motion:
                motion["id"] = sid
                motion["label"] = row["label"]
                results.append(motion)

    print(f"Dev: {len(dev_csv)} done")

    df = pd.DataFrame(results)

    # 통계 출력
    print(f"\n=== 움직임 통계 ===")
    for label in ["unstable", "stable"]:
        sub = df[df.label == label]
        print(f"\n{label} ({len(sub)}개):")
        print(f"  max_diff: mean={sub.max_diff.mean():.2f}, min={sub.max_diff.min():.2f}, max={sub.max_diff.max():.2f}")
        print(f"  last_diff: mean={sub.last_diff.mean():.2f}, min={sub.last_diff.min():.2f}, max={sub.last_diff.max():.2f}")

    # 경계 케이스 분석
    unstable = df[df.label == "unstable"]
    stable = df[df.label == "stable"]
    print(f"\n=== 경계 케이스 ===")
    print(f"unstable 중 움직임 적은 것 (max_diff < 5): {(unstable.max_diff < 5).sum()}개")
    print(f"unstable 중 움직임 적은 것 (max_diff < 10): {(unstable.max_diff < 10).sum()}개")
    print(f"stable 중 움직임 있는 것 (max_diff > 3): {(stable.max_diff > 3).sum()}개")

    # Soft label 생성
    df = create_soft_labels(df)

    print(f"\n=== Soft Label 분포 ===")
    print(f"unstable soft_label: mean={df.loc[df.label=='unstable','soft_label'].mean():.3f}")
    print(f"stable soft_label: mean={df.loc[df.label=='stable','soft_label'].mean():.3f}")

    # 저장
    save_path = os.path.join(SAVE_DIR, "soft_labels.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSoft labels saved to {save_path}")
    print(f"Total: {len(df)} samples")


if __name__ == "__main__":
    main()
