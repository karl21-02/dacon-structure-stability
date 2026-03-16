"""
Step 1: 전략적 프레임 추출 (1, 10, 20, 30)

exp015는 프레임 1~5 (front.png와 거의 동일)
exp016은 프레임 1, 10, 20, 30 (더 넓은 간격)
- stable: 모든 프레임 동일 → 자연 augmentation
- unstable: 후반 프레임에서 기울기 시작 → 불안정 초기 징후 학습
"""
import os
import cv2
import pandas as pd

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
FRAMES_TO_EXTRACT = [1, 10, 20, 30]


def extract_frames(video_path, output_dir, sample_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    saved = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in FRAMES_TO_EXTRACT:
            save_path = os.path.join(output_dir, sample_id, f"front_frame{frame_idx}.png")
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, frame)
            saved.append(frame_idx)
        if frame_idx > max(FRAMES_TO_EXTRACT):
            break
        frame_idx += 1

    cap.release()
    return saved


def main():
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    train_dir = os.path.join(DATA_ROOT, "train")

    total_saved = 0
    for i, row in train_csv.iterrows():
        sid = row["id"]
        video_path = os.path.join(train_dir, sid, "simulation.mp4")
        if os.path.exists(video_path):
            saved = extract_frames(video_path, train_dir, sid)
            total_saved += len(saved)
        if (i + 1) % 200 == 0:
            print(f"{i+1}/1000 done ({total_saved} frames saved)")

    print(f"\nComplete! {total_saved} frames extracted")
    print(f"Frames: {FRAMES_TO_EXTRACT}")


if __name__ == "__main__":
    main()
