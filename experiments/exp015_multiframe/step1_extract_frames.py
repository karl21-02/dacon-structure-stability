"""
Step 1: 영상에서 초반 프레임 추출 → 학습 이미지 증강

각 train 샘플의 simulation.mp4에서 초반 프레임(1~5)을 추출해서
front_frame1.png ~ front_frame5.png로 저장.
(frame 0 ≈ front.png이므로 frame 1부터)

결과: 샘플당 6장 (front.png + 5프레임) → 학습 데이터 6배
"""
import os
import cv2
import pandas as pd

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_TO_EXTRACT = [1, 2, 3, 4, 5]  # 추출할 프레임 번호 (0은 front.png와 거의 동일)


def extract_frames(video_path, output_dir, sample_id):
    """영상에서 초반 프레임을 추출해서 PNG로 저장"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: Cannot open {video_path}")
        return []

    saved = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in FRAMES_TO_EXTRACT:
            save_path = os.path.join(output_dir, sample_id, f"front_frame{frame_idx}.png")
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

    print(f"\nComplete! {total_saved} frames extracted from {len(train_csv)} videos")
    print(f"Frames per sample: {FRAMES_TO_EXTRACT}")


if __name__ == "__main__":
    main()
