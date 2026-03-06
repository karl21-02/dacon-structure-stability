# Exp 011: Dual View (front + top) Side-by-Side

## 설정

- front.png와 top.png를 가로로 나란히 붙여서 하나의 이미지로 입력 (224x448)
- ConvNeXt-Small backbone (입력 크기에 유연)
- 5-Fold, train+dev 1,100샘플
- augmentation에서 좌우반전 제거 (front/top 위치 관계 보존)

## 아이디어

- front view: 높이, 기울기, 쌓인 형태
- top view: 밑면 면적, 대칭성, 블록 배치
- 두 뷰를 동시에 보면 3D 구조를 추론 가능
- exp002 실패 원인(두 backbone 분리 + concat)을 개선: 하나의 backbone이 두 뷰 관계를 자연스럽게 학습

## 결과

| Fold | Val LogLoss | Val Acc |
|------|------------|---------|
| 1 | 0.0491 | 99.1% |
| 2 | 0.0446 | 99.6% |
| 3 | 0.0458 | 100% |
| 4 | 0.0517 | 99.6% |
| 5 | 0.0539 | 99.6% |
| **Mean** | **0.0490** | |

- Fold 3 초기 학습 실패(0.6934) → 시드 변경 재학습으로 0.0458 달성
- 추론: 5 folds x 3 TTA = 15개 예측 평균

## 비교

| 실험 | 입력 | CV LogLoss |
|------|------|------------|
| exp006 | front만 (224x224) | 0.0522 |
| **exp011** | **front+top (224x448)** | **0.0490 (6% 개선)** |
