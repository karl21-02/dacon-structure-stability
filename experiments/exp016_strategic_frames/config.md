# Exp 016: Strategic Frames + SWA + Dual View + Pseudo-Label

## 개요

exp015 개선:
1. 프레임 1~5 → 1, 10, 20, 30 (불안정 초기 징후 포함)
2. 에폭 30→50 + SWA (Stochastic Weight Averaging)
3. Early stopping patience 10→15

## 핵심

- Frame 10, 20, 30에서 unstable 구조물은 살짝 기울기 시작
- Stable 구조물은 전 프레임 동일 → 순수 augmentation
- SWA: 학습 후반부(35~50 에폭) 가중치 평균 → 더 일반화된 모델

## 결과

- 대기 중
