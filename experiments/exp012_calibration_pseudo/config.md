# Exp 012: Temperature Scaling + Pseudo-Labeling

## 개요

두 가지 기법을 결합:
1. **Temperature Scaling**: 모델의 과한 확신/불확신을 보정 → LogLoss 최적화
2. **Pseudo-Labeling**: test 데이터 중 확신 높은 예측을 가짜 라벨로 사용해 재학습 → 도메인 갭 감소

## Step 1: Temperature Scaling

- exp011 모델의 logits에 온도(T) 적용: softmax(logits / T)
- T < 1.0: 확률이 더 극단적 (확신 증가)
- T > 1.0: 확률이 더 부드러움 (확신 감소)
- CV 최적 T=0.5 (0.0490 → 0.0057) 하지만 과적합 위험
- 실전에서는 T=0.7~0.8 추천

## Step 2: Pseudo-Labeling

- exp011 예측 중 확신도 >= 90%인 test 샘플을 가짜 라벨로 사용
- 847개 pseudo-label 추가 (unstable: 438, stable: 409)
- 총 학습 데이터: 1,100 → 1,947 (77% 증가!)

## 결과

| Fold | exp011 (원본만) | exp012 (pseudo 추가) |
|------|---------------|---------------------|
| 1 | 0.0491 | 0.0479 |
| 2 | 0.0446 | 0.0471 |
| 3 | 0.0458 | 0.0472 |
| 4 | 0.0517 | 0.0488 |
| 5 | 0.0539 | 0.0449 |
| **Mean** | **0.0490** | **0.0472** |

## Dacon 제출 결과

- **Dacon LogLoss: 0.04003**
- 이전 최고(exp010): 0.1111 → **72% 개선!**

## 제출 파일

| 파일 | Temperature | 설명 |
|------|------------|------|
| submission_pseudo_T0.7.csv | 0.7 | 추천 1순위 |
| submission_pseudo_T0.8.csv | 0.8 | 추천 2순위 |
| submission_pseudo_T1.0.csv | 1.0 | 안전한 선택 |
| submission_pseudo_T0.5.csv | 0.5 | 공격적 (위험) |
