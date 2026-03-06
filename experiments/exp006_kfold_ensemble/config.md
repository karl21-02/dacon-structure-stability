# Exp 006: 5-Fold Ensemble (ConvNeXt-Small, train+dev)

## 설정

- exp003과 동일 설정, K-Fold 앙상블 적용
- 데이터: train 1,000 + dev 100 = 1,100 샘플 전체 활용
- 5-Fold Stratified Split (seed=42)
- 각 fold: 880 train / 220 val
- 최종 예측: 5개 모델 예측 평균

## 결과

| Fold | Val LogLoss | Val Acc |
|------|------------|---------|
| 1 | 0.0497 | ~99.5% |
| 2 | 0.0490 | 100% |
| 3 | 0.0631 | 99.1% |
| 4 | 0.0500 | 99.5% |
| 5 | 0.0493 | 100% |
| **Mean** | **0.0522** | |

## 비교

| 실험 | Dev LogLoss | 비고 |
|------|------------|------|
| exp003 ConvNeXt-Small | 0.1239 | 단일 모델 |
| **exp006 5-Fold Ensemble** | **0.0522 (CV)** | **57% 개선** |

## 관찰

- 모든 fold에서 99%+ 정확도 달성
- Fold 3이 상대적으로 높은 loss (0.0631) → 해당 val split에 어려운 샘플 포함 추정
- train+dev 합쳐서 데이터 10% 증가 (1000→1100)도 기여
- 5개 모델 평균으로 예측 안정화
