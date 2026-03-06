# Exp 009: EfficientNet-B3 5-Fold

## 설정

- EfficientNet-B3: ConvNeXt와 완전히 다른 아키텍처
- IMG_SIZE: 300 (EfficientNet-B3 기본 입력 크기)
- 나머지 설정은 exp006과 동일 (5-Fold, train+dev, label_smoothing 등)
- 추론: 5-Fold x 5-TTA = 25개 예측 평균

## 목적

- ConvNeXt와 다른 관점으로 이미지를 분석하는 모델 확보
- 최종: ConvNeXt(exp006) + EfficientNet(exp009) 다중 아키텍처 앙상블

## 결과

| Fold | Val LogLoss | Val Acc |
|------|------------|---------|
| 1 | 0.0779 | ~98.6% |
| 2 | 0.0665 | ~99.6% |
| 3 | 0.0792 | ~99.1% |
| 4 | 0.0853 | ~99.1% |
| 5 | 0.0746 | ~99.1% |
| **Mean** | **0.0767** | |

- 추론: 5-Fold x 5-TTA = 25개 예측 평균
- 다중 아키텍처 앙상블: ConvNeXt(60%) + EfficientNet(40%) 가중 평균 생성
- 동일 가중치(50/50) 앙상블도 생성
- ConvNeXt vs EfficientNet 예측이 198개 샘플에서 크게 다름 → 앙상블 가치 높음

## 비교

| 실험 | 모델 | CV LogLoss |
|------|------|------------|
| exp006 | ConvNeXt-Small (224) | 0.0522 |
| exp008 | ConvNeXt-Small (384) | 0.0622 |
| **exp009** | **EfficientNet-B3 (300)** | **0.0767** |

## 관찰

- EfficientNet 단독 CV는 ConvNeXt보다 높음 (0.0767 vs 0.0522)
- 하지만 다른 아키텍처로 다른 관점의 예측 → 앙상블 다양성 확보
- 92개 샘플에서 두 모델의 예측 차이 > 0.2 → 이런 경계 케이스에서 앙상블이 도움
