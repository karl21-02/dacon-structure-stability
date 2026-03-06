# Exp 005: ConvNeXt-Base (Front Only)

## 설정

- exp003과 동일, backbone만 ConvNeXt-Small → ConvNeXt-Base로 변경
- Model ID: `convnext_base.fb_in22k_ft_in1k`
- 나머지 설정 동일 (batch 8, grad_accum 4, fp16, label_smoothing 0.1)

## 결과

| 지표 | 값 |
|------|-----|
| Best Dev LogLoss | 0.1963 |
| Best Dev Accuracy | 89% |
| Best Epoch | 7 / 17 (early stop) |

## 비교

| 실험 | Backbone | Dev LogLoss |
|------|----------|------------|
| exp003 | ConvNeXt-Small | **0.1239** |
| exp005 | ConvNeXt-Base | 0.1963 |

## 관찰

- ConvNeXt-Base가 오히려 성능 하락
- 모델이 커서 (89M vs 50M params) 데이터 1,000개로는 overfitting 심화
- dev score 변동이 큼 (0.19 ~ 0.65) → 학습 불안정
- **결론: 데이터 양 대비 ConvNeXt-Small이 적절한 크기**
