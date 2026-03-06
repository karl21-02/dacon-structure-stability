# Exp 003: ConvNeXt-Small + 강한 Augmentation (Front Only)

## 설정

| 항목 | 값 |
|------|-----|
| Model | ConvNeXt-Small (ImageNet-22k → ImageNet-1k pretrained) |
| Model ID | `convnext_small.fb_in22k_ft_in1k` |
| Input | front.png only (224x224) |
| Batch Size | 8 (+ Gradient Accumulation 4 = effective 32) |
| LR | 3e-4 |
| Optimizer | AdamW (weight_decay=1e-2) |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Epochs | 30 (Early Stopping patience=10, stopped at 20) |
| Mixed Precision | FP16 (torch.amp) |

## Augmentation (강화)

- Resize(256) → RandomCrop(224)
- RandomHorizontalFlip
- RandomVerticalFlip
- RandomRotation(15)
- RandomAffine(translate=0.1, scale=0.9~1.1)
- ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
- GaussianBlur(kernel=3, sigma=0.1~2.0)
- RandomErasing(p=0.2)

## 결과

| 지표 | 값 |
|------|-----|
| **Best Dev LogLoss** | **0.1239** |
| **Best Dev Accuracy** | **96%** |
| Best Epoch | 10 / 20 (early stop) |

## 학습 로그

```
Epoch 01 | Train Loss: 0.6024 Acc: 0.7240 | Dev LogLoss: 0.4023 Acc: 0.8400
Epoch 02 | Train Loss: 0.3631 Acc: 0.9020 | Dev LogLoss: 0.3303 Acc: 0.8500
Epoch 03 | Train Loss: 0.3110 Acc: 0.9360 | Dev LogLoss: 0.2126 Acc: 0.9100
Epoch 04 | Train Loss: 0.2715 Acc: 0.9580 | Dev LogLoss: 0.2167 Acc: 0.9200
Epoch 05 | Train Loss: 0.2671 Acc: 0.9600 | Dev LogLoss: 0.1647 Acc: 0.9500
Epoch 06 | Train Loss: 0.2547 Acc: 0.9670 | Dev LogLoss: 0.3101 Acc: 0.8700
Epoch 07 | Train Loss: 0.2882 Acc: 0.9490 | Dev LogLoss: 0.1794 Acc: 0.9600
Epoch 08 | Train Loss: 0.2534 Acc: 0.9710 | Dev LogLoss: 0.4009 Acc: 0.8300
Epoch 09 | Train Loss: 0.2466 Acc: 0.9710 | Dev LogLoss: 0.1696 Acc: 0.9600
Epoch 10 | Train Loss: 0.2386 Acc: 0.9700 | Dev LogLoss: 0.1239 Acc: 0.9600 <- Best
Epoch 11 | Train Loss: 0.2434 Acc: 0.9750 | Dev LogLoss: 0.2864 Acc: 0.8900
...
Epoch 20 | Early stopping
```

## exp001 대비 개선 요인

1. **ConvNeXt-Small**: ResNet50 대비 더 강력한 feature 추출 (IN-22k pretrained)
2. **강한 Augmentation**: 도메인 갭(광원/카메라 변동) 시뮬레이션으로 overfitting 완화
3. **Label Smoothing 0.1**: 경계 케이스에 대한 과도한 확신 방지
4. **Weight Decay 1e-2**: 더 강한 정규화
5. **Early Stopping**: overfitting 전 학습 중단

## 비교

| 실험 | Dev LogLoss | Dev Acc | 비고 |
|------|------------|---------|------|
| exp001 ResNet50 front | 0.4178 | 82% | 베이스라인 |
| exp002 ResNet50 multi-view | 0.5956 | 72% | 성능 하락 |
| **exp003 ConvNeXt front** | **0.1239** | **96%** | **70% 개선** |
