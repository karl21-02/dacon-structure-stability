# Exp 002: Multi-View ResNet50 (Shared Backbone)
- Model: ResNet50 shared backbone (front + top)
- Batch: 16, LR: 1e-4, Epochs: 20
- Augmentation: HorizontalFlip, ColorJitter(0.3)
- Best Dev LogLoss: 0.5956 (Epoch 7, Acc 72%)
- 결과: exp001보다 성능 하락. shared backbone이 시점별 특성 구분 못함
