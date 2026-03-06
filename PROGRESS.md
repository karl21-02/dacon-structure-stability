# 구조물 안정성 물리 추론 AI 경진대회 - 진행 기록

## 실험 요약

| 실험 | 모델 | 입력 | Dev LogLoss | Dev Acc | 순위 |
|------|------|------|------------|---------|------|
| exp001 | ResNet50 | front | 0.4178 | 82% | 70등 |
| exp002 | ResNet50 (shared backbone) | front + top | 0.5956 | 72% | 미제출 |
| **exp003** | **ConvNeXt-Small** | **front** | **0.1239** | **96%** | **31등** |

각 실험의 상세 기록은 `experiments/exp00X/config.md` 참조.

---

## 핵심 인사이트

1. **Backbone이 가장 큰 영향**: ResNet50 → ConvNeXt-Small로 LogLoss 70% 개선
2. **강한 Augmentation 필수**: 도메인 갭(고정→무작위 환경) 대응에 효과적
3. **Multi-View는 단순 결합으로는 역효과**: shared backbone + concat은 오히려 성능 하락
4. **Label Smoothing 효과적**: 경계 케이스에 대한 과도한 확신 방지
5. **Overfitting 주의**: 데이터 1,000개로 적어서 정규화 필수

## 환경 정보

| 항목 | 값 |
|------|-----|
| GPU | RTX 5090 x2 (32GB each, 여유 ~6GB) |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu128 |
| timm | 1.0.25 |
| venv | `.venv/` |

## 프로젝트 구조

```
dacon/
├── PLAN.md                          # 전략 계획서
├── PROGRESS.md                      # 이 문서
├── data/open (1)/                   # 데이터
├── experiments/
│   ├── exp001_resnet50_front/       # ResNet50, front only
│   │   ├── train.py, best_model.pth, submission.csv, config.md
│   ├── exp002_resnet50_multiview/   # ResNet50, front+top (shared)
│   │   ├── train.py, config.md
│   └── exp003_convnext_front/       # ConvNeXt-Small, front only
│       ├── train.py, inference.py, best_model.pth, submission.csv, config.md
└── .venv/                           # 가상환경
```

## 다음 단계

- [ ] ConvNeXt + Multi-View (front + top)
- [ ] 더 큰 backbone (ConvNeXt-Base, EVA-02)
- [ ] K-Fold CV + 앙상블
- [ ] TTA (Test Time Augmentation)
- [ ] 확률 보정 (Temperature Scaling)
