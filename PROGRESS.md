# 구조물 안정성 물리 추론 AI 경진대회 - 진행 기록

## 실험 요약

| 실험 | 모델 | 방식 | CV LogLoss | Dacon 점수 |
|------|------|------|-----------|-----------|
| exp001 | ResNet50 | 단일, front | 0.4178 | 0.5016 (70등) |
| exp002 | ResNet50 | 멀티뷰 front+top | 0.5956 | 미제출 |
| exp003 | ConvNeXt-Small | 단일, front | 0.1239 | 0.1304 (31등) |
| exp004 | ConvNeXt-Small | exp003 + TTA | - | 미제출 |
| exp005 | ConvNeXt-Base | 단일, front | 0.1963 | 미제출 |
| exp006 | ConvNeXt-Small | 5-Fold 앙상블 | 0.0522 | 미제출 |
| exp007 | ConvNeXt-Small | exp006 + TTA | - | 미제출 |
| exp008 | ConvNeXt-Small | 5-Fold, 384px | 0.0622 | 미제출 |
| exp009 | EfficientNet-B3 | 5-Fold, 300px | 0.0767 | 미제출 |
| exp010 | ConvNeXt-Small | 10-Fold x 3-Seed | 0.0524 | 0.1111 |
| exp011 | ConvNeXt-Small | **듀얼뷰 (front+top)** | 0.0490 | 미제출 |
| **exp012** | **ConvNeXt-Small** | **듀얼뷰 + Pseudo + TempScale** | **0.0472** | **0.0400** |

- 1등 점수: 0.01062
- **현재 최고: exp012 (Dacon 0.0400)**
- 각 실험의 상세 기록은 `experiments/exp00X/config.md` 참조.

---

## 핵심 인사이트

1. **Backbone이 가장 큰 영향**: ResNet50 → ConvNeXt-Small로 LogLoss 70% 개선
2. **강한 Augmentation 필수**: 도메인 갭(고정→무작위 환경) 대응에 효과적
3. **Multi-View는 단순 결합으로는 역효과**: shared backbone + concat은 오히려 성능 하락
4. **Label Smoothing 효과적**: 경계 케이스에 대한 과도한 확신 방지
5. **Overfitting 주의**: 데이터 1,000개로 적어서 정규화 필수
6. **K-Fold 앙상블이 큰 효과**: 단일 모델 0.1239 → 5-Fold 0.0522 (58% 개선)
7. **큰 모델/해상도가 항상 좋진 않음**: ConvNeXt-Base(0.1963), 384px(0.0622) 모두 Small 224보다 나쁨
8. **다중 아키텍처 앙상블은 성능 차이가 크면 역효과**: EfficientNet(0.0767)을 섞으면 오히려 하락
9. **듀얼 뷰(front+top)가 효과적**: side-by-side로 붙이면 단일 backbone으로 두 뷰 관계 학습 가능
10. **Pseudo-Labeling으로 도메인 갭 해결**: test 데이터 847개를 가짜 라벨로 추가 → 1,100→1,947 (77% 증가)
11. **Temperature Scaling이 LogLoss에 큰 효과**: 확률 보정만으로 Dacon 점수 대폭 개선
12. **CV와 Dacon 점수 사이에 갭 존재**: CV 0.0472 vs Dacon 0.0400 — 도메인 갭 줄이면 실제 점수가 더 좋을 수 있음

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
├── PLAN.md                              # 전략 계획서
├── PROGRESS.md                          # 이 문서
├── data/open (1)/                       # 데이터
├── experiments/
│   ├── exp001_resnet50_front/           # ResNet50, front only
│   ├── exp002_resnet50_multiview/       # ResNet50, front+top
│   ├── exp003_convnext_front/           # ConvNeXt-Small, front only
│   ├── exp004_convnext_tta/             # exp003 + TTA
│   ├── exp005_convnext_base/            # ConvNeXt-Base (오버피팅)
│   ├── exp006_kfold_ensemble/           # 5-Fold ConvNeXt-Small ★ 최고
│   ├── exp007_kfold_tta/                # exp006 + TTA
│   ├── exp008_kfold_384/                # 5-Fold, 384 해상도
│   ├── exp009_efficientnet_kfold/       # EfficientNet-B3 5-Fold
│   ├── exp010_10fold_seed_ensemble/     # 10-Fold x 3-Seed
│   ├── exp011_dualview_physics/         # 듀얼뷰 (front+top)
│   └── exp012_calibration_pseudo/       # Pseudo-Label + Temperature Scaling ★ 최고
└── .venv/                               # 가상환경
```

## 완료된 단계

- [x] ResNet50 baseline (exp001)
- [x] Multi-View 실험 (exp002)
- [x] ConvNeXt-Small backbone 변경 (exp003)
- [x] TTA (exp004)
- [x] ConvNeXt-Base 시도 (exp005, 실패)
- [x] K-Fold CV + 앙상블 (exp006)
- [x] K-Fold + TTA (exp007)
- [x] 해상도 384px (exp008)
- [x] 다중 아키텍처 앙상블 EfficientNet (exp009, 효과 없음)
- [x] 10-Fold x 3-Seed (exp010)
- [x] 듀얼 뷰 front+top (exp011)
- [x] Temperature Scaling + Pseudo-Labeling (exp012)

## 다음 단계

- [ ] 다른 Temperature 값 제출 비교 (T=0.5, 0.8)
- [ ] Pseudo-Label 2라운드 (exp012 모델로 다시 pseudo → 재학습)
- [ ] 영상 멀티프레임 학습 데이터 증강
- [ ] 6채널 입력 (front RGB + top RGB)
