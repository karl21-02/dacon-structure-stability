# 구조물 안정성 물리 추론 AI 경진대회 - 전략 계획서

## 1. 대회 개요

| 항목 | 내용 |
|------|------|
| **대회명** | 월간 데이콘 - 구조물 안정성 물리 추론 AI 경진대회 |
| **목표** | 2가지 시점(front, top) 이미지로 구조물의 안정/불안정 상태 확률 예측 |
| **평가 지표** | LogLoss (낮을수록 좋음) |
| **제출 형식** | `id`, `unstable_prob`, `stable_prob` (합 = 1) |
| **일일 제출** | 3회 |
| **리더보드** | Public 50% / Private 100% |

## 2. 데이터 분석

### 2.1 데이터 구성

| 데이터셋 | 샘플 수 | 이미지 | 영상 | 라벨 | 환경 |
|----------|---------|--------|------|------|------|
| **train** | 1,000 | front.png + top.png | simulation.mp4 (10초) | O | 고정 (실험실) |
| **dev** | 100 | front.png + top.png | X | O | 무작위 (광원/카메라) |
| **test** | 1,000 | front.png + top.png | X | X | 무작위 (광원/카메라) |

### 2.2 데이터 상세 스펙 (실측 확인)

| 항목 | 값 |
|------|-----|
| 이미지 크기 | 384 x 384 RGB PNG |
| 영상 스펙 | 384x384, 30fps, 10초, mpeg4 (mp4v), ~750KB/파일 |
| 데이터 경로 | `data/open (1)/` |

**ID 네이밍 주의 (불일치)**:
- train: `TRAIN_0001` (4자리 zero-pad)
- dev: `DEV_001` (3자리 zero-pad)
- test: `TEST_0001` (4자리 zero-pad)

### 2.3 라벨 분포

| 데이터셋 | unstable | stable | 비율 |
|----------|----------|--------|------|
| **train** | 500 | 500 | 50:50 (완벽 균형) |
| **dev** | 52 | 48 | 52:48 (거의 균형) |

### 2.4 라벨 정의

- **stable**: 10초 동안 의미 있는 이동/변형 없음
- **unstable**: 10초 이내 누적 이동 거리 >= 1.5cm 또는 구조적 붕괴 발생
- 일부 샘플은 **경계(Boundary) 케이스** -> 정밀 추론 필요

### 2.5 시각적 관찰

- **unstable 구조물**: 블록이 넓게 쌓여있고, 상단이 비대칭적으로 기울어짐. 무게중심 편향 뚜렷
- **stable 구조물**: 블록이 좁고 수직으로 균형 있게 정렬. 무게중심이 중앙에 위치
- **배경**: 체커보드 패턴 바닥 (train/dev/test 공통)

### 2.6 핵심 도메인 갭 (Domain Gap) - 실측 확인

- **train**: 고정 환경 - 광원 위치 일정, 카메라 앵글 일정, 그림자 방향 일관
- **dev/test**: 무작위 환경 - 광원 위치 변동(좌측 강한 빛 등), 카메라 앵글 변동, 그림자 방향 다양
- **-> 색상/밝기/그림자에 의존하지 않는 구조적 특징 학습이 핵심**

## 3. GPU 환경

| GPU | 모델 | VRAM | 현재 여유 |
|-----|------|------|----------|
| GPU 0 | RTX 5090 | 32GB | ~6.7GB |
| GPU 1 | RTX 5090 | 32GB | ~1.4GB |

- 학습 시 다른 프로세스 정리 또는 메모리 효율적 학습 전략 필요
- 두 GPU 모두 활용 가능 시 최대 64GB VRAM

## 4. 전략 수립

### 4.1 접근 방식 후보

#### A. Multi-View Vision Encoder + Classifier (기본)
- front/top 이미지를 각각 인코딩 후 fusion하여 분류
- Backbone: EfficientNet-V2, ConvNeXt-V2, EVA-02 등
- 장점: 구현 간단, 빠른 실험 가능
- 단점: 물리적 추론 능력 제한적

#### B. Vision-Language Model (VLM) 기반 추론
- 사전학습된 VLM(예: InternVL, LLaVA)을 활용한 물리 추론
- 이미지를 입력으로 "이 구조물이 안정적인가?" 추론
- 장점: 물리적 인과관계 추론에 강점, zero-shot 가능성
- 단점: 추론 속도 느림, fine-tuning 비용 높음

#### C. Video Feature 활용 (train only)
- simulation.mp4에서 붕괴 패턴/물리 특징 추출
- 비디오 feature를 teacher로 사용하여 이미지 모델 knowledge distillation
- 장점: 풍부한 물리 정보 활용 가능
- 단점: test에는 영상 없음 -> distillation 필수

#### D. 앙상블 전략
- 다양한 모델 + 다양한 augmentation 결과를 앙상블
- LogLoss 특성상 보정된(calibrated) 확률이 중요 -> Temperature Scaling

### 4.2 권장 전략 (단계별)

```
Phase 1: 베이스라인 구축 & 데이터 탐색
Phase 2: 강력한 단일 모델 개발
Phase 3: 비디오 Knowledge Distillation
Phase 4: 앙상블 & 확률 보정
```

## 5. 상세 실행 계획

### Phase 1: 베이스라인 (1-2일)

- [ ] 데이터 다운로드 및 EDA
  - train/dev 라벨 분포 확인
  - 이미지 해상도, 구조물 형태 시각화
  - simulation.mp4 샘플 분석
- [ ] 간단한 Multi-View ResNet/EfficientNet 베이스라인 구현
- [ ] train으로 학습 -> dev로 검증 -> 첫 제출
- [ ] 목표: 대회 제출 파이프라인 완성

### Phase 2: 단일 모델 고도화 (3-5일)

- [ ] **강력한 Backbone 실험**
  - ConvNeXt-V2, EfficientNet-V2-L, EVA-02, SigLIP 등
  - ImageNet pretrained weights 활용
- [ ] **도메인 갭 해소 전략**
  - 강력한 Data Augmentation: ColorJitter, RandomLighting, CameraAngle 시뮬레이션
  - train → dev 스타일 맞추기 (광원/카메라 변동 시뮬레이션)
  - MixUp, CutMix 적용
- [ ] **Multi-View Fusion 최적화**
  - Early Fusion vs Late Fusion vs Cross-Attention Fusion 비교
  - front/top 이미지의 상호 보완적 정보 활용
- [ ] **학습 전략**
  - train + dev 합쳐서 학습 (K-Fold CV)
  - Cosine Annealing + Warmup
  - Label Smoothing (경계 케이스 대응)
  - SAM (Sharpness-Aware Minimization) optimizer

### Phase 3: 비디오 Knowledge Distillation (2-3일)

- [ ] **Teacher 모델**: train 영상(simulation.mp4)으로 Video Model 학습
  - VideoMAE, TimeSformer 등 활용
  - 붕괴 과정의 시간적 패턴 학습
- [ ] **Student 모델**: 이미지 기반 모델이 Teacher의 soft label 학습
  - Teacher의 예측 확률을 pseudo-label로 활용
  - 영상 정보를 이미지 모델에 전이
- [ ] train 영상에서 물리적 feature 추출
  - 초기 프레임 vs 최종 프레임 비교 -> 이동량 측정
  - 이를 보조 feature로 활용

### Phase 4: 앙상블 & 보정 (1-2일)

- [ ] 다양한 모델의 예측 결과 앙상블
  - 서로 다른 backbone, 서로 다른 augmentation
  - Weighted Average / Stacking
- [ ] **확률 보정 (Calibration)**
  - LogLoss는 확률 보정이 핵심
  - Temperature Scaling, Platt Scaling 적용
  - dev set 기반 최적 temperature 탐색
- [ ] **최종 제출 전략**
  - Public LB 기준 최적 모델 선정
  - 다양성 확보를 위해 2-3개 다른 전략의 제출물 유지

## 6. 핵심 포인트 & 주의사항

### 승리를 위한 핵심

1. **도메인 갭 극복**: train(고정 환경) -> test(무작위 환경) 일반화가 최우선
2. **확률 보정**: LogLoss 평가이므로 confident한 틀린 예측은 치명적
3. **경계 케이스 처리**: 불확실한 샘플에 대해 0.5에 가까운 확률 출력
4. **비디오 정보 활용**: 영상에서 물리 정보를 이미지 모델로 전이
5. **데이터 증강**: 광원/카메라 변동을 시뮬레이션하는 augmentation

### 주의사항

- 일일 제출 3회 제한 -> 제출 전 dev에서 충분히 검증
- train에만 overfitting 금지 -> dev 성능을 주요 지표로 활용
- API 사용 불가 -> 모든 모델 로컬 실행 필수
- test 데이터 학습 금지 (pseudo-labeling도 주의)

## 7. 프로젝트 디렉토리 구조

```
dacon/
├── PLAN.md                # 이 문서
├── data/
│   └── open (1)/          # 실제 데이터 위치
│       ├── train/         # TRAIN_0001 ~ TRAIN_1000
│       ├── dev/           # DEV_001 ~ DEV_100
│       ├── test/          # TEST_0001 ~ TEST_1000
│       ├── train.csv
│       ├── dev.csv
│       └── sample_submission.csv
├── notebooks/             # EDA 및 실험 노트북
├── src/                   # 소스 코드
│   ├── dataset.py         # 데이터셋 & 데이터로더
│   ├── models/            # 모델 정의
│   ├── train.py           # 학습 스크립트
│   ├── inference.py       # 추론 스크립트
│   └── utils.py           # 유틸리티
├── configs/               # 실험 설정 파일
├── outputs/               # 학습 결과 & 체크포인트
└── submissions/           # 제출 파일
```

## 8. 기술 스택

- **Framework**: PyTorch + timm + transformers
- **실험 관리**: wandb 또는 간단한 로그
- **영상 처리**: OpenCV, decord
- **기타**: albumentations (augmentation), scikit-learn (calibration)
