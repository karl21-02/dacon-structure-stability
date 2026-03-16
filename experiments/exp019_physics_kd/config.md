# Exp 019: Video Teacher → Ultra-Clean Pseudo-Label

## 개요

영상 전체를 보는 Video 모델로 test pseudo-label을 생성 → 이미지 모델을 재학습.
Video 모델은 10초간의 동적 거동을 직접 관찰하므로 이미지 모델보다 압도적으로 정확.

## 왜 이 접근인가

### 기존 Pseudo-Label의 한계
- exp012/017의 이미지 모델(CV 0.047)로 만든 pseudo-label → 노이즈 포함
- 이미지만으로는 "이 구조물이 흔들릴 것인가"를 확신하기 어려움
- 노이즈가 있는 pseudo-label → 학습에 악영향

### Video 모델의 압도적 우위
- 영상에는 구조물의 **실제 거동**이 담겨 있음
  - 흔들리는가? 무너지는가? 수렴하는가?
  - 이건 "추론"이 아니라 "관찰"에 가까움
- 1,100개 영상으로 학습한 Video 모델은 CV 0.01~0.02 가능할 수 있음
- 이 모델이 test를 예측하면 → 거의 정답에 가까운 pseudo-label

### KD 대비 장점
- KD: Teacher의 지식을 soft label로 간접 전달 (정보 손실 있음)
- Video Pseudo-Label: 직접 고품질 라벨을 제공 (정보 손실 최소)
- 구현도 더 간단하고 직관적

## 데이터 현황

| 항목 | Train/Dev | Test |
|------|-----------|------|
| 샘플 수 | 1,100 | 847 |
| 이미지 | front.png, top.png | front.png, top.png |
| 영상 | simulation.mp4 (30fps, 300frame, 10s, 384x384) | **없음** |
| 라벨 | unstable 552 / stable 548 | 없음 |

## 파이프라인

### Step 1: Video 모델 학습 (step1_video_model.py)

**모델: Video Swin Transformer (Swin3D-Small)**
- 시간축 정보를 window attention으로 처리
- ImageNet-22K + Kinetics-400 pretrained
- 구조물 안정성 = 시간에 따른 변형 패턴 → video classification에 딱 맞음

**입력:**
- simulation.mp4 → 균등 샘플링 32프레임 (300프레임 중 32개 추출)
- 해상도: 224x224 (384→224 resize)
- 형태: (B, 3, 32, 224, 224)

**학습 설정:**
- 5-Fold Stratified CV
- LR: 1e-4 (pretrained이므로 낮게)
- Epochs: 30, Patience: 10
- Label Smoothing: 0.05 (영상 정보가 명확하므로 낮게)
- Batch: 4 (3D 모델은 메모리 많이 씀)
- Grad Accum: 8 (effective batch 32)

**기대 CV:**
- 이미지 모델 CV 0.047 대비 크게 낮을 것 (0.01~0.03 목표)
- 영상을 직접 보므로 거의 확실한 판단 가능

**Augmentation:**
- Temporal: 랜덤 시작점 + 균등 샘플링 (시간축 augmentation)
- Spatial: RandomResizedCrop, RandomHorizontalFlip
- ColorJitter (밝기, 대비)
- 구조물 영상이므로 과도한 augmentation 불필요

### Step 2: Video 모델로 Test Pseudo-Label 생성 (step2_video_pseudo.py)

**문제: Test에 영상이 없다**

→ 이건 Video 모델로 test를 직접 예측하는 게 아님!

**실제 활용:**
- Video 모델의 OOF prediction → Train/Dev에 대한 초정밀 soft label
- 이 soft label로 이미지 모델을 학습 (soft-label training)
- 기존 이미지 모델의 pseudo-label(exp018)은 그대로 사용

**핵심 아이디어:**
- Train 데이터의 hard label (0 or 1) 대신 Video 모델의 soft prediction 사용
- 예: "이 구조물은 unstable이지만 꽤 안정적" → [0.7, 0.3]
- 이런 미묘한 정보가 LogLoss 개선에 직접적

### Step 3: Soft-Label 이미지 모델 학습 (step3_soft_image_model.py)

**Train 데이터 라벨 교체:**
- 기존: hard label (unstable=0, stable=1)
- 변경: Video 모델의 soft prediction (연속값)

**Loss:**
```python
# Soft Label은 KL Divergence로 학습
loss = KL_div(image_model_logits, video_soft_labels)
```

**Pseudo-Label 처리:**
- Test pseudo-label (exp018에서 생성)은 그대로 사용
- 이건 이미지 모델이 만든 거라 hard label과 큰 차이 없음

**모델:**
- ConvNeXt-Small (기존과 동일)
- 듀얼뷰 (front+top)
- 5-Fold CV

**기대:**
- Video soft label의 미묘한 확률 정보 → 이미지 모델이 경계 케이스를 더 잘 처리
- CV가 직접적으로 낮아지진 않을 수 있지만, LB(실제 점수)에서 개선 기대

### Step 4: 추론 + 최종 블렌딩 (step4_inference.py)

**추론:**
- Soft-label 이미지 모델 5-Fold × 3-TTA
- Temperature Scaling

**최종 앙상블 (exp018 + exp019):**
```
exp018: ConvNeXt (hard label)     ─┐
exp018: Swin (hard label)         ─┤
exp018: EVA-02 (hard label)       ─┼→ Stacking (LogisticRegression)
exp019: ConvNeXt (video soft label)─┘
```

4개 모델의 OOF로 meta-learner 학습 → 최적 가중치 자동 결정

## 설정 요약

| 항목 | Video Teacher | Soft-Label Student |
|------|--------------|-------------------|
| 모델 | Video Swin-S (3D) | ConvNeXt-Small (2D) |
| 입력 | 32프레임 영상 | 이미지 (front+top) |
| LR | 1e-4 | 3e-4 |
| Epochs | 30 | 40 |
| Batch | 4 (accum 8) | 16 (accum 2) |
| Label | Hard label | Video soft label |
| GPU 메모리 | ~20GB (3D 모델) | ~4GB |

## 리스크 & 대안

### 1. Video Swin이 timm에 없을 수 있음
- 대안: torchvision의 VideoResNet, 또는 slowfast_r50 (pytorchvideo)
- 최악의 경우: 3D CNN을 직접 구성 (ResNet3D)
- 또는: 2D backbone + temporal pooling (프레임별 특징 추출 → LSTM/평균)

### 2. Video 모델이 GPU 메모리 부족
- 32프레임 → 16프레임으로 줄이기
- 해상도 224 → 160
- Gradient checkpointing 사용

### 3. Video 모델 CV가 기대보다 안 좋으면
- 문제: 1,100개가 video 모델 학습에도 적을 수 있음
- 대안: pretrained video 모델을 freeze하고 head만 학습 (linear probing)
- 또는: 2D backbone으로 프레임별 특징 추출 → temporal aggregation

### 4. Soft label이 hard label 대비 효과 없으면
- Temperature 조절 (soft label을 더 부드럽게/날카롭게)
- Mixup: hard label과 soft label을 비율로 섞어서 학습
- Label smoothing과 결합

## 성공 기준

| 단계 | 목표 |
|------|------|
| Video 모델 CV | < 0.030 (이미지 0.047 대비 크게 개선) |
| Soft-label 이미지 모델 CV | < 0.045 |
| 최종 블렌딩 LB | < 0.030 (현재 0.036 대비 개선) |

## 실행 순서

```bash
# Step 1: Video 모델 학습 (~2-3시간)
python experiments/exp019_physics_kd/step1_video_model.py

# Step 2: OOF soft label 생성 (~10분)
python experiments/exp019_physics_kd/step2_video_pseudo.py

# Step 3: Soft-label 이미지 모델 학습 (~1시간)
python experiments/exp019_physics_kd/step3_soft_image_model.py

# Step 4: 추론 + 블렌딩 (~15분)
python experiments/exp019_physics_kd/step4_inference.py
```
