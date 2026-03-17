# Exp 020: 배경 제거 + 구조 피처 결합

## 개요

1. 체커보드 배경 제거 → CNN이 구조물에만 집중
2. 이미지에서 기하학적 피처 명시적 추출 (lean, hw_ratio 등)
3. CNN backbone + 구조 피처 결합 모델 학습
4. exp018 모델들과 stacking

## 핵심 발견

| 피처 | Unstable | Stable | 구분력 |
|------|----------|--------|--------|
| lean (기울기) | 20.8 | 5.3 | ★★★ |
| w (너비) | 147 | 124 | ★★ |
| top view pixels | 4558 | 2981 | ★★ |
| hw_ratio | 1.43 | 1.88 | ★ |

## 파이프라인

### Step 1: 전처리 (step1_preprocess.py)
- 전체 train/dev/test에 대해:
  - 배경 제거 이미지 생성 (구조물만 남김)
  - 구조 피처 추출 (lean, pixels, h, w, hw_ratio, top_base_ratio 등)
- 저장: structural_features.csv + masked images

### Step 2: 모델 학습 (step2_train.py)
- 입력: 배경 제거 이미지 + 구조 피처 벡터
- ConvNeXt backbone → image feat + struct feat → concat → head
- 5-Fold CV, Pseudo-Label R2, OOF 저장

### Step 3: 추론 + 최종 stacking (step3_inference.py)
- exp020 모델 추론
- exp018 (ConvNeXt, Swin, EVA-02) + exp020 → 4종 stacking
