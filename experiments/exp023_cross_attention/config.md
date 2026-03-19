# Exp 023: Cross-Attention + Layer Decay + Focal Loss

## 3가지 핵심 개선

### 1. Cross-Attention (front↔top 상호 참조)
- front 특징이 top 특징을 질문 → "밑면 상태 어때?"
- top 특징이 front 특징을 질문 → "무게중심 어디야?"
- concat(독립적)보다 관계 학습이 명시적

### 2. Layer Decay (레이어별 LR 차등)
- 얕은 레이어 (기본 패턴): LR 낮게 → 사전학습 지식 보존
- 깊은 레이어 (의미 해석): LR 높게 → 우리 태스크에 맞게 조정
- decay_rate = 0.65

### 3. Focal Loss (어려운 샘플 집중)
- 쉬운 샘플 (확률 0.95): loss 대폭 감소
- 어려운 샘플 (확률 0.55): loss 유지
- gamma = 2.0

## 설정
- exp020 기반 (배경 제거 + 구조 피처 + pseudo-label + 멀티프레임)
- Head LR: 3e-4, Backbone LR: 1e-5
- 40 에폭, patience 12

## 결과
- 대기 중
