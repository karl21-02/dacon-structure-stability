# Exp 017: Swin Transformer + Multi-Arch Blending

## 개요

ConvNeXt(CNN 계열)와 Swin Transformer(Transformer 계열) 블렌딩.
서로 다른 구조가 서로 다른 패턴을 포착 → 보완 효과.

## 설정

- Swin-Base (88M params): swin_base_patch4_window7_224
- 듀얼뷰 + Pseudo-Label + 전략적 멀티프레임 (exp016과 동일 데이터)
- LR: 2e-4 (Swin에 맞게 조정)
- 40 에폭, patience 12

## 블렌딩 전략

- ConvNeXt(exp015) + Swin(exp017) 예측을 가중 평균
- 비율: 60/40, 50/50, 70/30 비교
- Temperature Scaling 적용

## 결과

- 대기 중
