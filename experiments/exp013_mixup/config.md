# Exp 013: Dual View + Pseudo-Label + Mixup

## 설정

- exp012 기반 (듀얼뷰 + pseudo-label)
- Mixup 추가: 50% 확률로 두 이미지를 섞어서 학습
- Mixup alpha=0.4 (Beta 분포 파라미터)
- 나머지: ConvNeXt-Small, 5-Fold, Label Smoothing 0.1

## Mixup이란?

- 두 학습 이미지를 비율에 따라 섞음
- 라벨도 같은 비율로 섞음 (soft label)
- 모델이 "확실한 것"뿐 아니라 "애매한 것"도 학습 → 확률 예측이 부드러워짐
- LogLoss에서 과도한 확신으로 인한 벌점을 줄여줌

## 결과

- 대기 중
