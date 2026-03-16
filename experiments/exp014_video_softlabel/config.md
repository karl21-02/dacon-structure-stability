# Exp 014: Video Soft Label + Dual View + Pseudo-Label

## 개요

영상(simulation.mp4)에서 구조물의 움직임을 측정해서 soft label 생성.
hard label(0/1) 대신 soft label(0.0~1.0)로 학습.

## Step 1: 영상 분석

- 각 영상의 첫 프레임 vs 이후 프레임의 픽셀 차이 측정
- max_diff: 최대 변화량 → 구조물이 얼마나 무너졌는지
- soft label로 변환:
  - unstable: max_diff 기반 0.55~0.99
  - stable: max_diff 기반 0.01~0.15

## Step 2: Soft Label 학습

- KLDivLoss 사용 (CrossEntropy 대신)
- 듀얼 뷰 (front+top) + Pseudo-Label 유지
- 모델이 "애매한 확률"을 더 정확하게 학습

## 결과

- 대기 중
