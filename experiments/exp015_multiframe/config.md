# Exp 015: Multi-Frame Data Augmentation + Dual View + Pseudo-Label

## 개요

영상(simulation.mp4) 초반 프레임(1~5)을 추가 학습 이미지로 활용.
라벨은 hard label 유지 (exp014 soft label 실패 교훈).

## 핵심 아이디어

- train 데이터에만 simulation.mp4 존재
- 초반 프레임(1~5): 구조물이 아직 서있는 상태 → test 이미지와 유사
- 같은 구조물의 살짝 다른 렌더링 → 자연스러운 데이터 증강
- 학습 데이터: ~880 x 6 + 847 pseudo = ~6,127 per fold (기존 대비 3.5배)

## 결과

- 대기 중
