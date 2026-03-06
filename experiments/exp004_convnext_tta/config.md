# Exp 004: ConvNeXt-Small + TTA (Test Time Augmentation)

## 설정

- 모델: exp003의 best_model.pth 그대로 사용 (추가 학습 없음)
- TTA 5종: 원본, 좌우반전, 상하반전, 중앙크롭(256→224), 밝기변경
- 5개 예측의 평균으로 최종 확률 산출

## 결과

- 제출 여부: 미제출
- Dev 검증: 미실시 (inference만 수행)

## 비고

- exp003 대비 예측이 안정화됨 (확률이 덜 극단적)
- 예: TEST_0005 unstable_prob 0.314 → 0.251
