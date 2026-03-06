import os
import torch
import torch.nn as nn
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

# ── 설정 ──
# 데이터가 어디 있는지 경로 설정
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
# 모델을 저장할 폴더 (이 파일이 있는 폴더)
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
# GPU가 있으면 GPU 사용, 없으면 CPU 사용
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS = 30           # 최대 30번 반복 학습 (early stopping으로 보통 더 일찍 끝남)
BATCH_SIZE = 8        # 한 번에 8장씩 이미지를 묶어서 학습 (GPU 메모리 제한)
LR = 3e-4             # 학습률: 모델이 얼마나 빠르게 배울지 결정 (너무 크면 불안정, 너무 작으면 느림)
IMG_SIZE = 224         # 이미지를 224x224 크기로 줄여서 사용
LABEL_SMOOTHING = 0.1  # 정답을 "100% 확실"이 아니라 "90% 확실"로 부드럽게 만듦 (과적합 방지)
MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"  # 사용할 모델: ConvNeXt-Small (이미 22,000개 카테고리로 사전학습됨)
GRAD_ACCUM = 4        # 4번 모아서 한 번에 업데이트 → 실제로는 8x4=32장씩 학습하는 효과
N_FOLDS = 5           # 데이터를 5등분해서 5개의 모델을 만들 것


# ── Dataset (데이터를 읽어오는 도구) ──
# PyTorch가 이미지를 읽고 라벨을 가져올 수 있도록 만든 클래스
class StructureDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)  # 인덱스 번호를 0부터 새로 매김
        self.data_dir = data_dir
        self.transform = transform    # 이미지에 적용할 변환 (크기 조절, 뒤집기 등)
        self.is_test = is_test
        self.label_map = {"unstable": 0, "stable": 1}  # 문자를 숫자로 변환 (컴퓨터는 숫자를 좋아함)

    def __len__(self):
        return len(self.df)  # 전체 데이터 개수

    def __getitem__(self, idx):
        # idx번째 데이터를 가져오는 함수 (PyTorch가 자동으로 호출)
        row = self.df.iloc[idx]
        sample_id = row["id"]
        # 각 샘플의 이미지가 있는 폴더 경로 (train 폴더 또는 dev 폴더)
        img_dir = row.get("img_dir", self.data_dir)
        front_path = os.path.join(img_dir, sample_id, "front.png")
        img = Image.open(front_path).convert("RGB")  # 이미지 열기
        if self.transform:
            img = self.transform(img)  # 이미지 변환 적용
        if self.is_test:
            return img, sample_id      # 테스트면 이미지와 ID만 반환
        label = self.label_map[row["label"]]
        return img, label              # 학습이면 이미지와 정답 반환


# ── Augmentation (이미지 변형 = 데이터 뻥튀기) ──
# 같은 이미지를 뒤집고, 돌리고, 밝기 바꿔서 "다른 이미지인 척" 하는 기법
# 왜? 데이터가 1,100개밖에 없어서, 변형해서 더 많은 패턴을 보여줘야 모델이 잘 배움
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),  # 약간 크게 만들고
            transforms.RandomCrop(IMG_SIZE),                     # 랜덤하게 잘라냄 (위치 다양화)
            transforms.RandomHorizontalFlip(),   # 50% 확률로 좌우 반전
            transforms.RandomVerticalFlip(),     # 50% 확률로 상하 반전
            transforms.RandomRotation(15),       # -15도 ~ +15도 랜덤 회전
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 살짝 이동, 살짝 확대/축소
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 밝기/색감 랜덤 변경 (광원 변동 시뮬레이션)
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 살짝 흐릿하게 (카메라 초점 변동)
            transforms.ToTensor(),               # 이미지를 숫자 배열로 변환 (PyTorch가 이해할 수 있는 형태)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet 기준으로 정규화
            transforms.RandomErasing(p=0.2),     # 20% 확률로 이미지 일부를 검은색으로 지움 (가려진 상황 대비)
        ])
    # 검증/테스트 시에는 변형 없이 깔끔하게
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── Model (모델 = 이미지를 보고 판단하는 AI) ──
class ConvNeXtClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True, num_classes=2):
        super().__init__()
        # backbone: 이미지에서 특징을 뽑아내는 부분 (이미 수백만 장의 이미지로 사전학습됨)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 뽑아낸 특징의 크기 (768차원 벡터)
        # head: 뽑아낸 특징을 보고 "안정/불안정"을 결정하는 부분
        self.head = nn.Sequential(
            nn.Dropout(0.3),                   # 30% 확률로 뉴런을 꺼서 과적합 방지
            nn.Linear(feat_dim, num_classes),   # 768차원 → 2차원 (unstable, stable 점수)
        )

    def forward(self, x):
        features = self.backbone(x)  # 이미지 → 특징 벡터
        return self.head(features)   # 특징 벡터 → 예측 점수


# ── Utils (도구 함수들) ──

# LogLoss: 대회 평가 지표. 예측 확률이 정답에 가까울수록 점수가 낮음 (낮을수록 좋음)
# 예: 정답이 unstable인데 0.9로 예측하면 좋은 점수, 0.1로 예측하면 나쁜 점수
def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)             # 0이나 1이 되면 log가 터지니까 살짝 보정
    pred = pred / pred.sum(axis=1, keepdims=True)   # 확률 합이 1이 되도록 정규화
    loss = -np.sum(true * np.log(pred), axis=1)
    return np.mean(loss)


# 한 에폭(전체 데이터 1회 학습) 진행하는 함수
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()  # 학습 모드 ON (Dropout 등 활성화)
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()  # 기울기(gradient) 초기화
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)  # GPU로 보냄

        # FP16 (반정밀도): 메모리를 절반만 쓰면서 거의 같은 정확도 → GPU 메모리 절약
        with torch.amp.autocast("cuda"):
            outputs = model(images)                         # 이미지 → 예측
            loss = criterion(outputs, labels) / GRAD_ACCUM  # 손실 계산 (4로 나눠서 나중에 합산)

        # 역전파: "이 예측이 틀렸으니 이 방향으로 고쳐!" 라는 정보를 계산
        scaler.scale(loss).backward()

        # GRAD_ACCUM번 모아서 한 번에 모델 업데이트 (작은 배치로도 큰 배치 효과)
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)   # 모델 파라미터 업데이트
            scaler.update()
            optimizer.zero_grad()    # 기울기 다시 초기화

        total_loss += loss.item() * GRAD_ACCUM * images.size(0)
        _, predicted = outputs.max(1)                      # 가장 높은 점수의 클래스 선택
        correct += predicted.eq(labels).sum().item()       # 맞힌 개수
        total += labels.size(0)
    return total_loss / total, correct / total


# 검증 함수: 학습 없이 현재 모델이 얼마나 잘하는지 측정
@torch.no_grad()  # 기울기 계산 안 함 (메모리 절약, 속도 향상)
def evaluate(model, loader):
    model.eval()  # 평가 모드 ON (Dropout 비활성화)
    all_probs, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
        # softmax: 점수를 확률(0~1)로 변환. 예: [2.1, -0.5] → [0.93, 0.07]
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    # 정답을 one-hot으로 변환: 0 → [1,0], 1 → [0,1]
    true_onehot = np.zeros_like(all_probs)
    true_onehot[np.arange(len(all_labels)), all_labels] = 1
    score = logloss(true_onehot, all_probs)
    acc = (all_probs.argmax(1) == all_labels).mean()
    return score, acc


# ── 한 개의 Fold를 학습하는 함수 ──
# K-Fold란? 데이터를 K등분해서, 매번 다른 부분을 검증용으로 사용
# 예: 5-Fold면 데이터를 5등분 → 5번 학습 → 5개의 모델 생성
#
# Fold 1: [검증][학습][학습][학습][학습]
# Fold 2: [학습][검증][학습][학습][학습]
# Fold 3: [학습][학습][검증][학습][학습]
# Fold 4: [학습][학습][학습][검증][학습]
# Fold 5: [학습][학습][학습][학습][검증]
#
# 장점: 모든 데이터가 한 번씩은 검증에 사용됨 → 더 공정하고 안정적
# 최종: 5개 모델의 예측을 평균 → 한 모델보다 훨씬 안정적 (앙상블)
def train_fold(fold, train_df, val_df, train_dir, dev_dir):
    print(f"\n{'='*50}")
    print(f"Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*50}")

    train_ds = StructureDataset(train_df, train_dir, transform=get_transforms(True))
    val_ds = StructureDataset(val_df, dev_dir, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 매 Fold마다 새로운 모델을 처음부터 만듦 (사전학습 가중치에서 시작)
    model = ConvNeXtClassifier(pretrained=True).to(DEVICE)
    # 손실 함수: 모델의 예측이 정답과 얼마나 다른지 계산
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    # 옵티마이저: 손실을 줄이는 방향으로 모델을 조금씩 수정하는 도구
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    # 스케줄러: 학습률을 천천히 줄여감 (처음엔 크게 배우고, 나중엔 미세 조정)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda")  # FP16 학습을 위한 도구
    best_score = float("inf")  # 지금까지 가장 좋은 점수 (작을수록 좋음)
    patience = 0               # 점수가 안 좋아진 횟수
    max_patience = 10          # 10번 연속 안 좋아지면 학습 중단 (과적합 방지)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_score, val_acc = evaluate(model, val_loader)
        scheduler.step()  # 학습률 조정

        print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

        if val_score < best_score:
            # 검증 점수가 좋아졌으면 모델 저장!
            best_score = val_score
            patience = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth"))
            print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")
        else:
            # 검증 점수가 안 좋아졌으면 참을성 -1
            patience += 1
            if patience >= max_patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break  # 더 학습해도 나빠지기만 하니까 그만!

    print(f"Fold {fold+1} complete. Best Val LogLoss: {best_score:.4f}")
    return best_score


# ── 메인 함수: 전체 흐름 ──
def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Folds: {N_FOLDS}")

    # 1단계: train 데이터(1,000개)와 dev 데이터(100개)를 합침 → 총 1,100개
    # 왜? 데이터가 많을수록 모델이 더 잘 배우니까!
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")

    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")

    # 각 샘플이 어느 폴더에 있는지 기록 (train 이미지는 train/, dev 이미지는 dev/)
    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    all_df = pd.concat([train_csv, dev_csv], ignore_index=True)
    print(f"Total samples: {len(all_df)} (train: {len(train_csv)}, dev: {len(dev_csv)})")

    # 2단계: Stratified K-Fold로 데이터를 5등분
    # "Stratified" = unstable/stable 비율을 각 fold에서 동일하게 유지
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    labels = all_df["label"].values

    # 3단계: 5번 학습! 매번 다른 데이터로 검증
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_df, labels)):
        train_fold_df = all_df.iloc[train_idx]  # 이번 fold의 학습 데이터 (880개)
        val_fold_df = all_df.iloc[val_idx]      # 이번 fold의 검증 데이터 (220개)
        score = train_fold(fold, train_fold_df, val_fold_df, train_dir, dev_dir)
        fold_scores.append(score)

    # 4단계: 결과 요약
    print(f"\n{'='*50}")
    print(f"All folds complete!")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean: {np.mean(fold_scores):.4f}")
    print(f"{'='*50}")
    # → 이제 inference.py에서 5개 모델의 예측을 평균내면 최종 결과!


if __name__ == "__main__":
    main()
