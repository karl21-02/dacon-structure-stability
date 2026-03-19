"""
Exp023: Cross-Attention + Layer Decay + Focal Loss

3가지 핵심 개선:
1. Cross-Attention: front↔top 특징이 서로 참조 → 어려운 샘플에서 효과적
2. Layer Decay: backbone 레이어별 다른 LR → 사전학습 지식 보존
3. Focal Loss: 어려운 샘플에 집중 학습 → 경계 케이스 개선
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

# ── 경로 설정 ──
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "open (1)")
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP020_DIR = os.path.join(SAVE_DIR, "..", "exp020_structural_features")  # 배경제거 이미지 + 구조 피처가 여기 있음
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── 하이퍼파라미터 ──
IMG_SIZE = 224         # 입력 이미지 크기 (가로세로 224px)
N_FOLDS = 5            # K-Fold 수 (데이터를 5등분)
EPOCHS = 40            # 최대 학습 반복 횟수
BATCH_SIZE = 16        # 한 번에 처리하는 이미지 수
GRAD_ACCUM = 2         # 2번 모아서 한 번에 업데이트 → 실효 배치 = 16×2 = 32

# ★ 개선 1: Layer Decay 관련 설정
HEAD_LR = 3e-4         # head(판단부)의 학습률 → 처음부터 배워야 하니 빠르게
BACKBONE_LR = 1e-5     # backbone(특징추출)의 기본 학습률 → 이미 학습됐으니 살살
LAYER_DECAY = 0.65     # 감쇠율: Stage 0은 LR이 0.65^4배, Stage 3은 0.65^0배
                       # = 얕은 레이어일수록 LR이 낮아짐 (기존 지식 보존)

# ★ 개선 3: Focal Loss 관련 설정
FOCAL_GAMMA = 2.0      # 집중도: 클수록 쉬운 샘플의 loss를 더 많이 깎음
                       # gamma=0이면 일반 CrossEntropy와 동일
                       # gamma=2이면 확률 0.95인 쉬운 샘플의 loss가 400배 줄어듦

MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"  # 사전학습된 ConvNeXt-Small 모델
EXTRA_FRAMES = [1, 10, 20, 30]  # 영상에서 추출할 추가 프레임 번호

# 구조 피처 컬럼 (exp020에서 이미지 분석으로 추출한 20개 수치)
# front_lean = 기울기, front_hw_ratio = 높이/너비 비율 등
FEAT_COLS = [
    "front_pixels", "front_h", "front_w", "front_hw_ratio", "front_lean",
    "front_top_base_ratio", "front_cy_ratio", "front_cx_ratio",
    "front_fill_ratio", "front_symmetry",
    "top_pixels", "top_h", "top_w", "top_hw_ratio", "top_lean",
    "top_top_base_ratio", "top_cy_ratio", "top_cx_ratio",
    "top_fill_ratio", "top_symmetry",
]
N_FEATS = len(FEAT_COLS)  # 20개


# ══════════════════════════════════════════════════
# ★ 개선 3: Focal Loss
# ══════════════════════════════════════════════════
# 일반 CrossEntropy: -log(p) → 모든 샘플 동등
# Focal Loss: -log(p) × (1-p)^γ → 쉬운 샘플의 loss를 깎음
#
# 예시 (gamma=2):
#   쉬운 샘플 p=0.95: -log(0.95) × (0.05)^2 = 0.05 × 0.0025 = 0.000125
#   어려운 샘플 p=0.55: -log(0.55) × (0.45)^2 = 0.60 × 0.2025 = 0.1215
#   → 쉬운 것의 loss가 ~1000배 작아짐 → 어려운 것에 집중!
# ══════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=FOCAL_GAMMA, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma                  # 집중도 파라미터
        self.label_smoothing = label_smoothing  # 라벨 부드럽게 (1.0 → 0.9)

    def forward(self, logits, targets):
        # label smoothing: 정답을 [1, 0] 대신 [0.9, 0.1]으로 부드럽게
        # 왜? 모델이 100% 확신하는 걸 방지 → 과적합 줄임
        n_classes = logits.size(1)  # 2 (unstable, stable)
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.label_smoothing / (n_classes - 1))  # 오답 클래스에 0.1
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)  # 정답 클래스에 0.9

        # softmax → 확률 변환 (log 버전으로 수치 안정성 확보)
        log_probs = F.log_softmax(logits, dim=1)  # log(확률)
        probs = torch.exp(log_probs)               # 확률 자체

        # ★ focal weight: 쉬운 샘플일수록 (1-p)가 작아짐 → weight가 작아짐
        focal_weight = (1 - probs) ** self.gamma

        # 최종 loss: 정답 × 가중치 × log(확률)
        loss = -smooth_targets * focal_weight * log_probs
        return loss.sum(dim=1).mean()  # 배치 평균


# ══════════════════════════════════════════════════
# Dataset: 이미지 + 구조 피처를 함께 로드
# (exp020과 동일 — 배경 제거 이미지 사용)
# ══════════════════════════════════════════════════
class StructuralDataset(Dataset):
    def __init__(self, df, masked_dir, orig_dir, feat_mean, feat_std, transform=None):
        self.df = df.reset_index(drop=True)
        self.masked_dir = masked_dir    # 배경 제거된 이미지 폴더
        self.orig_dir = orig_dir        # 원본 이미지 폴더
        self.transform = transform
        self.label_map = {"unstable": 0, "stable": 1}
        self.feat_mean = feat_mean      # 피처 정규화용 평균
        self.feat_std = feat_std        # 피처 정규화용 표준편차

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = row["id"]
        front_file = row.get("front_file", "front.png")

        # --- 이미지 로드 ---
        # front.png면 배경제거 버전 사용, 멀티프레임이면 원본 사용
        if front_file == "front.png":
            img_dir = row.get("masked_dir", self.masked_dir)
        else:
            img_dir = row.get("img_dir", self.orig_dir)

        # 이미지 경로 찾기 (없으면 대체 경로 시도)
        front_path = os.path.join(img_dir, sample_id, front_file)
        if not os.path.exists(front_path):
            front_path = os.path.join(img_dir, sample_id, "front.png")
        if not os.path.exists(front_path):
            orig_dir = row.get("img_dir", self.orig_dir)
            front_path = os.path.join(orig_dir, sample_id, "front.png")

        # top 이미지도 배경제거 버전 우선
        top_masked = os.path.join(row.get("masked_dir", self.masked_dir), sample_id, "top.png")
        if os.path.exists(top_masked):
            top_path = top_masked
        else:
            orig_dir = row.get("img_dir", self.orig_dir)
            top_path = os.path.join(orig_dir, sample_id, "top.png")

        # 이미지 열기 → 224×224로 리사이즈
        front = Image.open(front_path).convert("RGB")
        top = Image.open(top_path).convert("RGB")
        front = front.resize((IMG_SIZE, IMG_SIZE))
        top = top.resize((IMG_SIZE, IMG_SIZE))

        # front + top을 나란히 붙여서 하나의 이미지로 (224×448)
        combined = Image.new("RGB", (IMG_SIZE * 2, IMG_SIZE))
        combined.paste(front, (0, 0))       # 왼쪽에 front
        combined.paste(top, (IMG_SIZE, 0))   # 오른쪽에 top

        # 이미지 변환 (augmentation 등)
        if self.transform:
            combined = self.transform(combined)

        # --- 구조 피처 로드 ---
        # 20개 수치를 배열로 만들고, 정규화 (평균=0, 표준편차=1로 맞춤)
        feats = np.array([row.get(c, 0) for c in FEAT_COLS], dtype=np.float32)
        feats = np.nan_to_num(feats, 0)  # NaN 값은 0으로 대체
        feats = (feats - self.feat_mean) / (self.feat_std + 1e-8)  # 정규화
        feats = torch.tensor(feats, dtype=torch.float32)

        # --- 라벨 ---
        label = row["label"]
        if isinstance(label, str):
            label = self.label_map[label]  # "unstable"→0, "stable"→1
        else:
            label = int(label)

        return combined, feats, label  # 이미지, 구조피처, 정답


# ══════════════════════════════════════════════════
# ★ 개선 2: Cross-Attention 모델
# ══════════════════════════════════════════════════
# 기존 (exp020): front특징 + top특징 → 그냥 이어붙이기 (concat)
# 개선 (exp023): front특징 ↔ top특징 서로 참조 (cross-attention)
#
# 비유:
#   concat = "front 소견서 + top 소견서를 스테이플러로 묶어서 판사에게 전달"
#   cross-attention = "front 의사와 top 의사가 서로 상의한 후 판사에게 전달"
# ══════════════════════════════════════════════════
class CrossAttentionClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=True, n_feats=N_FEATS, num_classes=2):
        super().__init__()

        # ── 부품 1: Backbone (이미지 → 특징 추출) ──
        # ImageNet에서 수백만 장으로 사전학습된 ConvNeXt-Small
        # front와 top 이미지가 같은 backbone을 공유 (같은 "눈"으로 봄)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features  # 768차원 특징 벡터

        # ── 부품 2: Cross-Attention (front↔top 상호 참조) ──
        # nn.MultiheadAttention = PyTorch가 제공하는 Attention 연산 도구
        # embed_dim=768: 입력 특징 크기
        # num_heads=8: 8개의 "관점"에서 동시에 참조 (multi-head)
        #   → 한 head는 "높이 관련", 다른 head는 "대칭성 관련" 등으로 자동 분화
        self.cross_attn_f2t = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, batch_first=True)
        # f2t = front가 top을 참조 (front→top 방향)
        self.cross_attn_t2f = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=8, batch_first=True)
        # t2f = top이 front를 참조 (top→front 방향)

        # Attention 결과를 안정화하는 정규화 레이어
        self.norm_front = nn.LayerNorm(feat_dim)
        self.norm_top = nn.LayerNorm(feat_dim)

        # ── 부품 3: 구조 피처 처리 네트워크 ──
        # 20개 수치 → 64 → 32차원으로 압축
        self.feat_net = nn.Sequential(
            nn.Linear(n_feats, 64),     # 20 → 64
            nn.BatchNorm1d(64),         # 값 분포 안정화
            nn.ReLU(),                  # 비선형 변환 (음수를 0으로)
            nn.Dropout(0.2),            # 20% 랜덤 끄기 (과적합 방지)
            nn.Linear(64, 32),          # 64 → 32
            nn.ReLU(),
        )

        # ── 부품 4: Head (최종 판단) ──
        # cross-attention으로 업데이트된 front(768) + top(768) + 구조피처(32) = 1568
        # → 256 → 2 (unstable 점수, stable 점수)
        self.head = nn.Sequential(
            nn.Dropout(0.3),                        # 30% 랜덤 끄기
            nn.Linear(feat_dim * 2 + 32, 256),      # 1568 → 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),             # 256 → 2
        )

    def forward(self, x, struct_feats):
        """
        데이터가 모델을 통과하는 파이프라인:
        이미지 → backbone → cross-attention → head → 예측
        """

        # ── Step 1: 이미지 분리 ──
        # 입력 x는 front+top이 붙어있는 이미지 (224×448)
        # 왼쪽 224px = front, 오른쪽 224px = top
        front = x[:, :, :, :IMG_SIZE]    # (배치, 3채널, 224, 224)
        top = x[:, :, :, IMG_SIZE:]      # (배치, 3채널, 224, 224)

        # ── Step 2: Backbone으로 특징 추출 ──
        # 224×224 이미지 → 768개의 숫자 (특징 벡터)
        front_feat = self.backbone(front)  # (배치, 768)
        top_feat = self.backbone(top)      # (배치, 768)

        # ── Step 3: ★ Cross-Attention ──
        # MultiheadAttention은 3D 입력이 필요: (배치, 시퀀스길이, 특징크기)
        # 우리 특징은 벡터 1개이므로 시퀀스길이=1로 차원 추가
        f = front_feat.unsqueeze(1)  # (배치, 768) → (배치, 1, 768)
        t = top_feat.unsqueeze(1)    # (배치, 768) → (배치, 1, 768)

        # front가 top을 참조:
        #   Query = front ("내가 알고 싶은 것")
        #   Key = top ("참조할 대상의 색인")
        #   Value = top ("실제로 가져올 정보")
        #   → front가 top에서 관련 정보를 가져와서 자신을 업데이트
        f_updated, _ = self.cross_attn_f2t(query=f, key=t, value=t)

        # top이 front를 참조 (반대 방향):
        t_updated, _ = self.cross_attn_t2f(query=t, key=f, value=f)

        # 잔차 연결 (Residual Connection):
        # 원본 특징 + attention 결과 → 원본 정보를 잃지 않으면서 새 정보 추가
        # 그 후 LayerNorm으로 값 분포 안정화
        front_out = self.norm_front(front_feat + f_updated.squeeze(1))
        top_out = self.norm_top(top_feat + t_updated.squeeze(1))

        # ── Step 4: 구조 피처 처리 ──
        struct_feat = self.feat_net(struct_feats)  # 20개 → 32개

        # ── Step 5: 결합 → 최종 판단 ──
        # front(768) + top(768) + 구조피처(32) = 1568차원
        combined = torch.cat([front_out, top_out, struct_feat], dim=1)
        return self.head(combined)  # 1568 → 256 → 2 (unstable/stable 점수)


# ══════════════════════════════════════════════════
# ★ 개선 1: Layer Decay — 레이어별 LR 설정
# ══════════════════════════════════════════════════
# ConvNeXt 내부 구조:
#   stem (가장 얕음) → stage_0 → stage_1 → stage_2 → stage_3 (가장 깊음)
#
# 얕은 레이어: "모서리, 색상" 같은 범용 패턴 → 거의 안 건드림 (LR 낮게)
# 깊은 레이어: "물체의 의미" → 우리 태스크에 맞게 수정 (LR 높게)
#
# decay_rate=0.65일 때 LR 계산:
#   stem:    1e-5 × 0.65^4 = 1.8e-6  (거의 동결)
#   stage_0: 1e-5 × 0.65^3 = 2.7e-6
#   stage_1: 1e-5 × 0.65^2 = 4.2e-6
#   stage_2: 1e-5 × 0.65^1 = 6.5e-6
#   stage_3: 1e-5 × 0.65^0 = 1.0e-5  (backbone 기본 LR)
#   head:    3e-4                     (가장 빠르게)
# ══════════════════════════════════════════════════
def get_layer_decay_params(model, backbone_lr, head_lr, decay_rate):
    """backbone 내부 레이어별로 다른 LR을 설정하는 함수"""
    param_groups = []

    # backbone의 레이어들을 순서대로 수집
    backbone_layers = []

    # stem: 입력 이미지를 처음 처리하는 레이어 (가장 얕음)
    if hasattr(model.backbone, 'stem'):
        backbone_layers.append(('stem', model.backbone.stem))

    # stages: 점점 깊어지는 레이어들 (0→1→2→3)
    if hasattr(model.backbone, 'stages'):
        for i, stage in enumerate(model.backbone.stages):
            backbone_layers.append((f'stage_{i}', stage))

    n_layers = len(backbone_layers)  # 총 5개 (stem + 4 stages)

    # 각 레이어에 LR 할당
    for idx, (name, layer) in enumerate(backbone_layers):
        # 공식: LR = backbone_lr × decay^(총레이어수-1-현재인덱스)
        # idx=0 (stem): decay^4 → 가장 낮은 LR
        # idx=4 (stage_3): decay^0 = 1 → backbone_lr 그대로
        layer_lr = backbone_lr * (decay_rate ** (n_layers - 1 - idx))
        param_groups.append({
            "params": layer.parameters(),
            "lr": layer_lr,
            "name": name,
        })

    # backbone에서 위 레이어에 포함 안 된 파라미터 (norm 등)
    backbone_covered = set()
    for _, layer in backbone_layers:
        for p in layer.parameters():
            backbone_covered.add(id(p))

    remaining_backbone = [p for p in model.backbone.parameters() if id(p) not in backbone_covered]
    if remaining_backbone:
        param_groups.append({
            "params": remaining_backbone,
            "lr": backbone_lr,
            "name": "backbone_other",
        })

    # Cross-Attention, feat_net, head → 높은 LR (처음부터 배우는 부분)
    for name, module in [
        ("cross_attn_f2t", model.cross_attn_f2t),
        ("cross_attn_t2f", model.cross_attn_t2f),
        ("norm_front", model.norm_front),
        ("norm_top", model.norm_top),
        ("feat_net", model.feat_net),
        ("head", model.head),
    ]:
        param_groups.append({
            "params": module.parameters(),
            "lr": head_lr,
            "name": name,
        })

    return param_groups


# ══════════════════════════════════════════════════
# 이미지 변환 (Augmentation)
# ══════════════════════════════════════════════════
def get_transforms(is_train=True):
    if is_train:
        # 학습 시: 다양한 변형으로 데이터 뻥튀기
        return transforms.Compose([
            transforms.RandomVerticalFlip(),     # 50% 확률로 상하 반전
            transforms.RandomRotation(5),        # ±5도 회전
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 색감 변경
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 살짝 흐릿하게
            transforms.ToTensor(),               # 이미지 → 숫자 배열
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 정규화
            transforms.RandomErasing(p=0.2),     # 20% 확률로 일부 영역 지움
        ])
    # 검증/테스트 시: 변형 없이 깔끔하게
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def logloss(true, pred, eps=1e-15):
    """LogLoss 계산: 예측 확률이 정답에 가까울수록 점수 낮음 (낮을수록 좋음)"""
    pred = np.clip(pred, eps, 1 - eps)            # 0이나 1 방지
    pred = pred / pred.sum(axis=1, keepdims=True)  # 합이 1이 되도록
    return -np.mean(np.sum(true * np.log(pred), axis=1))


# ══════════════════════════════════════════════════
# 학습 함수: 1 에폭 = 전체 데이터 1회 통과
# ══════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()  # 학습 모드 ON (Dropout 등 활성화)
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()  # gradient 초기화

    for i, (images, feats, labels) in enumerate(loader):
        # GPU로 데이터 이동
        images, feats, labels = images.to(DEVICE), feats.to(DEVICE), labels.to(DEVICE)

        # FP16 혼합 정밀도: 메모리 절약 + 속도 향상
        with torch.amp.autocast("cuda"):
            outputs = model(images, feats)                     # 모델 예측
            loss = criterion(outputs, labels) / GRAD_ACCUM     # ★ Focal Loss 계산

        # 역전파: "이 방향으로 파라미터를 고쳐!" 정보 계산
        scaler.scale(loss).backward()

        # GRAD_ACCUM번 모아서 한 번에 업데이트
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)   # ★ Layer Decay가 적용된 optimizer로 업데이트
            scaler.update()          #   → 각 레이어가 자기 LR만큼만 변함
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM * images.size(0)
        _, predicted = outputs.max(1)                    # 가장 높은 점수의 클래스
        correct += predicted.eq(labels).sum().item()     # 맞힌 개수
        total += labels.size(0)

    return total_loss / total, correct / total


# ══════════════════════════════════════════════════
# 검증 함수: 학습 없이 현재 모델 성능 측정
# ══════════════════════════════════════════════════
@torch.no_grad()  # gradient 계산 안 함 → 메모리 절약
def evaluate(model, loader):
    model.eval()  # 평가 모드 ON (Dropout 비활성화)
    all_probs, all_labels = [], []

    for images, feats, labels in loader:
        images, feats = images.to(DEVICE), feats.to(DEVICE)
        with torch.amp.autocast("cuda"):
            outputs = model(images, feats)
        # softmax: 점수 → 확률 변환 (합=1)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # 정답을 one-hot으로: 0 → [1,0], 1 → [0,1]
    true_onehot = np.zeros_like(all_probs)
    true_onehot[np.arange(len(all_labels)), all_labels] = 1

    score = logloss(true_onehot, all_probs)
    acc = (all_probs.argmax(1) == all_labels).mean()
    return score, acc, all_probs


# ══════════════════════════════════════════════════
# 멀티프레임 확장: 영상에서 추출한 프레임을 추가 데이터로
# ══════════════════════════════════════════════════
def expand_with_frames(df, data_dir):
    """train 데이터에 영상 프레임을 추가해서 데이터 뻥튀기"""
    rows = []
    for _, row in df.iterrows():
        # 원본 (front.png)
        new_row = row.copy()
        new_row["front_file"] = "front.png"
        rows.append(new_row)

        # 추가 프레임 (front_frame1.png, front_frame10.png 등)
        if row["id"].startswith("TRAIN"):
            for frame_num in EXTRA_FRAMES:
                frame_file = f"front_frame{frame_num}.png"
                frame_path = os.path.join(data_dir, row["id"], frame_file)
                if os.path.exists(frame_path):
                    new_row = row.copy()
                    new_row["front_file"] = frame_file
                    rows.append(new_row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ══════════════════════════════════════════════════
# 메인 함수: 전체 학습 흐름
# ══════════════════════════════════════════════════
#
# 전체 파이프라인:
#
#   1. 데이터 로드 (train 1000개, dev 100개)
#   2. 구조 피처 로드 (이미지에서 뽑은 기울기, 면적 등 20개 수치)
#   3. Pseudo-Label 로드 (이전 모델이 test에 붙인 가짜 정답 ~900개)
#   4. 피처 정규화 (수치들의 평균=0, 표준편차=1로 맞춤)
#   5. 5-Fold 학습:
#      ┌────────────────────────────────────────────┐
#      │ Fold 1: [검증][학습][학습][학습][학습]     │
#      │ Fold 2: [학습][검증][학습][학습][학습]     │
#      │ Fold 3: [학습][학습][검증][학습][학습]     │
#      │ Fold 4: [학습][학습][학습][검증][학습]     │
#      │ Fold 5: [학습][학습][학습][학습][검증]     │
#      │                                            │
#      │ 각 Fold에서:                               │
#      │   - 모델 생성 (Cross-Attention)            │
#      │   - Layer Decay LR 설정                    │
#      │   - Focal Loss로 학습                      │
#      │   - 최고 모델 저장                         │
#      └────────────────────────────────────────────┘
#   6. 결과 요약 + OOF 예측 저장
#
# ══════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME} + Cross-Attention")
    print(f"Layer Decay: {LAYER_DECAY}")
    print(f"Focal Loss gamma: {FOCAL_GAMMA}")
    print(f"Head LR: {HEAD_LR}, Backbone LR: {BACKBONE_LR}")

    # ══════════════════════════════════════════════
    # 1단계: 데이터 로드
    # ══════════════════════════════════════════════
    # train.csv: 1000개 샘플 (id, label)
    # dev.csv: 100개 샘플 (id, label)
    # → 합치면 1100개가 우리의 "진짜 데이터"
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    train_dir = os.path.join(DATA_ROOT, "train")
    dev_dir = os.path.join(DATA_ROOT, "dev")
    test_dir = os.path.join(DATA_ROOT, "test")

    # 이미지가 어느 폴더에 있는지 기록
    train_csv["img_dir"] = train_dir
    dev_csv["img_dir"] = dev_dir

    # ══════════════════════════════════════════════
    # 2단계: 구조 피처 로드
    # ══════════════════════════════════════════════
    # exp020에서 미리 추출해둔 물리 수치들
    # 예: front_lean=20.8 (기울기), front_hw_ratio=1.5 (높이/너비 비율)
    # 이걸 각 샘플의 데이터프레임에 컬럼으로 추가
    struct_feats = pd.read_csv(os.path.join(EXP020_DIR, "structural_features.csv"))

    # id → 피처 딕셔너리로 변환 (빠른 검색용)
    # feat_map = {"TRAIN_0001": {"front_lean": 20.8, ...}, ...}
    feat_map = {}
    for _, row in struct_feats.iterrows():
        feat_map[row["id"]] = {c: row.get(c, 0) for c in FEAT_COLS}

    # train/dev 데이터프레임에 20개 피처 컬럼 추가
    for df in [train_csv, dev_csv]:
        for col in FEAT_COLS:
            df[col] = df["id"].map(lambda x, c=col: feat_map.get(x, {}).get(c, 0))

    # train(1000) + dev(100) = 1100개 합치기
    all_real = pd.concat([train_csv, dev_csv], ignore_index=True)

    # 배경 제거 이미지 경로 설정
    # exp020에서 체커보드 배경을 제거한 이미지가 masked_images/ 에 저장되어 있음
    masked_train = os.path.join(EXP020_DIR, "masked_images", "train")
    masked_dev = os.path.join(EXP020_DIR, "masked_images", "dev")
    masked_test = os.path.join(EXP020_DIR, "masked_images", "test")
    all_real["masked_dir"] = all_real["id"].apply(
        lambda x: masked_train if x.startswith("TRAIN") else masked_dev)

    # ══════════════════════════════════════════════
    # 3단계: Pseudo-Label 로드
    # ══════════════════════════════════════════════
    # 이전 모델(exp018)이 test 1000개에 대해 예측한 "가짜 정답"
    # 확신도 높은 ~900개만 사용
    # → 학습 데이터가 1100 → ~2000개로 증가
    # → test 도메인(조명/카메라가 다른)의 데이터를 학습에 포함시켜 도메인 갭 줄임
    pseudo_path = os.path.join(EXP020_DIR, "..", "exp018_triple_stack", "pseudo_labels_round2.csv")
    pseudo_df = pd.read_csv(pseudo_path)
    pseudo_df["img_dir"] = test_dir
    pseudo_df["masked_dir"] = masked_test
    pseudo_df["front_file"] = "front.png"
    if "confidence" in pseudo_df.columns:
        pseudo_df = pseudo_df.drop(columns=["confidence"])
    # pseudo-label 샘플에도 구조 피처 추가
    for col in FEAT_COLS:
        pseudo_df[col] = pseudo_df["id"].map(lambda x, c=col: feat_map.get(x, {}).get(c, 0))

    print(f"Real: {len(all_real)}, Pseudo: {len(pseudo_df)}")

    # ══════════════════════════════════════════════
    # 4단계: 피처 정규화 통계 계산
    # ══════════════════════════════════════════════
    # 20개 피처의 값 범위가 다름 (면적: ~5000, 비율: ~1.5, 기울기: ~20)
    # → 평균=0, 표준편차=1로 맞추면 모델이 공정하게 비교 가능
    # 예: front_lean=20.8 → 정규화 → 1.2 (평균보다 1.2 표준편차만큼 높음)
    feat_values = all_real[FEAT_COLS].values.astype(np.float32)
    feat_mean = np.nanmean(feat_values, axis=0)  # 각 피처의 평균 (20개)
    feat_std = np.nanstd(feat_values, axis=0)    # 각 피처의 표준편차 (20개)
    np.save(os.path.join(SAVE_DIR, "feat_mean.npy"), feat_mean)  # 추론 때도 같은 값 사용
    np.save(os.path.join(SAVE_DIR, "feat_std.npy"), feat_std)

    # ══════════════════════════════════════════════
    # 5단계: K-Fold 학습
    # ══════════════════════════════════════════════
    # 1100개 데이터를 5등분 → 5번 학습
    # 매번 다른 220개로 검증 → 5개의 모델 생성
    # 최종: 5개 모델의 예측을 평균 → 한 모델보다 훨씬 안정적
    real_labels = all_real["label"].map({"unstable": 0, "stable": 1}).values

    # StratifiedKFold: unstable/stable 비율을 각 fold에서 동일하게 유지
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # OOF (Out-of-Fold) 예측: 각 샘플이 "검증용"이었을 때의 예측을 모음
    # → 전체 1100개에 대한 "모델이 처음 보는 데이터에서의 예측"
    # → 나중에 블렌딩이나 보정에 활용
    oof_preds = np.zeros((len(all_real), 2))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_real, real_labels)):
        # train_idx: 이번 fold에서 학습에 쓸 샘플 인덱스 (880개)
        # val_idx: 이번 fold에서 검증에 쓸 샘플 인덱스 (220개)

        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*50}")

        # 검증 데이터: 원본 이미지만 사용 (공정한 평가를 위해 augmentation 없음)
        val_fold_df = all_real.iloc[val_idx].copy()
        val_fold_df["front_file"] = "front.png"

        # 학습 데이터 준비:
        # 1) 원본 880개를 멀티프레임으로 확장 (~4000개)
        # 2) pseudo-label (~900개) 추가
        # → 총 ~5000개로 학습
        train_real = all_real.iloc[train_idx]
        train_expanded = expand_with_frames(train_real, train_dir)  # 영상 프레임 추가
        train_fold_df = pd.concat([train_expanded, pseudo_df], ignore_index=True)

        # Dataset: 이미지 + 피처 + 라벨을 묶어서 관리
        # DataLoader: Dataset에서 배치 단위로 꺼내주는 도구
        train_ds = StructuralDataset(train_fold_df, masked_train, train_dir,
                                     feat_mean, feat_std, transform=get_transforms(True))
        val_ds = StructuralDataset(val_fold_df, masked_dev, dev_dir,
                                   feat_mean, feat_std, transform=get_transforms(False))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True)
        # shuffle=True: 학습 때 매 에폭마다 데이터 순서를 섞음 (과적합 방지)
        # num_workers=4: 4개 프로세스로 병렬 데이터 로딩 (속도↑)
        # pin_memory=True: GPU 전송 속도 향상
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True)

        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

        # ── 모델 생성 ──
        # ★ Cross-Attention이 포함된 우리의 새 모델!
        model = CrossAttentionClassifier(pretrained=True).to(DEVICE)
        # .to(DEVICE) = GPU로 모델 이동

        # ── ★ Layer Decay LR 설정 ──
        # 각 레이어에 다른 LR을 부여한 파라미터 그룹 생성
        # stem: 1.8e-6 (거의 동결) → ... → stage_3: 1e-5 → head: 3e-4
        param_groups = get_layer_decay_params(model, BACKBONE_LR, HEAD_LR, LAYER_DECAY)
        print("Layer-wise LR:")
        for pg in param_groups:
            print(f"  {pg['name']}: lr={pg['lr']:.6f}")

        # ── ★ Focal Loss 생성 ──
        # 쉬운 샘플의 loss를 줄이고, 어려운 샘플에 집중
        criterion = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=0.1)

        # ── Optimizer: 파라미터를 업데이트하는 도구 ──
        # AdamW: Adam + Weight Decay (가장 널리 쓰이는 optimizer)
        # param_groups를 넘기면 각 그룹이 자기 LR대로 업데이트됨
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2)

        # ── Scheduler: 학습률을 점점 줄이는 도구 ──
        # Cosine Annealing: LR을 cosine 곡선 형태로 서서히 감소
        # 처음엔 크게 배우고 → 나중엔 미세 조정
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

        # FP16 학습 도구: 메모리 절약 + 속도 향상
        scaler = torch.amp.GradScaler("cuda")

        best_score = float("inf")  # 지금까지 최고 점수 (낮을수록 좋음)
        patience = 0               # 점수 안 좋아진 횟수
        max_patience = 12          # 12번 연속 안 좋아지면 학습 중단

        # ── 에폭 반복: 전체 데이터를 여러 번 학습 ──
        for epoch in range(EPOCHS):
            # 1) 학습: 전체 train 데이터 1회 통과
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            # 2) 검증: val 데이터로 현재 성능 측정 (학습 없이)
            val_score, val_acc, val_probs = evaluate(model, val_loader)
            # 3) 학습률 조정
            scheduler.step()

            print(f"[Epoch {epoch+1:02d}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val LogLoss: {val_score:.4f} Acc: {val_acc:.4f}")

            # 검증 점수가 좋아졌으면 → 모델 저장!
            if val_score < best_score:
                best_score = val_score
                patience = 0  # 참을성 리셋
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth"))
                best_val_probs = val_probs.copy()  # 이 시점의 예측도 저장
                print(f"  -> Best model saved! (LogLoss: {best_score:.4f})")
            else:
                # 점수가 안 좋아졌으면 → 참을성 -1
                patience += 1
                if patience >= max_patience:
                    # 12번 연속 안 좋아지면 → 더 학습해도 나빠지기만 하니까 중단!
                    print(f"  -> Early stopping at epoch {epoch+1}")
                    break

        # 이 fold의 검증 예측을 OOF에 저장
        # val_idx 위치에 예측값 넣음 → 전체 1100개가 채워짐
        oof_preds[val_idx] = best_val_probs
        fold_scores.append(best_score)
        del model; torch.cuda.empty_cache()  # GPU 메모리 해제 (다음 fold를 위해)
        print(f"Fold {fold+1} complete. Best: {best_score:.4f}")

    # ══════════════════════════════════════════════
    # 6단계: 결과 요약
    # ══════════════════════════════════════════════
    print(f"\n{'='*50}")
    print(f"Cross-Attention Model Complete!")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    print(f"  Mean: {np.mean(fold_scores):.4f}")
    print(f"{'='*50}")

    # OOF 예측 저장
    # 나중에 다른 모델들과 블렌딩하거나, 확률 보정에 사용
    np.save(os.path.join(SAVE_DIR, "oof_cross_attention.npy"), oof_preds)


if __name__ == "__main__":
    main()
