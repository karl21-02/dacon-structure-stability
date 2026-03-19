"""
Exp022 v2: 전체 모델 최적 블렌딩 (probability 기반)

OOF가 probability로 저장되어 있으므로, probability 공간에서 블렌딩.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP018_DIR = os.path.join(SAVE_DIR, "..", "exp018_triple_stack")
EXP020_DIR = os.path.join(SAVE_DIR, "..", "exp020_structural_features")
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(SAVE_DIR)), "data", "open (1)")


def logloss(true_onehot, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true_onehot * np.log(pred), axis=1))


def main():
    # OOF probabilities 로드
    oofs = {
        "convnext": np.load(os.path.join(EXP018_DIR, "oof_convnext.npy")),
        "swin": np.load(os.path.join(EXP018_DIR, "oof_swin.npy")),
        "eva02": np.load(os.path.join(EXP018_DIR, "oof_eva02.npy")),
        "structural": np.load(os.path.join(EXP020_DIR, "oof_structural.npy")),
    }
    names = list(oofs.keys())

    # labels
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    all_labels = pd.concat([train_csv, dev_csv], ignore_index=True)["label"]
    labels = (all_labels == "stable").astype(int).values  # unstable=0, stable=1 (OOF와 맞춤)

    true_onehot = np.zeros((len(labels), 2))
    true_onehot[np.arange(len(labels)), labels] = 1

    # structural은 fold 0,1이 비어있음 → 채워진 것만 사용
    all_filled = np.ones(len(labels), dtype=bool)
    for oof in oofs.values():
        all_filled &= ~((oof[:, 0] == 0) & (oof[:, 1] == 0))

    print(f"전체: {len(labels)}, 모든 모델 채워진: {all_filled.sum()}")

    f_oofs = {k: v[all_filled] for k, v in oofs.items()}
    f_true = true_onehot[all_filled]

    # === 개별 모델 성능 (probability 기반) ===
    print("\n=== 개별 모델 OOF LogLoss ===")
    for name, oof in f_oofs.items():
        print(f"  {name}: {logloss(f_true, oof):.6f}")

    # === 최적 블렌딩 (probability 공간) ===
    oof_list = [f_oofs[n] for n in names]
    n_models = len(oof_list)

    def blend_loss(params):
        weights = np.array(params[:n_models])
        weights = np.abs(weights) / np.abs(weights).sum()  # 정규화

        # probability 공간에서 가중 평균
        blended = sum(weights[i] * oof_list[i] for i in range(n_models))
        return logloss(f_true, blended)

    def blend_loss_with_temp(params):
        weights = np.array(params[:n_models])
        temp = params[n_models]
        weights = np.abs(weights) / np.abs(weights).sum()

        blended_probs = sum(weights[i] * oof_list[i] for i in range(n_models))
        # temperature 적용: prob → logit → scale → prob
        logits = np.log(np.clip(blended_probs, 1e-15, 1))
        scaled = logits / max(temp, 0.01)
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return logloss(f_true, probs)

    # Grid Search
    print("\n=== Grid Search (10% 단위) ===")
    best_score = float("inf")
    best_weights = None
    from itertools import product as iprod
    steps = np.arange(0, 1.05, 0.1)
    for w in iprod(steps, repeat=n_models):
        if abs(sum(w) - 1.0) > 0.01:
            continue
        score = blend_loss(w)
        if score < best_score:
            best_score = score
            best_weights = w
    print(f"  Weights: {dict(zip(names, best_weights))}")
    print(f"  LogLoss: {best_score:.6f}")

    # Fine Grid (5% 단위, best 주변)
    print("\n=== Fine Grid (5% 단위) ===")
    steps5 = np.arange(0, 1.05, 0.05)
    for w in iprod(steps5, repeat=n_models):
        if abs(sum(w) - 1.0) > 0.01:
            continue
        score = blend_loss(w)
        if score < best_score:
            best_score = score
            best_weights = w
    print(f"  Weights: {dict(zip(names, best_weights))}")
    print(f"  LogLoss: {best_score:.6f}")

    # Differential Evolution + Temperature
    print("\n=== Differential Evolution + Temperature ===")
    bounds = [(0, 1)] * n_models + [(0.1, 2.0)]
    result = differential_evolution(
        blend_loss_with_temp, bounds, seed=42, maxiter=2000, tol=1e-12,
    )
    de_weights = np.abs(result.x[:n_models])
    de_weights = de_weights / de_weights.sum()
    de_temp = result.x[n_models]
    print(f"  Weights: {dict(zip(names, de_weights))}")
    print(f"  Temperature: {de_temp:.4f}")
    print(f"  LogLoss: {result.fun:.6f}")

    # === Test Submission 생성 ===
    print("\n=== Test Submission 생성 ===")

    # test predictions (T=1.0 submissions 사용)
    test_preds = {}
    for name, fname in [
        ("convnext", os.path.join(EXP018_DIR, "submission_stack_T1.0.csv")),
        ("structural", os.path.join(EXP020_DIR, "submission_struct_T1.0.csv")),
    ]:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            test_preds[name] = df[["unstable_prob", "stable_prob"]].values

    # 개별 모델 test submissions 찾기
    for name in names:
        test_sub_path = os.path.join(EXP018_DIR, f"submission_{name}_T1.0.csv")
        if os.path.exists(test_sub_path):
            df = pd.read_csv(test_sub_path)
            test_preds[name] = df[["unstable_prob", "stable_prob"]].values

    print(f"  Available test preds: {list(test_preds.keys())}")

    # exp018 stack + exp020 struct 블렌딩으로 대체
    stack_sub = pd.read_csv(os.path.join(EXP018_DIR, "submission_stack_T1.0.csv"))
    struct_sub = pd.read_csv(os.path.join(EXP020_DIR, "submission_struct_T1.0.csv"))
    ids = stack_sub["id"].values

    stack_probs = stack_sub[["unstable_prob", "stable_prob"]].values
    struct_probs = struct_sub[["unstable_prob", "stable_prob"]].values

    # 다양한 비율 + temperature 조합
    print("\n=== 2-Model 블렌딩 + Temperature ===")
    for w_stack in np.arange(0.3, 0.8, 0.05):
        w_struct = 1 - w_stack
        blended = w_stack * stack_probs + w_struct * struct_probs

        for t in [0.3, 0.35, 0.4, 0.45, 0.47, 0.5]:
            logits = np.log(np.clip(blended, 1e-15, 1))
            scaled = logits / t
            probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
            fname = f"submission_stack{int(w_stack*100)}_struct{int(w_struct*100)}_T{t:.2f}.csv"
            sub.to_csv(os.path.join(SAVE_DIR, fname), index=False)

    # 최적 T (DE 결과) 적용
    for w_stack in [0.5, 0.6, 0.7]:
        w_struct = 1 - w_stack
        blended = w_stack * stack_probs + w_struct * struct_probs
        logits = np.log(np.clip(blended, 1e-15, 1))
        scaled = logits / de_temp
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
        fname = f"submission_stack{int(w_stack*100)}_struct{int(w_struct*100)}_T{de_temp:.2f}.csv"
        sub.to_csv(os.path.join(SAVE_DIR, fname), index=False)
        print(f"  {fname}")

    print("\nDone!")


if __name__ == "__main__":
    main()
