"""
Exp022: 전체 모델 최적 블렌딩

모든 OOF 예측을 모아서 scipy.optimize로 최적 가중치 탐색.
단순 grid search가 아닌 Bayesian/Nelder-Mead 최적화.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from itertools import product

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP018_DIR = os.path.join(SAVE_DIR, "..", "exp018_triple_stack")
EXP020_DIR = os.path.join(SAVE_DIR, "..", "exp020_structural_features")
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(SAVE_DIR)), "data", "open (1)")


def logloss(true, pred, eps=1e-15):
    pred = np.clip(pred, eps, 1 - eps)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return -np.mean(np.sum(true * np.log(pred), axis=1))


def load_oof_predictions():
    """모든 OOF 예측 로드"""
    # exp018: ConvNeXt, Swin, EVA-02
    oof_convnext = np.load(os.path.join(EXP018_DIR, "oof_convnext.npy"))
    oof_swin = np.load(os.path.join(EXP018_DIR, "oof_swin.npy"))
    oof_eva = np.load(os.path.join(EXP018_DIR, "oof_eva02.npy"))

    # exp020: Structural
    oof_struct = np.load(os.path.join(EXP020_DIR, "oof_structural.npy"))

    # labels
    train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), encoding="utf-8-sig")
    dev_csv = pd.read_csv(os.path.join(DATA_ROOT, "dev.csv"), encoding="utf-8-sig")
    all_labels = pd.concat([train_csv, dev_csv], ignore_index=True)["label"]
    labels = (all_labels == "unstable").astype(int).values  # 0=unstable

    return {
        "convnext": oof_convnext,
        "swin": oof_swin,
        "eva02": oof_eva,
        "structural": oof_struct,
    }, labels


def load_test_predictions():
    """Test 예측 로드 (T=1.0 기준)"""
    preds = {}

    # exp018 submissions (T=1.0)
    for name, fname in [
        ("convnext", "submission_w502525_T0.7.csv"),  # 대표 하나
        ("stack", "submission_stack_T1.0.csv"),
    ]:
        path = os.path.join(EXP018_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            preds[name] = df[["unstable_prob", "stable_prob"]].values

    # exp020 structural (T=1.0)
    path = os.path.join(EXP020_DIR, "submission_struct_T1.0.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        preds["structural"] = df[["unstable_prob", "stable_prob"]].values

    # ids
    df = pd.read_csv(path)
    ids = df["id"].values

    return preds, ids


def main():
    oof_preds, labels = load_oof_predictions()

    # true onehot
    true_onehot = np.zeros((len(labels), 2))
    true_onehot[np.arange(len(labels)), labels] = 1

    # 비어있는 OOF 확인 (exp020 fold 0,1)
    names = list(oof_preds.keys())
    print("=== OOF 모델 목록 ===")
    for name, oof in oof_preds.items():
        filled = ~((oof[:, 0] == 0) & (oof[:, 1] == 0))
        print(f"  {name}: {filled.sum()}/{len(oof)} filled")

    # 모든 모델이 채워진 샘플만 사용
    all_filled = np.ones(len(labels), dtype=bool)
    for oof in oof_preds.values():
        all_filled &= ~((oof[:, 0] == 0) & (oof[:, 1] == 0))

    print(f"\n모든 모델 채워진 샘플: {all_filled.sum()}/{len(labels)}")

    # 필터링
    filtered_oofs = {k: v[all_filled] for k, v in oof_preds.items()}
    filtered_true = true_onehot[all_filled]
    filtered_labels = labels[all_filled]

    # === 개별 모델 성능 ===
    print("\n=== 개별 모델 OOF LogLoss ===")
    for name, oof in filtered_oofs.items():
        probs = np.exp(oof) / np.exp(oof).sum(axis=1, keepdims=True)
        score = logloss(filtered_true, probs)
        print(f"  {name}: {score:.6f}")

    # === 최적 가중치 탐색 ===
    oof_list = [filtered_oofs[n] for n in names]
    n_models = len(oof_list)

    def blend_logloss(weights):
        """가중 평균 → softmax → logloss"""
        w = np.array(weights)
        w = w / w.sum()  # 정규화

        blended_logits = sum(w[i] * oof_list[i] for i in range(n_models))
        probs = np.exp(blended_logits) / np.exp(blended_logits).sum(axis=1, keepdims=True)
        return logloss(filtered_true, probs)

    def blend_logloss_with_temp(params):
        """가중 평균 + temperature → softmax → logloss"""
        weights = params[:-1]
        temp = params[-1]
        w = np.array(weights)
        w = w / w.sum()

        blended_logits = sum(w[i] * oof_list[i] for i in range(n_models))
        scaled = blended_logits / temp
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        return logloss(filtered_true, probs)

    # 방법 1: Grid Search
    print("\n=== Grid Search (5% 단위) ===")
    best_grid_score = float("inf")
    best_grid_weights = None
    steps = np.arange(0, 1.05, 0.05)

    for w in product(steps, repeat=n_models):
        if abs(sum(w) - 1.0) > 0.01:
            continue
        if any(wi < 0 for wi in w):
            continue
        score = blend_logloss(w)
        if score < best_grid_score:
            best_grid_score = score
            best_grid_weights = w

    print(f"  Best weights: {dict(zip(names, best_grid_weights))}")
    print(f"  LogLoss: {best_grid_score:.6f}")

    # 방법 2: Differential Evolution (전역 최적화)
    print("\n=== Differential Evolution ===")
    bounds = [(0, 1)] * n_models + [(0.1, 2.0)]  # weights + temperature
    result = differential_evolution(
        blend_logloss_with_temp,
        bounds,
        seed=42,
        maxiter=1000,
        tol=1e-10,
    )
    opt_weights = result.x[:-1]
    opt_temp = result.x[-1]
    opt_weights = opt_weights / opt_weights.sum()

    print(f"  Best weights: {dict(zip(names, opt_weights))}")
    print(f"  Best temperature: {opt_temp:.4f}")
    print(f"  LogLoss: {result.fun:.6f}")

    # 방법 3: Nelder-Mead (로컬 최적화, grid search 결과부터)
    print("\n=== Nelder-Mead (fine-tuning) ===")
    x0 = list(best_grid_weights) + [0.5]
    result_nm = minimize(
        blend_logloss_with_temp,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-10, "fatol": 1e-10},
    )
    nm_weights = result_nm.x[:-1]
    nm_temp = result_nm.x[-1]
    nm_weights = nm_weights / nm_weights.sum()

    print(f"  Best weights: {dict(zip(names, nm_weights))}")
    print(f"  Best temperature: {nm_temp:.4f}")
    print(f"  LogLoss: {result_nm.fun:.6f}")

    # === Test submission 생성 ===
    print("\n=== Test Submission 생성 ===")

    # Test logits 로드
    test_logits = {}
    for name in names:
        if name == "structural":
            # exp020 test logits (5 fold 평균)
            test_sub = pd.read_csv(os.path.join(EXP020_DIR, "submission_struct_T1.0.csv"))
            probs = test_sub[["unstable_prob", "stable_prob"]].values
            test_logits[name] = np.log(np.clip(probs, 1e-15, 1))
        else:
            # exp018 individual model logits
            logit_path = os.path.join(EXP018_DIR, f"test_logits_{name}.npy")
            if os.path.exists(logit_path):
                test_logits[name] = np.load(logit_path)
            else:
                print(f"  WARNING: {logit_path} not found")

    # exp018 test logits 확인
    exp018_logit_files = [f for f in os.listdir(EXP018_DIR) if "test_logit" in f and f.endswith(".npy")]
    print(f"  exp018 logit files: {exp018_logit_files}")

    # 사용 가능한 모델만
    available = [n for n in names if n in test_logits]
    print(f"  Available for test: {available}")

    if len(available) == len(names):
        test_list = [test_logits[n] for n in names]
        ids = pd.read_csv(os.path.join(EXP020_DIR, "submission_struct_T1.0.csv"))["id"].values

        # DE 최적 가중치로 생성
        for label, weights, temp in [
            ("de", opt_weights, opt_temp),
            ("nm", nm_weights, nm_temp),
            ("grid", np.array(best_grid_weights), 0.47),
        ]:
            w = weights / weights.sum()
            blended = sum(w[i] * test_list[i] for i in range(len(names)))
            for t in [temp, 0.3, 0.4, 0.5]:
                scaled = blended / t
                probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
                sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
                fname = f"submission_{label}_T{t:.2f}.csv"
                sub.to_csv(os.path.join(SAVE_DIR, fname), index=False)
                print(f"  {fname}: range [{probs[:, 0].min():.6f}, {probs[:, 0].max():.6f}]")
    else:
        # 가용한 것만으로 블렌딩
        print("\n  일부 모델 logits 없음 → 가용한 submission으로 블렌딩")

        # 모든 T=1.0 submission 로드
        subs = {
            "exp018_stack": pd.read_csv(os.path.join(EXP018_DIR, "submission_stack_T1.0.csv")),
            "exp020_struct": pd.read_csv(os.path.join(EXP020_DIR, "submission_struct_T1.0.csv")),
        }

        ids = subs["exp018_stack"]["id"].values
        sub_probs = {k: v[["unstable_prob", "stable_prob"]].values for k, v in subs.items()}

        # 2모델 최적화
        def blend2_loss(params):
            w, t = params
            blended = w * np.log(np.clip(sub_probs["exp018_stack"], 1e-15, 1)) + \
                      (1 - w) * np.log(np.clip(sub_probs["exp020_struct"], 1e-15, 1))
            scaled = blended / t
            probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            return 0  # test엔 label 없으니 grid로만

        for w in np.arange(0.3, 0.8, 0.05):
            for t in [0.3, 0.4, 0.47, 0.5]:
                log_blend = w * np.log(np.clip(sub_probs["exp018_stack"], 1e-15, 1)) + \
                            (1 - w) * np.log(np.clip(sub_probs["exp020_struct"], 1e-15, 1))
                scaled = log_blend / t
                probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
                sub = pd.DataFrame({"id": ids, "unstable_prob": probs[:, 0], "stable_prob": probs[:, 1]})
                fname = f"submission_e18_{int(w*100)}_s20_{int((1-w)*100)}_T{t:.2f}.csv"
                sub.to_csv(os.path.join(SAVE_DIR, fname), index=False)

        print("  2-model blending submissions generated")

    print("\nDone!")


if __name__ == "__main__":
    main()
