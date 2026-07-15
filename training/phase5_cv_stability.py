"""
Phase 5 (계속): 최종 후보 QRF(onehot)의 5-Fold CV 안정성 확인.
튜닝된 하이퍼파라미터(n_estimators=363, max_depth=14, min_samples_leaf=10)를 고정하고
train 내에서 5-fold로 학습/평가를 반복해 Pinball/Coverage가 특정 분할에만 우연히 좋았던 게
아닌지 확인한다. (재튜닝 아님 — 이미 고른 하이퍼파라미터의 분산만 확인)
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from training.phase4_baselines import QUANTILES, evaluate  # noqa: E402
from training.phase5_train import NUMERIC_FEATURES, get_dong_cols  # noqa: E402

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
SEED = 42
BEST_PARAMS = {"n_estimators": 363, "max_depth": 14, "min_samples_leaf": 10}


def main():
    from quantile_forest import RandomForestQuantileRegressor

    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"))
    dong_cols = get_dong_cols(train)
    cols = NUMERIC_FEATURES + dong_cols
    X = train[cols].values.astype(float)
    y_log = train["target_log"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        model = RandomForestQuantileRegressor(random_state=SEED, n_jobs=-1, **BEST_PARAMS)
        model.fit(X[tr_idx], y_log[tr_idx])
        pred_log = model.predict(X[te_idx], quantiles=QUANTILES)  # (n, n_quantiles)

        preds_manwon = {q: np.expm1(pred_log[:, i]) for i, q in enumerate(QUANTILES)}
        y_true_manwon = np.expm1(y_log[te_idx])

        m = evaluate(y_true_manwon, preds_manwon)
        m["fold"] = fold_i
        fold_results.append(m)
        print(f"Fold {fold_i}: Pinball={m['mean_pinball']:.1f}  CRPS={m['crps_approx']:.1f}  "
              f"Coverage@90={m['coverage_90']:.3f}  MAE={m['mae']:.1f}")

    df = pd.DataFrame(fold_results)
    print("\n" + "=" * 60)
    print("5-Fold CV 요약 (mean ± std)")
    print("=" * 60)
    for col in ["mean_pinball", "crps_approx", "coverage_90", "interval_width", "mae", "rmse"]:
        print(f"  {col}: {df[col].mean():.2f} ± {df[col].std():.2f}")

    df.to_csv(os.path.join(REPO_ROOT, "data", "phase5_cv_results.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
