"""
Phase 6: 모델 해석.
최종 모델 QRF(onehot, n_estimators=363, max_depth=14, min_samples_leaf=10)에 대해
Feature Importance / SHAP / PDP / 오류분석을 수행한다.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from quantile_forest import RandomForestQuantileRegressor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from training.phase4_baselines import QUANTILES, evaluate  # noqa: E402
from training.phase5_train import NUMERIC_FEATURES, get_dong_cols  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
PLOTS_DIR = os.path.join(REPO_ROOT, "docs", "eda_plots")  # 기존 plots 폴더 재사용
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42
BEST_PARAMS = {"n_estimators": 363, "max_depth": 14, "min_samples_leaf": 10}


def load_and_fit():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val_features.csv"))
    dong_cols = get_dong_cols(train)
    cols = NUMERIC_FEATURES + dong_cols

    X_train = train[cols].values.astype(float)
    y_train = train["target_log"].values
    X_val = val[cols].values.astype(float)

    model = RandomForestQuantileRegressor(random_state=SEED, n_jobs=-1, **BEST_PARAMS)
    model.fit(X_train, y_train)
    return model, train, val, X_train, X_val, cols, dong_cols


def feature_importance(model, cols, dong_cols):
    print("\n" + "=" * 60)
    print("Feature Importance (MDI, gain 기반)")
    print("=" * 60)
    importances = pd.Series(model.feature_importances_, index=cols)
    non_dong = importances.drop(dong_cols)
    dong_agg = importances[dong_cols].sum()
    combined = pd.concat([non_dong, pd.Series({"지역(dong 합산)": dong_agg})]).sort_values(ascending=False)
    print(combined.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    combined.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Feature Importance (dong onehot 54개는 합산 표시)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=100)
    plt.close()
    return combined


def run_shap(model, X_train, X_val, cols, dong_cols, n_background=100, n_explain=200):
    print("\n" + "=" * 60)
    print("SHAP 분석 (median 예측 기준, 모델-불가지론적 Explainer)")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_train), size=n_background, replace=False)
    explain_idx = rng.choice(len(X_val), size=n_explain, replace=False)
    background = X_train[bg_idx]
    explain_set = X_val[explain_idx]

    def predict_median(X):
        return model.predict(X, quantiles=[0.5])

    explainer = shap.Explainer(predict_median, background, feature_names=cols)
    shap_values = explainer(explain_set)

    # dong onehot들을 하나로 합산해서 보기 쉽게(그림에는 원본 유지, 요약표만 합산)
    sv_df = pd.DataFrame(shap_values.values, columns=cols)
    non_dong_mean_abs = sv_df.drop(columns=dong_cols).abs().mean().sort_values(ascending=False)
    dong_mean_abs = sv_df[dong_cols].abs().sum(axis=1).mean()
    print("\n평균 |SHAP| (지역은 54개 dong 합산):")
    combined = pd.concat([non_dong_mean_abs, pd.Series({"지역(dong 합산)": dong_mean_abs})]).sort_values(ascending=False)
    print(combined.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    combined.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("SHAP 평균 절대값 (dong onehot 54개는 합산 표시)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"), dpi=100)
    plt.close()

    return shap_values, explain_idx, combined


def pdp(model, train, cols, dong_cols, feature, grid_size=20, sample_n=1500, seed=SEED):
    """전통적 PDP: 표본 sample_n개를 뽑아 feature를 grid로 스윕하며 나머지는 그대로 두고 평균 예측."""
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(train), size=min(sample_n, len(train)), replace=False)
    X_sample = train[cols].values.astype(float)[sample_idx]

    f_idx = cols.index(feature)
    lo, hi = np.percentile(train[feature], [2, 98])
    grid = np.linspace(lo, hi, grid_size)

    means = []
    for v in grid:
        X_mod = X_sample.copy()
        X_mod[:, f_idx] = v
        pred = model.predict(X_mod, quantiles=[0.5])
        means.append(np.expm1(pred).mean())
    return grid, np.array(means)


def error_analysis(model, val, X_val, cols):
    print("\n" + "=" * 60)
    print("오류 분석")
    print("=" * 60)
    pred_log = model.predict(X_val, quantiles=[0.5])
    pred_manwon = np.expm1(pred_log)
    actual_manwon = np.expm1(val["target_log"].values)
    residual = actual_manwon - pred_manwon
    abs_error = np.abs(residual)

    val = val.copy()
    val["pred_median"] = pred_manwon
    val["abs_error"] = abs_error
    val["residual"] = residual

    print(f"\n전체 MAE: {abs_error.mean():.1f}만원")

    dong_cols_present = [c for c in val.columns if c.startswith("dong_")]
    # onehot에서 실제 동 이름 복원
    dong_names = []
    for i in range(len(val)):
        active = [c.replace("dong_", "") for c in dong_cols_present if val.iloc[i][c] == 1]
        dong_names.append(active[0] if active else "기타")
    val["_dong"] = dong_names

    print("\n--- is_jeonse별 MAE ---")
    print(val.groupby("is_jeonse")["abs_error"].agg(["mean", "count"]))

    print("\n--- 동별 MAE (표본 20건 이상, 상위 10개 나쁜 순) ---")
    dong_mae = val.groupby("_dong")["abs_error"].agg(["mean", "count"])
    dong_mae = dong_mae[dong_mae["count"] >= 20].sort_values("mean", ascending=False)
    print(dong_mae.head(10))

    print("\n--- 절대오차 상위 10건 (가장 많이 틀린 케이스) ---")
    worst = val.nlargest(10, "abs_error")[["_dong", "totalFloorAr", "building_age", "is_jeonse",
                                            "target_log", "pred_median", "abs_error"]]
    worst["actual_manwon"] = np.expm1(worst["target_log"])
    print(worst[["_dong", "totalFloorAr", "building_age", "is_jeonse", "actual_manwon", "pred_median", "abs_error"]]
          .to_string(index=False))

    return val


def main():
    model, train, val, X_train, X_val, cols, dong_cols = load_and_fit()

    fi = feature_importance(model, cols, dong_cols)
    shap_values, explain_idx, shap_summary = run_shap(model, X_train, X_val, cols, dong_cols)

    print("\n" + "=" * 60)
    print("PDP (Partial Dependence)")
    print("=" * 60)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, feature in zip(axes, ["totalFloorAr", "building_age"]):
        grid, means = pdp(model, train, cols, dong_cols, feature)
        ax.plot(grid, means)
        ax.set_xlabel(feature)
        ax.set_ylabel("예측 보증금(만원, 평균)")
        ax.set_title(f"PDP: {feature}")
        print(f"\n{feature}: grid={grid.round(1).tolist()}")
        print(f"        pred={means.round(0).tolist()}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pdp.png"), dpi=100)
    plt.close()

    val_with_error = error_analysis(model, val, X_val, cols)


if __name__ == "__main__":
    main()
