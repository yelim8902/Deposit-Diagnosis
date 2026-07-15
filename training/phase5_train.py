"""
Phase 5: 모델 학습 및 비교.
Linear QR / LightGBM Quantile / XGBoost Quantile / CatBoost MultiQuantile / QRF / NGBoost
6개를 val 기준 Pinball Loss로 Optuna 튜닝 후 비교한다. MDN은 Phase 1에서 잠재 다봉분포가
확인되지 않아 후보에서 제외(EDA_결과.md 참고).

트리 기반 4개(LightGBM/XGBoost/CatBoost/QRF)는 Target Encoding과 One-hot 둘 다 비교한다.
Linear QR과 NGBoost는 Target Encoding만 사용한다 — 54개 One-hot 더미와 결합 시
Linear QR은 불안정/저속(regularization 없는 statsmodels QuantReg), NGBoost는 얕은 기본
트리 특성상 인코딩 효과 검증에 시간을 쓸 가치가 낮다고 판단(시간 대비 정보량 낮음, 명시적 스코프 축소).
"""

import os
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # noqa (참고용, 실제로는 quantile_forest 사용)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from training.phase4_baselines import QUANTILES, pinball_loss, evaluate  # noqa: E402

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
N_TRIALS = 20
SEED = 42

NUMERIC_FEATURES = ["totalFloorAr", "buildYear_imputed", "buildYear_missing",
                     "building_age", "time_index", "is_jeonse"]


def load_features():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val_features.csv"))
    return train, val


def get_dong_cols(df):
    return [c for c in df.columns if c.startswith("dong_")]


def build_X(df, dong_cols, variant):
    if variant == "target_enc":
        cols = NUMERIC_FEATURES + ["dong_target_enc"]
    else:  # onehot
        cols = NUMERIC_FEATURES + dong_cols
    return df[cols].values.astype(float), cols


def to_manwon_quantile_dict(pred_log_matrix):
    """pred_log_matrix: shape (n_samples, n_quantiles) 로그스케일 → 만원 스케일 dict로 변환"""
    return {q: np.expm1(pred_log_matrix[:, i]) for i, q in enumerate(QUANTILES)}


# ══════════════════════════════════════════════════════════════
# Linear Quantile Regression (statsmodels) — target_enc만
# ══════════════════════════════════════════════════════════════

def run_linear_qr(train, val, dong_cols):
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg

    X_train, cols = build_X(train, dong_cols, "target_enc")
    X_val, _ = build_X(val, dong_cols, "target_enc")
    y_train = train["target_log"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    X_train_s = sm.add_constant(X_train_s)
    X_val_s = sm.add_constant(X_val_s, has_constant="add")

    preds = np.zeros((len(X_val_s), len(QUANTILES)))
    for i, q in enumerate(QUANTILES):
        model = QuantReg(y_train, X_train_s).fit(q=q, max_iter=2000)
        preds[:, i] = model.predict(X_val_s)
    preds = np.sort(preds, axis=1)  # quantile crossing 방지(사후 정렬)
    return to_manwon_quantile_dict(preds)


# ══════════════════════════════════════════════════════════════
# LightGBM Quantile — 분위수별 개별 모델, alpha=0.5로 튜닝 후 재사용
# ══════════════════════════════════════════════════════════════

def run_lightgbm(train, val, dong_cols, variant):
    import lightgbm as lgb

    X_train, _ = build_X(train, dong_cols, variant)
    X_val, _ = build_X(val, dong_cols, variant)
    y_train = train["target_log"].values
    y_val_log = val["target_log"].values

    def objective(trial):
        params = {
            "objective": "quantile", "alpha": 0.5,
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "verbosity": -1, "random_state": SEED,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return pinball_loss(y_val_log, pred, 0.5)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params

    preds = np.zeros((len(X_val), len(QUANTILES)))
    for i, q in enumerate(QUANTILES):
        params = dict(best, objective="quantile", alpha=q, verbosity=-1, random_state=SEED)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds[:, i] = model.predict(X_val)
    preds = np.sort(preds, axis=1)
    return to_manwon_quantile_dict(preds), best


# ══════════════════════════════════════════════════════════════
# XGBoost Quantile — reg:quantileerror, 한 모델로 다중분위수 동시 예측
# ══════════════════════════════════════════════════════════════

def run_xgboost(train, val, dong_cols, variant):
    import xgboost as xgb

    X_train, _ = build_X(train, dong_cols, variant)
    X_val, _ = build_X(val, dong_cols, variant)
    y_train = train["target_log"].values
    y_val_log = val["target_log"].values

    def fit_predict(params):
        model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=QUANTILES,
                                  random_state=SEED, **params)
        model.fit(X_train, y_train)
        return model.predict(X_val)  # shape (n, n_quantiles)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        pred = fit_predict(params)
        loss = np.mean([pinball_loss(y_val_log, pred[:, i], q) for i, q in enumerate(QUANTILES)])
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params

    preds = fit_predict(best)
    preds = np.sort(preds, axis=1)
    return to_manwon_quantile_dict(preds), best


# ══════════════════════════════════════════════════════════════
# CatBoost MultiQuantile — 한 모델로 다중분위수 동시 예측
# ══════════════════════════════════════════════════════════════

def run_catboost(train, val, dong_cols, variant):
    from catboost import CatBoostRegressor

    X_train, _ = build_X(train, dong_cols, variant)
    X_val, _ = build_X(val, dong_cols, variant)
    y_train = train["target_log"].values
    y_val_log = val["target_log"].values

    alpha_str = ",".join(str(q) for q in QUANTILES)

    def fit_predict(params):
        model = CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alpha_str}",
                                   random_state=SEED, verbose=False, **params)
        model.fit(X_train, y_train)
        return np.array(model.predict(X_val))  # shape (n, n_quantiles)

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 200, 600),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        }
        pred = fit_predict(params)
        loss = np.mean([pinball_loss(y_val_log, pred[:, i], q) for i, q in enumerate(QUANTILES)])
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params

    preds = fit_predict(best)
    preds = np.sort(preds, axis=1)
    return to_manwon_quantile_dict(preds), best


# ══════════════════════════════════════════════════════════════
# Quantile Regression Forest — 한 forest로 다중분위수 동시 예측
# ══════════════════════════════════════════════════════════════

def run_qrf(train, val, dong_cols, variant):
    from quantile_forest import RandomForestQuantileRegressor

    X_train, _ = build_X(train, dong_cols, variant)
    X_val, _ = build_X(val, dong_cols, variant)
    y_train = train["target_log"].values
    y_val_log = val["target_log"].values

    def fit_predict(params):
        model = RandomForestQuantileRegressor(random_state=SEED, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        return model.predict(X_val, quantiles=QUANTILES)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        pred = fit_predict(params)
        loss = np.mean([pinball_loss(y_val_log, pred[:, i], q) for i, q in enumerate(QUANTILES)])
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_params

    preds = fit_predict(best)
    preds = np.sort(preds, axis=1)
    return to_manwon_quantile_dict(preds), best


# ══════════════════════════════════════════════════════════════
# NGBoost — 분포(Normal) 자체를 예측, ppf로 분위수 추출. target_enc만.
# ══════════════════════════════════════════════════════════════

def run_ngboost(train, val, dong_cols):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal

    X_train, _ = build_X(train, dong_cols, "target_enc")
    X_val, _ = build_X(val, dong_cols, "target_enc")
    y_train = train["target_log"].values
    y_val_log = val["target_log"].values

    def fit_predict(params):
        model = NGBRegressor(Dist=Normal, random_state=SEED, verbose=False, **params)
        model.fit(X_train, y_train)
        dist = model.pred_dist(X_val)
        return np.stack([dist.ppf(q) for q in QUANTILES], axis=1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        }
        pred = fit_predict(params)
        loss = np.mean([pinball_loss(y_val_log, pred[:, i], q) for i, q in enumerate(QUANTILES)])
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    best = study.best_params

    preds = fit_predict(best)
    preds = np.sort(preds, axis=1)
    return to_manwon_quantile_dict(preds), best


# ══════════════════════════════════════════════════════════════

def main():
    train, val = load_features()
    dong_cols = get_dong_cols(train)
    y_true = np.expm1(val["target_log"].values)

    results = []
    best_params_log = {}

    def record(name, preds):
        m = evaluate(y_true, preds)
        m["model"] = name
        results.append(m)
        print(f"  -> Pinball={m['mean_pinball']:.1f}  CRPS={m['crps_approx']:.1f}  Cov90={m['coverage_90']:.3f}")

    t0 = time.time()

    print("[1/6] Linear Quantile Regression (target_enc)...")
    record("Linear QR (target_enc)", run_linear_qr(train, val, dong_cols))

    for variant in ["target_enc", "onehot"]:
        print(f"[2/6] LightGBM Quantile ({variant})...")
        preds, best = run_lightgbm(train, val, dong_cols, variant)
        best_params_log[f"lightgbm_{variant}"] = best
        record(f"LightGBM Quantile ({variant})", preds)

    for variant in ["target_enc", "onehot"]:
        print(f"[3/6] XGBoost Quantile ({variant})...")
        preds, best = run_xgboost(train, val, dong_cols, variant)
        best_params_log[f"xgboost_{variant}"] = best
        record(f"XGBoost Quantile ({variant})", preds)

    for variant in ["target_enc", "onehot"]:
        print(f"[4/6] CatBoost MultiQuantile ({variant})...")
        preds, best = run_catboost(train, val, dong_cols, variant)
        best_params_log[f"catboost_{variant}"] = best
        record(f"CatBoost MultiQuantile ({variant})", preds)

    for variant in ["target_enc", "onehot"]:
        print(f"[5/6] Quantile Regression Forest ({variant})...")
        preds, best = run_qrf(train, val, dong_cols, variant)
        best_params_log[f"qrf_{variant}"] = best
        record(f"QRF ({variant})", preds)

    print("[6/6] NGBoost (target_enc)...")
    preds, best = run_ngboost(train, val, dong_cols)
    best_params_log["ngboost"] = best
    record("NGBoost (target_enc)", preds)

    elapsed = time.time() - t0
    print(f"\n전체 학습·튜닝 소요시간: {elapsed/60:.1f}분")

    df = pd.DataFrame(results)[["model", "mean_pinball", "crps_approx", "coverage_90", "interval_width", "mae", "rmse"]]
    df.columns = ["모델", "Pinball↓", "CRPS_근사↓", "Coverage@90", "IntervalWidth↓", "MAE↓", "RMSE↓"]
    df = df.sort_values("Pinball↓")

    print("\n" + "=" * 100)
    print("Phase 5 모델 비교표 (val, 만원 단위) — Baseline 3 기준선: Pinball 712.6 / CRPS 1425.1")
    print("=" * 100)
    print(df.to_string(index=False))

    out_dir = os.path.join(REPO_ROOT, "data")
    df.to_csv(os.path.join(out_dir, "phase5_model_comparison.csv"), index=False, encoding="utf-8-sig")

    print("\n튜닝된 하이퍼파라미터:")
    for k, v in best_params_log.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
