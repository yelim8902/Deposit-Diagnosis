"""
Phase 7 준비: (1) Test셋 최종 1회 평가 (2) 프로덕션 배포용 모델 학습·저장.

프로토콜 원칙("Test는 최종 1회만 사용")에 따라, 지금까지 val로만 모델을 선택했고
test는 한 번도 안 봤다. 여기서 딱 한 번 평가하고 그 결과를 그대로 기록한다.
그 다음, 실제 서비스에 넣을 모델은 이제 더 이상 모델 선택이 남아있지 않으므로
train+val+test 전체를 합쳐 재학습해서 데이터를 최대한 활용한다 (일반적인 배포 관행).
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from training.phase4_baselines import QUANTILES, evaluate  # noqa: E402
from training.phase5_train import NUMERIC_FEATURES, get_dong_cols  # noqa: E402

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_PATH = os.path.join(REPO_ROOT, "models", "module_a_qrf.joblib")
SEED = 42
BEST_PARAMS = {"n_estimators": 363, "max_depth": 14, "min_samples_leaf": 10}


def main():
    from quantile_forest import RandomForestQuantileRegressor

    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val_features.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test_features.csv"))
    dong_cols = get_dong_cols(train)
    cols = NUMERIC_FEATURES + dong_cols

    # ── (1) Test 최종 1회 평가: train으로만 학습한 기존 선택 모델을 test에 적용 ──
    X_train = train[cols].values.astype(float)
    y_train = train["target_log"].values
    X_test = test[cols].values.astype(float)
    y_test_manwon = np.expm1(test["target_log"].values)

    model_for_test = RandomForestQuantileRegressor(random_state=SEED, n_jobs=-1, **BEST_PARAMS)
    model_for_test.fit(X_train, y_train)
    pred_log = model_for_test.predict(X_test, quantiles=QUANTILES)
    preds_manwon = {q: np.expm1(pred_log[:, i]) for i, q in enumerate(QUANTILES)}

    test_metrics = evaluate(y_test_manwon, preds_manwon)
    print("=" * 60)
    print("Test셋 최종 1회 평가 (train만으로 학습한 모델 → test 예측)")
    print("=" * 60)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.2f}")

    # ── (2) 배포용 최종 모델: train+val+test 전체로 재학습 ──
    full = pd.concat([train, val, test], ignore_index=True)
    X_full = full[cols].values.astype(float)
    y_full = full["target_log"].values

    final_model = RandomForestQuantileRegressor(random_state=SEED, n_jobs=-1, **BEST_PARAMS)
    final_model.fit(X_full, y_full)

    # is_jeonse 샘플링용 전역 p_jeonse, buildYear 결측 대체용 median 등 메타 정보 저장
    meta = {
        "feature_cols": cols,
        "numeric_features": NUMERIC_FEATURES,
        "dong_cols": dong_cols,
        "quantiles": QUANTILES,
        "global_p_jeonse": float(full["is_jeonse"].mean()),
        "buildyear_median": float(train["buildYear_imputed"].median()),
        "best_params": BEST_PARAMS,
        "test_metrics": test_metrics,
        "trained_on_n": len(full),
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final_model, "meta": meta}, MODEL_PATH)
    print(f"\n최종 모델 저장: {MODEL_PATH} (train+val+test 총 {len(full)}건으로 재학습)")

    with open(os.path.join(REPO_ROOT, "models", "module_a_qrf_meta.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in meta.items() if k != "test_metrics"} | {"test_metrics": test_metrics},
                   f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
