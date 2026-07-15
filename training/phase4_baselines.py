"""
Phase 4: 베이스라인 설정.
Baseline 0(전체 평균) / 1(동별) / 2(동×면적구간별) / 3(현재 A모듈의 지역 KDE 샘플링, 이걸 이겨야 함)
전부 같은 지표(Pinball Loss, Coverage@90, Interval Width, CRPS 근사, MAE/RMSE)로 val에서 측정한다.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def load():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    train_feat = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"))
    val_feat = pd.read_csv(os.path.join(PROCESSED_DIR, "val_features.csv"))
    # area_band는 원본 train/val에는 있지만 *_features.csv에는 dong_onehot으로만 남아있으므로 원본에서 다시 가져옴
    train_feat["area_band"] = train["totalFloorAr"].pipe(
        lambda s: pd.cut(s, bins=[0, 20, 40, np.inf], labels=["원룸(~20㎡)", "투룸(20~40㎡)", "쓰리룸+(40㎡~)"])
    )
    val_feat["area_band"] = val["totalFloorAr"].pipe(
        lambda s: pd.cut(s, bins=[0, 20, 40, np.inf], labels=["원룸(~20㎡)", "투룸(20~40㎡)", "쓰리룸+(40㎡~)"])
    )
    train_feat["umdNm"] = train["umdNm"]
    val_feat["umdNm"] = val["umdNm"]
    train_feat["is_jeonse"] = train["is_jeonse"]
    val_feat["is_jeonse"] = val["is_jeonse"]
    return train_feat, val_feat


# ── 지표 ──────────────────────────────────────────────────────────

def pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def evaluate(y_true_manwon, quantile_preds_manwon):
    """quantile_preds_manwon: dict{quantile_level: array} (만원 단위, 원 스케일)"""
    pinballs = {q: pinball_loss(y_true_manwon, quantile_preds_manwon[q], q) for q in QUANTILES}
    mean_pinball = np.mean(list(pinballs.values()))
    crps_approx = 2 * mean_pinball  # discretized CRPS 근사 (문서에 명시)

    lo, hi = quantile_preds_manwon[0.05], quantile_preds_manwon[0.95]
    coverage_90 = np.mean((y_true_manwon >= lo) & (y_true_manwon <= hi))
    interval_width = np.mean(hi - lo)

    median_pred = quantile_preds_manwon[0.5]
    mae = np.mean(np.abs(y_true_manwon - median_pred))
    rmse = np.sqrt(np.mean((y_true_manwon - median_pred) ** 2))

    return {
        "mean_pinball": mean_pinball, "crps_approx": crps_approx,
        "coverage_90": coverage_90, "interval_width": interval_width,
        "mae": mae, "rmse": rmse,
    }


# ── Baseline 0: 전체 평균(=전체 분포) ─────────────────────────────

def baseline_0(train, val):
    train_quantiles_log = train["target_log"].quantile(QUANTILES)
    preds = {q: np.full(len(val), np.expm1(train_quantiles_log[q])) for q in QUANTILES}
    return preds


# ── Baseline 1: 법정동별 분포 ──────────────────────────────────────

def baseline_1(train, val):
    global_q = train["target_log"].quantile(QUANTILES)
    dong_q = train.groupby("umdNm")["target_log"].quantile(QUANTILES).unstack()

    preds = {q: np.zeros(len(val)) for q in QUANTILES}
    for i, dong in enumerate(val["umdNm"].values):
        row = dong_q.loc[dong] if dong in dong_q.index else global_q
        for q in QUANTILES:
            preds[q][i] = np.expm1(row[q])
    return preds


# ── Baseline 2: 법정동 × 면적구간별 분포 (표본 부족시 동 → 전체 fallback) ──

def baseline_2(train, val):
    global_q = train["target_log"].quantile(QUANTILES)
    dong_q = train.groupby("umdNm")["target_log"].quantile(QUANTILES).unstack()
    group = train.groupby(["umdNm", "area_band"], observed=True)["target_log"]
    counts = group.count()
    combo_q = group.quantile(QUANTILES).unstack()

    preds = {q: np.zeros(len(val)) for q in QUANTILES}
    n_fallback_to_dong = 0
    n_fallback_to_global = 0
    for i, (dong, band) in enumerate(zip(val["umdNm"].values, val["area_band"].values)):
        key = (dong, band)
        if key in combo_q.index and counts.get(key, 0) >= 10:
            row = combo_q.loc[key]
        elif dong in dong_q.index:
            row = dong_q.loc[dong]
            n_fallback_to_dong += 1
        else:
            row = global_q
            n_fallback_to_global += 1
        for q in QUANTILES:
            preds[q][i] = np.expm1(row[q])
    print(f"  (동×면적 표본<10 → 동 평균 fallback: {n_fallback_to_dong}건, 전체 fallback: {n_fallback_to_global}건)")
    return preds


# ── Baseline 3: 현재 A모듈의 지역 KDE 샘플링 (src/module_a_deposit_dist.py와 동일 방식) ──

def baseline_3(train, val, n_resample=20000, seed=42):
    rng = np.random.default_rng(seed)
    jeonse_deposits = train.loc[train.is_jeonse, "target_log"].pipe(np.expm1).values
    wolse_deposits = train.loc[~train.is_jeonse, "target_log"].pipe(np.expm1).values

    kde_jeonse = gaussian_kde(jeonse_deposits)
    kde_wolse = gaussian_kde(wolse_deposits)

    jeonse_samples = np.clip(kde_jeonse.resample(n_resample, seed=rng)[0], 0, None)
    wolse_samples = np.clip(kde_wolse.resample(n_resample, seed=rng)[0], 0, None)

    jeonse_q = {q: np.quantile(jeonse_samples, q) for q in QUANTILES}
    wolse_q = {q: np.quantile(wolse_samples, q) for q in QUANTILES}

    preds = {q: np.zeros(len(val)) for q in QUANTILES}
    for i, is_j in enumerate(val["is_jeonse"].values):
        source = jeonse_q if is_j else wolse_q
        for q in QUANTILES:
            preds[q][i] = source[q]
    return preds


def main():
    train, val = load()
    y_true = np.expm1(val["target_log"].values)

    baselines = {
        "Baseline 0 (전체 평균)": baseline_0,
        "Baseline 1 (동별 중앙값 분포)": baseline_1,
        "Baseline 2 (동×면적구간별 분포)": baseline_2,
        "Baseline 3 (현재 A모듈: 지역 KDE)": baseline_3,
    }

    results = []
    for name, fn in baselines.items():
        print(f"\n{name} 실행 중...")
        preds = fn(train, val)
        metrics = evaluate(y_true, preds)
        metrics["model"] = name
        results.append(metrics)

    df = pd.DataFrame(results)[["model", "mean_pinball", "crps_approx", "coverage_90", "interval_width", "mae", "rmse"]]
    df.columns = ["모델", "Pinball↓", "CRPS_근사↓", "Coverage@90", "IntervalWidth↓", "MAE↓", "RMSE↓"]
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}" if abs(x) > 10 else f"{x:.3f}")
    print("\n" + "=" * 100)
    print("베이스라인 성능표 (val, 만원 단위)")
    print("=" * 100)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(PROCESSED_DIR, "..", "baseline_results.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
