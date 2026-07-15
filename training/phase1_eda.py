"""
Phase 1: EDA (탐색적 데이터 분석).
data/raw/rent_deals.csv를 분석해서 docs/eda_plots/*.png 와 docs/EDA_결과.md를 만든다.
숫자는 전부 이 스크립트의 실제 출력에서 나온 것만 문서에 옮긴다 (지어내지 않음).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "rent_deals.csv")
PLOTS_DIR = os.path.join(REPO_ROOT, "docs", "eda_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams["font.family"] = "AppleGothic"  # macOS 한글 폰트. 없으면 깨질 수 있음(라벨은 영문 병기)
plt.rcParams["axes.unicode_minus"] = False


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["deposit_manwon"] = df["deposit"].astype(str).str.replace(",", "").astype(float)
    df["is_jeonse"] = df["monthlyRent"].fillna(0) == 0
    df["time_index"] = df["dealYear"] * 12 + df["dealMonth"]
    return df


def section_1_1_profiling(df):
    print("\n" + "=" * 60)
    print("1-1. 기본 프로파일링")
    print("=" * 60)
    print(f"행 수: {len(df)}, 열 수: {df.shape[1]}")
    print("\n컬럼별 결측치 비율(%):")
    print((df.isna().mean() * 100).round(2))
    print(f"\n중복 행 수: {df.duplicated().sum()}")
    print("\n컬럼별 고유값 수:")
    print(df.nunique())


def section_1_2_target(df):
    print("\n" + "=" * 60)
    print("1-2. 타겟 변수(보증금) 분석")
    print("=" * 60)

    d = df["deposit_manwon"]
    print("\n[전체] 기술통계 (단위: 만원)")
    print(d.describe())
    print(f"skew: {d.skew():.3f}, kurtosis: {d.kurtosis():.3f}")
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    print(f"\n분위수: {dict(zip(qs, d.quantile(qs).round(1)))}")

    for label, sub in [("전세", df[df.is_jeonse]), ("월세", df[~df.is_jeonse])]:
        s = sub["deposit_manwon"]
        print(f"\n[{label}] n={len(s)}, mean={s.mean():.0f}, median={s.median():.0f}, "
              f"std={s.std():.0f}, skew={s.skew():.2f}")

    log_d = np.log1p(d)
    print(f"\n[로그변환 후] skew: {log_d.skew():.3f}, kurtosis: {log_d.kurtosis():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(d[d < d.quantile(0.99)], bins=60)
    axes[0, 0].set_title("Deposit (raw, <99pct)")
    axes[0, 1].hist(log_d, bins=60)
    axes[0, 1].set_title("Deposit (log1p)")
    axes[1, 0].hist(df[df.is_jeonse]["deposit_manwon"].clip(upper=df["deposit_manwon"].quantile(0.99)), bins=60, alpha=0.7, label="jeonse")
    axes[1, 0].hist(df[~df.is_jeonse]["deposit_manwon"].clip(upper=df["deposit_manwon"].quantile(0.99)), bins=60, alpha=0.7, label="wolse")
    axes[1, 0].legend()
    axes[1, 0].set_title("Deposit by contract type (clipped at 99pct)")
    axes[1, 1].hist(np.log1p(df[df.is_jeonse]["deposit_manwon"]), bins=60, alpha=0.7, label="jeonse")
    axes[1, 1].hist(np.log1p(df[~df.is_jeonse]["deposit_manwon"]), bins=60, alpha=0.7, label="wolse")
    axes[1, 1].legend()
    axes[1, 1].set_title("log1p(Deposit) by contract type")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "target_distribution.png"), dpi=100)
    plt.close()

    return {
        "skew_raw": d.skew(), "kurtosis_raw": d.kurtosis(),
        "skew_log": log_d.skew(), "kurtosis_log": log_d.kurtosis(),
        "n_jeonse": int(df.is_jeonse.sum()), "n_wolse": int((~df.is_jeonse).sum()),
        "jeonse_mean": df[df.is_jeonse]["deposit_manwon"].mean(),
        "wolse_mean": df[~df.is_jeonse]["deposit_manwon"].mean(),
        "quantiles": d.quantile(qs).round(1).to_dict(),
    }


def section_1_3_outliers(df):
    print("\n" + "=" * 60)
    print("1-3. 이상치 탐지")
    print("=" * 60)
    d = df["deposit_manwon"]

    n_zero = (d == 0).sum()
    print(f"보증금 0원: {n_zero}건 ({n_zero/len(d)*100:.2f}%)")

    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr_outlier = ((d < lo) | (d > hi)).sum()
    print(f"IQR 기준 이상치 (< {lo:.0f} or > {hi:.0f}): {n_iqr_outlier}건 ({n_iqr_outlier/len(d)*100:.2f}%)")

    z = np.abs(stats.zscore(d))
    n_z_outlier = (z > 3).sum()
    print(f"z-score>3 이상치: {n_z_outlier}건 ({n_z_outlier/len(d)*100:.2f}%)")

    print(f"최댓값: {d.max():.0f}만원, 상위 5개: {sorted(d.values)[-5:]}")
    return {"n_zero": int(n_zero), "iqr_bounds": (lo, hi), "n_iqr_outlier": int(n_iqr_outlier),
            "n_z_outlier": int(n_z_outlier), "max_deposit": d.max()}


def section_1_4_relationships(df):
    print("\n" + "=" * 60)
    print("1-4. 피처별 관계 분석")
    print("=" * 60)

    d = df["deposit_manwon"]
    log_d = np.log1p(d)

    area_corr = df["totalFloorAr"].corr(d)
    area_corr_log = df["totalFloorAr"].corr(log_d)
    print(f"전용면적 vs 보증금 상관계수: raw={area_corr:.3f}, log={area_corr_log:.3f}")

    year_corr = df["buildYear"].corr(d)
    print(f"건축연도 vs 보증금 상관계수: {year_corr:.3f}")

    time_corr = df["time_index"].corr(d)
    print(f"계약시점(time_index) vs 보증금 상관계수: {time_corr:.3f}")

    monthly_median = df.groupby("time_index")["deposit_manwon"].median()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sample = df.sample(min(5000, len(df)), random_state=42)
    axes[0, 0].scatter(sample["totalFloorAr"], sample["deposit_manwon"], s=3, alpha=0.3)
    axes[0, 0].set_xlim(0, 150)
    axes[0, 0].set_ylim(0, df["deposit_manwon"].quantile(0.99))
    axes[0, 0].set_title(f"Area vs Deposit (r={area_corr:.2f})")

    axes[0, 1].scatter(sample["buildYear"], sample["deposit_manwon"], s=3, alpha=0.3)
    axes[0, 1].set_ylim(0, df["deposit_manwon"].quantile(0.99))
    axes[0, 1].set_title(f"BuildYear vs Deposit (r={year_corr:.2f})")

    axes[1, 0].plot(monthly_median.index, monthly_median.values)
    axes[1, 0].set_title("Median deposit over time")

    dong_counts = df["umdNm"].value_counts()
    top_dongs = dong_counts.head(15).index
    box_data = [df[df.umdNm == dong]["deposit_manwon"].clip(upper=df["deposit_manwon"].quantile(0.95)) for dong in top_dongs]
    axes[1, 1].boxplot(box_data, labels=top_dongs, vert=False)
    axes[1, 1].set_title("Deposit by dong (top 15 by count)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_relationships.png"), dpi=100)
    plt.close()

    numeric_cols = ["totalFloorAr", "buildYear", "time_index", "monthlyRent", "deposit_manwon"]
    corr_matrix = df[numeric_cols].corr()
    print("\n상관행렬:")
    print(corr_matrix.round(3))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_matrix.png"), dpi=100)
    plt.close()

    return {"area_corr": area_corr, "area_corr_log": area_corr_log,
            "year_corr": year_corr, "time_corr": time_corr,
            "corr_matrix": corr_matrix}


def section_1_5_cardinality(df):
    print("\n" + "=" * 60)
    print("1-5. 카디널리티 / 희소성")
    print("=" * 60)
    dong_counts = df["umdNm"].value_counts()
    print(f"법정동 고유값 수: {df['umdNm'].nunique()}")
    print(f"동별 표본 수 - min={dong_counts.min()}, median={dong_counts.median():.0f}, max={dong_counts.max()}")
    n_sparse = (dong_counts < 10).sum()
    print(f"표본 10건 미만 동: {n_sparse}개 / 전체 {len(dong_counts)}개")
    print("\n표본 10건 미만 동 목록:")
    print(dong_counts[dong_counts < 10])
    return {"n_dong": df["umdNm"].nunique(), "n_sparse_dong": int(n_sparse),
            "dong_counts": dong_counts}


def main():
    df = load_data()
    section_1_1_profiling(df)
    target_result = section_1_2_target(df)
    outlier_result = section_1_3_outliers(df)
    rel_result = section_1_4_relationships(df)
    card_result = section_1_5_cardinality(df)
    print("\n" + "=" * 60)
    print(f"플롯 저장 위치: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
