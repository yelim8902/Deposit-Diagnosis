"""
Phase 3: 피처 엔지니어링.
train에서만 통계를 계산해 val/test에 적용한다(누수 방지). 지역 인코딩은 Target Encoding과
One-hot 둘 다 만들어서 Phase 5에서 실제 모델 성능으로 비교한다.
"""

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

# 면적 구간 컷오프 — 국내 임대 매물 관행상 통용되는 기준(원룸/투룸/쓰리룸+). 데이터로 적합한 값이 아니라
# 도메인 관행에 따른 것임을 명시.
AREA_BAND_BINS = [0, 20, 40, np.inf]
AREA_BAND_LABELS = ["원룸(~20㎡)", "투룸(20~40㎡)", "쓰리룸+(40㎡~)"]

TARGET_ENCODING_SMOOTHING_K = 20  # James-Stein류 축소 인코딩의 스무딩 강도


def load_splits():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train, val, test


def add_basic_features(df, buildyear_median):
    df = df.copy()
    df["buildYear_missing"] = df["buildYear"].isna().astype(int)
    df["buildYear_imputed"] = df["buildYear"].fillna(buildyear_median)
    df["building_age"] = df["dealYear"] - df["buildYear_imputed"]
    df["area_band"] = pd.cut(df["totalFloorAr"], bins=AREA_BAND_BINS, labels=AREA_BAND_LABELS)
    df["target_log"] = np.log1p(df["deposit_manwon"])
    return df


def fit_dong_target_encoding(train_df, smoothing_k=TARGET_ENCODING_SMOOTHING_K):
    """
    train에서만 학습하는 지역(법정동) Target Encoding.
    희소한 동은 구(sigungu) 평균 쪽으로 축소(shrinkage)해서 과적합을 막는다.
    encoded = (n_dong * mean_dong + k * mean_sigungu) / (n_dong + k)
    """
    global_mean = train_df["target_log"].mean()
    sigungu_stats = train_df.groupby("sigungu_name")["target_log"].mean()
    dong_stats = train_df.groupby(["sigungu_name", "umdNm"])["target_log"].agg(["mean", "count"])

    encoding_map = {}
    for (sigungu, dong), row in dong_stats.iterrows():
        sigungu_mean = sigungu_stats.get(sigungu, global_mean)
        n = row["count"]
        encoded = (n * row["mean"] + smoothing_k * sigungu_mean) / (n + smoothing_k)
        encoding_map[(sigungu, dong)] = encoded

    return encoding_map, sigungu_stats, global_mean


def apply_dong_target_encoding(df, encoding_map, sigungu_stats, global_mean):
    df = df.copy()

    def lookup(row):
        key = (row["sigungu_name"], row["umdNm"])
        if key in encoding_map:
            return encoding_map[key]
        return sigungu_stats.get(row["sigungu_name"], global_mean)

    df["dong_target_enc"] = df.apply(lookup, axis=1)
    return df


def add_onehot_dong(train, val, test):
    """One-hot은 train의 동 목록을 기준으로 컬럼을 고정하고 val/test에 동일하게 적용(없는 동은 전부 0)."""
    dong_categories = sorted(train["umdNm"].unique())

    def onehot(df):
        d = pd.get_dummies(df["umdNm"], prefix="dong")
        for cat in dong_categories:
            col = f"dong_{cat}"
            if col not in d.columns:
                d[col] = 0
        return d[[f"dong_{c}" for c in dong_categories]]

    return onehot(train), onehot(val), onehot(test)


def main():
    train, val, test = load_splits()
    buildyear_median = train["buildYear"].median()
    print(f"buildYear 결측 대체값(train median): {buildyear_median}")

    train = add_basic_features(train, buildyear_median)
    val = add_basic_features(val, buildyear_median)
    test = add_basic_features(test, buildyear_median)

    print("\n면적 구간 분포 (train):")
    print(train["area_band"].value_counts())

    encoding_map, sigungu_stats, global_mean = fit_dong_target_encoding(train)
    train = apply_dong_target_encoding(train, encoding_map, sigungu_stats, global_mean)
    val = apply_dong_target_encoding(val, encoding_map, sigungu_stats, global_mean)
    test = apply_dong_target_encoding(test, encoding_map, sigungu_stats, global_mean)

    print(f"\nTarget Encoding — train 동 개수: {len(encoding_map)}, global_mean(log): {global_mean:.3f}")
    val_unseen = set(zip(val["sigungu_name"], val["umdNm"])) - set(encoding_map.keys())
    test_unseen = set(zip(test["sigungu_name"], test["umdNm"])) - set(encoding_map.keys())
    print(f"val에서 train에 없던 (구,동) 조합: {len(val_unseen)}개 → 구 평균으로 fallback")
    print(f"test에서 train에 없던 (구,동) 조합: {len(test_unseen)}개 → 구 평균으로 fallback")

    train_oh, val_oh, test_oh = add_onehot_dong(train, val, test)
    print(f"\nOne-hot 지역 피처 차원: {train_oh.shape[1]}개 컬럼")

    feature_cols = ["totalFloorAr", "buildYear_imputed", "buildYear_missing",
                     "building_age", "time_index", "is_jeonse", "dong_target_enc", "target_log"]

    for name, df, oh in [("train", train, train_oh), ("val", val, val_oh), ("test", test, test_oh)]:
        out = pd.concat([df[feature_cols].reset_index(drop=True), oh.reset_index(drop=True)], axis=1)
        out.to_csv(os.path.join(PROCESSED_DIR, f"{name}_features.csv"), index=False, encoding="utf-8-sig")
        print(f"{name}_features.csv 저장: {out.shape}")

    print("\n최종 피처 목록 (Target Encoding 버전):")
    for c in feature_cols[:-1]:
        print(f"  - {c}")
    print(f"  - dong_onehot × {train_oh.shape[1]} (One-hot 버전, 별도 컬럼으로 같은 파일에 포함)")
    print("타겟: target_log (= log1p(deposit_manwon))")


if __name__ == "__main__":
    main()
