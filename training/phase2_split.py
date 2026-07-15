"""
Phase 2: 데이터 분할.
Phase 1에서 결정한 대로: 중복/0원 제거 → 시간 기반 분할(과거 60% train / 다음 20% val / 최근 20% test).
Test는 이후 Phase 5에서 최종 1회만 사용한다.
"""

import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "rent_deals.csv")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEED = 42


def load_and_clean():
    df = pd.read_csv(RAW_PATH)
    df["deposit_manwon"] = df["deposit"].astype(str).str.replace(",", "").astype(float)
    df["is_jeonse"] = df["monthlyRent"].fillna(0) == 0
    df["time_index"] = df["dealYear"] * 12 + df["dealMonth"]

    n_before = len(df)
    df = df.drop_duplicates()
    n_after_dedup = len(df)
    df = df[df["deposit_manwon"] > 0]
    n_after_zero = len(df)

    print(f"원본: {n_before}건")
    print(f"중복 제거 후: {n_after_dedup}건 (-{n_before - n_after_dedup})")
    print(f"0원 제거 후: {n_after_zero}건 (-{n_after_dedup - n_after_zero})")
    return df.reset_index(drop=True)


def time_based_split(df, train_ratio=0.6, val_ratio=0.2):
    df = df.sort_values("time_index").reset_index(drop=True)
    n = len(df)
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_cut]
    val = df.iloc[train_cut:val_cut]
    test = df.iloc[val_cut:]
    return train, val, test


def describe_split(name, split_df):
    return {
        "split": name,
        "n": len(split_df),
        "time_range": f"{split_df['time_index'].min()}~{split_df['time_index'].max()}",
        "deposit_mean": split_df["deposit_manwon"].mean(),
        "deposit_median": split_df["deposit_manwon"].median(),
        "jeonse_ratio": split_df["is_jeonse"].mean(),
        "area_mean": split_df["totalFloorAr"].mean(),
    }


def main():
    df = load_and_clean()
    train, val, test = time_based_split(df)

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        split_df.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False, encoding="utf-8-sig")

    print("\n분할 결과:")
    rows = [describe_split(n, d) for n, d in [("train", train), ("val", val), ("test", test)]]
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    print(f"\n비율: train {len(train)/len(df)*100:.1f}% / val {len(val)/len(df)*100:.1f}% / test {len(test)/len(df)*100:.1f}%")
    print(f"저장 위치: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
