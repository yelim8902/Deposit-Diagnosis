"""
Phase 0: 데이터 수집.
단독/다가구 전월세 실거래가를 수원시 4개 구, 최근 3년치로 최대한 넓게 수집해서
data/raw/rent_deals.csv 로 저장한다 (API 재호출 없이 반복 실험 가능하게).
"""

import os
import sys
import time
from datetime import date

import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import SERVICE_KEY, RENT_URL

SIGUNGU_LIST = [
    ("41111", "장안구"),
    ("41113", "권선구"),
    ("41115", "팔달구"),
    ("41117", "영통구"),
]


def year_months(n_months, today=None):
    today = today or date.today()
    months = []
    y, m = today.year, today.month
    for _ in range(n_months):
        months.append(f"{y}{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return months


def fetch_month(sigungu_cd, ym):
    params = {
        "serviceKey": SERVICE_KEY,
        "LAWD_CD": sigungu_cd,
        "DEAL_YMD": ym,
        "numOfRows": "1000",
        "pageNo": "1",
    }
    r = requests.get(RENT_URL, params=params, timeout=15)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r.text)
    code = root.find(".//resultCode")
    if code is not None and code.text not in ("00", "0", "000"):
        return []
    return [{c.tag: (c.text or "").strip() for c in item} for item in root.iter("item")]


def main():
    months = year_months(36)
    print(f"수집 기간: {months[-1]} ~ {months[0]} (총 {len(months)}개월)")
    print(f"수집 지역: {[name for _, name in SIGUNGU_LIST]}")

    all_rows = []
    for sigungu_cd, name in SIGUNGU_LIST:
        gu_count = 0
        for ym in months:
            items = fetch_month(sigungu_cd, ym)
            for it in items:
                it["sigungu_cd"] = sigungu_cd
                it["sigungu_name"] = name
            all_rows.extend(items)
            gu_count += len(items)
            time.sleep(0.05)  # 공공데이터포털 과호출 방지
        print(f"  {name}({sigungu_cd}): {gu_count}건")

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(REPO_ROOT, "data", "raw", "rent_deals.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n총 수집 건수: {len(df)}")
    print(f"저장 위치: {out_path}")

    if len(df) < 1000:
        print("\n*** 중단 조건 발동: 전체 표본 1,000건 미만. Phase 1로 넘어갈 수 없음. ***")
        return

    print("\n--- 연도별 건수 ---")
    print(df["dealYear"].value_counts().sort_index())

    print("\n--- 구별 건수 ---")
    print(df["sigungu_name"].value_counts())

    print("\n--- 주택유형(houseType)별 건수 ---")
    print(df["houseType"].value_counts())

    print("\n--- 법정동(umdNm) 고유값 수 ---")
    print(df["umdNm"].nunique())


if __name__ == "__main__":
    main()
