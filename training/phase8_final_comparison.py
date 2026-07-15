"""
Phase 8: 최종 검증.
같은 건물(우만동 39-7)·같은 실거래 데이터·같은 seed로 기존 KDE 방식과 신규 학습모델 방식을
나란히 돌려서 결과가 얼마나 달라지는지 비교한다.
"""

import importlib.util
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.api_client import get_building_title, get_unit_areas, get_rent_transactions, filter_by_dong  # noqa: E402
from src import module_a_deposit_dist as new_module  # noqa: E402


def _load_legacy_module():
    path = os.path.join(REPO_ROOT, "archive", "module_a_deposit_dist_kde_legacy.py")
    spec = importlib.util.spec_from_file_location("legacy_module_a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recent_year_months(n=6):
    from datetime import date
    today = date.today()
    months, y, m = [], today.year, today.month
    for _ in range(n):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
        months.append(f"{y}{m:02d}")
    return months


def main():
    SIGUNGU, BJDONG, BUN, JI, DONG_NAME = "41115", "14000", "0039", "0007", "우만동"
    N_SIM, SEED = 10000, 42

    building = get_building_title(SIGUNGU, BJDONG, BUN, JI)
    n_units = int(building.get("fmlyCnt") or building.get("hhldCnt") or 1)
    areas, _ = get_unit_areas(SIGUNGU, BJDONG, BUN, JI)
    deals = get_rent_transactions(SIGUNGU, recent_year_months(6))
    filtered_deals, _ = filter_by_dong(deals, DONG_NAME)

    use_apr_day = building.get("useAprDay", "")
    build_year = int(use_apr_day[:4]) if use_apr_day[:4].isdigit() else None

    print(f"대상: {building.get('platPlc')} / 가구수 N={n_units} / 준공년도={build_year}")
    print(f"실거래 표본 {len(filtered_deals)}건, n_sim={N_SIM}, seed={SEED}")

    legacy = _load_legacy_module()
    legacy_kde_info = legacy.fit_deposit_kde(filtered_deals)
    legacy_sim = legacy.simulate_priority_deposits(legacy_kde_info, n_units, n_sim=N_SIM, seed=SEED)
    legacy_summary = legacy.summarize(legacy_sim)

    new_kde_info = new_module.fit_deposit_kde(filtered_deals)
    new_sim = new_module.simulate_priority_deposits(
        new_kde_info, n_units, area_list=areas, build_year=build_year, dong_name=DONG_NAME,
        n_sim=N_SIM, seed=SEED,
    )
    new_summary = new_module.summarize(new_sim)

    def fmt(d):
        return (f"평균 {d['mean']/1e8:.3f}억 / 90%구간 {d['p05']/1e8:.3f}억~{d['p95']/1e8:.3f}억 "
                f"/ 상위5% {d['worst_p95']/1e8:.3f}억")

    print("\n" + "=" * 70)
    print("기존 KDE (지역·전세월세만 구분)  vs  신규 QRF 학습모델 (면적·연식·지역 조건화)")
    print("=" * 70)
    print(f"[기존 KDE]   {fmt(legacy_summary)}")
    print(f"[신규 모델]  {fmt(new_summary)}")

    diff_pct = (new_summary["mean"] - legacy_summary["mean"]) / legacy_summary["mean"] * 100
    print(f"\n평균 차이: {diff_pct:+.1f}%")
    print(f"90% 구간 폭 — 기존: {(legacy_summary['p95']-legacy_summary['p05'])/1e8:.3f}억, "
          f"신규: {(new_summary['p95']-new_summary['p05'])/1e8:.3f}억")


if __name__ == "__main__":
    main()
