"""
[A] 숨은 선순위 보증금 분포 추정 — KDE + 몬테카를로.

한계 (기획안 원칙 "숫자를 지어내지 마라"에 따라 명시):
- 면적대별로 조건화하지 않고 지역·전세/월세 구분만으로 KDE를 적합한다.
  (실거래 표본이 충분치 않으면 면적 조건화 시 표본이 더 희박해지기 때문)
"""

import numpy as np
from scipy.stats import gaussian_kde


def _to_float(value):
    if value is None:
        return None
    v = value.strip().replace(",", "")
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def split_jeonse_wolse(deals):
    """deposit/monthlyRent 필드로 전세/월세 분리. 단위: 만원."""
    jeonse, wolse = [], []
    for d in deals:
        deposit = _to_float(d.get("deposit"))
        rent = _to_float(d.get("monthlyRent"))
        if deposit is None:
            continue
        if rent is None or rent == 0:
            jeonse.append(deposit)
        else:
            wolse.append(deposit)
    return jeonse, wolse


def fit_deposit_kde(deals):
    """
    전세/월세 보증금 KDE를 적합한다.
    표본이 너무 적은(<5) 쪽은 KDE 대신 나머지 쪽 분포로 대체(부트스트랩 불가하므로 명시적 fallback).
    """
    jeonse, wolse = split_jeonse_wolse(deals)
    p_jeonse = len(jeonse) / (len(jeonse) + len(wolse)) if (jeonse or wolse) else 0.0

    def _kde_or_none(sample):
        return gaussian_kde(sample) if len(sample) >= 5 else None

    kde_jeonse = _kde_or_none(jeonse)
    kde_wolse = _kde_or_none(wolse)

    if kde_jeonse is None and kde_wolse is None:
        raise ValueError("전세/월세 보증금 표본이 모두 부족해 KDE를 적합할 수 없음")
    if kde_jeonse is None:
        kde_jeonse = kde_wolse
    if kde_wolse is None:
        kde_wolse = kde_jeonse

    return {
        "p_jeonse": p_jeonse,
        "kde_jeonse": kde_jeonse,
        "kde_wolse": kde_wolse,
        "n_jeonse": len(jeonse),
        "n_wolse": len(wolse),
    }


def simulate_priority_deposits(kde_info, n_units, n_sim=10000, seed=None):
    """
    앞선 (n_units - 1)세대의 총 선순위 보증금을 몬테카를로로 시뮬레이션.
    반환: 시뮬레이션 결과 배열 (만원 단위), n_sim개.
    """
    rng = np.random.default_rng(seed)
    n_prior = max(n_units - 1, 0)
    if n_prior == 0:
        return np.zeros(n_sim)

    p_jeonse = kde_info["p_jeonse"]
    kde_jeonse = kde_info["kde_jeonse"]
    kde_wolse = kde_info["kde_wolse"]

    results = np.zeros(n_sim)
    for i in range(n_sim):
        is_jeonse = rng.random(n_prior) < p_jeonse
        n_j = int(is_jeonse.sum())
        n_w = n_prior - n_j
        total = 0.0
        if n_j:
            total += np.clip(kde_jeonse.resample(n_j, seed=rng)[0], 0, None).sum()
        if n_w:
            total += np.clip(kde_wolse.resample(n_w, seed=rng)[0], 0, None).sum()
        results[i] = total
    return results


def summarize(results_manwon):
    """results: 만원 단위 배열. 요약 통계 반환 (원 단위로 환산)."""
    won = results_manwon * 10_000
    return {
        "mean": float(np.mean(won)),
        "p05": float(np.percentile(won, 5)),
        "p95": float(np.percentile(won, 95)),
        "worst_p95": float(np.percentile(won, 95)),  # 상위 5%(=95th pct) 값
    }
