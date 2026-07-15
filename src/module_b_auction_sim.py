"""
[B] 경매 배당 시뮬레이션 — 사고 시 실제 회수액 분포.

한계 (명시):
- 최우선변제 총액은 개별 선순위 세대의 보증금을 하나씩 추적하지 않고,
  실거래 표본에서 "소액임차인 기준 이하 보증금 비율(p_small)"을 구해
  n_prior * p_small 세대가 최우선변제 대상이라고 근사한다.
- 낙찰가율은 경기도 연립·다세대 평균(config.py 주석 참고)으로 근사 — 단독/다가구 전용 통계 아님.
"""

import numpy as np

from .config import (
    AUCTION_RATE_MEAN,
    AUCTION_RATE_STD,
    AUCTION_COST_RATE,
    PRIORITY_REPAYMENT_TABLE,
    TARGET_BUILDING_REGION_CLASS,
)


def _region_repayment_params(region_class=TARGET_BUILDING_REGION_CLASS):
    for row in PRIORITY_REPAYMENT_TABLE:
        if row["region"] == region_class:
            return row
    raise ValueError(f"알 수 없는 지역구분: {region_class}")


def estimate_small_tenant_ratio(deposits_manwon, deposit_limit_won):
    """실거래 보증금 표본 중 소액임차인 기준(원) 이하 비율."""
    if not deposits_manwon:
        return 0.0
    limit_manwon = deposit_limit_won / 10_000
    below = sum(1 for d in deposits_manwon if d <= limit_manwon)
    return below / len(deposits_manwon)


def simulate_recovery(
    my_deposit_won,
    market_price_won,
    mortgage_won,
    priority_deposits_manwon,   # module_a의 시뮬레이션 결과 배열
    n_prior,
    small_tenant_ratio,
    region_class=TARGET_BUILDING_REGION_CLASS,
    n_sim=10000,
    seed=None,
):
    rng = np.random.default_rng(seed)
    region = _region_repayment_params(region_class)
    max_repayment_per_unit = region["max_repayment"]

    count_small = round(n_prior * small_tenant_ratio)
    priority_total_won = priority_deposits_manwon * 10_000

    # module_a 시뮬레이션 결과와 개수를 맞춰 페어링 (같은 인덱스로 상관관계 유지)
    idx = rng.integers(0, len(priority_total_won), size=n_sim) if len(priority_total_won) != n_sim else np.arange(n_sim)
    priority_won = priority_total_won[idx]

    auction_rates = np.clip(rng.normal(AUCTION_RATE_MEAN, AUCTION_RATE_STD, n_sim), 0.3, 1.3)
    winning_bid = market_price_won * auction_rates
    auction_cost = winning_bid * AUCTION_COST_RATE

    max_repayment_total = np.minimum(count_small * max_repayment_per_unit, winning_bid * 0.5)

    distributable = winning_bid - auction_cost - max_repayment_total - mortgage_won - priority_won
    distributable = np.clip(distributable, 0, None)

    my_recovery = np.clip(distributable, 0, my_deposit_won)
    return my_recovery


def summarize(recovery_won, my_deposit_won):
    full = np.isclose(recovery_won, my_deposit_won, rtol=0.01)
    zero = recovery_won <= (my_deposit_won * 0.01)
    partial = ~full & ~zero
    return {
        "full_recovery_pct": float(full.mean() * 100),
        "partial_recovery_pct": float(partial.mean() * 100),
        "total_loss_pct": float(zero.mean() * 100),
        "expected_recovery_won": float(recovery_won.mean()),
        "partial_recovery_avg_won": float(recovery_won[partial].mean()) if partial.any() else 0.0,
    }
