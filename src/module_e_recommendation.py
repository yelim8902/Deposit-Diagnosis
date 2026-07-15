"""[E] KB 금융상품 액션 — 기대손실과 상품 비용을 직접 손익 비교."""

from .config import HUG_GUARANTEE_PREMIUM_RATE, BUTTIMOK_LOAN_RATE


def recommend(expected_loss_won, my_deposit_won):
    premium = my_deposit_won * HUG_GUARANTEE_PREMIUM_RATE
    net_benefit = expected_loss_won - premium

    actions = []
    actions.append({
        "name": "HUG 전세보증보험",
        "cost_won": premium,
        "benefit_won": expected_loss_won,
        "net_won": net_benefit,
        "recommended": net_benefit > 0,
    })
    return actions
