"""
[C] 사고 확률 예측 — 안선영·이상엽(2025, 주택금융연구 9(2)) 로지스틱 회귀 오즈비 + HUG 통계 캘리브레이션.

*** 중요한 한계 (반드시 리포트/발표에서 밝힐 것) ***
이것은 우리가 직접 학습시킨 분류기가 아니다. HUG 악성임대인 명단(127명)은 주소 단위
구조화 데이터가 없어 개별 건물에 매칭할 학습셋을 만들 수 없었다. 대신:
  1. 부채비율 구간별 승수는 안선영·이상엽(2025)이 HF·HUG 실제 보증사고 데이터 453,122건으로
     추정한 로지스틱 회귀 오즈비(Exp(B), 표10, 전부 p<0.001)를 그대로 가져다 쓴다 — 우리가
     지어낸 값이 아니라 논문에 실린 값이다.
  2. 이 오즈비를 우리가 계산한 "실질부채비율"(등기부 근저당 + A모듈 추정 선순위보증금)에 적용한다.
     논문 원문의 "부채비율"은 등기부 기준(근저당/시세)만 반영하므로, 다가구 특유의 숨은
     선순위를 포함한 실질부채비율에 그대로 적용하는 것은 근사(외삽)이며 논문이 검증한 범위를
     벗어난다는 점을 명시한다.
  3. 논문에는 "단독다가구" 건물유형 더미가 오즈비 0.815(음수 계수, p<0.01)로 나온다 — 즉 등기부
     기준 부채비율을 통제하면 다가구가 오히려 위험이 낮게 나온다. 그래서 "다가구니까 위험 가중치를
     추가로 곱한다"는 식의 별도 보정은 넣지 않는다 — 실질부채비율 자체가 이미 다가구의 숨은 위험을
     반영하도록 설계했으므로, 건물유형 더미를 중복으로 얹으면 논문 결과와 모순되고 이중계산이 된다.
  4. 전세가율 승수는 논문에 없는 지표(논문은 log 전세보증금 절대액을 씀)라 여전히 사람이 정한
     휴리스틱이다. 방향성 참고용일 뿐, 검증된 계수가 아니다.
  5. HUG 공개 전국 평균 사고율을 baseline으로 곱해 스케일만 맞춘다(캘리브레이션) — 오즈비를
     절대확률로 바꾸는 엄밀한 방법이 아니라 상대위험 근사임을 유의할 것.
PU Learning으로 실제 분류기를 학습하는 것, 혹은 논문 저자 데이터를 직접 확보해 재현하는 것이
다음 단계 과제다.
"""

from .config import HUG_NATIONAL_ACCIDENT_RATE

# 출처: 안선영·이상엽(2025), "전세보증금 미반환에 영향을 미치는 주요요인 연구", 주택금융연구 9(2), 표10.
# 기준집단(reference): 부채비율 60% 미만
DEBT_RATIO_ODDS_RATIO = [
    (0.60, 1.0),     # 60% 미만 (기준집단)
    (0.70, 1.508),   # 60~70% 미만
    (0.80, 2.59),    # 70~80% 미만
    (0.90, 5.489),   # 80~90% 미만
    (float("inf"), 29.923),  # 90% 이상
]


def _debt_ratio_multiplier(real_debt_ratio):
    for upper_bound, odds_ratio in DEBT_RATIO_ODDS_RATIO:
        if real_debt_ratio < upper_bound:
            return odds_ratio
    return DEBT_RATIO_ODDS_RATIO[-1][1]


def _jeonse_ratio_multiplier(jeonse_ratio):
    """논문에 없는 휴리스틱 — 사람이 방향성만 보고 정한 근사치."""
    if jeonse_ratio >= 0.9:
        return 2.0
    if jeonse_ratio >= 0.7:
        return 1.3
    return 0.8


def score(mortgage_won, priority_deposit_mean_won, my_deposit_won, market_price_won):
    """
    반환: dict(real_debt_ratio, jeonse_ratio, accident_probability)
    accident_probability는 HUG_NATIONAL_ACCIDENT_RATE 대비 상대위험을 곱한 근사치이며,
    상한 60%로 클리핑한다(무한정 발산 방지).
    """
    if market_price_won <= 0:
        raise ValueError("market_price_won은 0보다 커야 함")

    real_debt_ratio = (mortgage_won + priority_deposit_mean_won) / market_price_won
    registry_debt_ratio = mortgage_won / market_price_won
    jeonse_ratio = my_deposit_won / market_price_won

    multiplier = _debt_ratio_multiplier(real_debt_ratio) * _jeonse_ratio_multiplier(jeonse_ratio)

    probability = min(HUG_NATIONAL_ACCIDENT_RATE * multiplier, 0.60)

    return {
        "registry_debt_ratio": registry_debt_ratio,
        "real_debt_ratio": real_debt_ratio,
        "jeonse_ratio": jeonse_ratio,
        "risk_multiplier": multiplier,
        "accident_probability": probability,
    }
