"""
세이프홈 (SafeHome) — CLI 진입점. A~D 모듈을 순서대로 실행하고 진단 리포트를 출력한다.
([E] 금융상품 추천은 일단 범위 제외 — 직방 대응 진단 리포트부터 완성하기로 함)

사용 예:
    python report.py --my-deposit 50000000 --mortgage 0                     # 시세 자동추정 시도
    python report.py --my-deposit 50000000 --mortgage 0 --market-price 250000000  # 시세 직접입력

*** 현재 버전의 알려진 한계 ***
- 근저당(등기부) 금액: CODEF 등기부 API가 사업자등록번호를 요구해서 --mortgage 인자로 직접 입력받는다.
- 시세: --market-price 생략 시 매매 실거래 API(RTMSDataSvcSHTrade)로 유사 면적대 매매가 중앙값을
  자동 추정한다(2026-07-15 활용신청·승인 확인 완료). 표본이 부족한 지역/기간이면 실패할 수 있고,
  그 경우 --market-price로 직접 입력해야 한다.
- 종합의견: OPENAI_API_KEY 환경변수가 설정돼 있으면 OpenAI(gpt-4o-mini)로 실제 자연어 요약을
  생성한다. 미설정이거나 호출 실패 시 규칙 기반 문장 조립으로 자동 폴백한다(둘 다 숫자를
  지어내지 않고 이미 계산된 값만 사용).
- [A] 선순위 보증금은 기본적으로 N-1세대 전원이 입주해 있다고 가정하고 몬테카를로로 추정한다.
  확정일자 부여현황으로 이 중 일부를 실측 확인했다면 --known-tenants / --known-priority-deposit-won
  으로 그 부분만 고정값으로 대체할 수 있다(나머지 세대만 추정). 다만 확정일자 부여현황 열람은
  주택임대차보호법 제3조의6에 따라 임대인 동의가 필요해 API 자동조회는 불가능하고(계약 전 접근이
  제한적인 경우가 많음), 수동 입력만 지원한다 — 이 접근성 문제 자체가 A모듈이 필요한 이유이기도 하다.
- [B] 국세·지방세 체납액(조세채권)은 --tax-arrears-won으로 수동 입력받는다. 당해세 등은 근저당보다도
  우선순위가 높을 수 있어 회수액에서 최우선으로 차감하지만, 이 정보(납세증명서)도 확정일자와 같은
  이유로 임대인 동의가 있어야 계약 시점에만 확인 가능해(주택임대차보호법 제3조의7) 자동조회는 불가능하다.
- [B] 회수 결과는 단일 확률/평균 대신 10,000회 몬테카를로 분포의 p10/p50/p90을 "보수적/기준/낙관적"
  시나리오로 제시한다(conservative_recovery_won/base_recovery_won/optimistic_recovery_won).
"""

import argparse
import json
import os
import sys
from datetime import date

import numpy as np

from src import config
from src.api_client import (
    get_building_title,
    get_unit_areas,
    get_rent_transactions,
    get_sale_transactions,
    filter_by_dong,
)
from src.module_a_deposit_dist import fit_deposit_kde, simulate_priority_deposits, summarize as summarize_a
from src.module_b_auction_sim import estimate_small_tenant_ratio, simulate_recovery, summarize as summarize_b
from src.module_c_risk_score import score
from src.module_d_expected_loss import expected_loss
from src.safety_info import cctv_summary
# [E] 금융상품 추천(module_e_recommendation)은 일단 리포트 범위에서 제외 — 직방 대응 진단
# 리포트(기본진단+다가구특별진단)부터 먼저 완성하기로 함. 모듈 파일 자체는 남겨둠.


# training/phase3_features.py의 면적 구간 기준과 동일 (참고 시세 산정 시 같은 기준으로 비교)
AREA_BAND_BINS = [0, 20, 40, float("inf")]
AREA_BAND_LABELS = ["원룸(~20㎡)", "투룸(20~40㎡)", "쓰리룸+(40㎡~)"]


def _to_float(value):
    if value is None:
        return None
    v = str(value).strip().replace(",", "")
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _area_band(area):
    for i, hi in enumerate(AREA_BAND_BINS[1:]):
        if area <= hi:
            return AREA_BAND_LABELS[i]
    return AREA_BAND_LABELS[-1]


def market_reference(filtered_deals, target_area_avg):
    """
    대상 건물과 유사 면적대(같은 구간)의 지역 실거래로 참고 시세를 계산.
    전세/월세 둘 다 구하고, 전월세 전환율(보증금↔월세 트레이드오프)도 함께 근사한다.
    표본이 부족하면 그 사실을 그대로 노출한다(숫자를 지어내지 않음).
    """
    band = _area_band(target_area_avg)
    band_idx = AREA_BAND_LABELS.index(band)
    lo, hi = AREA_BAND_BINS[band_idx], AREA_BAND_BINS[band_idx + 1]

    jeonse_deposits, wolse_deposits, wolse_rents = [], [], []
    for d in filtered_deals:
        area = _to_float(d.get("totalFloorAr"))
        deposit = _to_float(d.get("deposit"))
        rent = _to_float(d.get("monthlyRent"))
        if area is None or deposit is None or not (lo < area <= hi):
            continue
        if rent is None or rent == 0:
            jeonse_deposits.append(deposit)
        else:
            wolse_deposits.append(deposit)
            wolse_rents.append(rent)

    result = {"band": band, "n_jeonse": len(jeonse_deposits), "n_wolse": len(wolse_deposits)}

    if jeonse_deposits:
        result["jeonse_q25"] = np.percentile(jeonse_deposits, 25)
        result["jeonse_q50"] = np.percentile(jeonse_deposits, 50)
        result["jeonse_q75"] = np.percentile(jeonse_deposits, 75)

    if wolse_deposits:
        result["wolse_deposit_q25"] = np.percentile(wolse_deposits, 25)
        result["wolse_deposit_q50"] = np.percentile(wolse_deposits, 50)
        result["wolse_deposit_q75"] = np.percentile(wolse_deposits, 75)
        result["wolse_rent_q25"] = np.percentile(wolse_rents, 25)
        result["wolse_rent_q50"] = np.percentile(wolse_rents, 50)
        result["wolse_rent_q75"] = np.percentile(wolse_rents, 75)

    # 전월세 전환율(보증금을 올릴수록 월세가 내려가는 관계) 근사 — 전세·월세 둘 다 있어야 계산 가능
    if jeonse_deposits and wolse_deposits:
        gap = result["jeonse_q50"] - result["wolse_deposit_q50"]
        if gap > 0:
            annual_rent = result["wolse_rent_q50"] * 12
            result["conversion_rate_pct"] = annual_rent / gap * 100

    return result


# 매매 실거래 응답의 거래금액 필드명 — dealAmount로 실제 응답에서 확인 완료(2026-07-15).
# 혹시 모를 스키마 변동 대비로 후보 목록 형태는 유지.
SALE_AMOUNT_FIELD_CANDIDATES = ["dealAmount", "dealAmt", "amount"]


def estimate_market_price(sale_deals, target_area_avg):
    """
    유사 면적대의 매매 실거래 중앙값으로 시세(원)를 추정.
    표본이 없거나 거래금액 필드를 못 찾으면 None 반환 (호출부에서 수동입력으로 폴백).
    """
    if not sale_deals:
        return None

    field = next((f for f in SALE_AMOUNT_FIELD_CANDIDATES if f in sale_deals[0]), None)
    if field is None:
        return None

    band = _area_band(target_area_avg)
    band_idx = AREA_BAND_LABELS.index(band)
    lo, hi = AREA_BAND_BINS[band_idx], AREA_BAND_BINS[band_idx + 1]

    prices_manwon = []
    for d in sale_deals:
        area = _to_float(d.get("totalFloorAr"))
        price = _to_float(d.get(field))
        if area is None or price is None or not (lo < area <= hi):
            continue
        prices_manwon.append(price)

    if len(prices_manwon) < 5:
        return None

    return np.percentile(prices_manwon, 50) * 10_000  # 만원 → 원


def recent_year_months(n=6, today=None):
    today = today or date.today()
    months = []
    y, m = today.year, today.month
    for _ in range(n):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
        months.append(f"{y}{m:02d}")
    return months


def parse_args():
    p = argparse.ArgumentParser(description="세이프홈 — 다가구주택 전세사기 위험 예측 리포트")
    p.add_argument("--sigungu", default="41115", help="시군구코드 (기본: 수원시 팔달구)")
    p.add_argument("--bjdong", default="14000", help="법정동코드 (기본: 우만동)")
    p.add_argument("--bun", default="0039")
    p.add_argument("--ji", default="0007")
    p.add_argument("--dong-name", default="우만동")
    p.add_argument("--my-deposit", type=int, required=True, help="내 보증금 (원)")
    p.add_argument("--mortgage", type=int, required=True, help="근저당 금액 (원, 등기부 확인값 직접 입력)")
    p.add_argument("--market-price", type=int, default=None,
                    help="건물 시세 추정치 (원). 생략하면 매매 실거래 API로 자동 추정 시도, 실패 시 에러로 직접 입력 요청")
    p.add_argument("--known-tenants", type=int, default=0,
                    help="확정일자 부여현황 등으로 직접 확인한 선순위 임차인 수(관측된 하한). "
                         "API로 자동 조회할 방법이 없어 수동 입력만 지원 (임대인 동의 필요, 계약 전 접근 제한적)")
    p.add_argument("--known-priority-deposit-won", type=int, default=0,
                    help="--known-tenants만큼의 실제 확인된 선순위 보증금 총액(원). "
                         "이 금액은 추정이 아니라 실측값으로 고정 반영되고, 나머지 세대만 몬테카를로 시뮬레이션한다")
    p.add_argument("--tax-arrears-won", type=int, default=0,
                    help="임대인 국세·지방세 체납액(원) — 납세증명서로 직접 확인한 값(원, 등기부처럼 자동조회 불가). "
                         "당해세 등은 근저당보다도 우선순위가 높을 수 있어 경매 회수액에서 우선 차감된다")
    p.add_argument("--n-sim", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", action="store_true", help="사람이 읽는 텍스트 대신 구조화된 JSON을 stdout에 출력")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. 건축물대장 표제부
    building = get_building_title(args.sigungu, args.bjdong, args.bun, args.ji)
    # fmlyCnt(가구수) > 0 이면 다가구/다세대류(비구분등기, 세입자 여러명이 한 등기에 얽힘).
    # 그 외(구분등기 공동주택 — 아파트/연립/다세대 등)는 세대별로 등기가 따로라 "숨은 다른
    # 세입자" 개념 자체가 없음 → A모듈(선순위 보증금)을 적용하지 않고 B/C/D만 수행.
    fmlycnt_raw = building.get("fmlyCnt", "0")
    is_multi_household = fmlycnt_raw.isdigit() and int(fmlycnt_raw) > 0
    if is_multi_household:
        n_units = int(fmlycnt_raw)
    else:
        # "0"은 non-empty string이라 `or` 폴백으로는 안 걸러짐 — 양수인 첫 값을 직접 찾는다.
        n_units = next(
            (int(v) for v in (building.get("hoCnt"), building.get("hhldCnt")) if v and v.isdigit() and int(v) > 0),
            1,
        )

    # 2. 호별 면적 (fallback 여부만 기록, 이번 버전 리포트에는 참고용으로만 노출)
    areas, area_is_fallback = get_unit_areas(args.sigungu, args.bjdong, args.bun, args.ji)

    # 3. 실거래가 (최근 6개월, 동 표본 부족시 구 단위로 자동 확대)
    deals = get_rent_transactions(args.sigungu, recent_year_months(6))
    filtered_deals, dong_expanded = filter_by_dong(deals, args.dong_name)

    # 3-1. 참고 시세 (전세/월세, 유사 면적대) — A모듈 결과의 sanity-check용
    target_area_avg = sum(areas) / len(areas) if areas else 30.0
    market_ref = market_reference(filtered_deals, target_area_avg)

    # 3-1b. 참고 치안정보 (CCTV, 수원시 한정 — data.go.kr 파일데이터, API 아님)
    safety_ref = cctv_summary(args.dong_name)

    # 3-2. 시세 확보 — --market-price 생략 시 매매 실거래 API로 자동 추정 시도
    if args.market_price is not None:
        market_price_won = args.market_price
        market_price_source = "manual"
    else:
        market_price_won = None
        try:
            sale_deals = get_sale_transactions(args.sigungu, recent_year_months(6))
            filtered_sales, _ = filter_by_dong(sale_deals, args.dong_name)
            market_price_won = estimate_market_price(filtered_sales, target_area_avg)
        except Exception:
            market_price_won = None  # API 활용신청 반영 지연 등 — 아래에서 에러로 안내

        if market_price_won is None:
            raise SystemExit(
                "시세를 자동으로 못 가져왔습니다 (매매 실거래 API 활용신청 반영 지연 또는 표본 부족). "
                "--market-price로 직접 입력해주세요."
            )
        market_price_source = "auto_trade_api"

    # 4. [A] 선순위 보증금 분포 — 다가구/다세대(비구분등기)일 때만. 구분등기 공동주택은 해당 없음.
    use_apr_day = building.get("useAprDay", "")
    build_year = int(use_apr_day[:4]) if use_apr_day[:4].isdigit() else None

    if is_multi_household:
        if args.known_tenants > max(n_units - 1, 0):
            raise SystemExit(
                f"--known-tenants({args.known_tenants})가 선순위 임차인 최대 인원({n_units - 1}명)보다 큽니다."
            )
        known_deposit_manwon = args.known_priority_deposit_won / 10_000
        kde_info = fit_deposit_kde(filtered_deals)
        priority_sim = simulate_priority_deposits(
            kde_info, n_units,
            area_list=areas, build_year=build_year, dong_name=args.dong_name,
            n_sim=args.n_sim, seed=args.seed,
            known_tenants=args.known_tenants, known_deposit_manwon=known_deposit_manwon,
        )
        a_summary = summarize_a(priority_sim)
    else:
        priority_sim = np.zeros(args.n_sim)
        a_summary = {"mean": 0.0, "p05": 0.0, "p95": 0.0, "worst_p95": 0.0}

    # 5. [B] 경매 배당 시뮬레이션 (구분등기 건물은 n_prior=0 → 선순위 항 자동으로 0)
    region_row = config.PRIORITY_REPAYMENT_TABLE[1]  # 과밀억제권역·세종·용인·화성 (수원)
    all_deposits_manwon = [
        float(d["deposit"].replace(",", "")) for d in filtered_deals if d.get("deposit")
    ]
    small_ratio = estimate_small_tenant_ratio(all_deposits_manwon, region_row["deposit_limit"])
    recovery = simulate_recovery(
        my_deposit_won=args.my_deposit,
        market_price_won=market_price_won,
        mortgage_won=args.mortgage,
        priority_deposits_manwon=priority_sim,
        n_prior=max(n_units - 1, 0) if is_multi_household else 0,
        small_tenant_ratio=small_ratio,
        n_sim=args.n_sim,
        seed=args.seed,
        tax_arrears_won=args.tax_arrears_won,
    )
    b_summary = summarize_b(recovery, args.my_deposit)

    # 6. [C] 사고 확률 (규칙 기반 스코어링)
    c_result = score(
        mortgage_won=args.mortgage,
        priority_deposit_mean_won=a_summary["mean"],
        my_deposit_won=args.my_deposit,
        market_price_won=market_price_won,
    )

    # 7. [D] 기대손실
    loss = expected_loss(c_result["accident_probability"], args.my_deposit, b_summary["expected_recovery_won"])

    # [E] 금융상품 추천은 일단 범위 제외 (직방 대응 진단 리포트 먼저 완성)

    summary, summary_source = get_summary_opinion(building, n_units, is_multi_household, a_summary,
                                                    c_result, b_summary, loss, args.my_deposit, market_price_source)

    if args.json:
        result = build_result_dict(building, n_units, areas, area_is_fallback, dong_expanded,
                                    a_summary, market_ref, c_result, b_summary, loss,
                                    build_year, args, market_price_won, market_price_source,
                                    is_multi_household, summary, summary_source, safety_ref)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_report(building, n_units, areas, area_is_fallback, dong_expanded,
                     a_summary, b_summary, c_result, loss, args, market_ref,
                     market_price_won, market_price_source, is_multi_household,
                     summary, summary_source, safety_ref)


def build_result_dict(building, n_units, areas, area_is_fallback, dong_expanded,
                       a_summary, market_ref, c_result, b_summary, loss,
                       build_year, args, market_price_won, market_price_source,
                       is_multi_household, summary, summary_source, safety_ref):
    """README.md '입력/출력 스키마' 절과 동일한 구조로 결과를 조립한다."""
    market_ref_out = {
        "area_band": market_ref["band"],
        "jeonse": {
            "q25": market_ref.get("jeonse_q25"), "q50": market_ref.get("jeonse_q50"),
            "q75": market_ref.get("jeonse_q75"), "n": market_ref["n_jeonse"],
        },
        "wolse": {
            "deposit_q25": market_ref.get("wolse_deposit_q25"), "deposit_q50": market_ref.get("wolse_deposit_q50"),
            "deposit_q75": market_ref.get("wolse_deposit_q75"),
            "rent_q25": market_ref.get("wolse_rent_q25"), "rent_q50": market_ref.get("wolse_rent_q50"),
            "rent_q75": market_ref.get("wolse_rent_q75"), "n": market_ref["n_wolse"],
        },
    }
    if "conversion_rate_pct" in market_ref:
        market_ref_out["conversion_rate_pct"] = market_ref["conversion_rate_pct"]

    return {
        "building": {
            "address": building.get("platPlc", ""),
            "purpose": building.get("mainPurpsCdNm", ""),
            "purpose_detail": building.get("etcPurps", ""),
            "is_multi_household": is_multi_household,
            "n_units": n_units,
            "areas_sqm": areas,
            "area_is_fallback": area_is_fallback,
            "build_year": build_year,
        },
        "module_a": {
            "applicable": is_multi_household,  # 구분등기 공동주택은 해당 없음 (선순위 개념 없음)
            "priority_deposit_mean_won": a_summary["mean"],
            "priority_deposit_p05_won": a_summary["p05"],
            "priority_deposit_p95_won": a_summary["p95"],
            "priority_deposit_worst_p95_won": a_summary["worst_p95"],
            "dong_sample_expanded": dong_expanded,
        },
        "market_reference": market_ref_out,
        "safety_reference": safety_ref,  # CCTV 참고정보 (수원시 한정, 표본 없으면 null)
        "module_c": c_result,
        "module_b": b_summary,
        "module_d": {"expected_loss_won": loss},
        "summary_opinion": {"text": summary, "source": summary_source},
        "inputs_used": {
            "my_deposit_won": args.my_deposit,
            "mortgage_won": args.mortgage,
            "market_price_won": market_price_won,
            "market_price_source": market_price_source,
            "known_tenants": args.known_tenants,
            "known_priority_deposit_won": args.known_priority_deposit_won,
            "tax_arrears_won": args.tax_arrears_won,
        },
    }


def generate_rule_based_summary(building, n_units, is_multi_household, a_summary,
                                 c_result, b_summary, loss, my_deposit_won, market_price_source):
    """
    종합의견 — 규칙 기반으로 문장을 조립하는 자연어 요약 (LLM 호출 실패/키 미설정 시 폴백용).
    """
    p = c_result["accident_probability"]
    addr = building.get("platPlc", "이 건물")
    sentences = []

    if is_multi_household:
        sentences.append(
            f"{addr}은(는) {n_units}가구가 함께 사는 다가구주택으로, 등기부에는 나오지 않는 "
            f"앞선 세대들의 선순위 보증금이 평균 {_man(a_summary['mean'])}원 "
            f"(90% 구간 {_man(a_summary['p05'])}~{_man(a_summary['p95'])}원) 있는 것으로 추정됩니다."
        )
        sentences.append(
            f"이를 반영한 실질 부채비율은 {c_result['real_debt_ratio']*100:.0f}%로, "
            f"등기부에만 보이는 값({c_result['registry_debt_ratio']*100:.0f}%)보다 높게 나타납니다."
        )
    else:
        sentences.append(
            f"{addr}은(는) 구분등기 건물이라 다른 세대의 보증금이 내 계약에 영향을 주지 않고, "
            f"등기부 기준 부채비율은 {c_result['registry_debt_ratio']*100:.0f}%입니다."
        )

    if p < 0.02:
        tone = "사고 발생 가능성이 낮은 편으로 보입니다."
    elif p < 0.05:
        tone = "사고 발생 가능성은 대체로 낮지만, 참고할 필요는 있습니다."
    elif p < 0.15:
        tone = "사고 발생 가능성이 낮지 않아 주의가 필요합니다."
    else:
        tone = "사고 발생 가능성이 상당히 높은 편이라 신중한 검토가 필요합니다."
    sentences.append(f"사고 발생 확률은 {p*100:.1f}%로, {tone}")

    sentences.append(
        f"만약 경매까지 간다고 가정하면 보증금 {_man(my_deposit_won)}원 중 평균 "
        f"{_man(b_summary['expected_recovery_won'])}원 회수가 예상되고, "
        f"이를 종합한 기대손실은 {_man(loss)}원입니다."
    )

    if market_price_source == "manual":
        sentences.append("(시세는 자동 조회가 아닌 직접 입력값 기준으로 계산되었습니다.)")

    return " ".join(sentences)


def _build_summary_prompt(building, n_units, is_multi_household, a_summary,
                           c_result, b_summary, loss, my_deposit_won, market_price_source):
    addr = building.get("platPlc", "이 건물")
    facts = {
        "주소": addr,
        "다가구_여부": is_multi_household,
        "가구수_또는_세대수": n_units,
        "등기부_기준_부채비율_퍼센트": round(c_result["registry_debt_ratio"] * 100, 1),
        "실질_부채비율_퍼센트": round(c_result["real_debt_ratio"] * 100, 1) if is_multi_household else None,
        "선순위_보증금_평균_원": a_summary["mean"] if is_multi_household else None,
        "선순위_보증금_90퍼센트구간_원": [a_summary["p05"], a_summary["p95"]] if is_multi_household else None,
        "사고_발생_확률_퍼센트": round(c_result["accident_probability"] * 100, 1),
        "내_보증금_원": my_deposit_won,
        "경매_가정시_예상회수액_원": b_summary["expected_recovery_won"],
        "기대손실_원": loss,
        "시세_출처": "자동추정(매매실거래)" if market_price_source == "auto_trade_api" else "사용자 직접입력",
    }
    return (
        "너는 부동산 전세사기 위험 진단 리포트의 마지막에 들어갈 '종합의견'을 쓰는 어시스턴트다. "
        "아래 JSON 사실관계만 근거로, 3~4문장짜리 한국어 종합의견을 써라. "
        "숫자는 절대 지어내지 말고 주어진 값만 자연스러운 문장으로 풀어써라. "
        "다가구_여부가 false면 선순위 보증금 관련 내용은 언급하지 마라. "
        "과장하지 말고 담백하게, 사고확률 수치에 맞는 톤(낮으면 안심, 높으면 신중)으로 써라.\n\n"
        f"{json.dumps(facts, ensure_ascii=False, indent=2)}"
    )


def generate_llm_summary(building, n_units, is_multi_household, a_summary,
                          c_result, b_summary, loss, my_deposit_won, market_price_source):
    """실제 OpenAI API 호출로 종합의견을 생성한다. 키 미설정/호출 실패 시 예외를 던진다(호출부에서 폴백)."""
    import openai

    client = openai.OpenAI()
    prompt = _build_summary_prompt(building, n_units, is_multi_household, a_summary,
                                    c_result, b_summary, loss, my_deposit_won, market_price_source)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 필요시 최신/선호 모델로 교체 가능
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def get_summary_opinion(building, n_units, is_multi_household, a_summary,
                         c_result, b_summary, loss, my_deposit_won, market_price_source):
    """LLM 호출을 시도하고, 키 미설정/API 오류 시 규칙 기반 요약으로 폴백한다."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            text = generate_llm_summary(building, n_units, is_multi_household, a_summary,
                                         c_result, b_summary, loss, my_deposit_won, market_price_source)
            return text, "llm"
        except Exception as e:
            print(f"(LLM 호출 실패, 규칙 기반으로 폴백: {e})", file=sys.stderr)

    text = generate_rule_based_summary(building, n_units, is_multi_household, a_summary,
                                        c_result, b_summary, loss, my_deposit_won, market_price_source)
    return text, "rule_based"


def _man(won):
    return f"{won / 10_000:,.0f}만"


def _man_manwon(v):
    return f"{v:,.0f}만"


def print_market_reference(market_ref, dong_name):
    print(f"\n참고: {dong_name} 유사면적({market_ref['band']}) 시세")
    if market_ref["n_jeonse"] > 0:
        print(f"    전세 보증금       {_man_manwon(market_ref['jeonse_q25'])}~{_man_manwon(market_ref['jeonse_q75'])}원 "
              f"(중앙값 {_man_manwon(market_ref['jeonse_q50'])}원, n={market_ref['n_jeonse']}건)")
    else:
        print("    전세 보증금       표본 부족 (해당 면적대 전세 실거래 없음)")

    if market_ref["n_wolse"] > 0:
        print(f"    월세 보증금/월세  {_man_manwon(market_ref['wolse_deposit_q25'])}원/{_man_manwon(market_ref['wolse_rent_q25'])}원"
              f" ~ {_man_manwon(market_ref['wolse_deposit_q75'])}원/{_man_manwon(market_ref['wolse_rent_q75'])}원"
              f" (중앙값 {_man_manwon(market_ref['wolse_deposit_q50'])}원/{_man_manwon(market_ref['wolse_rent_q50'])}원, "
              f"n={market_ref['n_wolse']}건)")
    else:
        print("    월세 보증금/월세  표본 부족 (해당 면적대 월세 실거래 없음)")

    if "conversion_rate_pct" in market_ref:
        print(f"    ※ 전월세 전환율 약 {market_ref['conversion_rate_pct']:.1f}% — 보증금을 올릴수록 월세가 내려가는"
              f" 트레이드오프 관계이니, 위 두 줄은 서로 대체 가능한 조합으로 참고할 것 (참고용 근사치)")


def print_safety_reference(safety_ref, dong_name):
    print(f"\n참고: {dong_name} 방범 CCTV 현황 (행정안전부 표준데이터, 수원시 한정)")
    if safety_ref is None:
        print("    표본 부족 (해당 동 CCTV 데이터 없음)")
    else:
        print(f"    설치 {safety_ref['cctv_count']}개소 / 카메라 {safety_ref['camera_total']}대")


def print_report(building, n_units, areas, area_is_fallback, dong_expanded,
                  a_summary, b_summary, c_result, loss, args, market_ref,
                  market_price_won, market_price_source, is_multi_household,
                  summary, summary_source, safety_ref):
    print("=" * 60)
    print(f"[기본 진단] {building.get('platPlc', '')}")
    print("=" * 60)
    print(f"  건물 용도       : {building.get('mainPurpsCdNm', '')} ({building.get('etcPurps', '')})")
    unit_label = "총 가구수(N)" if is_multi_household else "세대수      "
    print(f"  {unit_label}   : {n_units}")
    if area_is_fallback:
        print(f"  호별 면적       : 전유부 없음 → 층별개요 {len(areas)}건으로 대체 (균등분할 아님, 층별 면적)")
    else:
        print(f"  호별 면적       : 전유부 {len(areas)}건 확인됨")

    if is_multi_household:
        print("\n" + "━" * 60)
        print(f"⚠️  다가구 특별 진단 · 총 {n_units}가구")
        print("━" * 60)
        print(f"숨은 선순위 임차보증금 (등기부 미기재)")
        print(f"    평균 {_man(a_summary['mean'])}원  /  90% 구간 {_man(a_summary['p05'])} ~ {_man(a_summary['p95'])}원")
        print(f"    상위 5%(최악) {_man(a_summary['worst_p95'])}원 이상")
        if dong_expanded:
            print(f"    ※ '{args.dong_name}' 표본 부족으로 구(시군구) 단위 실거래로 확대 추정")
        if args.known_tenants > 0:
            print(f"    ※ 확정일자 부여현황 등으로 확인된 {args.known_tenants}세대(실측 {_man(args.known_priority_deposit_won)}원)는 "
                  f"고정값 반영, 나머지 {max(n_units - 1 - args.known_tenants, 0)}세대만 추정")
    else:
        print("\n" + "━" * 60)
        print("보증금 안전도 분석")
        print("━" * 60)
        print("이 건물은 구분등기(아파트/연립·다세대 등)라 다가구 특유의 '숨은 선순위 세입자'")
        print("문제가 구조적으로 없습니다(세대별로 등기가 분리되어 있어 다른 세입자의 보증금이")
        print("내 세대에 영향을 주지 않음). 그래서 아래는 등기부상 근저당만으로 계산됩니다.")

    print_market_reference(market_ref, args.dong_name)
    print_safety_reference(safety_ref, args.dong_name)

    print(f"\n실질 부채비율")
    print(f"    등기부 기준(근저당/시세)   {c_result['registry_debt_ratio']*100:.1f}%")
    if is_multi_household:
        print(f"    선순위 반영(A모듈 평균)    {c_result['real_debt_ratio']*100:.1f}%")

    print(f"\n사고 발생 확률(규칙기반 근사)   {c_result['accident_probability']*100:.1f}%")
    print(f"[사고(경매) 발생을 가정했을 때] 보증금 회수 예상   {_man(b_summary['expected_recovery_won'])} / {_man(args.my_deposit)}원")
    print(f"  전액회수 {b_summary['full_recovery_pct']:.1f}% / 일부회수 {b_summary['partial_recovery_pct']:.1f}% "
          f"/ 전액손실 {b_summary['total_loss_pct']:.1f}%  (※ 경매까지 갈 확률 자체는 위 {c_result['accident_probability']*100:.1f}%)")
    print(f"  시나리오별 회수율(10,000회 시뮬레이션 분위수) — "
          f"보수적 {b_summary['conservative_recovery_pct_of_deposit']:.0f}% / "
          f"기준 {b_summary['base_recovery_pct_of_deposit']:.0f}% / "
          f"낙관적 {b_summary['optimistic_recovery_pct_of_deposit']:.0f}%")
    if args.tax_arrears_won > 0:
        print(f"  ※ 임대인 국세·지방세 체납 {_man(args.tax_arrears_won)}원을 우선 차감해 반영(당해세는 근저당보다도 우선순위가 높을 수 있음)")
    print("─" * 60)
    print(f"기대손실                        {_man(loss)}원")
    print("━" * 60)

    tag = "OpenAI API" if summary_source == "llm" else "규칙 기반 폴백 — OPENAI_API_KEY 미설정"
    print(f"\n[종합의견] ({tag})")
    print(f"  {summary}")

    print("\n" + "=" * 60)
    price_note = "자동추정(매매 실거래 API)" if market_price_source == "auto_trade_api" else "사용자 직접입력"
    print(f"한계: 근저당·국세체납은 사용자 입력값(등기부·납세증명서 미연동), 시세는 {price_note}({_man(market_price_won)}원).")
    print("낙찰가율은 경기 연립·다세대 평균치로 근사, 사고확률은 학습된 분류기가 아닌 규칙기반")
    print("스코어링입니다.")
    if is_multi_household:
        print("선순위 보증금(A모듈)은 학습된 QRF 모델 기반이며, 90㎡ 이상 고가 세대가 섞인")
        print("건물은 추정치가 보수적(과소)일 수 있습니다 (docs/모델_해석.md 참고).")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
