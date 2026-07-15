"""
공공데이터포털 API 래퍼 (test_api.py에서 검증한 로직을 함수화).
"""

import requests
import xml.etree.ElementTree as ET

from .config import (
    SERVICE_KEY,
    BUILDING_TITLE_URL,
    EXPOS_PUBUSE_URL,
    FLR_OULN_URL,
    RENT_URL,
    TRADE_URL,
    MIN_SAMPLE_SIZE,
)


def _parse(xml_text):
    root = ET.fromstring(xml_text)
    code = root.find(".//resultCode")
    if code is not None and code.text not in ("00", "0", "000"):
        msg = root.find(".//resultMsg")
        raise RuntimeError(f"API 오류 [{code.text}] {msg.text if msg is not None else ''}")
    return [{c.tag: (c.text or "").strip() for c in item} for item in root.iter("item")]


def get_building_title(sigungu_cd, bjdong_cd, bun, ji):
    """표제부: 가구수(N), 연면적, 용도 등."""
    params = {
        "serviceKey": SERVICE_KEY,
        "sigunguCd": sigungu_cd,
        "bjdongCd": bjdong_cd,
        "bun": bun,
        "ji": ji,
        "numOfRows": "10",
        "pageNo": "1",
    }
    r = requests.get(BUILDING_TITLE_URL, params=params, timeout=15)
    items = _parse(r.text)
    if not items:
        raise ValueError("표제부 조회 결과 없음 — 법정동코드/본번/부번 확인 필요")
    return items[0]


def get_unit_areas(sigungu_cd, bjdong_cd, bun, ji):
    """
    호별 전유면적. 다가구는 구분등기가 아니라 안 나올 수 있음.
    반환: (면적 리스트, is_fallback) — is_fallback=True면 층별개요로 대체한 것.
    """
    params = {
        "serviceKey": SERVICE_KEY,
        "sigunguCd": sigungu_cd,
        "bjdongCd": bjdong_cd,
        "bun": bun,
        "ji": ji,
        "numOfRows": "100",
        "pageNo": "1",
    }
    r = requests.get(EXPOS_PUBUSE_URL, params=params, timeout=15)
    items = _parse(r.text)
    exclusive = [it for it in items if it.get("exposPubuseGbCdNm") == "전유"]
    if exclusive:
        areas = [float(it["area"]) for it in exclusive if it.get("area")]
        return areas, False

    # Fallback: 층별개요
    params["numOfRows"] = "50"
    r = requests.get(FLR_OULN_URL, params=params, timeout=15)
    items = _parse(r.text)
    areas = [float(it["area"]) for it in items if it.get("area")]
    return areas, True


def get_rent_transactions(lawd_cd, year_months):
    """여러 달치 단독/다가구 전월세 실거래 병합."""
    all_deals = []
    for ym in year_months:
        params = {
            "serviceKey": SERVICE_KEY,
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": ym,
            "numOfRows": "1000",
            "pageNo": "1",
        }
        r = requests.get(RENT_URL, params=params, timeout=15)
        all_deals.extend(_parse(r.text))
    return all_deals


def get_sale_transactions(lawd_cd, year_months):
    """
    여러 달치 단독/다가구 매매 실거래 병합 (--market-price 자동화용).

    2026-07-15 활용신청 → 승인 반영 확인 완료. 거래금액 필드는 dealAmount(만원, 콤마 포함
    문자열)로 실제 응답에서 검증함. 그 외 필드: buildYear, totalFloorAr, umdNm, houseType
    (단독/다가구), dealYear/Month/Day 등 — RTMSDataSvcSHRent(전월세)와 거의 동일한 구조.
    """
    all_deals = []
    for ym in year_months:
        params = {
            "serviceKey": SERVICE_KEY,
            "LAWD_CD": lawd_cd,
            "DEAL_YMD": ym,
            "numOfRows": "1000",
            "pageNo": "1",
        }
        r = requests.get(TRADE_URL, params=params, timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"매매 실거래 API 호출 실패 (HTTP {r.status_code}) — 활용신청 반영 지연 가능성")
        all_deals.extend(_parse(r.text))
    return all_deals


def filter_by_dong(deals, dong_name, min_sample=MIN_SAMPLE_SIZE):
    """
    동 단위 표본이 부족하면 구 전체(입력 deals 그대로)로 확대.
    반환: (필터된 거래 리스트, 확대여부)
    """
    subset = [d for d in deals if dong_name in d.get("umdNm", "")]
    if len(subset) >= min_sample:
        return subset, False
    return deals, True
