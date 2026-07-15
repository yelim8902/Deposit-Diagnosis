"""
KB AI Challenge - 데이터 확보 가능성 검증
대상: 경기 수원시 팔달구 우만동 39-7

확인 목표:
  [1] 건축물대장 → 가구수 N, 호별 면적 나오는가?
  [2] 실거래가   → 팔달구 다가구 전월세 표본 충분한가?
"""

import os
import requests
import xml.etree.ElementTree as ET
from collections import Counter

# ─────────────────────────────────────────
SERVICE_KEY = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")

SIGUNGU_CD = "41115"      # 수원시 팔달구
BJDONG_CD  = "14000"      # 우만동 (행정표준코드 4111514000 확인 완료)
BUN        = "0039"       # 본번 39
JI         = "0007"       # 부번 7
# ─────────────────────────────────────────


def parse(xml_text):
    """XML → item 리스트"""
    root = ET.fromstring(xml_text)
    # 에러 체크
    msg = root.find(".//resultMsg")
    code = root.find(".//resultCode")
    if code is not None and code.text not in ("00", "0", "000"):
        print(f"  ⚠️  API 오류 [{code.text}] {msg.text if msg is not None else ''}")
        return []
    items = []
    for item in root.iter("item"):
        items.append({c.tag: (c.text or "").strip() for c in item})
    return items


def show(d, keys):
    for k in keys:
        if k in d:
            print(f"    {k:<20} : {d[k]}")


# ═══════════════════════════════════════════════════
# [1-A] 표제부 — 가구수 N 확인
# ═══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[1-A] 건축물대장 표제부 — 가구수 확인")
print("═" * 60)

url = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
params = {
    "serviceKey": SERVICE_KEY,
    "sigunguCd": SIGUNGU_CD,
    "bjdongCd": BJDONG_CD,
    "bun": BUN,
    "ji": JI,
    "numOfRows": "10",
    "pageNo": "1",
}

r = requests.get(url, params=params, timeout=15)
items = parse(r.text)

if not items:
    print("  ❌ 결과 없음 — 법정동코드/본번/부번 확인 필요")
    print(f"  raw: {r.text[:400]}")
else:
    for it in items:
        print(f"\n  ▶ {it.get('platPlc', '')}")
        show(it, ["mainPurpsCdNm", "etcPurps", "hhldCnt", "fmlyCnt",
                  "hoCnt", "totArea", "grndFlrCnt", "useAprDay"])
        print(f"\n  ★ 가구수(fmlyCnt) = {it.get('fmlyCnt', '?')}")
        print(f"  ★ 세대수(hhldCnt) = {it.get('hhldCnt', '?')}")
        print(f"  ★ 호수(hoCnt)     = {it.get('hoCnt', '?')}")


# ═══════════════════════════════════════════════════
# [1-B] 전유공용면적 — 호별 면적 나오는가? (핵심)
# ═══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[1-B] 전유공용면적 — 호별 면적 (★ 최대 관심사)")
print("═" * 60)

url = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrExposPubuseAreaInfo"
params = {
    "serviceKey": SERVICE_KEY,
    "sigunguCd": SIGUNGU_CD,
    "bjdongCd": BJDONG_CD,
    "bun": BUN,
    "ji": JI,
    "numOfRows": "100",
    "pageNo": "1",
}

r = requests.get(url, params=params, timeout=15)
items = parse(r.text)

if not items:
    print("  ❌ 전유부 없음 → 다가구는 구분등기가 아니라서 안 나올 가능성 높음")
    print("  → Fallback: 층별개요(1-C)로 대체")
else:
    print(f"  ✅ {len(items)}건 조회됨\n")
    exclusive = [i for i in items if i.get("exposPubuseGbCdNm") == "전유"]
    print(f"  전유 항목: {len(exclusive)}건\n")
    for it in items[:15]:
        print(f"    {it.get('flrNoNm',''):>6} {it.get('hoNm',''):>8}호 "
              f"| {it.get('exposPubuseGbCdNm',''):>4} "
              f"| {it.get('mainPurpsCdNm',''):<12} "
              f"| {it.get('area','')}㎡")


# ═══════════════════════════════════════════════════
# [1-C] 층별개요 — Fallback
# ═══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[1-C] 층별개요 (Fallback)")
print("═" * 60)

url = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrFlrOulnInfo"
params = {
    "serviceKey": SERVICE_KEY,
    "sigunguCd": SIGUNGU_CD,
    "bjdongCd": BJDONG_CD,
    "bun": BUN,
    "ji": JI,
    "numOfRows": "50",
    "pageNo": "1",
}

r = requests.get(url, params=params, timeout=15)
items = parse(r.text)

if not items:
    print("  ❌ 없음")
else:
    print(f"  ✅ {len(items)}건\n")
    for it in items:
        print(f"    {it.get('flrNoNm',''):>8} | {it.get('mainPurpsCdNm',''):<15} "
              f"| {it.get('area','')}㎡")


# ═══════════════════════════════════════════════════
# [2] 단독/다가구 전월세 실거래가 — 표본 충분한가?
# ═══════════════════════════════════════════════════
print("\n" + "═" * 60)
print("[2] 단독/다가구 전월세 실거래 — 팔달구 표본")
print("═" * 60)

url = "https://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent"

all_deals = []
for ym in ["202606", "202605", "202604", "202603", "202602", "202601"]:
    params = {
        "serviceKey": SERVICE_KEY,
        "LAWD_CD": SIGUNGU_CD,
        "DEAL_YMD": ym,
        "numOfRows": "1000",
        "pageNo": "1",
    }
    r = requests.get(url, params=params, timeout=15)
    items = parse(r.text)
    all_deals.extend(items)
    print(f"  {ym}: {len(items):>4}건")

print(f"\n  ─────────────────────")
print(f"  총 {len(all_deals)}건\n")

if all_deals:
    print("  샘플 5건:")
    for d in all_deals[:5]:
        print(f"    {d.get('umdNm',''):<8} {d.get('totalFloorAr','?'):>7}㎡ "
              f"| 보증금 {d.get('deposit','').strip():>8}만 "
              f"| 월세 {d.get('monthlyRent','').strip():>5}만")

    # 전세/월세 비율
    jeonse = sum(1 for d in all_deals
                 if d.get("monthlyRent", "0").strip().replace(",", "") in ("0", ""))
    print(f"\n  ★ 전세: {jeonse}건 ({jeonse/len(all_deals)*100:.1f}%)")
    print(f"  ★ 월세: {len(all_deals)-jeonse}건 ({(len(all_deals)-jeonse)/len(all_deals)*100:.1f}%)")

    # 동별 분포
    print(f"\n  동별 건수 (상위 5):")
    for dong, cnt in Counter(d.get("umdNm", "?") for d in all_deals).most_common(5):
        print(f"    {dong:<10} {cnt}건")

    # 우만동만
    wooman = [d for d in all_deals if "우만" in d.get("umdNm", "")]
    print(f"\n  ★★ 우만동 표본: {len(wooman)}건")
    if len(wooman) < 30:
        print("     ⚠️  30건 미만 → 동 단위 분포 추정 어려움. 구 단위로 확대 필요")
    else:
        print("     ✅ 분포 추정 가능")


print("\n" + "═" * 60)
print("검증 완료")
print("═" * 60)
