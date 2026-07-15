"""
공통 설정값. 숫자는 전부 출처를 주석에 남긴다 (지어내지 않음).
"""

import os

# data.go.kr에서 발급받은 서비스키. 공개 저장소에 실제 키를 커밋하지 않도록 환경변수로 주입한다.
# 사용 전 `export DATA_GO_KR_SERVICE_KEY="..."` 로 설정할 것 (README 참고).
SERVICE_KEY = os.environ.get("DATA_GO_KR_SERVICE_KEY", "")

BUILDING_TITLE_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
EXPOS_PUBUSE_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrExposPubuseAreaInfo"
FLR_OULN_URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrFlrOulnInfo"
RENT_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcSHRent/getRTMSDataSvcSHRent"
# 국토교통부_단독/다가구 매매 실거래가 자료 — 시세(--market-price) 자동화용. 2026-07-15 활용신청·승인
# 반영 확인 완료(같은 서비스키로 정상 호출됨). 거래금액 필드는 dealAmount.
TRADE_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcSHTrade/getRTMSDataSvcSHTrade"

# ── CODEF (근저당/등기부등본 자동조회) ──────────────────────────────
# 아래 값은 https://codef.io 가입 → 상품 "부동산등기부등본 열람/발급" 신청 후
# https://codef.io/#/account/keys 에서 확인해 채워 넣으면 된다.
# CONNECTED_ID는 최초 1회 계정 등록(add) API 호출로 발급받아야 함 (그 가이드도 로그인 후 대시보드에 있음).
CODEF_CLIENT_ID = ""       # TODO: 가입 후 채우기
CODEF_CLIENT_SECRET = ""  # TODO: 가입 후 채우기
CODEF_CONNECTED_ID = ""   # TODO: 가입 후 채우기
# 아래 두 값은 로그인 후 "부동산등기부등본 열람/발급" 상품 문서 페이지에서
# 실제 요청 경로(path)와 organization 코드를 그대로 복사해서 채워야 한다.
# (문서가 로그인 후에만 렌더링돼서 여기서는 정확한 값을 추측해 넣지 않았음)
CODEF_REGISTER_PATH = ""       # TODO: 예) "/v1/kr/public/ck/..." 형태, 문서에서 확인
CODEF_ORGANIZATION_CODE = ""   # TODO: 대법원 인터넷등기소 organization 코드

# ── VWorld (개별주택가격 조회) ───────────────────────────────────────
# https://vworld.kr 가입 → 오픈API > 인증키 발급 (승인까지 이메일 인증 필요)
VWORLD_API_KEY = ""  # TODO: 가입 후 채우기
# PNU(19자리 필지고유번호) = 법정동코드(10) + 산여부(1) + 본번(4) + 부번(4)
# 예: 우만동 39-7 → "4111514000" + "1"(대지=1) + "0039" + "0007"
VWORLD_HOUSE_PRICE_URL = "https://api.vworld.kr/ned/data/getIndvdHousePriceAttr"  # TODO: 로그인 후 문서에서 정확한 경로 재확인

# 실거래 조회 시 동일 동 표본이 부족하면 구 단위로 확대하는 최소 표본 기준
MIN_SAMPLE_SIZE = 30

# ── 최우선변제금 기준표 ──────────────────────────────────────────
# 출처: 주택임대차보호법 시행령 (2023.2.21 개정), 지역별 소액임차인 보증금 한도·최우선변제액
# 수원시(팔달구 포함)는 수도권정비계획법 시행령 별표1상 "과밀억제권역"에 해당
PRIORITY_REPAYMENT_TABLE = [
    # (지역, 소액임차인 보증금 상한, 최우선변제액)
    {"region": "서울특별시", "deposit_limit": 165_000_000, "max_repayment": 55_000_000},
    {"region": "과밀억제권역·세종·용인·화성", "deposit_limit": 145_000_000, "max_repayment": 48_000_000},
    {"region": "광역시(인천제외)·안산·김포·광주·파주", "deposit_limit": 85_000_000, "max_repayment": 28_000_000},
    {"region": "그 밖의 지역", "deposit_limit": 75_000_000, "max_repayment": 25_000_000},
]
TARGET_BUILDING_REGION_CLASS = "과밀억제권역·세종·용인·화성"  # 수원시 팔달구

# ── 경매 낙찰가율 분포 파라미터 ────────────────────────────────────
# 출처: 지지옥션 2026년 5월 동향 - 경기도 연립·다세대 낙찰가율 평균 80.9%
# (단독/다가구 전용 통계는 확인하지 못해 연립·다세대 통계로 근사. 리포트에 한계 명시)
AUCTION_RATE_MEAN = 0.809
AUCTION_RATE_STD = 0.07  # 공식 표준편차 미확인 — 최근 관측 범위(약 75~89%)를 근거로 한 근사치
AUCTION_COST_RATE = 0.03  # 경매비용은 통상 낙찰가의 약 2~3% 수준으로 근사

# ── HUG 전세보증사고 통계 ────────────────────────────────────────
# 출처: HUG 공시(언론 종합) - 대위변제 사고율(신규 보증 대비) 추이
# 2023.5 8.1% → 2025.8 2.2% (하락 추세). 지역별 세분 통계는 확인하지 못해 전국 평균 사용
HUG_NATIONAL_ACCIDENT_RATE = 0.022

# ── KB 금융상품 파라미터 ─────────────────────────────────────────
# 실제 상품 요율은 공모전 발표용 근사치이며, 실제 HUG 보증료율(개인/부부합산소득·보증금별 차등)로 교체 필요
HUG_GUARANTEE_PREMIUM_RATE = 0.003  # 연 0.03~0.09% 수준의 대표값 근사
BUTTIMOK_LOAN_RATE = 0.021  # 버팀목 전세자금대출 최저금리대 근사
