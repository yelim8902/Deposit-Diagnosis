"""
치안 정보 참고자료 — 수원시 CCTV 표준데이터(행정안전부, data.go.kr) 기반.
API가 아니라 파일(수동 다운로드) 데이터라 수원시로 지역이 이미 한정되어 있음.
숨은 선순위 보증금(A모듈)과 무관한 참고용 부가정보이며, 사고확률(C모듈) 계산에는 반영하지 않는다.
"""

import os

import pandas as pd

CCTV_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cctv_suwon.csv")

_cctv_df = None


def _load_cctv():
    global _cctv_df
    if _cctv_df is None:
        _cctv_df = pd.read_csv(CCTV_CSV_PATH, encoding="utf-8")
    return _cctv_df


def cctv_summary(dong_name):
    """
    법정동명으로 소재지지번주소를 매칭해 해당 동의 방범 CCTV 설치 현황을 집계한다.
    표본이 없으면 None을 반환한다(숫자를 지어내지 않음).
    """
    df = _load_cctv()
    subset = df[df["소재지지번주소"].astype(str).str.contains(dong_name, na=False)]
    if subset.empty:
        return None

    return {
        "dong_name": dong_name,
        "cctv_count": int(len(subset)),
        "camera_total": int(subset["카메라대수"].fillna(0).sum()),
    }
