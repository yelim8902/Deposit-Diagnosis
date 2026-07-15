"""
[A] 숨은 선순위 보증금 분포 추정 — 학습된 QRF(Quantile Regression Forest) 모델 기반.

Phase 0~7(training/phase0~7_*.py, docs/EDA_결과.md ~ docs/모델_해석.md) 과정을 거쳐
기존 KDE(지역·전세월세만 구분) 방식을 실제 학습모델로 교체했다. 기존 KDE 버전은
archive/module_a_deposit_dist_kde_legacy.py에 그대로 보존(Phase 8 비교용).

★ 모델이 예측하는 건 "선순위 보증금"이 아니라 그냥 "세대별 보증금"이다.
"선순위"라는 개념은 이 파일의 시뮬레이션 단계(N-1번 샘플링)에서 붙이는 것 — 학습 데이터에
"이 계약이 선순위다"라는 라벨은 애초에 없다 (docs/IMPLEMENTATION.md "A~E 모듈 요약" 참고).

한계 (docs/모델_해석.md 오류분석 결과 그대로 반영):
- 90㎡ 이상·고가 전세 세대에서 모델이 구조적으로 과소예측하는 경향이 있음.
  대형 세대가 섞인 건물은 이 추정치가 보수적(과소)일 수 있음.
- 지역(동) 원핫 인코딩은 학습 당시 수원시 4개구 55개 동 기준. 대상 동이 그 목록에
  없으면 전부 0벡터로 처리되어(=지역 정보 없이) 예측 정밀도가 떨어질 수 있음.
- 분위수 7개(5/10/25/50/75/90/95%)로만 분포를 근사하므로, 5%/95% 바깥 꼬리는
  마지막 구간 기울기로 flat 외삽한다(진짜 분포의 극단 꼬리와는 다를 수 있음).
"""

import os

import joblib
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "module_a_qrf.joblib")

_bundle_cache = None


def _load_bundle():
    global _bundle_cache
    if _bundle_cache is None:
        _bundle_cache = joblib.load(MODEL_PATH)
    return _bundle_cache["model"], _bundle_cache["meta"]


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
    """deposit/monthlyRent 필드로 전세/월세 분리. 단위: 만원. (지역 p_jeonse 추정용으로 계속 사용)"""
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
    이름은 하위호환을 위해 유지(report.py 인터페이스 보존). 실제로는 KDE를 적합하지 않고,
    (1) 학습된 QRF 모델을 로드하고 (2) 이번 조회된 지역 실거래로 로컬 p_jeonse만 추정한다.
    지역 표본이 없으면 모델 학습 시 전체 데이터의 p_jeonse(meta['global_p_jeonse'])로 대체.
    """
    model, meta = _load_bundle()
    jeonse, wolse = split_jeonse_wolse(deals)
    n_jeonse, n_wolse = len(jeonse), len(wolse)
    if n_jeonse + n_wolse > 0:
        p_jeonse = n_jeonse / (n_jeonse + n_wolse)
    else:
        p_jeonse = meta["global_p_jeonse"]

    return {
        "model": model, "meta": meta, "p_jeonse": p_jeonse,
        "n_jeonse": n_jeonse, "n_wolse": n_wolse,
    }


def _quantile_inverse_sample(pred_log_matrix, quantiles, u):
    """
    분위수 격자(quantiles, x축)와 예측값(pred_log_matrix, 행별 y값)으로 역CDF 샘플링.
    u는 각 행에 대응하는 균등분포(0,1) 난수. u가 quantiles 범위 밖이면 마지막 구간 기울기로
    flat 외삽(클리핑)한다.
    """
    q = np.asarray(quantiles)
    u_clipped = np.clip(u, q[0], q[-1])
    idx = np.clip(np.searchsorted(q, u_clipped, side="right"), 1, len(q) - 1)
    n = pred_log_matrix.shape[0]
    q_lo, q_hi = q[idx - 1], q[idx]
    y_lo = pred_log_matrix[np.arange(n), idx - 1]
    y_hi = pred_log_matrix[np.arange(n), idx]
    frac = (u_clipped - q_lo) / (q_hi - q_lo)
    return y_lo + frac * (y_hi - y_lo)


def _build_dong_onehot(dong_name, dong_cols):
    vec = np.zeros(len(dong_cols))
    col = f"dong_{dong_name}"
    if col in dong_cols:
        vec[dong_cols.index(col)] = 1.0
    return vec


def simulate_priority_deposits(kde_info, n_units, area_list, build_year, dong_name,
                                n_sim=10000, seed=None, known_tenants=0, known_deposit_manwon=0.0):
    """
    앞선 (n_units - 1)세대의 총 선순위 보증금을 몬테카를로로 시뮬레이션.
    학습모델로 세대별(면적·건물연식·지역·전세여부 조건부) 분위수를 예측한 뒤 역CDF 샘플링하고,
    N-1세대만큼 합산한다. 반환 형식(만원 단위 배열)은 기존 KDE 버전과 동일.

    area_list: 대상 건물의 알려진 호별(또는 층별 fallback) 면적 리스트 — 프리테넌트 면적을
               이 목록에서 복원추출로 샘플링한다(개별 세대 면적을 알 수 없으므로).
    build_year: 대상 건물 준공연도 (건축물대장 표제부 useAprDay 앞 4자리). None이면 학습시 중앙값 사용.
    dong_name: 대상 건물의 법정동명 (예: "우만동").
    known_tenants: 확정일자 부여현황 등으로 사용자가 직접 확인한 "선순위 임차인 수"(관측된 하한).
                   API로 자동 조회할 방법이 없어 사용자 수동 입력만 지원한다(임대인 동의 필요 —
                   report.py 상단 docstring 참고). 0이면 기존과 동일하게 N-1세대 전부를 시뮬레이션.
    known_deposit_manwon: known_tenants만큼의 실제 확인된 보증금 총액(만원). 시뮬레이션 결과에
                   고정값으로 더해진다(추정이 아니라 실측값이므로 몬테카를로 대상에서 제외).
    """
    from datetime import date

    rng = np.random.default_rng(seed)
    n_prior = max(n_units - 1 - max(known_tenants, 0), 0)
    if n_prior == 0:
        return np.full(n_sim, known_deposit_manwon)

    model, meta = kde_info["model"], kde_info["meta"]
    p_jeonse = kde_info["p_jeonse"]
    quantiles = meta["quantiles"]
    dong_cols = meta["dong_cols"]
    numeric_features = meta["numeric_features"]

    if build_year is not None:
        build_year_imputed = float(build_year)
        build_year_missing = 0.0
    else:
        build_year_imputed = meta["buildyear_median"]
        build_year_missing = 1.0

    today = date.today()
    time_index = today.year * 12 + today.month
    building_age = today.year - build_year_imputed

    total_draws = n_sim * n_prior
    area_samples = rng.choice(area_list, size=total_draws, replace=True) if len(area_list) > 0 \
        else np.full(total_draws, 30.0)  # area_list가 비면 임의값이 아니라 최후 fallback으로만 사용, 상위에서 항상 채워 넣도록 되어있음
    is_jeonse_samples = (rng.random(total_draws) < p_jeonse).astype(float)

    dong_vec = _build_dong_onehot(dong_name, dong_cols)

    # 피처 순서: numeric_features + dong_cols (src/module_a training과 동일 순서로 맞춤)
    n_features = len(numeric_features) + len(dong_cols)
    X = np.zeros((total_draws, n_features))
    feat_idx = {name: i for i, name in enumerate(numeric_features)}
    X[:, feat_idx["totalFloorAr"]] = area_samples
    X[:, feat_idx["buildYear_imputed"]] = build_year_imputed
    X[:, feat_idx["buildYear_missing"]] = build_year_missing
    X[:, feat_idx["building_age"]] = building_age
    X[:, feat_idx["time_index"]] = time_index
    X[:, feat_idx["is_jeonse"]] = is_jeonse_samples
    X[:, len(numeric_features):] = dong_vec  # 모든 행에 동일한 지역 벡터(같은 건물이므로)

    pred_log = model.predict(X, quantiles=quantiles)  # (total_draws, n_quantiles)

    u = rng.random(total_draws)
    sampled_log = _quantile_inverse_sample(pred_log, quantiles, u)
    sampled_manwon = np.clip(np.expm1(sampled_log), 0, None)

    results = sampled_manwon.reshape(n_sim, n_prior).sum(axis=1) + known_deposit_manwon
    return results


def summarize(results_manwon):
    """results: 만원 단위 배열. 요약 통계 반환 (원 단위로 환산). 기존과 동일한 출력 형식 유지."""
    won = results_manwon * 10_000
    return {
        "mean": float(np.mean(won)),
        "p05": float(np.percentile(won, 5)),
        "p95": float(np.percentile(won, 95)),
        "worst_p95": float(np.percentile(won, 95)),  # 상위 5%(=95th pct) 값
    }
