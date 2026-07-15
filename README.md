# 세이프홈 (SafeHome) — 다가구주택 전세사기 위험 예측

KB국민은행 제8회 AI Challenge (주제 1. 청년 주거 금융 도우미) 출품 프로젝트. 슬로건: "같은 집, 다른 결론."

## 한 줄 요약

다가구주택은 등기가 건물 전체로 1개라서, 근저당은 등기부에 보이지만 **앞서 입주한 다른 세대들의 임차보증금(선순위 보증금)은 등기부 어디에도 안 보인다.** 이 프로젝트는 그 숨은 선순위 보증금을 확률적으로 추정하고, 실제 사고 시 회수 가능액과 사고 확률, 기대손실 금액까지 계산해서 "직방 지킴진단"류 서비스가 놓치는 부분을 보완하는 리포트를 만든다.

자세한 문제정의와 설계 배경은 [`docs/KB_AI_Challenge_기획안_v2 (1).md`](docs/KB_AI_Challenge_기획안_v2%20(1).md)에 있다. 이 README는 **팀원이 처음 봐도 프로젝트 전체를 파악하고 바로 실행해볼 수 있게** 만든 문서다.

---

## 왜 이 프로젝트를 시작했나 (과정 요약)

1. **실측 검증**: 직방 '지킴진단 리포트'를 실제로 발급받아 확인함. 대상은 수원 팔달구 우만동의 실제 9가구 다가구주택. 결과는 "매물 진단 양호 / 집주인 진단 양호 / 근저당 없음" — **9가구 건물인데 앞선 8세대의 보증금 얘기는 리포트 어디에도 없었다.**
2. **API 검증**: 이 문제를 풀려면 (a) 대상 건물의 가구수, (b) 호별 면적, (c) 지역 전월세 실거래 표본, (d) 전세/월세 비율이 실제로 API로 조회되는지 먼저 확인해야 했다. 건축물대장 API(표제부·전유공용면적·층별개요)와 국토부 전월세 실거래가 API로 검증한 결과: 가구수(9)는 나오지만 **전유부(호별 면적)는 다가구 특성상 안 나오고**(층별개요로 fallback), 실거래 표본은 충분했다.
3. **선행연구 조사**: "아무도 사고확률을 예측하지 않는다"는 초기 가설을 검증하기 위해 선행연구를 찾아본 결과, 이미 유사한 시도(안선영·이상엽 2025, 민병철 2023/2024 등)가 있다는 걸 확인. 문제정의를 "다가구 한 건물 내 여러 세입자가 경합하는 누적 선순위 보증금을 정량 모델링한 사례는 없다"로 좁혔다 (근거 상세는 기획안 2.2절).
4. **파이프라인 구현**: 기획안의 A~E 모듈을 실제 동작하는 Python 코드로 구현 (아래 "모듈 구조" 참고).
5. **모듈 C 고도화**: 안선영·이상엽(2025) 논문 원문에서 실제 로지스틱 회귀 오즈비(HF·HUG 실사고 데이터 45만여 건 기반)를 확인하고, 손으로 정했던 임의 승수를 논문의 실제 계수로 교체.
6. **모듈 A 학습모델 리빌드 (완료)**: "AI Challenge인데 학습된 모델이 없다"는 문제의식에 따라, 유일하게 라벨(실거래 보증금)이 존재하는 모듈 A를 정식 ML 파이프라인(EDA → 베이스라인 → 6개 quantile 회귀 모델 비교 → 해석 → 재구현 → 최종검증, Phase 0~8)으로 재구축했다. 수원 4개구 3년치(49,112건)로 학습한 QRF(Quantile Regression Forest)가 val Pinball 712.6(기존 KDE) → 408.5로 약 43% 개선, test에서도 390.25로 재확인됨. 전체 과정은 [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)에 Phase별로, 최종 결론은 [`docs/최종검증_결과.md`](docs/최종검증_결과.md)에 기록.
7. **시세 자동화 + 전체 주택유형 지원 (2026-07-15)**: 국토교통부_단독/다가구 매매 실거래가 API를 활용신청해서 `--market-price`를 자동 추정하도록 연결(더 이상 VWorld 불필요). 동시에, 건축물대장의 `fmlyCnt`(가구수) 값으로 건물 유형을 판별해 **아파트·연립·다세대(구분등기) 건물도 리포트 대상에 포함**시켰다 — 다만 구분등기 건물은 "숨은 다른 세입자" 개념이 구조적으로 없으므로 A모듈(선순위 보증금)은 적용하지 않고, B·C·D모듈(경매회수·사고확률·기대손실)만으로 "보증금 안전도"를 진단한다.

---

## 시스템 구조

```
직방 파이프라인                     본 프로젝트
────────────────────────────    ─────────────────────────
1. 등기부·건축물대장 조회      →  동일
2. 집주인 재무·채무            →  동일
3. 시세·보증금 안전성          →  [A] 선순위 분포 추정
                                  [B] 경매 배당 시뮬레이션
4. 치안·생활편의               →  동일
5. AI 요약                     →  동일 (LLM)
                               →  [C] 사고 확률 예측
                               →  [D] 기대손실 산출
6. 전문가 검토                 →  자동화로 대체
7. 맞춤 특약                   →  [E] KB 금융상품 액션
8. 리포트 발급                 →  동일
```

### 모듈별 상세

| 모듈 | 뭘 하나 | 어떻게(현재) | 성격 |
|---|---|---|---|
| **[A]** 선순위 보증금 분포 | 나보다 먼저 들어온 N-1세대가 걸어둔 보증금 총액을 확률분포로 추정 | 학습된 QRF(Quantile Regression Forest)로 세대별(면적·건물연식·지역·전세여부) 조건부 분위수 예측 → 역CDF 샘플링 → 몬테카를로 1만회 합산 | **학습된 지도학습 모델** (수원 4개구 48,047건, val Pinball 408.5 / test 390.25). 기존 KDE는 `archive/module_a_deposit_dist_kde_legacy.py`로 보존 |
| **[B]** 경매 배당 시뮬레이션 | 경매로 넘어갔을 때 내 보증금 실제 회수액 분포 | 낙찰가율 샘플링 → 낙찰가에서 경매비용·최우선변제금(법정기준표)·근저당·A모듈 선순위 순서대로 차감 | 규칙 기반 시뮬레이션 (실제 법령 수치 사용) |
| **[C]** 사고 확률 예측 | 이 건물에서 보증금을 못 돌려받을 확률 | 실질부채비율을 안선영·이상엽(2025) 논문의 실제 로지스틱회귀 오즈비(45만 건 데이터 기반)에 대입 → HUG 전국 사고율로 캘리브레이션 | 논문의 실제 학습된 계수를 재사용 |
| **[D]** 기대손실 | 등급이 아니라 원 단위 금액으로 위험 표현 | `P(사고, C) × (내 보증금 − 예상회수액, B)` | 단순 수식 |
| **[E]** KB 금융상품 액션 | 기대손실과 HUG 보증료를 비교해 가입 여부 판단 | 단순 비용-편익 비교 | **현재 리포트 범위에서 제외** (일단 직방 대응 진단 리포트부터 완성하기로 함). 코드(`src/module_e_recommendation.py`)는 남아있고 `report.py`에서만 호출 안 함 |

> **중요한 구분**: A모듈이 예측하는 건 "선순위 보증금"이 아니라 그냥 "세대별 보증금"이다. "선순위"라는 딱지는 몬테카를로 조립 단계(N-1번 샘플링)에서 붙이는 것 — 실거래 데이터에는 애초에 "이 계약이 선순위다"라는 라벨이 없다. 발표 문구에서 "선순위 보증금 예측 모델"이라고 하면 부정확하고, "세대별 보증금 예측 + 몬테카를로 조립"이 정확한 표현.

> **주택유형별 분기**: 건축물대장의 `fmlyCnt`(가구수)가 0보다 크면 다가구/다세대류(비구분등기)로 판단해 A~D 전 모듈을 적용한다. 반대로 아파트·연립·다세대 같은 **구분등기 공동주택은 세대별로 등기가 분리돼 있어 "숨은 다른 세입자" 문제 자체가 없으므로 A모듈을 건너뛰고 B·C·D모듈만으로 보증금 안전도를 진단**한다 (`report.py`의 `is_multi_household` 분기, `src/config.py` 사용).

---

## 폴더/파일 구조

```
.
├── README.md                          이 문서
├── requirements.txt                   전체 의존성 (API 호출 + ML 스택)
├── report.py                          CLI 진입점 — 전체 파이프라인 실행 + 리포트 출력 (여기서 실행)
│
├── docs/                              문서 (기획·진행기록·Phase별 결과)
│   ├── IMPLEMENTATION.md                구현 진행 로그 (Phase 0~8 전부 기록, 계속 갱신됨)
│   ├── KB_AI_Challenge_기획안_v2 (1).md 전체 기획안 (문제정의·설계 원문)
│   ├── CLAUDE_프롬프트.md               초기 인수인계 문서 (참고용, 최신 내용은 기획안_v2 우선)
│   ├── EDA_결과.md / 데이터분할_결과.md / 피처엔지니어링_결과.md
│   ├── 베이스라인_결과.md / 모델_비교표.md / 모델_해석.md / 최종검증_결과.md   Phase 1~8 각 단계 상세 결과
│   └── eda_plots/                      그래프(타겟분포, 상관행렬, feature importance, SHAP, PDP 등)
│
├── src/                                A~E 파이프라인 모듈 (패키지, 실제 서비스가 매번 실행하는 코드)
│   ├── config.py                        API 키·법정 기준표·통계 파라미터 (전부 출처 주석)
│   ├── api_client.py                     건축물대장·실거래가 API 호출 함수
│   ├── module_a_deposit_dist.py          [A] 선순위 보증금 분포 — 학습된 QRF 모델 로드해서 예측
│   ├── module_b_auction_sim.py           [B] 경매 배당 시뮬레이션
│   ├── module_c_risk_score.py            [C] 사고확률 스코어링
│   ├── module_d_expected_loss.py         [D] 기대손실 계산
│   └── module_e_recommendation.py        [E] KB 금융상품 액션 (현재 report.py에서 미사용, 코드만 보존)
│
├── training/                           모듈 A 학습모델 리빌드 파이프라인 (Phase 0~8, 한 번 돌리고 결과물만 쓰면 됨)
│   ├── phase0_collect_data.py            데이터 수집 (API → data/raw/)
│   ├── phase1_eda.py                     탐색적 분석
│   ├── phase2_split.py                   시간 기반 train/val/test 분할
│   ├── phase3_features.py                피처 엔지니어링 (Target Enc / One-hot)
│   ├── phase4_baselines.py               베이스라인 4종 비교
│   ├── phase5_train.py                   모델 6종 Optuna 튜닝·비교
│   ├── phase5_cv_stability.py            최종후보 5-Fold CV 안정성 확인
│   ├── phase6_interpret.py               Feature Importance/SHAP/PDP/오류분석
│   ├── phase7_finalize_model.py          Test 최종평가 + 배포용 모델 학습·저장
│   └── phase8_final_comparison.py        기존 KDE vs 신규 모델 최종 비교
│
├── models/
│   └── module_a_qrf.joblib             프로덕션 모델 (train+val+test 48,047건 학습, ~79MB) + module_a_qrf_meta.json
│
├── data/
│   ├── raw/rent_deals.csv              수집 원본 (수원 4개구, 36개월, 49,112건)
│   └── processed/                      train/val/test.csv, *_features.csv
│
└── archive/
    ├── test_api.py                     초기 API 검증용 스크립트 (기능은 src/api_client.py로 대체됨)
    └── module_a_deposit_dist_kde_legacy.py   교체 전 KDE 버전 (Phase 8 비교용으로 보존)
```

`report.py`와 `training/*.py`는 항상 **저장소 루트에서** 실행한다 (`python3 report.py ...`, `python3 training/phase0_collect_data.py`). `src/`는 패키지라서 `from src.config import ...` 식으로 참조된다.

### 학습모델 재생성이 필요하면

`models/module_a_qrf.joblib`은 이미 학습이 끝난 결과물이라 `report.py`는 이것만 로드해서 쓰면 된다. 데이터가 바뀌었거나 모델을 처음부터 다시 만들고 싶으면 `training/phase0_collect_data.py`부터 `training/phase7_finalize_model.py`까지 순서대로 실행(각 Phase는 이전 Phase의 산출물을 읽음). 전체 소요시간은 약 20~30분(Phase 0 데이터 수집이 대부분).

---

## 실행 방법

macOS 시스템 파이썬은 externally-managed라 pip 직접 설치가 막혀 있다. **가상환경(.venv)을 만들어서 쓴다**:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 필수: data.go.kr에서 발급받은 서비스키 (건축HUB·실거래가 API 공통, 저장소에는 커밋 안 함)
export DATA_GO_KR_SERVICE_KEY="발급받은_서비스키"
# 선택: 종합의견을 실제 LLM으로 생성하고 싶으면 (없으면 규칙 기반 문장으로 자동 폴백)
export OPENAI_API_KEY="sk-..."

python3 report.py \
  --my-deposit 50000000 \
  --mortgage 0
  # --market-price 생략하면 매매 실거래 API로 자동 추정 (아래 참고)
```

기본 대상 건물은 수원시 팔달구 우만동 39-7 (직방 리포트로 실측 검증한 그 건물). 다른 건물을 보려면 `--sigungu --bjdong --bun --ji --dong-name`을 바꾸면 된다.

### 지금 당장 알아야 할 한계 2가지 (직접 입력이 필요한 값)

| 값 | 이유 |
|---|---|
| `--mortgage` (근저당) | 등기부 자동조회(CODEF)가 사업자등록번호를 요구해서 보류. [iros.go.kr](https://www.iros.go.kr)에서 건당 약 700원으로 직접 열람해서 을구 채권최고액을 입력하면 됨 |
| 사고확률(C모듈) | 학습된 분류기가 아니라 논문 계수 기반 근사치. HUG 악성임대인 명단(127명)이 주소 매칭이 안 돼서 자체 학습셋을 못 만듦 |

**시세(`--market-price`)는 2026-07-15부로 자동화됨** — 국토교통부_단독/다가구 매매 실거래가 API(`RTMSDataSvcSHTrade`)로 유사 면적대 매매가 중앙값을 자동 추정한다. 표본이 부족하면 에러를 내고 직접 입력을 요구하니, 그때만 `--market-price`를 수동으로 넣으면 된다.

---

## 입력/출력 스키마

팀원이 이 서비스를 다른 곳(프론트엔드, API 등)에 연결하려면 이 부분이 제일 중요하다. `report.py`는 기본적으로 사람이 읽는 텍스트를 출력하지만, **`--json` 플래그를 주면 아래 구조 그대로 JSON을 stdout에 출력**한다 (`python3 report.py --my-deposit ... --mortgage ... --market-price ... --json`).

### 입력

| 값 | 필수 | 기본값 | 비고 |
|---|---|---|---|
| `--sigungu` | 아니오 | `41115` (수원 팔달구) | 시군구코드 |
| `--bjdong` | 아니오 | `14000` (우만동) | 법정동코드 |
| `--bun` / `--ji` | 아니오 | `0039` / `0007` | 본번/부번 |
| `--dong-name` | 아니오 | `우만동` | 실거래 필터링용 동 이름 |
| `--my-deposit` | **예** | — | 내 보증금 (원) |
| `--mortgage` | **예** | — | 근저당 금액 (원) — 등기부 직접 확인 필요 (위 한계표 참고) |
| `--market-price` | 아니오 | 자동추정(매매 실거래 API) | 생략하면 자동 추정, 실패 시(표본 부족 등) 에러 발생 → 그때 직접 입력 |
| `--known-tenants` | 아니오 | `0` | 확정일자 부여현황 등으로 직접 확인한 선순위 임차인 수(관측된 하한). API 자동조회 불가(임대인 동의 필요) — 수동 입력만 지원 |
| `--known-priority-deposit-won` | 아니오 | `0` | `--known-tenants`만큼의 실측 확인된 선순위 보증금 총액(원). 고정값으로 반영되고 나머지 세대만 몬테카를로 추정 |
| `--n-sim` | 아니오 | `10000` | 몬테카를로 시뮬레이션 횟수 |
| `--seed` | 아니오 | `42` | 랜덤 시드 |
| `--json` | 아니오 | (미지정 시 텍스트) | 켜면 사람이 읽는 텍스트 대신 아래 스키마의 JSON을 출력 |

### 출력

```jsonc
{
  "building": {                          // 건축물대장 표제부 원본 (기본 진단)
    "address": "경기도 수원시 팔달구 우만동 39-7번지",
    "purpose": "단독주택",
    "purpose_detail": "주택9가구",
    "n_units": 9,                        // 가구수
    "areas_sqm": [140.22, 28.32, 108.15, 140.22],  // 호별/층별 면적(㎡)
    "area_is_fallback": true,            // 전유부 없어서 층별개요로 대체했는지
    "build_year": 1996
  },

  "module_a": {                          // [A] 학습된 QRF 모델 결과 — 원(KRW) 단위
    "priority_deposit_mean_won": 235630000,
    "priority_deposit_p05_won": 79720000,
    "priority_deposit_p95_won": 454720000,
    "priority_deposit_worst_p95_won": 454720000,
    "dong_sample_expanded": false        // 동 표본 부족으로 구 단위로 확대했는지
  },

  "market_reference": {                  // 참고 시세 (만원 단위)
    "area_band": "쓰리룸+(40㎡~)",
    "jeonse": {"q25": 8650, "q50": 10400, "q75": 13000, "n": 35},
    "wolse":  {"deposit_q25": 500, "deposit_q50": 1000, "deposit_q75": 4200,
               "rent_q25": 39, "rent_q50": 50, "rent_q75": 70, "n": 71},
    "conversion_rate_pct": 6.4           // 전월세 전환율 (전세·월세 둘 다 있을 때만 존재)
  },

  "safety_reference": {                  // 참고 치안정보 (행안부 CCTV 표준데이터, 수원시 한정)
    "dong_name": "우만동",
    "cctv_count": 115,
    "camera_total": 417
  },                                      // 해당 동에 표본이 없으면 null

  "module_c": {                          // [C] 사고확률 스코어링
    "registry_debt_ratio": 0.0,          // 등기부 기준(근저당/시세)
    "real_debt_ratio": 0.785,            // 선순위 반영(A모듈 평균 사용)
    "jeonse_ratio": 0.167,
    "risk_multiplier": 2.09,
    "accident_probability": 0.046
  },

  "module_b": {                          // [B] 경매 배당 시뮬레이션 (조건부: 경매 발생 가정)
    "full_recovery_pct": 2.4,
    "partial_recovery_pct": 11.9,
    "total_loss_pct": 85.8,
    "expected_recovery_won": 3890000,
    "partial_recovery_avg_won": 12300000
  },

  "module_d": {                          // [D] 기대손실
    "expected_loss_won": 2100000
  },

  "summary_opinion": {                   // 종합의견 (OpenAI 실호출, 실패/키없으면 규칙기반 폴백)
    "text": "본 건물은 9가구로 구성된 다가구 주택으로...",
    "source": "llm"                      // "llm" 또는 "rule_based"
  },

  // [E] 금융상품 액션은 현재 리포트 범위에서 제외 (직방 대응 진단 리포트부터 완성하기로 함)

  "inputs_used": {                       // 사용자가 넣은 값 + 시세 출처 기록(재현성/감사용)
    "my_deposit_won": 50000000,
    "mortgage_won": 0,
    "market_price_won": 510000000,
    "market_price_source": "auto_trade_api",  // "auto_trade_api" 또는 "manual"
    "known_tenants": 0,                  // 확정일자 부여현황 등으로 실측 확인한 선순위 임차인 수 (기본 0)
    "known_priority_deposit_won": 0      // 그 세대들의 실측 확인된 보증금 총액 (기본 0)
  }
}
```

각 블록이 어느 모듈/함수에서 나오는지: `building`은 `src/api_client.get_building_title`, `module_a`는 `src/module_a_deposit_dist.summarize`, `market_reference`는 `report.py`의 `market_reference()`, `safety_reference`는 `src/safety_info.cctv_summary()`, `module_b`는 `src/module_b_auction_sim.summarize`, `module_c`는 `src/module_c_risk_score.score`, `module_d`는 `src/module_d_expected_loss.expected_loss`가 그대로 반환하는 딕셔너리다. `summary_opinion`은 `report.py`의 `get_summary_opinion()` — `OPENAI_API_KEY` 환경변수가 있으면 실제 OpenAI(`gpt-4o-mini`) 호출, 없거나 실패하면 규칙 기반 문장 조립으로 자동 폴백한다. ([E]는 현재 미사용)

---

## 현재 진행 상황

- ✅ 기본 파이프라인(A~D) 구현 및 검증 완료 ([E] 금융상품 추천은 일단 범위 제외)
- ✅ 모듈 C 논문 계수 반영 완료
- ✅ **모듈 A 학습모델 리빌드 완료** (Phase 0~8 전부 완료 — QRF 모델로 교체, val Pinball 408.5 / test 390.25, 기존 KDE 대비 검증된 개선)
- ✅ **시세(`--market-price`) 자동화 완료** (2026-07-15, 매매 실거래 API 활용신청·연동 완료 — 더 이상 VWorld 불필요)
- ✅ **종합의견(AI 요약) 완료** — `OPENAI_API_KEY` 설정 시 OpenAI(`gpt-4o-mini`) 실호출로 자연어 종합의견 생성, 미설정/호출 실패 시 규칙 기반 문장 조립으로 자동 폴백 (숫자는 항상 계산된 값만 사용, LLM이 지어내지 않도록 프롬프트에 명시)
- ✅ **참고 치안정보(CCTV) 추가 완료** — 행안부 CCTV 표준데이터(수원시, 사용자가 직접 다운로드한 파일 기반) 연동, 법정동명으로 필터링해 설치개소·카메라대수를 참고정보로 표시
- ✅ **A모듈 확정일자 수동 override 완료** — `--known-tenants`/`--known-priority-deposit-won`으로 확정일자 부여현황 등 실측 확인값을 고정 반영하고 나머지 세대만 몬테카를로 추정. 확정일자 부여현황은 주택임대차보호법 제3조의6상 임대인 동의가 필요해 API 자동조회가 불가능(계약 전 접근 제한적)하므로 수동 입력만 지원 — 이 접근성 한계 자체가 A모듈의 존재 이유이기도 함

**아직 남은 것 (향후 과제)**:
- C모듈 PU Learning 실제 학습 (HUG 명단 127명 주소매칭 불가 문제 미해결)
- CODEF(근저당) API 연동 — 사업자등록번호 문제로 보류, 현재는 수동 입력
- 학습 데이터 지역 확장(현재 수원 4개구 한정)
- [E] 금융상품 추천 리포트 재도입 여부 검토
- 경찰청 지구대·파출소 데이터(Open API, 활용신청 필요 + 주소 지오코딩 필요) 반영 — CCTV 대비 마찰이 커서 아직 미착수

상세 진행 로그와 각 Phase의 "한 일 / 발견 / 결정 근거"는 [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)를, 최종 결론은 [`docs/최종검증_결과.md`](docs/최종검증_결과.md)를 참고할 것 — 두 문서는 계속 갱신되는 작업 일지이고, README는 안정적인 전체 그림만 유지한다.

---

## 참고 문헌 / 데이터 출처

- 안선영·이상엽 (2025), "전세보증금 미반환에 영향을 미치는 주요요인 연구", 주택금융연구 9(2) — 모듈 C의 부채비율 오즈비 출처
- 주택임대차보호법 시행령 (2023.2.21 개정) — 최우선변제금 기준표
- 지지옥션 2026년 5월 동향 — 경기도 연립·다세대 낙찰가율
- HUG 공시 — 전세보증금 대위변제 사고율 추이
- 국토교통부 건축HUB(건축물대장), 국토교통부 단독/다가구 전월세 실거래가, 공공데이터포털
