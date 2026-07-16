"""
등기사항전부증명서(등기부등본) PDF를 OpenAI로 직접 읽어서 권리관계를 구조화 추출한다.

*** 중요한 한계 ***
- 실제 등기부에서 "말소사항 포함" 문서는 말소(취소)된 권리를 실선(취소선)으로 표시하는데,
  pypdf 텍스트 추출은 취소선 서식을 인식하지 못한다. 즉 이미 말소된 근저당/압류가 여전히
  "유효한" 것으로 잘못 추출될 위험이 있다 — 사람이 원본 PDF를 육안으로 대조 확인하는 걸
  권장하며, 이 리포트에도 그 한계를 명시한다.
- 근저당 설정 여부·금액, 압류/가압류/신탁 여부, 소유권이전 횟수는 LLM이 텍스트에서 읽어낸
  값이라 OCR/파싱 오류 가능성이 있다 — 법적 효력이 있는 판단에는 원본 등기부를 직접 확인할 것.
- PDF에는 실소유자 실명·주민등록번호가 포함돼 있어, OpenAI로 전송하기 전에 정규식으로
  이름+주민번호 패턴을 레닥션한다(완벽하지 않을 수 있음 — 표준 형식만 커버).
"""

import json
import re

import pypdf


def extract_text(pdf_path):
    reader = pypdf.PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)


# 관찰된 실제 형식: "이석재  601121-*******" (한글 이름 + 공백 + 생년월일6자리-마스킹7자리)
_NAME_RRN_PATTERN = re.compile(r"[가-힣]{2,5}\s+\d{6}-\*+")


def redact_pii(text):
    """이름+주민등록번호 패턴을 제거한다. 표준 형식만 커버하는 정규식 기반 레닥션이라 완벽하지 않다."""
    return _NAME_RRN_PATTERN.sub("[개인정보 제거]", text)


_EXTRACTION_PROMPT = """\
아래는 대한민국 등기사항전부증명서(등기부등본)에서 추출한 텍스트다(개인정보는 이미 제거됨).
이 문서의 갑구(소유권)·을구(소유권 이외 권리) 내용만 근거로 다음 JSON 스키마 그대로 답하라.
문서에 없는 내용은 절대 지어내지 말고 사실 그대로만 채워라.

{{
  "mortgage_won": <을구에 유효한(말소 안 된) 근저당권 채권최고액 합계, 원 단위 정수. 없으면 0>,
  "has_attachment": <압류가 하나라도 있으면 true, 없으면 false>,
  "has_provisional_attachment": <가압류가 하나라도 있으면 true, 없으면 false>,
  "has_trust": <신탁 등기가 있으면 true, 없으면 false>,
  "ownership_transfer_count": <갑구에서 소유권보존/이전/경정 등 소유권 관련 등기 건수>,
  "notes": "<특이사항이 있으면 한 문장으로, 없으면 빈 문자열>"
}}

등기부 텍스트:
---
{registry_text}
---
"""


def extract_registry_info(pdf_path):
    """
    등기부 PDF에서 권리관계를 구조화 추출한다. OPENAI_API_KEY가 없거나 호출 실패 시 예외를 던진다
    (--registry-pdf는 opt-in 기능이라 조용한 폴백 없이 호출부에서 에러로 안내한다).
    """
    import openai

    text = extract_text(pdf_path)
    redacted = redact_pii(text)

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": _EXTRACTION_PROMPT.format(registry_text=redacted)}],
    )
    result = json.loads(response.choices[0].message.content)
    result["source_pdf"] = pdf_path
    return result
