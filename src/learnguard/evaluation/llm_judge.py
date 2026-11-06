"""LLM-as-judge cho Gate 3 — Gemini chấm chất lượng nội dung advisor brief.

Khác với 8 check còn lại của LLM gate (chỉ phân tích dữ liệu đã ghi DB),
check `judge_quality` gọi Gemini REST trực tiếp (`urllib.request`, KHÔNG thêm
dependency — litellm thuộc backend) để chấm từng brief theo rubric 4 tiêu chí:
grounded / useful / no_fabrication / clarity, mỗi tiêu chí 1–5.

API key KHÔNG BAO GIỜ được ghi vào scorecard hay log — scorecard chỉ chứa
`judge_model` + số call. Mỗi brief 1 call, timeout 30s; call lỗi (network /
parse) được đếm và bỏ qua ở tầng gate, không crash.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from statistics import mean
from typing import Any

JUDGE_DEFAULT_MODEL = "gemini-3.1-flash-lite"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
JUDGE_TIMEOUT_S = 30
JUDGE_MAX_OUTPUT_TOKENS = 1000
JUDGE_CRITERIA: tuple[str, ...] = (
    "grounded", "useful", "no_fabrication", "clarity",
)


class JudgeCallError(RuntimeError):
    """Một lần gọi judge thất bại (network / HTTP / JSON hỏng / schema lệch)."""


@dataclass(frozen=True)
class JudgeVerdict:
    """Kết quả judge cho một brief — 4 điểm rubric + trung bình + ghi chú."""

    brief_id: int
    grounded: int
    useful: int
    no_fabrication: int
    clarity: int
    overall: float
    issues: str


def build_judge_prompt(
    content: dict[str, Any],
    citations: list[dict[str, Any]],
    top_signals: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> str:
    """Prompt tiếng Việt: brief + evidence gốc, rubric 4 tiêu chí, JSON schema chặt."""
    return (
        "Bạn là giám khảo độc lập (LLM-as-judge) chấm chất lượng một brief cố vấn "
        "học tập mà hệ thống sinh ra cho một sinh viên.\n\n"
        "## BRIEF CẦN CHẤM (nội dung hệ thống sinh)\n"
        f"summary:\n{_render(content.get('summary'))}\n\n"
        f"reasons:\n{_render(content.get('reasons'))}\n\n"
        f"support_suggestions:\n{_render(content.get('support_suggestions'))}\n\n"
        f"limitations:\n{_render(content.get('limitations'))}\n\n"
        "## EVIDENCE GỐC (chỉ được dựa vào các dữ kiện dưới đây)\n"
        f"citations mà brief tuyên bố:\n{_render(list(citations))}\n\n"
        f"top_signals của prediction cùng khóa:\n{_render(list(top_signals))}\n\n"
        "## RUBRIC — chấm MỖI tiêu chí một số nguyên từ 1 đến 5\n"
        "1. grounded — mọi phát biểu trong brief có bám evidence đã cho không? "
        "(5 = mọi phát biểu đều có evidence, 1 = hầu như không bám evidence)\n"
        "2. useful — đề xuất khả thi, có hành động cụ thể mà cố vấn làm được không?\n"
        "3. no_fabrication — brief có bịa số liệu/fact KHÔNG có trong evidence không? "
        "Nếu có bịa bất kỳ số liệu/fact nào → tiêu chí này PHẢI ≤ 2.\n"
        "4. clarity — ngôn ngữ tiếng Việt rõ ràng, phù hợp để gửi cho giáo viên?\n\n"
        "Chỉ trả về MỘT object JSON đúng schema sau, không thêm chữ nào khác:\n"
        '{"grounded": <1-5>, "useful": <1-5>, "no_fabrication": <1-5>, '
        '"clarity": <1-5>, "overall": <trung bình 4 điểm, làm tròn 1 chữ số>, '
        '"issues": "<mô tả ngắn các vấn đề hoặc chuỗi rỗng>"}'
    )


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def call_gemini_json(
    prompt: str,
    api_key: str,
    model: str = JUDGE_DEFAULT_MODEL,
    timeout_s: int = JUDGE_TIMEOUT_S,
) -> dict[str, Any]:
    """POST generateContent REST thuần, trả về JSON object judge trả ra.

    generationConfig: temperature=0, response_mime_type="application/json",
    maxOutputTokens=1000. Mọi lỗi network/HTTP/parse → `JudgeCallError`.
    """
    if not api_key:
        raise JudgeCallError("judge_api_key_missing")
    url = (
        f"{GEMINI_API_BASE}/models/{model}:generateContent"
        f"?key={urllib.parse.quote(api_key, safe='')}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "maxOutputTokens": JUDGE_MAX_OUTPUT_TOKENS,
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Body lỗi Gemini không chứa key — đọc an toàn để chẩn đoán (403/429/...).
        try:
            reason = exc.read().decode("utf-8", "replace")[:200].replace("\n", " ")
        except Exception:  # noqa: BLE001
            reason = ""
        raise JudgeCallError(
            f"judge_http_error:HTTP{exc.code}:{reason}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — mọi lỗi transport gói lại đồng nhất
        raise JudgeCallError(f"judge_http_error:{type(exc).__name__}") from exc
    text = _extract_text(payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeCallError("judge_json_parse_failed") from exc
    if not isinstance(parsed, dict):
        raise JudgeCallError("judge_json_not_object")
    return parsed


def _extract_text(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise JudgeCallError("judge_response_invalid")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise JudgeCallError("judge_response_empty")
    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not parts:
        raise JudgeCallError("judge_response_no_parts")
    texts = [
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict)
    ]
    text = "".join(texts).strip()
    if not text:
        raise JudgeCallError("judge_response_blank")
    return text


def parse_judge_verdict(raw: dict[str, Any], brief_id: int) -> JudgeVerdict:
    """Validate schema chặt: 4 tiêu chí số nguyên 1–5; overall tự tính từ 4 điểm."""
    scores: dict[str, int] = {}
    for criterion in JUDGE_CRITERIA:
        value = raw.get(criterion)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JudgeCallError(
                f"judge_score_invalid:{criterion}:{value!r}"
            )
        if isinstance(value, float) and not value.is_integer():
            raise JudgeCallError(
                f"judge_score_invalid:{criterion}:{value!r}"
            )
        score = int(value)
        if not 1 <= score <= 5:
            raise JudgeCallError(f"judge_score_out_of_range:{criterion}:{score}")
        scores[criterion] = score
    issues = raw.get("issues", "")
    if not isinstance(issues, str):
        issues = str(issues)
    return JudgeVerdict(
        brief_id=brief_id,
        grounded=scores["grounded"],
        useful=scores["useful"],
        no_fabrication=scores["no_fabrication"],
        clarity=scores["clarity"],
        overall=round(mean(scores[c] for c in JUDGE_CRITERIA), 1),
        issues=issues,
    )


def judge_brief(
    brief: Any,
    evidence: dict[str, Any],
    *,
    api_key: str,
    model: str = JUDGE_DEFAULT_MODEL,
    timeout_s: int = JUDGE_TIMEOUT_S,
) -> JudgeVerdict:
    """Chấm một brief: build prompt từ content + evidence, gọi Gemini, validate.

    `evidence` = {"citations": list dict citation của brief,
                  "top_signals": tuple dict top_signals của prediction cùng khóa}.
    """
    prompt = build_judge_prompt(
        brief.content if isinstance(brief.content, dict) else {},
        evidence.get("citations", []),
        evidence.get("top_signals", ()),
    )
    raw = call_gemini_json(prompt, api_key, model=model, timeout_s=timeout_s)
    return parse_judge_verdict(raw, brief.brief_id)


__all__ = [
    "JUDGE_CRITERIA",
    "JUDGE_DEFAULT_MODEL",
    "JUDGE_MAX_OUTPUT_TOKENS",
    "JUDGE_TIMEOUT_S",
    "JudgeCallError",
    "JudgeVerdict",
    "build_judge_prompt",
    "call_gemini_json",
    "judge_brief",
    "parse_judge_verdict",
]
