"""Gate 3 — `learnguard evaluate llm`: đánh giá chất lượng LLM sinh advisor brief.

9 check với id cố định: llm_store_reachable, call_success_rate, latency_budget,
cost_budget, format_compliance, fallback_rate, grounding, advisor_feedback,
judge_quality.
KHÔNG ghi DB — chỉ đọc dữ liệu Gemini/advisor đã ghi vào PostgreSQL chia sẻ
qua `evaluation/db_reader.py` và phân tích. Riêng `judge_quality` (LLM-as-judge)
gọi Gemini REST trực tiếp qua `evaluation/llm_judge.py` khi có API key; không
key / không brief / --judge-sample 0 → check đó SKIPPED.

Không có DSN (`--pg-dsn` + env `LEARNGUARD_PG_DSN` đều trống) → toàn bộ check
SKIPPED. DSN có nhưng không kết nối được → llm_store_reachable FAIL, các check
còn lại SKIPPED. WARN không fail, FAIL do CLI đổi exit code (scorecard).
"""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from learnguard.evaluation.db_reader import BriefRow, LlmStoreData, LlmStoreReader
from learnguard.evaluation.llm_judge import (
    JUDGE_DEFAULT_MODEL,
    JudgeVerdict,
    judge_brief,
)
from learnguard.evaluation.scorecard import GateCheck, Scorecard
from learnguard.ml.preprocessing import _validate_run_id

DEFAULT_SCAN_LIMIT = 5000
DEFAULT_JUDGE_SAMPLE = 10
DSN_ENV_VAR = "LEARNGUARD_PG_DSN"
LLM_API_KEY_ENV_VARS: tuple[str, ...] = ("LLM_API_KEY", "GEMINI_API_KEY")

REQUIRED_CONTENT_FIELDS: tuple[str, ...] = (
    "summary", "reasons", "support_suggestions", "limitations",
)
GENERATED_STATUS = "generated"

CHECK_IDS: tuple[str, ...] = (
    "llm_store_reachable", "call_success_rate", "latency_budget",
    "cost_budget", "format_compliance", "fallback_rate", "grounding",
    "advisor_feedback", "judge_quality",
)

CHECK_NAMES: dict[str, str] = {
    "llm_store_reachable": "Kết nối PostgreSQL + bảng llm_calls/briefs/decisions",
    "call_success_rate": "Tỉ lệ LLM call status 'ok'",
    "latency_budget": "p95 latency của call ok trong ngân sách",
    "cost_budget": "Chi phí trung bình mỗi call trong ngân sách",
    "format_compliance": "Brief đủ 4 trường nội dung theo hợp đồng",
    "fallback_rate": "Tỉ lệ brief generation_status == 'generated'",
    "grounding": "Citations của brief ⊆ top_signals của prediction cùng khóa",
    "advisor_feedback": "Phản hồi advisor (accept/reject/edited)",
    "judge_quality": "LLM-as-judge: Gemini chấm chất lượng nội dung brief theo rubric",
}


@dataclass(frozen=True)
class LlmGateThresholds:
    """Ngưỡng warn/fail của LLM gate — chốt theo hợp đồng scorecard v1.0."""

    success_rate_warn: float = 0.90
    success_rate_fail: float = 0.70
    latency_warn_ms: float = 5000.0
    latency_fail_ms: float = 15000.0
    cost_warn_usd: float = 0.01
    format_warn: float = 0.05
    format_fail: float = 0.20
    fallback_warn: float = 0.10
    fallback_fail: float = 0.30
    grounding_fail: float = 0.20
    reject_rate_warn: float = 0.50
    # judge_quality (LLM-as-judge): PASS ≥ 4.0; WARN 3.0–3.9; FAIL < 3.0.
    # Bịa evidence (no_fabrication ≤ 2) nghiêm trọng nhất: ≥ 30% brief → FAIL trực tiếp.
    # Lỗi hạ tầng judge (network/parse) > 50% call → WARN, không FAIL.
    judge_pass: float = 4.0
    judge_warn: float = 3.0
    judge_fabrication_ratio: float = 0.30
    judge_error_ratio: float = 0.50


THRESHOLDS = LlmGateThresholds()

# Signature judge có thể tiêm: (brief, evidence) -> JudgeVerdict. evidence =
# {"citations": brief.citations, "top_signals": top_signal_details cùng khóa}.
JudgeFn = Callable[[BriefRow, dict[str, Any]], JudgeVerdict]


def resolve_pg_dsn(flag_value: str | None, environ: dict[str, str] | None = None) -> str:
    """`--pg-dsn` ưu tiên, fallback env `LEARNGUARD_PG_DSN`; rỗng = chưa cấu hình."""
    env = os.environ if environ is None else environ
    return flag_value or env.get(DSN_ENV_VAR, "")


def resolve_llm_api_key(flag_value: str | None, environ: dict[str, str] | None = None) -> str:
    """API key judge: `--llm-api-key` ưu tiên, fallback env `LLM_API_KEY` rồi
    `GEMINI_API_KEY`; rỗng = chưa cấu hình → check judge_quality SKIPPED.
    Key không bao giờ được ghi vào scorecard hay log."""
    env = os.environ if environ is None else environ
    if flag_value:
        return flag_value
    for name in LLM_API_KEY_ENV_VARS:
        value = env.get(name, "")
        if value:
            return value
    return ""


def run_llm_gate(
    dsn: str,
    run_id: str,
    limit: int = DEFAULT_SCAN_LIMIT,
    reader: LlmStoreReader | None = None,
    judge_sample: int = DEFAULT_JUDGE_SAMPLE,
    llm_api_key: str | None = None,
    judge_model: str | None = None,
    judge_fn: JudgeFn | None = None,
) -> Scorecard:
    """Chạy 9 check LLM gate và trả về `Scorecard` subject="llm".

    `reader` cho phép tiêm reader giả trong test; None = đọc PostgreSQL thật
    qua `dsn` (dsn rỗng → mọi check skipped). `judge_fn` tiêm judge giả
    (brief, evidence) -> JudgeVerdict; None = gọi Gemini thật qua
    `evaluation/llm_judge.py` khi có `llm_api_key`. Key không bao giờ nằm
    trong scorecard — chỉ `judge_model` + số call.
    """
    run_id = _validate_run_id(run_id)
    if limit < 1:
        raise ValueError(f"evaluation_llm_limit_invalid:{limit}")
    if judge_sample < 0:
        raise ValueError(f"evaluation_llm_judge_sample_invalid:{judge_sample}")
    resolved_model = judge_model or JUDGE_DEFAULT_MODEL
    card = Scorecard(
        run_id=run_id,
        subject="llm",
        inputs={
            "pg_dsn": "configured" if dsn else None,
            "limit": str(limit),
            "judge_sample": str(judge_sample),
            "judge_model": resolved_model,
        },
    )
    if reader is None and not dsn:
        card.add(_skipped(
            "llm_store_reachable", "no_pg_dsn",
            "Không có --pg-dsn và LEARNGUARD_PG_DSN trống → bỏ qua LLM gate",
        ))
        for check_id in CHECK_IDS[1:]:
            card.add(_skipped(check_id, "no_pg_dsn", "Bỏ qua vì chưa cấu hình DSN"))
        return card
    store = reader if reader is not None else LlmStoreReader(dsn)
    try:
        with store:
            data = store.fetch(limit)
    except Exception as exc:  # noqa: BLE001 — mọi lỗi store → FAIL + phần còn lại skipped
        card.add(GateCheck(
            "llm_store_reachable", CHECK_NAMES["llm_store_reachable"],
            "fail", {"limit": limit}, None,
            f"Không đọc được store PostgreSQL: {exc}",
        ))
        for check_id in CHECK_IDS[1:]:
            card.add(_skipped(check_id, "store_unreachable", "Bỏ qua vì store không đọc được"))
        return card
    card.add(_check_store_reachable(data))
    card.add(_check_call_success_rate(data))
    card.add(_check_latency_budget(data))
    card.add(_check_cost_budget(data))
    card.add(_check_format_compliance(data))
    card.add(_check_fallback_rate(data))
    card.add(_check_grounding(data))
    card.add(_check_advisor_feedback(data))
    card.add(_check_judge_quality(
        data,
        api_key=resolve_llm_api_key(llm_api_key),
        judge_model=resolved_model,
        judge_sample=judge_sample,
        judge_fn=judge_fn,
    ))
    return card


# ---- 1. llm_store_reachable --------------------------------------------------

def _check_store_reachable(data: LlmStoreData) -> GateCheck:
    observed: dict[str, object] = {
        "tables": list(LlmStoreReader.REQUIRED_TABLES),
        "llm_calls_rows": len(data.llm_calls),
        "briefs_rows": len(data.briefs),
        "decisions_rows": len(data.decisions),
        "prediction_keys": len(data.top_signals),
    }
    return GateCheck(
        "llm_store_reachable", CHECK_NAMES["llm_store_reachable"],
        "pass", observed, "kết nối được store + đủ bảng bắt buộc",
        f"Store đọc được: {len(data.llm_calls)} llm_calls, "
        f"{len(data.briefs)} briefs, {len(data.decisions)} decisions, "
        f"{len(data.top_signals)} prediction keys",
    )


# ---- 2. call_success_rate ----------------------------------------------------

def _check_call_success_rate(data: LlmStoreData) -> GateCheck:
    if not data.llm_calls:
        return _skipped(
            "call_success_rate", "no_llm_calls",
            "llm_calls rỗng — chưa có call nào để đo",
        )
    status_counts = Counter(call.status for call in data.llm_calls)
    total = len(data.llm_calls)
    ok_count = status_counts.get("ok", 0)
    ok_rate = ok_count / total
    observed: dict[str, object] = {
        "total_calls": total,
        "ok_calls": ok_count,
        "api_error_calls": status_counts.get("api_error", 0),
        "validation_failed_calls": status_counts.get("validation_failed", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "distinct_models": sorted({call.model for call in data.llm_calls}),
        "ok_rate": round(ok_rate, 4),
    }
    if ok_rate < THRESHOLDS.success_rate_fail:
        return GateCheck(
            "call_success_rate", CHECK_NAMES["call_success_rate"],
            "fail", observed, f"ok-rate ≥ {THRESHOLDS.success_rate_fail}",
            f"ok-rate={ok_rate:.3f} < {THRESHOLDS.success_rate_fail} "
            f"({ok_count}/{total} call ok)",
        )
    if ok_rate < THRESHOLDS.success_rate_warn:
        return GateCheck(
            "call_success_rate", CHECK_NAMES["call_success_rate"],
            "warn", observed, f"ok-rate ≥ {THRESHOLDS.success_rate_warn}",
            f"ok-rate={ok_rate:.3f} < {THRESHOLDS.success_rate_warn} "
            f"({ok_count}/{total} call ok)",
        )
    return GateCheck(
        "call_success_rate", CHECK_NAMES["call_success_rate"],
        "pass", observed, f"ok-rate ≥ {THRESHOLDS.success_rate_warn}",
        f"ok-rate={ok_rate:.3f} trên {total} call "
        f"({len(observed['distinct_models'])} model)",
    )


# ---- 3. latency_budget -------------------------------------------------------

def _check_latency_budget(data: LlmStoreData) -> GateCheck:
    latencies = np.asarray(
        [call.latency_ms for call in data.llm_calls if call.status == "ok"],
        dtype=np.float64,
    )
    if latencies.size == 0:
        return _skipped(
            "latency_budget", "no_ok_calls",
            "Không có call ok nào — không đo được latency",
        )
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    max_latency = float(latencies.max())
    observed: dict[str, object] = {
        "ok_calls": int(latencies.size),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(max_latency, 1),
    }
    if p95 > THRESHOLDS.latency_fail_ms:
        return GateCheck(
            "latency_budget", CHECK_NAMES["latency_budget"],
            "fail", observed, f"p95 ≤ {THRESHOLDS.latency_fail_ms:.0f}ms",
            f"p95 latency={p95:.0f}ms > {THRESHOLDS.latency_fail_ms:.0f}ms "
            "— LLM quá chậm cho quy trình advisor",
        )
    if p95 > THRESHOLDS.latency_warn_ms:
        return GateCheck(
            "latency_budget", CHECK_NAMES["latency_budget"],
            "warn", observed, f"p95 ≤ {THRESHOLDS.latency_warn_ms:.0f}ms",
            f"p95 latency={p95:.0f}ms > {THRESHOLDS.latency_warn_ms:.0f}ms",
        )
    return GateCheck(
        "latency_budget", CHECK_NAMES["latency_budget"],
        "pass", observed, f"p95 ≤ {THRESHOLDS.latency_warn_ms:.0f}ms",
        f"p50={p50:.0f}ms, p95={p95:.0f}ms, max={max_latency:.0f}ms",
    )


# ---- 4. cost_budget ----------------------------------------------------------

def _check_cost_budget(data: LlmStoreData) -> GateCheck:
    costs = [call.cost_usd for call in data.llm_calls if call.cost_usd is not None]
    if not costs:
        return _skipped(
            "cost_budget", "no_cost_data",
            "Không có dữ liệu cost_usd (cột thiếu hoặc toàn NULL) — bỏ qua",
        )
    total_cost = sum(costs)
    avg_cost = total_cost / len(costs)
    max_cost = max(costs)
    observed: dict[str, object] = {
        "calls_with_cost": len(costs),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(avg_cost, 6),
        "max_cost_usd": round(max_cost, 6),
    }
    if avg_cost > THRESHOLDS.cost_warn_usd:
        return GateCheck(
            "cost_budget", CHECK_NAMES["cost_budget"],
            "warn", observed, f"avg cost ≤ ${THRESHOLDS.cost_warn_usd}/call",
            f"avg cost/call=${avg_cost:.6f} > ${THRESHOLDS.cost_warn_usd} "
            f"(tổng ${total_cost:.6f} trên {len(costs)} call)",
        )
    return GateCheck(
        "cost_budget", CHECK_NAMES["cost_budget"],
        "pass", observed, f"avg cost ≤ ${THRESHOLDS.cost_warn_usd}/call",
        f"avg cost/call=${avg_cost:.6f}, tổng ${total_cost:.6f}, "
        f"max ${max_cost:.6f}",
    )


# ---- 5. format_compliance ----------------------------------------------------

def _check_format_compliance(data: LlmStoreData) -> GateCheck:
    if not data.briefs:
        return _skipped(
            "format_compliance", "no_briefs",
            "briefs rỗng — chưa có brief nào để quét format",
        )
    malformed = [
        brief.brief_id for brief in data.briefs
        if not _brief_format_ok(brief.content)
    ]
    total = len(data.briefs)
    malformed_rate = len(malformed) / total
    version_counts = Counter(str(brief.version) for brief in data.briefs)
    observed: dict[str, object] = {
        "total_briefs": total,
        "malformed_count": len(malformed),
        "malformed_brief_ids": malformed[:20],
        "malformed_rate": round(malformed_rate, 4),
        "version_distribution": dict(sorted(version_counts.items())),
        "required_fields": list(REQUIRED_CONTENT_FIELDS),
    }
    if malformed_rate > THRESHOLDS.format_fail:
        return GateCheck(
            "format_compliance", CHECK_NAMES["format_compliance"],
            "fail", observed, f"tỉ lệ lệch chuẩn ≤ {THRESHOLDS.format_fail}",
            f"{len(malformed)}/{total} brief thiếu/trường sai kiểu "
            f"(> {THRESHOLDS.format_fail:.0%}) — LLM không tuân thủ hợp đồng nội dung",
        )
    if malformed_rate > THRESHOLDS.format_warn:
        return GateCheck(
            "format_compliance", CHECK_NAMES["format_compliance"],
            "warn", observed, f"tỉ lệ lệch chuẩn ≤ {THRESHOLDS.format_warn}",
            f"{len(malformed)}/{total} brief lệch chuẩn "
            f"(> {THRESHOLDS.format_warn:.0%}): brief_id {malformed[:10]}",
        )
    return GateCheck(
        "format_compliance", CHECK_NAMES["format_compliance"],
        "pass", observed, f"tỉ lệ lệch chuẩn ≤ {THRESHOLDS.format_warn}",
        f"Cả {total} brief đủ {len(REQUIRED_CONTENT_FIELDS)} trường bắt buộc",
    )


def _brief_format_ok(content: dict[str, object]) -> bool:
    """Brief hợp lệ: summary string không rỗng + 3 list đúng kiểu."""
    if not isinstance(content, dict):
        return False
    summary = content.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False
    for field in ("reasons", "support_suggestions", "limitations"):
        if not isinstance(content.get(field), list):
            return False
    return True


# ---- 6. fallback_rate --------------------------------------------------------

def _check_fallback_rate(data: LlmStoreData) -> GateCheck:
    if not data.briefs:
        return _skipped(
            "fallback_rate", "no_briefs",
            "briefs rỗng — chưa có brief nào để đo fallback",
        )
    status_counts = Counter(
        str(brief.content.get("generation_status", "missing"))
        for brief in data.briefs
    )
    total = len(data.briefs)
    generated = status_counts.get(GENERATED_STATUS, 0)
    fallback = total - generated
    fallback_rate = fallback / total
    observed: dict[str, object] = {
        "total_briefs": total,
        "generated": generated,
        "fallback": fallback,
        "generation_status_counts": dict(sorted(status_counts.items())),
        "fallback_rate": round(fallback_rate, 4),
    }
    if fallback_rate > THRESHOLDS.fallback_fail:
        return GateCheck(
            "fallback_rate", CHECK_NAMES["fallback_rate"],
            "fail", observed, f"fallback-rate ≤ {THRESHOLDS.fallback_fail}",
            f"{fallback}/{total} brief không sinh bởi LLM "
            f"(> {THRESHOLDS.fallback_fail:.0%}) — pipeline sinh brief suy sụp",
        )
    if fallback_rate > THRESHOLDS.fallback_warn:
        return GateCheck(
            "fallback_rate", CHECK_NAMES["fallback_rate"],
            "warn", observed, f"fallback-rate ≤ {THRESHOLDS.fallback_warn}",
            f"{fallback}/{total} brief fallback "
            f"(> {THRESHOLDS.fallback_warn:.0%}): "
            f"{dict(sorted(status_counts.items()))}",
        )
    return GateCheck(
        "fallback_rate", CHECK_NAMES["fallback_rate"],
        "pass", observed, f"fallback-rate ≤ {THRESHOLDS.fallback_warn}",
        f"{generated}/{total} brief sinh trực tiếp bởi LLM",
    )


# ---- 7. grounding ------------------------------------------------------------

def _check_grounding(data: LlmStoreData) -> GateCheck:
    if not data.briefs:
        return _skipped(
            "grounding", "no_briefs",
            "briefs rỗng — chưa có brief nào để cross-check citation",
        )
    total = len(data.briefs)
    orphan_brief_ids: list[int] = []
    violations: list[dict[str, object]] = []
    matched = 0
    for brief in data.briefs:
        key = (brief.code_module, brief.code_presentation, brief.student_key)
        signals = data.top_signals.get(key)
        if signals is None:
            orphan_brief_ids.append(brief.brief_id)
            continue
        matched += 1
        cited = {
            str(item.get("feature_name"))
            for item in brief.citations
            if isinstance(item, dict) and item.get("feature_name")
        }
        unknown = sorted(cited - set(signals))
        if unknown:
            violations.append({
                "brief_id": brief.brief_id,
                "unknown_features": unknown,
            })
    violation_rate = len(violations) / total
    observed: dict[str, object] = {
        "total_briefs": total,
        "matched_predictions": matched,
        "orphan_count": len(orphan_brief_ids),
        "orphan_brief_ids": orphan_brief_ids[:20],
        "violating_briefs": len(violations),
        "violation_rate": round(violation_rate, 4),
        "violation_examples": violations[:5],
    }
    if violation_rate > THRESHOLDS.grounding_fail:
        return GateCheck(
            "grounding", CHECK_NAMES["grounding"],
            "fail", observed, f"tỉ lệ brief vi phạm ≤ {THRESHOLDS.grounding_fail}",
            f"{len(violations)}/{total} brief cite feature ngoài top_signals "
            f"(> {THRESHOLDS.grounding_fail:.0%}) — hallucination grounding",
        )
    if violations:
        return GateCheck(
            "grounding", CHECK_NAMES["grounding"],
            "warn", observed, f"tỉ lệ brief vi phạm ≤ {THRESHOLDS.grounding_fail}",
            f"{len(violations)}/{total} brief cite feature lạ "
            f"(orphan: {len(orphan_brief_ids)} brief không tìm thấy prediction)",
        )
    detail = (
        f"Mọi brief ({matched}) chỉ cite feature trong top_signals"
        if orphan_brief_ids == []
        else f"Mọi brief khớp prediction ({matched}) chỉ cite feature trong "
        f"top_signals; {len(orphan_brief_ids)} orphan brief không có prediction"
    )
    return GateCheck(
        "grounding", CHECK_NAMES["grounding"],
        "pass", observed, f"tỉ lệ brief vi phạm ≤ {THRESHOLDS.grounding_fail}",
        detail,
    )


# ---- 8. advisor_feedback -----------------------------------------------------

def _check_advisor_feedback(data: LlmStoreData) -> GateCheck:
    if not data.decisions:
        return _skipped(
            "advisor_feedback", "no_decisions",
            "decisions rỗng — advisor chưa quyết định brief nào",
        )
    action_counts = Counter(decision.action for decision in data.decisions)
    accept = action_counts.get("accept", 0)
    reject = action_counts.get("reject", 0)
    decided_ids = {decision.brief_id for decision in data.decisions}
    undecided = [brief.brief_id for brief in data.briefs if brief.brief_id not in decided_ids]
    edited = [
        brief.brief_id for brief in data.briefs
        if brief.content.get("edited_at") is not None
    ]
    decided_total = accept + reject
    reject_rate = reject / decided_total if decided_total else 0.0
    observed: dict[str, object] = {
        "accept": accept,
        "reject": reject,
        "other_actions": {
            action: count for action, count in sorted(action_counts.items())
            if action not in {"accept", "reject"}
        },
        "undecided_briefs": len(undecided),
        "edited_briefs": len(edited),
        "reject_rate": round(reject_rate, 4),
    }
    if reject_rate > THRESHOLDS.reject_rate_warn:
        return GateCheck(
            "advisor_feedback", CHECK_NAMES["advisor_feedback"],
            "warn", observed, f"reject-rate ≤ {THRESHOLDS.reject_rate_warn}",
            f"reject-rate={reject_rate:.3f} > {THRESHOLDS.reject_rate_warn} "
            f"({reject}/{decided_total}) — chất lượng brief kém, advisor từ chối nhiều",
        )
    return GateCheck(
        "advisor_feedback", CHECK_NAMES["advisor_feedback"],
        "pass", observed, f"reject-rate ≤ {THRESHOLDS.reject_rate_warn} (quan sát)",
        f"accept={accept}, reject={reject}, undecided={len(undecided)}, "
        f"edited={len(edited)}",
    )


# ---- 9. judge_quality (LLM-as-judge) ------------------------------------------

JUDGE_THRESHOLD_TEXT = (
    "avg_overall ≥ 4.0 pass; 3.0–3.9 warn; < 3.0 fail; "
    "no_fabrication ≤ 2 trên ≥ 30% brief → fail trực tiếp"
)


def _brief_recency_key(brief: BriefRow) -> datetime:
    """Khóa sắp xếp brief theo created_at mới nhất (None → epoch cũ nhất)."""
    timestamp = brief.created_at
    if timestamp is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _check_judge_quality(
    data: LlmStoreData,
    api_key: str,
    judge_model: str,
    judge_sample: int,
    judge_fn: JudgeFn | None,
) -> GateCheck:
    """Chấm tối đa `judge_sample` brief mới nhất bằng Gemini theo rubric 4 tiêu chí.

    SKIPPED khi --judge-sample 0 / không API key / không brief. Call lỗi
    (network/parse) → brief bỏ qua; > 50% call lỗi → WARN (lỗi hạ tầng, không
    FAIL). Bịa evidence (no_fabrication ≤ 2) ≥ 30% brief chấm được → FAIL.
    """
    if judge_sample == 0:
        return _skipped(
            "judge_quality", "judge_sample_zero",
            "--judge-sample 0 → tắt LLM-as-judge",
        )
    if not api_key:
        return _skipped(
            "judge_quality", "no_api_key",
            "Không có API key (--llm-api-key / env LLM_API_KEY / GEMINI_API_KEY) "
            "→ bỏ qua judge_quality",
        )
    if not data.briefs:
        return _skipped(
            "judge_quality", "no_briefs",
            "briefs rỗng — không có brief nào để judge chấm",
        )
    sampled = sorted(
        data.briefs, key=_brief_recency_key, reverse=True,
    )[:judge_sample]
    if judge_fn is None:
        def judge_fn(brief: BriefRow, evidence: dict[str, Any]) -> JudgeVerdict:
            return judge_brief(
                brief, evidence, api_key=api_key, model=judge_model,
            )

    verdicts: list[JudgeVerdict] = []
    errors: list[dict[str, str]] = []
    started = time.perf_counter()
    for brief in sampled:
        key = (brief.code_module, brief.code_presentation, brief.student_key)
        evidence = {
            "citations": brief.citations,
            "top_signals": data.top_signal_details.get(key, ()),
        }
        try:
            verdicts.append(judge_fn(brief, evidence))
        except Exception as exc:  # noqa: BLE001 — lỗi call không được crash gate
            errors.append({
                "brief_id": str(brief.brief_id),
                "error": f"{type(exc).__name__}: {exc}",
            })
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _judge_quality_gate_check(
        sampled, verdicts, errors, elapsed_ms, judge_model, judge_sample,
    )


def _judge_quality_gate_check(
    sampled: list[BriefRow],
    verdicts: list[JudgeVerdict],
    errors: list[dict[str, str]],
    elapsed_ms: float,
    judge_model: str,
    judge_sample: int,
) -> GateCheck:
    observed: dict[str, object] = {
        "judge_model": judge_model,
        "judge_sample": judge_sample,
        "n_sampled": len(sampled),
        "n_judged": len(verdicts),
        "n_errors": len(errors),
        "judge_elapsed_ms": round(elapsed_ms, 1),
        "avg_overall": (
            round(float(np.mean([v.overall for v in verdicts])), 2)
            if verdicts else None
        ),
        "fabrication_briefs": sum(
            1 for verdict in verdicts if verdict.no_fabrication <= 2
        ),
        "per_brief": [
            {
                "brief_id": verdict.brief_id,
                "overall": verdict.overall,
                "grounded": verdict.grounded,
                "useful": verdict.useful,
                "no_fabrication": verdict.no_fabrication,
                "clarity": verdict.clarity,
                "issues": verdict.issues,
            }
            for verdict in verdicts
        ],
        "errors": errors[:10],
    }
    if not verdicts:
        first_error = errors[0]["error"] if errors else "unknown"
        return GateCheck(
            "judge_quality", CHECK_NAMES["judge_quality"],
            "warn", observed, JUDGE_THRESHOLD_TEXT,
            f"Toàn bộ {len(sampled)} judge call lỗi — không chấm được brief nào "
            f"(lỗi hạ tầng judge, không FAIL). Lỗi đầu tiên: {first_error}",
        )
    fabrication = sum(1 for verdict in verdicts if verdict.no_fabrication <= 2)
    judged = len(verdicts)
    avg_overall = float(np.mean([verdict.overall for verdict in verdicts]))
    if fabrication / judged >= THRESHOLDS.judge_fabrication_ratio:
        return GateCheck(
            "judge_quality", CHECK_NAMES["judge_quality"],
            "fail", observed, JUDGE_THRESHOLD_TEXT,
            f"{fabrication}/{judged} brief bị judge phát hiện bịa evidence "
            f"(no_fabrication ≤ 2, ≥ {THRESHOLDS.judge_fabrication_ratio:.0%}) "
            f"— bịa là nghiêm trọng nhất, FAIL trực tiếp (avg_overall={avg_overall:.2f})",
        )
    if avg_overall < THRESHOLDS.judge_warn:
        details = (
            f"avg_overall={avg_overall:.2f} < {THRESHOLDS.judge_warn} "
            f"trên {judged} brief — chất lượng nội dung brief kém"
        )
    elif avg_overall < THRESHOLDS.judge_pass:
        details = (
            f"avg_overall={avg_overall:.2f} < {THRESHOLDS.judge_pass} "
            f"trên {judged} brief — chất lượng trung bình"
        )
    else:
        details = (
            f"avg_overall={avg_overall:.2f} trên {judged} brief — "
            f"chất lượng nội dung đạt rubric"
        )
    if errors and len(errors) / len(sampled) > THRESHOLDS.judge_error_ratio:
        return GateCheck(
            "judge_quality", CHECK_NAMES["judge_quality"],
            "warn", observed, JUDGE_THRESHOLD_TEXT,
            f"{len(errors)}/{len(sampled)} judge call lỗi "
            f"(> {THRESHOLDS.judge_error_ratio:.0%}) — hạ tầng judge bất ổn, "
            f"hạ mức WARN. {details}",
        )
    status = (
        "fail" if avg_overall < THRESHOLDS.judge_warn
        else "warn" if avg_overall < THRESHOLDS.judge_pass
        else "pass"
    )
    if errors:
        details += f"; {len(errors)} call lỗi (bỏ qua)"
    return GateCheck(
        "judge_quality", CHECK_NAMES["judge_quality"],
        status, observed, JUDGE_THRESHOLD_TEXT, details,
    )


# ---- helpers -----------------------------------------------------------------

def _skipped(check_id: str, reason: str, details: str) -> GateCheck:
    """Check bị bỏ qua (không đủ dữ liệu / chưa cấu hình DSN)."""
    return GateCheck(
        check_id, CHECK_NAMES[check_id], "skipped", {"reason": reason}, None,
        details,
    )
