"""Gate 3 — LLM evaluation gate tests (mock reader + DB thật read-only).

Unit tests dùng FakeLlmStoreReader trả dữ liệu chế sẵn (không đụng DB).
Integration test đọc PostgreSQL thật đang chạy, read-only, skip khi PG
unavailable — giống pattern tests/test_predictions_store.py nhưng không
tạo/bảng nào (module chỉ SELECT).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from learnguard.cli import main
from learnguard.evaluation import llm_gate as llm_gate_module
from learnguard.evaluation.db_reader import (
    BriefRow,
    DecisionRow,
    LlmCallRow,
    LlmStoreData,
)
from learnguard.evaluation.llm_gate import (
    CHECK_IDS,
    LlmStoreReader,
    resolve_llm_api_key,
    run_llm_gate,
)
from learnguard.evaluation.llm_judge import (
    JUDGE_DEFAULT_MODEL,
    JudgeCallError,
    JudgeVerdict,
    judge_brief,
)

PG_DSN = "postgresql://learnguard:learnguard@localhost:5432/learnguard"

STUDENT_KEY = "f5a05a3d6bd27c1e"
SIGNAL_FEATURES = ("clicks__n_clicks_before_cutoff", "numeric__studied_credits")


# ---- fake reader -------------------------------------------------------------

class FakeLlmStoreReader:
    """Reader giả — trả LlmStoreData chế sẵn, không đụng mạng/DB."""

    def __init__(self, data: LlmStoreData) -> None:
        self._data = data

    def fetch(self, limit: int) -> LlmStoreData:
        if limit < 1:
            raise ValueError(f"llm_store_limit_invalid:{limit}")
        return self._data

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakeLlmStoreReader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _call(
    call_id: int = 1,
    status: str = "ok",
    latency_ms: int = 2000,
    cost_usd: float | None = 0.0008,
    model: str = "gemini-test",
) -> LlmCallRow:
    return LlmCallRow(
        call_id=call_id, brief_id=None, model=model, status=status,
        latency_ms=latency_ms, cost_usd=cost_usd,
    )


def _brief(
    brief_id: int = 1,
    content: dict[str, Any] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> BriefRow:
    if content is None:
        content = {
            "summary": "Sinh viên rủi ro cao.",
            "reasons": [{"text": "Ít tương tác", "evidence_features": ["x"]}],
            "support_suggestions": ["Gọi điện"],
            "limitations": ["Dữ liệu thiếu"],
            "generation_status": "generated",
        }
    return BriefRow(
        brief_id=brief_id, code_module="DDD", code_presentation="2014B",
        student_key=STUDENT_KEY, version=1,
        content=content, citations=citations if citations is not None else [],
    )


def _data(
    calls: tuple[LlmCallRow, ...] = (_call(),),
    briefs: tuple[BriefRow, ...] = (_brief(),),
    decisions: tuple[DecisionRow, ...] = (DecisionRow(1, "accept"),),
    top_signals: dict[tuple[str, str, str], tuple[str, ...]] | None = None,
) -> LlmStoreData:
    return LlmStoreData(
        llm_calls=calls, briefs=briefs, decisions=decisions,
        top_signals=(
            top_signals if top_signals is not None
            else {("DDD", "2014B", STUDENT_KEY): SIGNAL_FEATURES}
        ),
    )


def _check(card: Any, check_id: str) -> Any:
    return next(check for check in card.checks if check.id == check_id)


def _statuses_by_id(card: Any) -> dict[str, str]:
    return {check.id: check.status for check in card.checks}


# ---- run_llm_gate: cấu trúc + skipping ---------------------------------------

def test_gate_emits_all_nine_checks_in_order():
    card = run_llm_gate("postgresql://x", "eval-llm-test", reader=FakeLlmStoreReader(_data()))
    assert card.subject == "llm"
    assert tuple(check.id for check in card.checks) == CHECK_IDS
    assert len(CHECK_IDS) == 9
    # 8 check phân tích dữ liệu: pass. judge_quality không key → skipped.
    assert all(check.status == "pass" for check in card.checks[:8])
    judge = _check(card, "judge_quality")
    assert judge.status == "skipped"
    assert judge.observed["reason"] == "no_api_key"


def test_no_dsn_skips_every_check():
    card = run_llm_gate("", "eval-llm-test")
    statuses = _statuses_by_id(card)
    assert statuses == {check_id: "skipped" for check_id in CHECK_IDS}
    assert _check(card, "llm_store_reachable").observed["reason"] == "no_pg_dsn"


def test_unreachable_store_fails_head_and_skips_rest():
    class BrokenReader:
        def fetch(self, limit: int) -> LlmStoreData:
            raise ValueError("llm_store_read_failed:OperationalError")

        def close(self) -> None:
            pass

        def __enter__(self) -> "BrokenReader":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            self.close()

    card = run_llm_gate("postgresql://x", "eval-llm-test", reader=BrokenReader())
    statuses = _statuses_by_id(card)
    assert statuses["llm_store_reachable"] == "fail"
    assert all(statuses[cid] == "skipped" for cid in CHECK_IDS[1:])
    assert card.has_failure()


def test_invalid_limit_rejected():
    with pytest.raises(ValueError, match="evaluation_llm_limit_invalid"):
        run_llm_gate("postgresql://x", "eval-llm-test", limit=0)


# ---- happy path ---------------------------------------------------------------

def test_happy_path_all_pass_no_failure():
    card = run_llm_gate("postgresql://x", "eval-llm-test", reader=FakeLlmStoreReader(_data()))
    assert not card.has_failure()
    assert card.worst_status() == "pass"
    success = _check(card, "call_success_rate")
    assert success.observed["ok_rate"] == 1.0
    latency = _check(card, "latency_budget")
    assert latency.observed["p95_ms"] == 2000.0
    cost = _check(card, "cost_budget")
    assert cost.observed["avg_cost_usd"] == 0.0008


# ---- call_success_rate ngưỡng -------------------------------------------------

def test_call_success_rate_warn_and_fail_thresholds():
    # 80% ok → warn (< 90%); 60% ok → fail (< 70%)
    warn_calls = tuple(
        _call(call_id=i, status="ok" if i % 5 != 0 else "api_error")
        for i in range(1, 11)  # 8/10 ok = 80%
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(calls=warn_calls)))
    assert _check(card, "call_success_rate").status == "warn"

    fail_calls = tuple(
        _call(call_id=i, status="ok" if i % 5 != 0 and i % 5 != 1 else "api_error")
        for i in range(1, 11)  # 6/10 ok = 60%
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(calls=fail_calls)))
    assert _check(card, "call_success_rate").status == "fail"
    assert card.has_failure()


def test_call_success_rate_skipped_when_no_calls():
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(calls=())))
    assert _check(card, "call_success_rate").status == "skipped"
    assert _check(card, "call_success_rate").observed["reason"] == "no_llm_calls"


# ---- latency / cost budgets ---------------------------------------------------

def test_latency_budget_warn_and_fail():
    warn_calls = tuple(_call(call_id=i, latency_ms=5100) for i in range(1, 5))
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(calls=warn_calls)))
    assert _check(card, "latency_budget").status == "warn"

    fail_calls = tuple(_call(call_id=i, latency_ms=16000) for i in range(1, 5))
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(calls=fail_calls)))
    assert _check(card, "latency_budget").status == "fail"


def test_cost_budget_skipped_without_cost_and_warn_over_budget():
    no_cost = (_call(cost_usd=None), _call(call_id=2, cost_usd=None))
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(calls=no_cost)))
    cost = _check(card, "cost_budget")
    assert cost.status == "skipped"
    assert cost.observed["reason"] == "no_cost_data"

    expensive = (_call(cost_usd=0.02),)
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(calls=expensive)))
    assert _check(card, "cost_budget").status == "warn"


# ---- format_compliance ---------------------------------------------------------

def _malformed_content() -> dict[str, Any]:
    return {
        "summary": "",  # rỗng → malformed
        "reasons": "không phải list",
        "support_suggestions": [],
        "limitations": [],
        "generation_status": "generated",
    }


def test_format_compliance_warn_over_5_percent_fail_over_20_percent():
    # 2/10 malformed = 20% → warn (không vượt 20%); 3/10 = 30% → fail
    briefs_warn = tuple(
        _brief(brief_id=i, content=_malformed_content() if i <= 2 else None)
        for i in range(1, 11)
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(briefs=briefs_warn)))
    check = _check(card, "format_compliance")
    assert check.status == "warn"
    assert check.observed["malformed_count"] == 2

    briefs_fail = tuple(
        _brief(brief_id=i, content=_malformed_content() if i <= 3 else None)
        for i in range(1, 11)
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(briefs=briefs_fail)))
    assert _check(card, "format_compliance").status == "fail"
    assert card.has_failure()


# ---- fallback_rate --------------------------------------------------------------

def test_fallback_rate_warn_and_fail():
    def _fallback_content(status: str) -> dict[str, Any]:
        content = _brief().content
        return {**content, "generation_status": status}

    warn_briefs = (
        _brief(brief_id=1, content=_fallback_content("fallback_review")),
        _brief(brief_id=2),
        _brief(brief_id=3),
        _brief(brief_id=4),
        _brief(brief_id=5),
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(briefs=warn_briefs)))
    check = _check(card, "fallback_rate")
    assert check.status == "warn"
    assert check.observed["generation_status_counts"] == {"fallback_review": 1, "generated": 4}

    fail_briefs = tuple(
        _brief(brief_id=i, content=_fallback_content("unavailable")) for i in range(1, 5)
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(briefs=fail_briefs)))
    assert _check(card, "fallback_rate").status == "fail"


# ---- grounding -------------------------------------------------------------------

def test_grounding_unknown_feature_warns_and_over_20_percent_fails():
    # 1/5 brief cite feature lạ → warn
    briefs = tuple(
        _brief(
            brief_id=i,
            citations=[{"feature_name": "made_up_feature", "contribution": 1.0}]
            if i == 1 else
            [{"feature_name": SIGNAL_FEATURES[0], "contribution": 1.0}],
        )
        for i in range(1, 6)
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(briefs=briefs)))
    check = _check(card, "grounding")
    assert check.status == "warn"
    assert check.observed["violating_briefs"] == 1
    assert check.observed["violation_examples"][0]["unknown_features"] == ["made_up_feature"]

    # 2/5 = 40% > 20% → fail
    briefs = tuple(
        _brief(
            brief_id=i,
            citations=[{"feature_name": "made_up_feature", "contribution": 1.0}]
            if i <= 2 else [],
        )
        for i in range(1, 6)
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(briefs=briefs)))
    assert _check(card, "grounding").status == "fail"


def test_grounding_orphan_briefs_counted_separately_not_violations():
    orphan = BriefRow(
        brief_id=1, code_module="XXX", code_presentation="2099J",
        student_key="orphan0000000000", version=1,
        content=_brief().content, citations=[{"feature_name": "any_feature"}],
    )
    matched = _brief(brief_id=2, citations=[{"feature_name": SIGNAL_FEATURES[0]}])
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(
        briefs=(orphan, matched), top_signals={("DDD", "2014B", STUDENT_KEY): SIGNAL_FEATURES},
    )))
    check = _check(card, "grounding")
    assert check.status == "pass"
    assert check.observed["orphan_count"] == 1
    assert check.observed["matched_predictions"] == 1


# ---- advisor_feedback --------------------------------------------------------------

def test_advisor_feedback_reject_rate_warn_and_undecided_edited_counts():
    decisions = (DecisionRow(1, "reject"), DecisionRow(2, "reject"), DecisionRow(3, "accept"))
    briefs = (
        _brief(brief_id=1, content={**_brief().content, "edited_at": "2026-11-05T00:00:00+00:00"}),
        _brief(brief_id=2),
        _brief(brief_id=3),
        _brief(brief_id=4),  # undecided
    )
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(briefs=briefs, decisions=decisions)))
    check = _check(card, "advisor_feedback")
    assert check.status == "warn"
    assert check.observed["reject_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert check.observed["undecided_briefs"] == 1
    assert check.observed["edited_briefs"] == 1


def test_advisor_feedback_skipped_without_decisions():
    card = run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data(decisions=())))
    check = _check(card, "advisor_feedback")
    assert check.status == "skipped"
    assert check.observed["reason"] == "no_decisions"


# ---- judge_quality (LLM-as-judge) ---------------------------------------------------

def _verdict(
    brief_id: int,
    grounded: int,
    useful: int,
    no_fabrication: int,
    clarity: int,
    issues: str = "",
) -> JudgeVerdict:
    return JudgeVerdict(
        brief_id=brief_id,
        grounded=grounded,
        useful=useful,
        no_fabrication=no_fabrication,
        clarity=clarity,
        overall=round((grounded + useful + no_fabrication + clarity) / 4, 1),
        issues=issues,
    )


def _judge_map(
    mapping: dict[int, JudgeVerdict],
    error_ids: set[int] | None = None,
    error: Exception | None = None,
) -> Any:
    """Judge giả: trả verdict theo brief_id; brief trong error_ids → raise."""
    raised = error if error is not None else JudgeCallError("judge_http_error:URLError")

    def judge_fn(brief: Any, evidence: dict[str, Any]) -> JudgeVerdict:
        if brief.brief_id in (error_ids or set()):
            raise raised
        return mapping[brief.brief_id]

    return judge_fn


def _three_briefs() -> tuple[BriefRow, BriefRow, BriefRow]:
    return (_brief(brief_id=1), _brief(brief_id=2), _brief(brief_id=3))


def test_judge_quality_happy_three_briefs_pass():
    mapping = {
        1: _verdict(1, grounded=5, useful=4, no_fabrication=5, clarity=4),
        2: _verdict(2, grounded=5, useful=5, no_fabrication=5, clarity=5),
        3: _verdict(3, grounded=4, useful=4, no_fabrication=5, clarity=5, issues="hơi dài"),
    }
    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=_three_briefs())),
        llm_api_key="test-key", judge_model="judge-model-x",
        judge_fn=_judge_map(mapping),
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "pass"
    assert judge.observed["n_judged"] == 3
    assert judge.observed["n_errors"] == 0
    assert judge.observed["avg_overall"] == pytest.approx(4.67, abs=0.01)
    assert judge.observed["judge_model"] == "judge-model-x"
    assert judge.observed["fabrication_briefs"] == 0
    per_brief = judge.observed["per_brief"]
    assert len(per_brief) == 3
    assert per_brief[0] == {
        "brief_id": 1, "overall": 4.5, "grounded": 5, "useful": 4,
        "no_fabrication": 5, "clarity": 4, "issues": "",
    }
    # Key KHÔNG BAO GIỜ nằm trong scorecard (observed lẫn inputs).
    assert "test-key" not in json.dumps(card.to_dict(), ensure_ascii=False)


def test_judge_quality_warn_avg_3_2_and_fail_avg_2_5():
    warn_mapping = {
        i: _verdict(i, grounded=3, useful=3, no_fabrication=3, clarity=4)
        for i in (1, 2, 3)
    }
    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=_three_briefs())),
        llm_api_key="k", judge_fn=_judge_map(warn_mapping),
    )
    assert _check(card, "judge_quality").status == "warn"
    assert pytest.approx(3.2, abs=0.01) == _check(card, "judge_quality").observed["avg_overall"]

    fail_mapping = {
        i: _verdict(i, grounded=2, useful=3, no_fabrication=2, clarity=3)
        for i in (1, 2, 3)
    }
    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=_three_briefs())),
        llm_api_key="k", judge_fn=_judge_map(fail_mapping),
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "fail"
    assert card.has_failure()


def test_judge_quality_fabrication_fail_direct_even_with_high_overall():
    # 2/3 brief bịa (no_fabrication ≤ 2) = 67% ≥ 30% → FAIL trực tiếp
    # dù avg_overall còn ≥ 4.0.
    mapping = {
        1: _verdict(1, grounded=5, useful=5, no_fabrication=2, clarity=5,
                    issues="bia số liệu không có trong evidence"),
        2: _verdict(2, grounded=5, useful=5, no_fabrication=1, clarity=5),
        3: _verdict(3, grounded=5, useful=5, no_fabrication=5, clarity=5),
    }
    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=_three_briefs())),
        llm_api_key="k", judge_fn=_judge_map(mapping),
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "fail"
    assert judge.observed["fabrication_briefs"] == 2
    assert judge.observed["avg_overall"] >= 4.0
    assert card.has_failure()


def test_judge_quality_skipped_without_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def must_not_be_called(brief: Any, evidence: dict[str, Any]) -> JudgeVerdict:
        raise AssertionError("judge không được gọi khi thiếu API key")

    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data()),
        judge_fn=must_not_be_called,
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "skipped"
    assert judge.observed["reason"] == "no_api_key"


def test_judge_quality_skipped_without_briefs():
    def must_not_be_called(brief: Any, evidence: dict[str, Any]) -> JudgeVerdict:
        raise AssertionError("judge không được gọi khi không có brief")

    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=())),
        llm_api_key="k", judge_fn=must_not_be_called,
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "skipped"
    assert judge.observed["reason"] == "no_briefs"


def test_judge_quality_skipped_when_sample_zero():
    def must_not_be_called(brief: Any, evidence: dict[str, Any]) -> JudgeVerdict:
        raise AssertionError("judge không được gọi khi judge_sample=0")

    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data()),
        llm_api_key="k", judge_sample=0, judge_fn=must_not_be_called,
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "skipped"
    assert judge.observed["reason"] == "judge_sample_zero"


def test_judge_quality_invalid_sample_rejected():
    with pytest.raises(ValueError, match="evaluation_llm_judge_sample_invalid"):
        run_llm_gate("x", "r", reader=FakeLlmStoreReader(_data()), judge_sample=-1)


def test_judge_quality_two_of_three_call_errors_still_warns():
    # 2/3 call lỗi → chỉ chấm được 1 brief; > 50% lỗi → WARN (lỗi hạ tầng).
    mapping = {3: _verdict(3, grounded=5, useful=5, no_fabrication=5, clarity=5)}
    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=_three_briefs())),
        llm_api_key="k", judge_fn=_judge_map(mapping, error_ids={1, 2}),
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "warn"
    assert judge.observed["n_judged"] == 1
    assert judge.observed["n_errors"] == 2
    assert judge.observed["per_brief"][0]["brief_id"] == 3
    assert len(judge.observed["errors"]) == 2


def test_judge_quality_one_of_three_errors_keeps_score_status():
    # 1/3 call lỗi ≤ 50% → vẫn chấm theo điểm; pass với n_errors=1.
    mapping = {
        2: _verdict(2, grounded=5, useful=5, no_fabrication=5, clarity=4),
        3: _verdict(3, grounded=4, useful=5, no_fabrication=5, clarity=4),
    }
    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=_three_briefs())),
        llm_api_key="k", judge_fn=_judge_map(mapping, error_ids={1}),
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "pass"
    assert judge.observed["n_judged"] == 2
    assert judge.observed["n_errors"] == 1


def test_judge_quality_all_calls_error_warns_not_fails():
    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=_three_briefs())),
        llm_api_key="k",
        judge_fn=_judge_map({}, error_ids={1, 2, 3},
                            error=ValueError("judge_json_parse_failed")),
    )
    judge = _check(card, "judge_quality")
    assert judge.status == "warn"
    assert not card.has_failure()
    assert judge.observed["n_judged"] == 0
    assert judge.observed["avg_overall"] is None


def test_judge_quality_samples_newest_briefs_by_created_at():
    base = datetime(2026, 11, 1, tzinfo=timezone.utc)
    briefs = tuple(
        BriefRow(
            brief_id=i, code_module="DDD", code_presentation="2014B",
            student_key=STUDENT_KEY, version=1,
            content=_brief().content, citations=[],
            created_at=datetime(2026, 11, i, tzinfo=timezone.utc) if i > 1 else base,
        )
        for i in range(1, 5)
    )
    called_ids: list[int] = []

    def judge_fn(brief: Any, evidence: dict[str, Any]) -> JudgeVerdict:
        called_ids.append(brief.brief_id)
        return _verdict(brief.brief_id, 5, 5, 5, 5)

    card = run_llm_gate(
        "x", "r", reader=FakeLlmStoreReader(_data(briefs=briefs)),
        llm_api_key="k", judge_sample=2, judge_fn=judge_fn,
    )
    # Mới nhất theo created_at: brief 4 (2026-11-04) và 3 (2026-11-03).
    assert called_ids == [4, 3]
    assert _check(card, "judge_quality").status == "pass"


def test_resolve_llm_api_key_precedence(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert resolve_llm_api_key(None) == ""
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    assert resolve_llm_api_key(None) == "g-key"
    monkeypatch.setenv("LLM_API_KEY", "l-key")
    assert resolve_llm_api_key(None) == "l-key"
    assert resolve_llm_api_key("flag-key") == "flag-key"


# ---- llm_judge: REST call + parse ----------------------------------------------------

class _FakeHttpResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _gemini_response(scores: dict[str, Any]) -> bytes:
    return json.dumps({
        "candidates": [{"content": {"parts": [{"text": json.dumps(scores)}]}}]
    }).encode("utf-8")


def test_judge_brief_calls_gemini_rest_and_parses_verdict(monkeypatch):
    from learnguard.evaluation import llm_judge as judge_module

    captured: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeHttpResponse:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeHttpResponse(_gemini_response({
            "grounded": 5, "useful": 4, "no_fabrication": 5, "clarity": 4,
            "overall": 4.5, "issues": "ổn",
        }))

    monkeypatch.setattr(judge_module.urllib.request, "urlopen", fake_urlopen)
    verdict = judge_brief(
        _brief(brief_id=7),
        {"citations": [{"feature_name": "x", "observed_value": 3}],
         "top_signals": ({"feature_name": "x"},)},
        api_key="fake-key", model="judge-model-y",
    )
    assert verdict == JudgeVerdict(
        brief_id=7, grounded=5, useful=4, no_fabrication=5, clarity=4,
        overall=4.5, issues="ổn",
    )
    assert "generativelanguage.googleapis.com" in captured["url"]
    assert "models/judge-model-y:generateContent" in captured["url"]
    assert captured["timeout"] == 30
    config = captured["body"]["generationConfig"]
    assert config["temperature"] == 0
    assert config["responseMimeType"] == "application/json"
    assert config["maxOutputTokens"] == 1000


def test_judge_brief_broken_json_counts_as_error_not_crash(monkeypatch):
    from learnguard.evaluation import llm_judge as judge_module

    monkeypatch.setattr(
        judge_module.urllib.request, "urlopen",
        lambda request, timeout: _FakeHttpResponse(
            json.dumps({
                "candidates": [{"content": {"parts": [
                    {"text": "không phải json"},
                ]}}]
            }).encode("utf-8")
        ),
    )
    with pytest.raises(JudgeCallError, match="judge_json_parse_failed"):
        judge_brief(_brief(), {"citations": [], "top_signals": ()}, api_key="k")


def test_judge_brief_http_error_wrapped(monkeypatch):
    from learnguard.evaluation import llm_judge as judge_module

    def boom(request: Any, timeout: float) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(judge_module.urllib.request, "urlopen", boom)
    with pytest.raises(JudgeCallError, match="judge_http_error"):
        judge_brief(_brief(), {"citations": [], "top_signals": ()}, api_key="k")


# ---- publish + CLI -----------------------------------------------------------------

def test_publish_llm_scorecard_and_cli_evaluate_llm(tmp_path, monkeypatch):
    from learnguard.evaluation.scorecard import publish_evaluation_run

    card = run_llm_gate("postgresql://x", "eval-llm-test", reader=FakeLlmStoreReader(_data()))
    run_dir = publish_evaluation_run(tmp_path, "eval-llm-test", [card])
    published = json.loads((run_dir / "eval_llm.json").read_text())
    assert published["subject"] == "llm"
    assert published["summary"]["total"] == len(CHECK_IDS)

    monkeypatch.setattr(
        llm_gate_module, "run_llm_gate",
        lambda dsn, run_id, limit=5000, **kwargs: run_llm_gate(
            "postgresql://x", run_id, reader=FakeLlmStoreReader(_data()),
        ),
    )
    exit_code = main([
        "evaluate", "llm", "--artifacts-dir", str(tmp_path / "cli"),
        "--run-id", "eval-llm-cli",
    ])
    assert exit_code == 0
    cli_published = json.loads(
        (tmp_path / "cli" / "evaluations" / "eval-llm-cli" / "eval_llm.json").read_text()
    )
    assert cli_published["run_id"] == "eval-llm-cli"


def test_cli_evaluate_llm_without_dsn_skips(tmp_path, monkeypatch):
    monkeypatch.delenv("LEARNGUARD_PG_DSN", raising=False)
    exit_code = main([
        "evaluate", "llm", "--artifacts-dir", str(tmp_path), "--run-id", "eval-llm-nodsn",
    ])
    assert exit_code == 0
    published = json.loads(
        (tmp_path / "evaluations" / "eval-llm-nodsn" / "eval_llm.json").read_text()
    )
    assert published["summary"]["skipped"] == len(CHECK_IDS)


# ---- DB thật (read-only) ------------------------------------------------------------

def test_real_postgres_read_only_gate():
    reader = LlmStoreReader(PG_DSN)
    try:
        data = reader.fetch(limit=5000)
    except Exception:  # noqa: BLE001
        pytest.skip("postgres unavailable")
    finally:
        reader.close()
    card = run_llm_gate(PG_DSN, "eval-llm-real", limit=5000)
    assert card.subject == "llm"
    assert tuple(check.id for check in card.checks) == CHECK_IDS
    assert _check(card, "llm_store_reachable").status == "pass"
    assert _check(card, "call_success_rate").observed["total_calls"] == len(data.llm_calls)
