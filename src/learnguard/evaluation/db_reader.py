"""Read-only store cho Gate 3 (`learnguard evaluate llm`).

Đọc dữ liệu LLM đã ghi sẵn vào PostgreSQL chia sẻ (llm_calls, briefs,
decisions) + top_signals của ml_predictions để cross-check grounding.
Module này CHỈ SELECT — không CREATE, không INSERT, không UPDATE: gate
đánh giá phải trung tính với dữ liệu nó đo.

Connection tái dùng đúng pattern `ml/predictions_store.py`: SQLAlchemy
engine trên driver psycopg 3, DSN từ `--pg-dsn` / env `LEARNGUARD_PG_DSN`,
không bao giờ echo DSN vào log hay artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, create_engine, text

from learnguard.ml.predictions_store import _sqlalchemy_dsn

BRIEF_KEY_COLUMNS = ("code_module", "code_presentation", "student_key")


@dataclass(frozen=True)
class LlmCallRow:
    """Một dòng `llm_calls` — metadata một lần gọi LLM sinh brief."""

    call_id: int
    brief_id: int | None
    model: str
    status: str
    latency_ms: int
    cost_usd: float | None


@dataclass(frozen=True)
class BriefRow:
    """Một dòng `briefs` — advisor brief Gemini sinh (content JSONB)."""

    brief_id: int
    code_module: str
    code_presentation: str
    student_key: str
    version: int
    content: dict[str, Any]
    citations: list[dict[str, Any]]
    # created_at phục vụ judge_quality sampling "M mới NHẤT" (additive, có thể None
    # khi dataclass được dựng tay trong test).
    created_at: datetime | None = None


@dataclass(frozen=True)
class DecisionRow:
    """Quyết định mới nhất của advisor cho một brief."""

    brief_id: int
    action: str


@dataclass(frozen=True)
class LlmStoreData:
    """Bộ dữ liệu thô cho 8 check của LLM gate."""

    llm_calls: tuple[LlmCallRow, ...]
    briefs: tuple[BriefRow, ...]
    decisions: tuple[DecisionRow, ...]
    # (code_module, code_presentation, student_key) -> tên feature trong top_signals
    top_signals: dict[tuple[str, str, str], tuple[str, ...]]
    # Cùng khóa → toàn bộ item top_signals (feature_name + contribution +
    # observed_value) làm evidence gốc cho check judge_quality (additive).
    top_signal_details: dict[
        tuple[str, str, str], tuple[dict[str, Any], ...]
    ] = field(default_factory=dict)


class LlmStoreReader:
    """Đọc một lần toàn bộ dữ liệu gate từ PostgreSQL chia sẻ (read-only)."""

    REQUIRED_TABLES: tuple[str, ...] = ("llm_calls", "briefs", "decisions")
    OPTIONAL_TABLES: tuple[str, ...] = ("ml_predictions",)

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("llm_store_dsn_missing")
        self._dsn = dsn
        self._engine: Engine | None = None

    def fetch(self, limit: int) -> LlmStoreData:
        """SELECT toàn bộ dữ liệu gate (mỗi bảng tối đa `limit` dòng mới nhất)."""
        if limit < 1:
            raise ValueError(f"llm_store_limit_invalid:{limit}")
        engine = self._ensure_engine()
        try:
            with engine.connect() as connection:
                self._require_tables(connection)
                calls = self._fetch_calls(connection, limit)
                briefs = self._fetch_briefs(connection, limit)
                decisions = self._fetch_decisions(connection)
                signals, signal_details = self._fetch_top_signals(connection)
        except Exception as exc:
            raise ValueError(
                f"llm_store_read_failed:{type(exc).__name__}"
            ) from exc
        return LlmStoreData(
            llm_calls=tuple(calls),
            briefs=tuple(briefs),
            decisions=tuple(decisions),
            top_signals=signals,
            top_signal_details=signal_details,
        )

    def _ensure_engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(_sqlalchemy_dsn(self._dsn))
        return self._engine

    def _require_tables(self, connection: Any) -> None:
        present = {
            row[0] for row in connection.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
        }
        missing = [name for name in self.REQUIRED_TABLES if name not in present]
        if missing:
            raise ValueError(f"llm_store_tables_missing:{','.join(missing)}")

    def _fetch_calls(self, connection: Any, limit: int) -> list[LlmCallRow]:
        has_cost = bool(connection.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'llm_calls' "
            "AND column_name = 'cost_usd'"
        )).scalar())
        cost_expr = "cost_usd" if has_cost else "NULL"
        rows = connection.execute(text(
            f"SELECT call_id, brief_id, model, status, latency_ms, {cost_expr} "
            f"AS cost_usd FROM llm_calls ORDER BY call_id DESC LIMIT :limit"
        ), {"limit": limit}).mappings().all()
        return [
            LlmCallRow(
                call_id=int(row["call_id"]),
                brief_id=None if row["brief_id"] is None else int(row["brief_id"]),
                model=str(row["model"]),
                status=str(row["status"]),
                latency_ms=int(row["latency_ms"]),
                cost_usd=None if row["cost_usd"] is None else float(row["cost_usd"]),
            )
            for row in rows
        ]

    def _fetch_briefs(self, connection: Any, limit: int) -> list[BriefRow]:
        rows = connection.execute(text(
            "SELECT brief_id, code_module, code_presentation, student_key, "
            "version, content, citations, created_at FROM briefs "
            "ORDER BY brief_id DESC LIMIT :limit"
        ), {"limit": limit}).mappings().all()
        return [
            BriefRow(
                brief_id=int(row["brief_id"]),
                code_module=str(row["code_module"]).strip(),
                code_presentation=str(row["code_presentation"]).strip(),
                student_key=str(row["student_key"]).strip(),
                version=int(row["version"]),
                content=dict(row["content"]) if row["content"] is not None else {},
                citations=list(row["citations"]) if row["citations"] is not None else [],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _fetch_decisions(self, connection: Any) -> list[DecisionRow]:
        # Quyết định MỚI NHẤT mỗi brief (brief có thể bị review nhiều lần).
        rows = connection.execute(text(
            "SELECT d.brief_id, d.action FROM decisions d "
            "JOIN (SELECT brief_id, max(decision_id) AS decision_id "
            "      FROM decisions GROUP BY brief_id) latest "
            "ON d.decision_id = latest.decision_id ORDER BY d.brief_id"
        )).mappings().all()
        return [
            DecisionRow(brief_id=int(row["brief_id"]), action=str(row["action"]))
            for row in rows
        ]

    def _fetch_top_signals(
        self, connection: Any,
    ) -> tuple[
        dict[tuple[str, str, str], tuple[str, ...]],
        dict[tuple[str, str, str], tuple[dict[str, Any], ...]],
    ]:
        import json

        has_table = bool(connection.execute(text(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = 'ml_predictions'"
        )).scalar())
        if not has_table:
            return {}, {}
        rows = connection.execute(text(
            "SELECT code_module, code_presentation, student_key, top_signals_json "
            "FROM ml_predictions"
        )).mappings().all()
        signals: dict[tuple[str, str, str], tuple[str, ...]] = {}
        details: dict[tuple[str, str, str], tuple[dict[str, Any], ...]] = {}
        for row in rows:
            raw = row["top_signals_json"]
            names: list[str] = []
            items: list[dict[str, Any]] = []
            if raw:
                try:
                    parsed = json.loads(str(raw))
                except json.JSONDecodeError:
                    parsed = []
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict) or not item.get("feature_name"):
                            continue
                        names.append(str(item.get("feature_name")))
                        items.append(dict(item))
            key = (
                str(row["code_module"]).strip(),
                str(row["code_presentation"]).strip(),
                str(row["student_key"]).strip(),
            )
            signals[key] = tuple(names)
            details[key] = tuple(items)
        return signals, details

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> "LlmStoreReader":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
