"""Production serving store for batch scoring results.

Integration mode agreed 2026-10-30: ml-data WRITES scoring results directly
into the shared PostgreSQL table `ml_predictions`; the backend repo is a
READ-ONLY consumer (no import service, no schema ownership for this table).
The `prediction-batch.v1` files published by `learnguard ml score` remain the
audit trail and the cross-team verification artifact.

Every score run upserts — the primary key (code_module, code_presentation,
student_key) keeps exactly one row per student, overwritten by the newest
run, so readers always see the CURRENT prediction. Schema changes to this
table are breaking changes for the backend consumer: coordinate through
docs/M5_PREDICTION_BATCH.md before altering.

All SQL is parameterized; the table name is the only interpolated identifier
and it is validated against a strict pattern (it comes from configuration,
never from user values).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sqlalchemy import Engine, create_engine, text

from learnguard.ml.preprocessing import _validate_run_id
from learnguard.ml.scoring import PREDICTION_COLUMNS

TABLE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
DEFAULT_TABLE = "ml_predictions"
DSN_ENV_VAR = "LEARNGUARD_PG_DSN"

# Physical schema of `ml_predictions`. Kept in one place so `ensure_schema`
# and the column list below can never drift apart.
_TABLE_COLUMNS: tuple[str, ...] = (
    "student_key", "code_module", "code_presentation", "split",
    "risk_score", "predicted_label", "threshold", "uncertainty",
    "status", "review_reason", "model_version", "snapshot_version",
    "cutoff_day", "top_signals_json", "features_json", "run_id", "scored_at",
)

_UPSERT_STATEMENT = """
INSERT INTO {table} (
    student_key, code_module, code_presentation, split,
    risk_score, predicted_label, threshold, uncertainty,
    status, review_reason, model_version, snapshot_version,
    cutoff_day, top_signals_json, features_json, run_id
) VALUES (
    :student_key, :code_module, :code_presentation, :split,
    :risk_score, :predicted_label, :threshold, :uncertainty,
    :status, :review_reason, :model_version, :snapshot_version,
    :cutoff_day, :top_signals_json, :features_json, :run_id
)
ON CONFLICT (code_module, code_presentation, student_key) DO UPDATE SET
    split = EXCLUDED.split,
    risk_score = EXCLUDED.risk_score,
    predicted_label = EXCLUDED.predicted_label,
    threshold = EXCLUDED.threshold,
    uncertainty = EXCLUDED.uncertainty,
    status = EXCLUDED.status,
    review_reason = EXCLUDED.review_reason,
    model_version = EXCLUDED.model_version,
    snapshot_version = EXCLUDED.snapshot_version,
    cutoff_day = EXCLUDED.cutoff_day,
    top_signals_json = EXCLUDED.top_signals_json,
    features_json = EXCLUDED.features_json,
    run_id = EXCLUDED.run_id,
    scored_at = now()
"""


@dataclass(frozen=True)
class PredictionStoreConfig:
    """Connection settings for the shared prediction store.

    `dsn` empty means "not configured" — callers must skip the database write
    (file export remains the audit trail). There is deliberately NO default
    credential fallback in code.
    """

    dsn: str
    table: str = DEFAULT_TABLE

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None,
                         table: str = DEFAULT_TABLE) -> "PredictionStoreConfig":
        """Read the DSN from `LEARNGUARD_PG_DSN` (empty when unset)."""
        env = os.environ if environ is None else environ
        return cls(dsn=env.get(DSN_ENV_VAR, ""), table=table)

    def is_configured(self) -> bool:
        return bool(self.dsn)


class PredictionStore:
    """Transactional upsert/read access to the shared `ml_predictions` table."""

    def __init__(self, config: PredictionStoreConfig) -> None:
        if not TABLE_NAME_PATTERN.fullmatch(config.table):
            raise ValueError(f"prediction_store_table_name_invalid:{config.table}")
        self._config = config
        self._engine: Engine | None = None

    @property
    def table(self) -> str:
        return self._config.table

    @property
    def engine(self) -> Engine:
        """Lazily created SQLAlchemy engine on the psycopg 3 driver."""
        if self._engine is None:
            if not self._config.dsn:
                raise ValueError("prediction_store_dsn_missing")
            self._engine = create_engine(_sqlalchemy_dsn(self._config.dsn))
        return self._engine

    def ensure_schema(self) -> None:
        """Create the table and indexes if missing (idempotent DDL)."""
        engine = self.engine  # raises prediction_store_dsn_missing when unconfigured
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self._config.table} (
            student_key varchar(16) NOT NULL,
            code_module varchar NOT NULL,
            code_presentation varchar NOT NULL,
            split varchar NOT NULL,
            risk_score double precision,
            predicted_label smallint,
            threshold double precision NOT NULL,
            uncertainty double precision,
            status varchar NOT NULL,
            review_reason varchar,
            model_version varchar NOT NULL,
            snapshot_version varchar NOT NULL,
            cutoff_day integer NOT NULL,
            top_signals_json text,
            features_json text,
            run_id varchar NOT NULL,
            scored_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (code_module, code_presentation, student_key)
        )
        """
        try:
            with engine.begin() as connection:
                connection.execute(text(ddl))
                # DB cũ (tạo trước khi có features_json) — nâng cấp tại chỗ.
                connection.execute(text(
                    f"ALTER TABLE {self._config.table} "
                    "ADD COLUMN IF NOT EXISTS features_json text"
                ))
                connection.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {self._config.table}_status_idx"
                    f" ON {self._config.table} (status)"
                ))
                connection.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {self._config.table}_model_version_idx"
                    f" ON {self._config.table} (model_version)"
                ))
        except Exception as exc:
            raise ValueError(f"prediction_store_schema_failed:{type(exc).__name__}") from exc

    def write_predictions(self, frame: pd.DataFrame, run_id: str) -> int:
        """Upsert every frame row in ONE transaction; return the row count.

        All-or-nothing: any failing row aborts the transaction and nothing is
        written. The frame must carry exactly the `prediction-batch.v1`
        parquet columns. `top_signals_json` of `"[]"` (no signals) is stored
        as SQL NULL.
        """
        run_id = _validate_run_id(run_id)
        _validate_frame_columns(frame)
        rows = _rows_for_write(frame, run_id)
        statement = text(_UPSERT_STATEMENT.format(table=self._config.table))
        try:
            with self.engine.begin() as connection:
                if rows:
                    connection.execute(statement, rows)
        except Exception as exc:
            # engine.begin() already rolled back; nothing partially written.
            raise ValueError(f"prediction_store_write_failed:{type(exc).__name__}") from exc
        return len(rows)

    def count_run(self, run_id: str) -> int:
        """Rows currently stored for `run_id` (post-write verification)."""
        query = text(f"SELECT count(*) FROM {self._config.table} WHERE run_id = :run_id")
        try:
            with self.engine.connect() as connection:
                return int(connection.execute(query, {"run_id": run_id}).scalar_one())
        except Exception as exc:
            raise ValueError(f"prediction_store_read_failed:{type(exc).__name__}") from exc

    def fetch_predictions(self, status: str | None = None) -> pd.DataFrame:
        """Read stored predictions, optionally filtered by status.

        Deterministic order (code_module, code_presentation, student_key) so a
        backend analog gets stable paging. Column order mirrors the parquet
        contract plus `run_id` and `scored_at`.
        """
        columns = ", ".join(
            name for name in _TABLE_COLUMNS if name != "scored_at"
        )
        query = f"SELECT {columns}, scored_at FROM {self._config.table}"
        params: dict[str, object] | None = None
        if status is not None:
            query += " WHERE status = :status"
            params = {"status": status}
        query += " ORDER BY code_module, code_presentation, student_key"
        try:
            with self.engine.connect() as connection:
                return pd.read_sql(text(query), connection, params=params)
        except Exception as exc:
            raise ValueError(f"prediction_store_read_failed:{type(exc).__name__}") from exc

    def drop_table(self) -> None:
        """Drop the table (operator/test cleanup — never part of scoring)."""
        try:
            with self.engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {self._config.table}"))
        except Exception as exc:
            raise ValueError(f"prediction_store_drop_failed:{type(exc).__name__}") from exc

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    def __enter__(self) -> "PredictionStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _sqlalchemy_dsn(dsn: str) -> str:
    """Force the psycopg 3 driver for plain `postgresql://` DSNs."""
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://"):]
    if dsn.startswith("postgresql+psycopg://"):
        return dsn
    raise ValueError("prediction_store_dsn_unsupported")


def _validate_frame_columns(frame: pd.DataFrame) -> None:
    if list(frame.columns) != list(PREDICTION_COLUMNS):
        raise ValueError(
            f"prediction_store_columns_invalid:expected={','.join(PREDICTION_COLUMNS)}"
            f":got={','.join(map(str, frame.columns))}"
        )


def _rows_for_write(frame: pd.DataFrame, run_id: str) -> list[dict[str, object]]:
    """Convert one parquet frame to parameterized rows with Python natives.

    psycopg does not adapt numpy/pandas extension scalars, so every value is
    reduced to `int`/`float`/`str`/`None` explicitly.
    """
    rows: list[dict[str, object]] = []
    for record in frame.to_dict(orient="records"):
        rows.append({
            "student_key": _text(record["student_key"]),
            "code_module": _text(record["code_module"]),
            "code_presentation": _text(record["code_presentation"]),
            "split": _text(record["split"]),
            "risk_score": _float(record["risk_score"]),
            "predicted_label": _int(record["predicted_label"]),
            "threshold": _float(record["threshold"]),
            "uncertainty": _float(record["uncertainty"]),
            "status": _text(record["status"]),
            "review_reason": _text(record["review_reason"]),
            "model_version": _text(record["model_version"]),
            "snapshot_version": _text(record["snapshot_version"]),
            "cutoff_day": _int(record["cutoff_day"]),
            "top_signals_json": _signals(record["top_signals_json"]),
            "features_json": _text(record["features_json"]),
            "run_id": run_id,
        })
    return rows


def _text(value: object) -> str | None:
    # NaN float (np.nan/float('nan')) phải map về NULL — str(nan) cho ra chuỗi
    # 'nan' rác trong DB (bug thật: 13,295 dòng scored bị dính).
    if value is None or pd.isna(value):
        return None
    return str(value)


def _float(value: object) -> float | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return None if pd.isna(number) else number
    raise ValueError(f"prediction_store_value_invalid:float:{type(value).__name__}")


def _int(value: object) -> int | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not pd.isna(value):
        return int(value)
    raise ValueError(f"prediction_store_value_invalid:int:{type(value).__name__}")


def _signals(value: object) -> str | None:
    """`"[]"` (no signals) → NULL; otherwise the JSON string as-is."""
    if value is None or value is pd.NA:
        return None
    payload = str(value)
    return None if payload == "[]" else payload
