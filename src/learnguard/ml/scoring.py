"""M5 batch scoring: score a snapshot and publish a `prediction-batch.v1` artifact.

The batch service contract: score every row of a feature snapshot with a
model loaded through the algorithm-agnostic `ScoringModel` interface, route
uncertain / insufficient-data rows to manual review, and atomically publish a
directory (`predictions.parquet` + `manifest.json` + `success.json`) that the
backend consumes as files — never by importing this code. Student identity is
reduced to a deterministic pseudonymous key before anything leaves the
process.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from learnguard.data.features import KEYS
from learnguard.data.loading import SPLITS, load_snapshot_partitions
from learnguard.ml.contracts import (
    PREDICTION_BATCH_SCHEMA_VERSION,
    PredictionBatchExporter,
    PredictionSignal,
)
from learnguard.ml.model_interface import (
    TOP_SIGNAL_COUNT,
    ScoringModel,
    load_scoring_model,
)
from learnguard.ml.preprocessing import (
    _file_hash,
    _prepare_split,
    _validate_run_id,
)


class PredictionStatus(str, Enum):
    SCORED = "scored"
    INSUFFICIENT_DATA = "insufficient_data"
    MANUAL_REVIEW = "manual_review"
    NO_PREDICTION = "no_prediction"


@dataclass
class PredictionRecord:
    student_key: str
    code_module: str
    code_presentation: str
    risk_score: float | None
    predicted_label: int | None
    threshold: float
    uncertainty: float | None
    status: PredictionStatus
    model_version: str
    snapshot_version: str
    cutoff_day: int
    top_signals: list[PredictionSignal] = field(default_factory=list)
    review_reason: str | None = None
    split: str = ""
    # Thống kê hành vi/học thuật GỐC của SV tại cutoff (không demographics —
    # quyết định PII: gender/region/disability/imd_band… không rời snapshot).
    features: dict[str, float | None] = field(default_factory=dict)

HIGH_UNCERTAINTY_THRESHOLD = 0.8
PREDICTIONS_FILENAME = "predictions.parquet"
MANIFEST_FILENAME = "manifest.json"
SUCCESS_FILENAME = "success.json"

# Default: score all three splits. Callers may narrow (e.g. ("holdout",)) to
# score only the not-yet-consumed presentation.
DEFAULT_SCORE_SPLITS: tuple[str, ...] = SPLITS

PREDICTION_COLUMNS: tuple[str, ...] = (
    "student_key", "code_module", "code_presentation", "split", "risk_score",
    "predicted_label", "threshold", "uncertainty", "status", "review_reason",
    "model_version", "snapshot_version", "cutoff_day", "top_signals_json",
    "features_json",
)

# Cột thống kê đưa vào `features_json`: hành vi học tập + học thuật. Cố ý
# LOẠI demographics (gender/region/highest_education/imd_band/age_band/
# disability) — giữ ranh giới PII đã chốt; demographics ở lại trong snapshot.
STATS_FEATURE_COLUMNS: tuple[str, ...] = (
    "n_clicks_before_cutoff",
    "click_trend_w3_w4",
    "n_submitted_before_cutoff",
    "avg_score_submitted",
    "num_of_prev_attempts",
    "studied_credits",
)

_REVIEW_REASONS: dict[PredictionStatus, str | None] = {
    PredictionStatus.SCORED: None,
    PredictionStatus.INSUFFICIENT_DATA: "insufficient_data",
    PredictionStatus.MANUAL_REVIEW: "high_uncertainty",
    # Defensive: emitted only if a row cannot be scored at all (e.g. a future
    # model failing on a row); the current pipeline never produces it.
    PredictionStatus.NO_PREDICTION: "no_prediction",
}


class BatchScorer:
    """Scores snapshot rows and routes each outcome to its review state."""

    def score(self, snapshot_path: Path | str, model_path: Path | str,
              preprocessing_artifact_dir: Path | str, output_path: Path | str,
              threshold: float = 0.5, run_id: str = "score",
              splits: Sequence[str] = DEFAULT_SCORE_SPLITS, *,
              release_id: str | None = None,
              release_manifest_sha256: str | None = None,
              target_alert_fraction: float | None = None) -> ScoringReport:
        """Score the requested splits and atomically publish the batch.

        `output_path` is the published `predictions.parquet` path; its parent
        directory becomes the batch directory receiving `manifest.json` and
        `success.json` (staged then `os.replace`d, same as training.py). Risk
        scores keep full float64 precision; `predicted_label` is derived from
        exactly the value written to the parquet so backend consumers can
        recompute it as `risk_score >= threshold`. `splits` selects which
        snapshot splits enter the batch (default: train, validation, holdout).
        """
        snapshot_path = Path(snapshot_path)
        output_path = Path(output_path)
        run_id = _validate_run_id(run_id)
        threshold = _validate_threshold(threshold)
        requested_splits = _validate_splits(DEFAULT_SCORE_SPLITS if splits is None else splits)
        batch_dir = output_path.parent
        if batch_dir.exists():
            raise FileExistsError(f"prediction_batch_run_id_exists:{run_id}")
        model = load_scoring_model(model_path, snapshot_path, preprocessing_artifact_dir)
        snapshot_splits = load_snapshot_partitions(snapshot_path, requested_splits)
        for split_name in requested_splits:
            if snapshot_splits.frame(split_name).empty:
                raise ValueError(f"empty_split:{split_name}")
        cutoff_day = _ledger_cutoff_day(snapshot_path)
        model_version = model.version
        feature_names = model.feature_names

        records: list[PredictionRecord] = []
        id_students: list[object] = []
        for split_name in requested_splits:
            prepared = _prepare_split(split_name, snapshot_splits.frame(split_name))
            # One transform feeds explain() so the signals always describe the
            # identical matrix the model consumes; predict_positive_proba runs
            # the same deterministic transform internally.
            matrix = model.transform(prepared.X)
            explanation_batch = model.explain_many(matrix) if hasattr(model, "explain_many") else None
            if explanation_batch is not None and explanation_batch.values.shape != matrix.shape:
                raise ValueError("scoring_explanation_shape_mismatch")
            probabilities = model.predict_positive_proba(prepared.X)
            insufficient = prepared.X["insufficient_data"].to_numpy(dtype=bool)
            # Cột thống kê GỐC (chưa transform) — đi thẳng vào features_json.
            stats_frame = prepared.X.loc[:, STATS_FEATURE_COLUMNS]
            modules = prepared.keys["code_module"].tolist()
            presentations = prepared.keys["code_presentation"].tolist()
            students = prepared.keys["id_student"].tolist()
            for index in range(len(probabilities)):
                risk_score = float(probabilities[index])
                uncertainty = 1.0 - abs(2.0 * risk_score - 1.0)
                data_status = "insufficient_data" if insufficient[index] else "ok"
                status = self.assign_review_state(risk_score, threshold, uncertainty, data_status)
                predicted_label = (
                    None if status is PredictionStatus.INSUFFICIENT_DATA
                    else int(risk_score >= threshold)
                )
                keys = (modules[index], presentations[index], students[index])
                records.append(PredictionRecord(
                    student_key=_pseudonymous_key(*keys),
                    code_module=str(modules[index]),
                    code_presentation=str(presentations[index]),
                    split=split_name,
                    risk_score=risk_score,
                    predicted_label=predicted_label,
                    threshold=threshold,
                    uncertainty=uncertainty,
                    status=status,
                    model_version=model_version,
                    snapshot_version=snapshot_splits.snapshot_version,
                    cutoff_day=cutoff_day,
                    top_signals=(
                        _top_signals(explanation_batch.feature_names, explanation_batch.values[index])
                        if explanation_batch is not None else model.explain(matrix[index])
                    ),
                    review_reason=_REVIEW_REASONS[status],
                    features=_features_json_values(stats_frame.iloc[index]),
                ))
                id_students.append(students[index])

        frame = _records_frame(records, id_students).sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
        frame = frame.drop(columns=["id_student"])
        raw_positive_count = int(np.count_nonzero(
            frame["risk_score"].to_numpy(dtype=np.float64) >= threshold
        ))
        raw_positive_rate = raw_positive_count / len(frame) if len(frame) else 0.0
        labeled_positive_count = int((frame["predicted_label"] == 1).sum())
        scored_alert_count = int((
            (frame["status"] == PredictionStatus.SCORED.value)
            & (frame["predicted_label"] == 1)
        ).sum())
        drift_status: str | None = None
        if target_alert_fraction is not None:
            deviation = abs(raw_positive_rate - float(target_alert_fraction))
            drift_status = (
                "critical_requires_review" if deviation > 0.10 else
                "warning" if deviation > 0.05 else "ok"
            )

        batch_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = batch_dir.parent / f".{batch_dir.name}.staging-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            frame.to_parquet(staging / PREDICTIONS_FILENAME, index=False)
            manifest = PredictionBatchExporter().export(
                staging / PREDICTIONS_FILENAME, staging / MANIFEST_FILENAME,
                run_id, snapshot_splits.snapshot_version, model_version, cutoff_day,
            )
            release_provenance_sha256: str | None = None
            if release_id is not None:
                if not release_manifest_sha256 or len(release_manifest_sha256) != 64:
                    raise ValueError("scoring_release_manifest_hash_invalid")
                provenance_path = staging / "release.provenance.json"
                provenance_path.write_text(json.dumps({
                    "release_id": release_id,
                    "release_manifest_sha256": release_manifest_sha256,
                    "production_threshold": threshold,
                    "target_alert_fraction": target_alert_fraction,
                    "raw_threshold_positive_count": raw_positive_count,
                    "raw_threshold_positive_rate": raw_positive_rate,
                    "labeled_positive_count": labeled_positive_count,
                    "scored_alert_count": scored_alert_count,
                    "manual_review_count": int((
                        frame["status"] == PredictionStatus.MANUAL_REVIEW.value
                    ).sum()),
                    "insufficient_data_count": int((
                        frame["status"] == PredictionStatus.INSUFFICIENT_DATA.value
                    ).sum()),
                    "alert_rate_drift_status": drift_status,
                    "monitoring_only": True,
                }, indent=2, sort_keys=True))
                release_provenance_sha256 = _file_hash(provenance_path)
            success_payload: dict[str, object] = {
                "schema_version": PREDICTION_BATCH_SCHEMA_VERSION,
                "run_id": run_id,
                "manifest_sha256": _file_hash(staging / MANIFEST_FILENAME),
                "predictions_sha256": manifest.predictions_sha256,
            }
            if release_provenance_sha256 is not None:
                success_payload["release_provenance_sha256"] = release_provenance_sha256
            (staging / SUCCESS_FILENAME).write_text(json.dumps(
                success_payload, indent=2, sort_keys=True,
            ))
            if batch_dir.exists():
                raise FileExistsError(f"prediction_batch_run_id_exists:{run_id}")
            os.replace(staging, batch_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        counts = {status: sum(record.status is status for record in records) for status in PredictionStatus}
        return ScoringReport(
            status="ok",
            model_version=model_version,
            snapshot_version=snapshot_splits.snapshot_version,
            input_rows=len(records),
            scored_rows=counts[PredictionStatus.SCORED],
            manual_review_rows=counts[PredictionStatus.MANUAL_REVIEW],
            insufficient_data_rows=counts[PredictionStatus.INSUFFICIENT_DATA],
            output_path=output_path,
            batch_dir=batch_dir,
            threshold=threshold,
            no_prediction_rows=counts[PredictionStatus.NO_PREDICTION],
            release_id=release_id,
            raw_threshold_positive_count=raw_positive_count,
            raw_threshold_positive_rate=raw_positive_rate,
            labeled_positive_count=labeled_positive_count,
            scored_alert_count=scored_alert_count,
            alert_rate_drift_status=drift_status,
        )

    def explain(self, model: ScoringModel, feature_row: np.ndarray) -> list[PredictionSignal]:
        """Return the top signals for ONE row in the TRANSFORMED feature space.

        Delegates to the model's own `explain`, so `BatchScorer` stays
        algorithm-agnostic: it never inspects coefficients or any other
        estimator internals. `feature_row` must be the post-preprocessing
        numeric row (what the model consumes, e.g. a row of
        `FittedPreprocessor.transform` output), NOT the raw snapshot frame.
        Returns at most `TOP_SIGNAL_COUNT` signals sorted by absolute
        contribution (descending, stable on ties); contributions are
        associations, not causal effects.
        """
        return model.explain(feature_row)

    def assign_review_state(self, risk_score: float, threshold: float,
                            uncertainty: float, data_status: str) -> PredictionStatus:
        """Route insufficient-data and high-uncertainty rows to manual review."""
        if data_status == "insufficient_data":
            return PredictionStatus.INSUFFICIENT_DATA
        if uncertainty >= HIGH_UNCERTAINTY_THRESHOLD:
            return PredictionStatus.MANUAL_REVIEW
        return PredictionStatus.SCORED


@dataclass
class ScoringReport:
    status: str
    model_version: str
    snapshot_version: str
    input_rows: int
    scored_rows: int
    manual_review_rows: int
    insufficient_data_rows: int
    output_path: Path
    batch_dir: Path
    threshold: float
    no_prediction_rows: int
    # Database serving side (set by the CLI after the file publish):
    # scoring itself never writes the DB, so the default reflects "no DSN".
    database_write: str = "skipped_no_dsn"
    database_rows: int | None = None
    release_id: str | None = None
    raw_threshold_positive_count: int | None = None
    raw_threshold_positive_rate: float | None = None
    labeled_positive_count: int | None = None
    scored_alert_count: int | None = None
    alert_rate_drift_status: str | None = None


def score_snapshot(snapshot_path: Path | str, model_path: Path | str,
                   preprocessing_artifact_dir: Path | str, output_path: Path | str,
                   threshold: float = 0.5, run_id: str = "score",
                   splits: Sequence[str] = DEFAULT_SCORE_SPLITS) -> ScoringReport:
    """M5 pipeline entry point: score the snapshot and publish the batch."""
    return BatchScorer().score(
        snapshot_path, model_path, preprocessing_artifact_dir,
        output_path, threshold, run_id, splits,
    )


def score_release_snapshot(
    snapshot_path: Path | str,
    release_artifact_dir: Path | str,
    output_path: Path | str,
    run_id: str,
    splits: Sequence[str],
) -> ScoringReport:
    """Score with a verified immutable release and its locked threshold."""
    from learnguard.ml.release import load_and_verify_model_release

    requested = _validate_splits(splits)
    release = load_and_verify_model_release(release_artifact_dir, snapshot_path)
    if (
        release.holdout_access_policy == "forbidden_prior_consumed"
        and "holdout" in requested
    ):
        raise ValueError("scoring_release_holdout_forbidden")
    target_fraction = release.manifest.get("target_alert_fraction")
    if (
        isinstance(target_fraction, bool)
        or not isinstance(target_fraction, (int, float))
    ):
        raise ValueError("scoring_release_alert_fraction_invalid")
    manifest_path = Path(release_artifact_dir) / "release.manifest.json"
    return BatchScorer().score(
        snapshot_path,
        release.model_artifact_dir,
        release.preprocessing_artifact_dir,
        output_path,
        release.production_threshold,
        run_id,
        requested,
        release_id=release.release_id,
        release_manifest_sha256=_file_hash(manifest_path),
        target_alert_fraction=float(target_fraction),
    )


def _validate_splits(splits: Sequence[str]) -> tuple[str, ...]:
    """Validate the split selection: known names, no duplicates, non-empty."""
    names = tuple(str(name) for name in splits)
    if not names:
        raise ValueError("empty_split_selection")
    for name in names:
        if name not in SPLITS:
            raise ValueError(f"unknown_split_name:{name}")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate_split_name:{','.join(names)}")
    return names


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("scoring_threshold_out_of_range")
    return float(threshold)


def _top_signals(feature_names: Sequence[str], contributions: np.ndarray) -> list[PredictionSignal]:
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or len(feature_names) != values.size or not np.isfinite(values).all():
        raise ValueError("scoring_explanation_invalid")
    order = np.argsort(-np.abs(values), kind="stable")[:TOP_SIGNAL_COUNT]
    return [
        PredictionSignal(
            feature_name=str(feature_names[index]),
            contribution=float(values[index]),
            direction=("increases_risk" if values[index] > 0 else
                       "decreases_risk" if values[index] < 0 else "neutral"),
        )
        for index in order
    ]


def _pseudonymous_key(code_module: str, code_presentation: str, id_student: object) -> str:
    """Stable pseudonymous key — never exposes the raw `id_student`."""
    raw = f"learnguard:prediction-batch.v1:{code_module}:{code_presentation}:{id_student}"
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def _ledger_cutoff_day(snapshot_path: Path) -> int:
    ledger = snapshot_path.parent / f"{snapshot_path.stem}.ledger.json"
    if not ledger.is_file():
        raise FileNotFoundError(f"missing_ledger:{ledger}")
    value = json.loads(ledger.read_text()).get("cutoff_day")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("ledger_cutoff_day_missing")
    return value


def _features_json_values(row: "pd.Series") -> dict[str, float | None]:
    """Giá trị GỐC của các cột thống kê cho 1 dòng snapshot.

    NaN → None (null trong JSON) — thống nhất với quy ước NULL của
    predictions_store; số được chuẩn về float Python gốc (parquet/JSON an toàn).
    """
    values: dict[str, float | None] = {}
    for name in STATS_FEATURE_COLUMNS:
        value = row[name]
        if value is None or pd.isna(value):
            values[name] = None
        else:
            values[name] = float(value)
    return values


def _records_frame(records: Sequence[PredictionRecord],
                   id_students: Sequence[object]) -> pd.DataFrame:
    """Materialize records into one frame; `id_student` is kept ONLY to sort
    by KEYS and is dropped by the caller before anything is written."""
    return pd.DataFrame({
        "student_key": [record.student_key for record in records],
        "code_module": [record.code_module for record in records],
        "code_presentation": [record.code_presentation for record in records],
        "id_student": list(id_students),
        "split": [record.split for record in records],
        "risk_score": np.asarray([record.risk_score for record in records], dtype=np.float64),
        "predicted_label": pd.array(
            [record.predicted_label for record in records], dtype="Int64"
        ),
        "threshold": np.asarray([record.threshold for record in records], dtype=np.float64),
        "uncertainty": np.asarray([record.uncertainty for record in records], dtype=np.float64),
        "status": [record.status.value for record in records],
        "review_reason": [record.review_reason for record in records],
        "model_version": [record.model_version for record in records],
        "snapshot_version": [record.snapshot_version for record in records],
        "cutoff_day": np.asarray([record.cutoff_day for record in records], dtype=np.int64),
        "top_signals_json": [
            json.dumps(
                [{"feature_name": signal.feature_name, "contribution": signal.contribution,
                  "direction": signal.direction} for signal in record.top_signals],
                separators=(",", ":"), sort_keys=True,
            )
            for record in records
        ],
        "features_json": [
            json.dumps(record.features, separators=(",", ":"), sort_keys=True,
                       allow_nan=False)
            for record in records
        ],
    })
