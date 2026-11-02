"""Versioned file contract used to hand scoring batches to the backend.

The exported directory IS the interface: the backend never imports this code,
it only reads `manifest.json` (schema `prediction-batch.v1`) and verifies the
`predictions.parquet` payload against the recorded sha256 checksum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from learnguard.ml.preprocessing import _file_hash, _validate_run_id

PREDICTION_BATCH_SCHEMA_VERSION = "prediction-batch.v1"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = (PREDICTION_BATCH_SCHEMA_VERSION,)


@dataclass(frozen=True)
class PredictionSignal:
    feature_name: str
    contribution: float
    direction: str


@dataclass(frozen=True)
class PredictionBatchManifest:
    schema_version: str
    run_id: str
    cutoff_day: int
    snapshot_version: str
    model_version: str
    created_at: datetime
    row_count: int
    predictions_path: Path
    predictions_sha256: str


class PredictionBatchExporter:
    def export(self, predictions_path: Path, manifest_path: Path,
               run_id: str, snapshot_version: str,
               model_version: str, cutoff_day: int) -> PredictionBatchManifest:
        """Write the deterministic batch manifest next to the predictions file.

        `predictions_path` must already exist (written into the staging
        directory by the caller); the manifest records its name relative to
        the batch directory so the exported tree stays relocatable.
        """
        predictions_path, manifest_path = Path(predictions_path), Path(manifest_path)
        run_id = _validate_run_id(run_id)
        if not predictions_path.is_file():
            raise FileNotFoundError(f"predictions_file_missing:{predictions_path}")
        row_count = int(pq.read_metadata(predictions_path).num_rows)
        predictions_sha256 = _file_hash(predictions_path)
        created_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "schema_version": PREDICTION_BATCH_SCHEMA_VERSION,
            "run_id": run_id,
            "cutoff_day": int(cutoff_day),
            "snapshot_version": snapshot_version,
            "model_version": model_version,
            "created_at": created_at,
            "row_count": row_count,
            "predictions_path": predictions_path.name,
            "predictions_sha256": predictions_sha256,
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return PredictionBatchManifest(
            schema_version=PREDICTION_BATCH_SCHEMA_VERSION,
            run_id=run_id,
            cutoff_day=int(cutoff_day),
            snapshot_version=snapshot_version,
            model_version=model_version,
            created_at=datetime.fromisoformat(created_at),
            row_count=row_count,
            predictions_path=Path(predictions_path.name),
            predictions_sha256=predictions_sha256,
        )


def load_prediction_batch_manifest(manifest_path: Path) -> PredictionBatchManifest:
    """Parse and validate a published manifest, verifying the predictions payload.

    Fails loudly with snake_case error codes on unsupported schema versions,
    missing payloads, or checksum/row-count mismatches — the backend must
    never consume a silently corrupted batch.
    """
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text())
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"prediction_batch_schema_unsupported:{schema_version}")
    for field in ("run_id", "snapshot_version", "model_version"):
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"prediction_batch_field_invalid:{field}")
    cutoff_day = payload.get("cutoff_day")
    if not isinstance(cutoff_day, int) or isinstance(cutoff_day, bool):
        raise ValueError("prediction_batch_cutoff_day_invalid")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("prediction_batch_created_at_invalid")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ValueError("prediction_batch_created_at_invalid") from exc
    row_count = payload.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ValueError("prediction_batch_row_count_invalid")
    relative = payload.get("predictions_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("prediction_batch_predictions_path_invalid")
    predictions_path = manifest_path.parent / relative
    if not predictions_path.is_file():
        raise ValueError(f"prediction_batch_predictions_file_missing:{predictions_path}")
    expected_sha256 = payload.get("predictions_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("prediction_batch_checksum_invalid")
    if _file_hash(predictions_path) != expected_sha256:
        raise ValueError("prediction_batch_checksum_mismatch")
    if int(pq.read_metadata(predictions_path).num_rows) != row_count:
        raise ValueError("prediction_batch_row_count_mismatch")
    return PredictionBatchManifest(
        schema_version=str(schema_version),
        run_id=str(payload["run_id"]),
        cutoff_day=int(cutoff_day),
        snapshot_version=str(payload["snapshot_version"]),
        model_version=str(payload["model_version"]),
        created_at=parsed_created_at,
        row_count=int(row_count),
        predictions_path=predictions_path,
        predictions_sha256=expected_sha256,
    )
