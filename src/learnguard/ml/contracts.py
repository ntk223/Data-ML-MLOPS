"""Versioned file contract used to hand scoring batches to the backend."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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
        """TODO: write deterministic batch manifest and checksums."""
        raise NotImplementedError


def load_prediction_batch_manifest(manifest_path: Path) -> PredictionBatchManifest:
    """TODO: parse and validate supported manifest schema versions."""
    raise NotImplementedError
