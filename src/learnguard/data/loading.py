"""Validated snapshot loader and temporal split handoff for the ML owner."""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from learnguard.data.snapshot_contract import (
    MODEL_FEATURE_COLUMNS,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_SCHEMA_VERSION,
    SPLITS,
    SPLIT_RULES,
    ledger_path,
    snapshot_content_hash,
    success_path,
    validate_ledger_metadata,
)

TARGET_COLUMN = "label"


@dataclass(frozen=True)
class SnapshotSplits:
    snapshot_version: str
    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame

    def frame(self, split: str) -> pd.DataFrame:
        if split not in SPLITS:
            raise ValueError(f"unknown_split:{split}:allowed={','.join(SPLITS)}")
        return getattr(self, split)

    def x_y(self, split: str) -> tuple[pd.DataFrame, pd.Series]:
        frame = self.frame(split)
        return frame[list(MODEL_FEATURE_COLUMNS)].copy(), frame[TARGET_COLUMN].copy()


def validate_snapshot_artifact(
    snapshot_path: Path | str, require_all_splits: bool = True
) -> tuple[pd.DataFrame, dict[str, object]]:
    snapshot_path = Path(snapshot_path)
    snapshot = pd.read_parquet(snapshot_path)
    actual_columns = list(snapshot.columns)
    if actual_columns != list(SNAPSHOT_COLUMNS):
        raise ValueError("snapshot_schema_or_column_order_mismatch")
    present = set(snapshot["split"].dropna().unique())
    invalid = present - set(SPLITS)
    if invalid:
        raise ValueError(f"split_out_of_domain:{','.join(sorted(invalid))}")
    if require_all_splits:
        empty = [split for split in SPLITS if split not in present]
        if empty:
            raise ValueError(f"split_empty:{','.join(empty)}")
    expected_split = snapshot["code_presentation"].map(SPLIT_RULES)
    if expected_split.isna().any() or not expected_split.equals(snapshot["split"]):
        raise ValueError("snapshot_split_rule_mismatch")

    metadata_path = ledger_path(snapshot_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing_ledger:{metadata_path}:rebuild snapshot")
    marker_path = success_path(snapshot_path)
    if not marker_path.is_file():
        raise FileNotFoundError(f"missing_snapshot_success_marker:{marker_path}")

    ledger = json.loads(metadata_path.read_text())
    if ledger.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported_snapshot_schema:{ledger.get('schema_version') or 'legacy'}:"
            "rebuild snapshot from canonical data"
        )
    metadata_errors = validate_ledger_metadata(ledger)
    if metadata_errors:
        raise ValueError(f"ledger_metadata_invalid:{','.join(metadata_errors)}")
    if ledger.get("output_schema") != list(SNAPSHOT_COLUMNS):
        raise ValueError("ledger_output_schema_mismatch")
    if ledger.get("row_count") != len(snapshot):
        raise ValueError("ledger_row_count_mismatch")
    if ledger.get("snapshot_content_hash") != snapshot_content_hash(snapshot):
        raise ValueError("ledger_snapshot_content_hash_mismatch")

    for field in ("input_hashes", "input_content_hashes"):
        hashes = ledger.get(field)
        if not isinstance(hashes, dict) or not hashes or any(
            not isinstance(value, str) or len(value) != 64 for value in hashes.values()
        ):
            raise ValueError(f"ledger_{field}_invalid")

    marker = json.loads(marker_path.read_text())
    if marker.get("snapshot_version") != ledger.get("snapshot_version") or marker.get(
        "snapshot_content_hash"
    ) != ledger.get("snapshot_content_hash"):
        raise ValueError("snapshot_success_marker_mismatch")

    return snapshot, ledger


def load_snapshot_splits(snapshot_path: Path | str) -> SnapshotSplits:
    snapshot, ledger = validate_snapshot_artifact(snapshot_path)
    return SnapshotSplits(
        snapshot_version=str(ledger["snapshot_version"]),
        train=snapshot[snapshot["split"] == "train"].copy(),
        validation=snapshot[snapshot["split"] == "validation"].copy(),
        holdout=snapshot[snapshot["split"] == "holdout"].copy(),
    )
