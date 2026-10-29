"""Test API tiêu thụ snapshot — loader chuẩn cho ML owner."""

import json

import pandas as pd
import pytest

from learnguard.data.features import FEATURE_CATALOG, build_feature_snapshot
from learnguard.data.loading import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_snapshot_splits,
)
from tests.test_features import base_tables, write_canonical


def _tables_all_presentations() -> dict[str, pd.DataFrame]:
    """base_tables + SV ở 2013J/2014B/2014J — để cả 3 split đều có dòng."""
    tables = base_tables()
    pres = ["2013J", "2014B", "2014J"]
    sids = [201, 202, 203, 204, 205, 206]  # 2 SV mỗi kỳ
    tables["courses"] = pd.concat([tables["courses"], pd.DataFrame({
        "code_module": ["AAA"] * 3, "code_presentation": pres,
        "module_presentation_length": [240] * 3})], ignore_index=True)
    tables["studentInfo"] = pd.concat([tables["studentInfo"], pd.DataFrame({
        "code_module": ["AAA"] * 6, "code_presentation": sum(([p] * 2 for p in pres), []),
        "id_student": sids, "gender": ["M", "F"] * 3, "region": ["Wales"] * 6,
        "highest_education": ["A Level or Equivalent"] * 6, "imd_band": ["10-20"] * 6,
        "age_band": ["0-35"] * 6, "num_of_prev_attempts": [0] * 6,
        "studied_credits": [60] * 6, "disability": ["N"] * 6,
        "final_result": ["Pass", "Fail", "Withdrawn", "Pass", "Fail", "Pass"],
    })], ignore_index=True)
    tables["studentRegistration"] = pd.concat([tables["studentRegistration"], pd.DataFrame({
        "code_module": ["AAA"] * 6, "code_presentation": sum(([p] * 2 for p in pres), []),
        "id_student": sids, "date_registration": [-5.0] * 6,
        "date_unregistration": [float("nan")] * 6,
    })], ignore_index=True)
    assessment_ids = [1011, 1012, 1013]
    tables["assessments"] = pd.concat([tables["assessments"], pd.DataFrame({
        "id_assessment": assessment_ids, "code_module": ["AAA"] * 3,
        "code_presentation": pres, "assessment_type": ["TMA"] * 3,
        "date": [20.0] * 3, "weight": [50.0] * 3,
    })], ignore_index=True)
    tables["studentAssessment"] = pd.concat([tables["studentAssessment"], pd.DataFrame({
        "id_assessment": sum(([value] * 2 for value in assessment_ids), []),
        "id_student": sids, "date_submitted": [15] * 6,
        "is_banked": [0] * 6, "score": [70.0, 80.0] * 3,
    })], ignore_index=True)
    return tables


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("loading")
    processed = write_canonical(tmp, _tables_all_presentations())
    result = build_feature_snapshot(processed, tmp / "snapshot.parquet", cutoff_day=27)
    return tmp, result


def test_load_returns_three_splits_with_version(snapshot):
    tmp, result = snapshot
    splits = load_snapshot_splits(result.snapshot_path)
    assert splits.snapshot_version == result.snapshot_version
    total = sum(len(getattr(splits, s)) for s in ("train", "validation", "holdout"))
    snap = pd.read_parquet(result.snapshot_path)
    assert total == len(snap)  # không mất dòng, không trùng


def test_x_y_returns_committed_columns_and_target(snapshot):
    tmp, result = snapshot
    splits = load_snapshot_splits(result.snapshot_path)
    X, y = splits.x_y("train")
    assert list(X.columns) == list(MODEL_FEATURE_COLUMNS)
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y) == len(splits.train)


def test_model_feature_columns_cover_catalog_and_flags():
    behavioral = {f.name for f in FEATURE_CATALOG} - {"module_presentation"}
    assert behavioral <= set(MODEL_FEATURE_COLUMNS)
    assert "module_presentation" in MODEL_FEATURE_COLUMNS
    assert {"imd_band_missing", "insufficient_data"} <= set(MODEL_FEATURE_COLUMNS)
    assert TARGET_COLUMN not in MODEL_FEATURE_COLUMNS  # target không lọt vào X


def test_unknown_split_rejected(snapshot):
    tmp, result = snapshot
    splits = load_snapshot_splits(result.snapshot_path)
    with pytest.raises(ValueError, match="unknown_split"):
        splits.frame("demo")


def test_missing_ledger_fails_loudly(snapshot):
    tmp, result = snapshot
    bare = tmp / "bare.parquet"
    pd.read_parquet(result.snapshot_path).to_parquet(bare, index=False)
    with pytest.raises(FileNotFoundError, match="missing_ledger"):
        load_snapshot_splits(bare)


def test_tampered_snapshot_content_is_rejected(snapshot):
    tmp, result = snapshot
    path = tmp / "tampered-content.parquet"
    frame = pd.read_parquet(result.snapshot_path)
    frame.loc[0, "n_clicks_before_cutoff"] += 1
    frame.to_parquet(path, index=False)
    (tmp / "tampered-content.ledger.json").write_text(result.ledger_path.read_text())
    (tmp / "tampered-content.success.json").write_text(
        (tmp / "snapshot.success.json").read_text()
    )
    with pytest.raises(ValueError, match="content_hash_mismatch"):
        load_snapshot_splits(path)


def test_legacy_ledger_is_rejected_with_rebuild_message(snapshot):
    tmp, result = snapshot
    path = tmp / "legacy.parquet"
    pd.read_parquet(result.snapshot_path).to_parquet(path, index=False)
    (tmp / "legacy.ledger.json").write_text(json.dumps({"snapshot_version": "snapshot-v1:x"}))
    (tmp / "legacy.success.json").write_text("{}")
    with pytest.raises(ValueError, match="unsupported_snapshot_schema"):
        load_snapshot_splits(path)


def test_tampered_ledger_rules_are_rejected(snapshot):
    tmp, result = snapshot
    path = tmp / "tampered-ledger.parquet"
    pd.read_parquet(result.snapshot_path).to_parquet(path, index=False)
    ledger = json.loads(result.ledger_path.read_text())
    ledger["split_rules"]["2014J"] = "validation"
    (tmp / "tampered-ledger.ledger.json").write_text(json.dumps(ledger))
    (tmp / "tampered-ledger.success.json").write_text(
        (tmp / "snapshot.success.json").read_text()
    )
    with pytest.raises(ValueError, match="ledger_metadata_invalid"):
        load_snapshot_splits(path)


def test_out_of_domain_split_fails_loudly(snapshot):
    tmp_path, result = snapshot
    df = pd.read_parquet(result.snapshot_path)
    df.loc[0, "split"] = "demo"
    path = tmp_path / "bad.parquet"
    df.to_parquet(path, index=False)
    (tmp_path / "bad.ledger.json").write_text(result.ledger_path.read_text())
    (tmp_path / "bad.success.json").write_text(
        (tmp_path / "snapshot.success.json").read_text()
    )
    with pytest.raises(ValueError, match="split_out_of_domain"):
        load_snapshot_splits(path)
