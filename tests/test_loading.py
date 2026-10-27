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


def test_out_of_domain_split_fails_loudly(tmp_path):
    df = pd.DataFrame({
        "code_module": ["AAA"], "code_presentation": ["2013B"], "id_student": [1],
        "label": [1], "split": ["demo"],
    })
    path = tmp_path / "bad.parquet"
    df.to_parquet(path, index=False)
    (tmp_path / "bad.ledger.json").write_text(json.dumps({"snapshot_version": "x"}))
    with pytest.raises(ValueError, match="split_out_of_domain"):
        load_snapshot_splits(path)
