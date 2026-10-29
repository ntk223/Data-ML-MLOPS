"""Test M3 — quality gates: schema, missingness, volume, PSI (drift)."""

import json
import numpy as np
import pandas as pd
import pytest

from learnguard.data.quality import DataQualityGate, QualityThresholds, validate_snapshot
from learnguard.data.snapshot_contract import (
    COHORT_RULES, DERIVED_RULES, HASH_ALGORITHM, SNAPSHOT_COLUMNS,
    SNAPSHOT_SCHEMA_VERSION, SPLIT_RULES, contract_hash, expected_snapshot_version,
    feature_rules, snapshot_content_hash,
)

KEYS = ["code_module", "code_presentation", "id_student"]


def make_snapshot(n: int = 100, clicks_scale: float = 1.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "code_module": rng.choice(["AAA", "BBB"], n),
            "code_presentation": "2013B",
            "id_student": np.arange(n),
            "gender": rng.choice(["M", "F"], n),
            "region": rng.choice(["Scotland", "London Region", "Wales"], n),
            "highest_education": "A Level or Equivalent",
            "imd_band": rng.choice(["20-30%", "40-50%", None], n, p=[0.5, 0.45, 0.05]),
            "age_band": "0-35",
            "num_of_prev_attempts": rng.integers(0, 3, n),
            "studied_credits": rng.integers(30, 200, n),
            "disability": rng.choice(["Y", "N"], n, p=[0.1, 0.9]),
            "imd_band_missing": False,
            "module_presentation": "",  # đồng bộ từ khóa ngay dưới
            "n_clicks_before_cutoff": (rng.gamma(2.0, 170.0, n) * clicks_scale).astype(int),
            "active_days_before_cutoff": rng.integers(0, 20, n),
            "click_trend_w3_w4": rng.normal(0, 50, n).astype(int),
            "n_submitted_before_cutoff": rng.integers(0, 3, n),
            "avg_score_submitted": rng.normal(72, 15, n).clip(0, 100),
            "insufficient_data": False,
            "final_result": rng.choice(["Pass", "Fail", "Withdrawn", "Distinction"], n),
            "split": rng.choice(["train", "validation", "holdout"], n),
        }
    )
    df["label"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    splits = np.resize(np.array(["train", "validation", "holdout"]), n)
    df["split"] = splits
    df["code_presentation"] = pd.Series(splits).map({
        "train": "2013B", "validation": "2014B", "holdout": "2014J",
    })
    df["module_presentation"] = df["code_module"] + "-" + df["code_presentation"]
    df["imd_band_missing"] = df["imd_band"].isna()
    return df[list(SNAPSHOT_COLUMNS)]


def save(tmp_path, name, df):
    path = tmp_path / f"{name}.parquet"
    df.to_parquet(path, index=False)
    source_hash = "a" * 64
    content_hash = snapshot_content_hash(df) if list(df.columns) == list(SNAPSHOT_COLUMNS) else ""
    ledger = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "cutoff_day": 27,
        "contract_hash": contract_hash(),
        "feature_list_hash": contract_hash(),
        "feature_rules": feature_rules(),
        "cohort_rules": COHORT_RULES,
        "derived_rules": DERIVED_RULES,
        "split_rules": SPLIT_RULES,
        "hash_algorithm": HASH_ALGORITHM,
        "source_hash": source_hash,
        "snapshot_version": expected_snapshot_version(source_hash),
        "output_schema": list(SNAPSHOT_COLUMNS),
        "row_count": len(df),
        "snapshot_content_hash": content_hash,
        "input_hashes": {"fixture": "b" * 64},
        "input_content_hashes": {"fixture": "c" * 64},
    }
    (tmp_path / f"{name}.ledger.json").write_text(json.dumps(ledger))
    (tmp_path / f"{name}.success.json").write_text(json.dumps({
        "snapshot_version": ledger["snapshot_version"],
        "snapshot_content_hash": content_hash,
    }))
    return path


def test_valid_snapshot_passes_all_gates(tmp_path):
    report = validate_snapshot(save(tmp_path, "s", make_snapshot()))
    assert report.status == "ok", report.errors
    assert report.row_count == 100
    assert report.volume_ratio is None  # không có baseline → không đo volume/PSI
    assert report.psi_by_feature == {}


def test_missing_label_column_detected(tmp_path):
    df = make_snapshot().drop(columns=["label"])
    report = validate_snapshot(save(tmp_path, "s", df))
    assert report.status == "failed"
    assert any(e.startswith("schema_missing_columns") for e in report.errors)


def test_label_out_of_domain_detected(tmp_path):
    df = make_snapshot()
    df.loc[0, "label"] = 5
    report = validate_snapshot(save(tmp_path, "s", df))
    assert "schema_label_out_of_domain" in report.errors


def test_duplicate_keys_detected(tmp_path):
    df = pd.concat([make_snapshot(), make_snapshot(n=100).iloc[[0]]], ignore_index=True)
    report = validate_snapshot(save(tmp_path, "s", df))
    assert any(e.startswith("schema_duplicate_keys") for e in report.errors)


def test_wrong_dtype_and_column_order_are_detected(tmp_path):
    wrong_dtype = make_snapshot()
    wrong_dtype["id_student"] = wrong_dtype["id_student"].astype(str)
    report = validate_snapshot(save(tmp_path, "dtype", wrong_dtype))
    assert "schema_dtype:id_student:expected=integer" in report.errors

    reordered = make_snapshot()[list(reversed(SNAPSHOT_COLUMNS))]
    report = validate_snapshot(save(tmp_path, "order", reordered))
    assert "schema_column_order_mismatch" in report.errors


def test_active_days_must_be_nonnegative_integer(tmp_path):
    df = make_snapshot()
    df.loc[0, "active_days_before_cutoff"] = -1
    report = validate_snapshot(save(tmp_path, "negative", df))
    assert "schema_negative_value:active_days_before_cutoff" in report.errors


def test_missingness_gate_catches_high_null_column(tmp_path):
    df = make_snapshot()
    df["region"] = None  # 100% null > ngưỡng 0.70
    report = validate_snapshot(save(tmp_path, "s", df))
    assert report.status == "failed"
    assert any(e.startswith("missingness_not_allowed:region") for e in report.errors)
    assert report.null_fraction_by_feature["region"] == 1.0


def test_volume_gate_bounds(tmp_path):
    prev = save(tmp_path, "prev", make_snapshot(n=100, seed=1))
    ok = validate_snapshot(save(tmp_path, "cur_ok", make_snapshot(n=95, seed=2)), prev)
    assert ok.status == "ok" and ok.volume_ratio == 0.95

    collapsed = validate_snapshot(save(tmp_path, "cur_bad", make_snapshot(n=5, seed=3)), prev)
    assert any(e.startswith("volume_out_of_bounds") for e in collapsed.errors)
    assert collapsed.status == "failed"


def test_explicit_missing_baseline_fails_instead_of_becoming_not_applicable(tmp_path):
    report = validate_snapshot(
        save(tmp_path, "current", make_snapshot()), tmp_path / "missing.parquet"
    )
    assert report.status == "failed"
    assert any(error.startswith("baseline_missing") for error in report.errors)
    assert report.gate_status["volume"] == "failed"
    assert report.gate_status["psi"] == "failed"


def test_psi_catches_distribution_drift(tmp_path):
    prev = save(tmp_path, "prev", make_snapshot(n=500, clicks_scale=1.0, seed=1))
    same = validate_snapshot(save(tmp_path, "cur_same", make_snapshot(n=500, clicks_scale=1.0, seed=7)), prev)
    assert same.status == "ok", same.errors
    assert same.psi_by_feature["n_clicks_before_cutoff"] < 0.2

    drifted = validate_snapshot(
        save(tmp_path, "cur_drift", make_snapshot(n=500, clicks_scale=100.0, seed=7)), prev
    )
    assert drifted.psi_by_feature["n_clicks_before_cutoff"] > 0.2
    assert any(e.startswith("psi_exceeded:n_clicks_before_cutoff") for e in drifted.errors)
    assert drifted.status == "failed"


def test_psi_categorical_detects_region_shift(tmp_path):
    prev = make_snapshot(n=500, seed=1)
    cur = make_snapshot(n=500, seed=2)
    cur["region"] = "London Region"  # 1 giá trị nuốt trọn → trôi cực đại
    report = validate_snapshot(save(tmp_path, "cur", cur), save(tmp_path, "prev", prev))
    assert report.psi_by_feature["region"] > 0.2
    assert any(e.startswith("psi_exceeded:region") for e in report.errors)


@pytest.mark.parametrize("missing_column", ["n_clicks_before_cutoff", "region"])
def test_psi_is_skipped_when_current_schema_is_invalid(tmp_path, missing_column):
    baseline = save(tmp_path, "baseline", make_snapshot(n=500, seed=1))
    malformed = make_snapshot(n=500, seed=2).drop(columns=[missing_column])

    report = validate_snapshot(save(tmp_path, "malformed", malformed), baseline)

    assert report.status == "failed"
    assert report.gate_status["schema"] == "failed"
    assert report.gate_status["psi"] == "not_applicable"
    assert report.psi_by_feature == {}
    assert any(
        error.startswith("schema_missing_columns") for error in report.errors
    )


def test_custom_thresholds_are_respected(tmp_path):
    strict = DataQualityGate(QualityThresholds(max_null_fraction=0.0))
    df = make_snapshot()
    df.loc[0, "region"] = None  # 1% null — default 0.70 cho qua, strict phải chặn
    report = strict.validate_snapshot(save(tmp_path, "s", df))
    assert report.status == "failed"
    assert any(e.startswith("missingness_not_allowed") for e in report.errors)
