"""Test M3 — quality gates: schema, missingness, volume, PSI (drift)."""

import numpy as np
import pandas as pd

from learnguard.data.quality import DataQualityGate, QualityThresholds, validate_snapshot

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
            "module_presentation": rng.choice(["AAA-2013B", "BBB-2013B"], n),
            "n_clicks_before_cutoff": (rng.gamma(2.0, 170.0, n) * clicks_scale).astype(int),
            "click_trend_w3_w4": rng.normal(0, 50, n).astype(int),
            "n_submitted_before_cutoff": rng.integers(0, 3, n),
            "avg_score_submitted": rng.normal(72, 15, n).clip(0, 100),
            "insufficient_data": False,
            "final_result": rng.choice(["Pass", "Fail", "Withdrawn", "Distinction"], n),
            "split": rng.choice(["train", "validation", "holdout"], n),
        }
    )
    df["label"] = df["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    return df


def save(tmp_path, name, df):
    path = tmp_path / f"{name}.parquet"
    df.to_parquet(path, index=False)
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


def test_missingness_gate_catches_high_null_column(tmp_path):
    df = make_snapshot()
    df["region"] = None  # 100% null > ngưỡng 0.70
    report = validate_snapshot(save(tmp_path, "s", df))
    assert report.status == "failed"
    assert any(e.startswith("missingness_exceeded:region") for e in report.errors)
    assert report.null_fraction_by_feature["region"] == 1.0


def test_volume_gate_bounds(tmp_path):
    prev = save(tmp_path, "prev", make_snapshot(n=100, seed=1))
    ok = validate_snapshot(save(tmp_path, "cur_ok", make_snapshot(n=95, seed=2)), prev)
    assert ok.status == "ok" and ok.volume_ratio == 0.95

    collapsed = validate_snapshot(save(tmp_path, "cur_bad", make_snapshot(n=5, seed=3)), prev)
    assert any(e.startswith("volume_out_of_bounds") for e in collapsed.errors)
    assert collapsed.status == "failed"


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


def test_custom_thresholds_are_respected(tmp_path):
    strict = DataQualityGate(QualityThresholds(max_null_fraction=0.0))
    df = make_snapshot()
    df.loc[0, "region"] = None  # 1% null — default 0.70 cho qua, strict phải chặn
    report = strict.validate_snapshot(save(tmp_path, "s", df))
    assert report.status == "failed"
    assert any(e.startswith("missingness_exceeded") for e in report.errors)
