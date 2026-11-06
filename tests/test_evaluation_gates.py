"""Evaluation gates — data gate, ML gate, publish ATOMIC (eval-report.v1).

Fixture tái dùng: snapshot nhỏ từ pipeline thật (test_loading) cho data gate,
snapshot synthetic 300 dòng/split cho ML gate (cần đủ mẫu cho PSI/fairness).
Model train nhanh đúng như tests/test_training.py: build_preprocessing_artifact
+ train_logistic_baseline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learnguard.data.features import SPLIT_RULES, build_feature_snapshot
from learnguard.evaluation.data_gate import run_data_gate
from learnguard.evaluation.ml_gate import run_ml_gate
from learnguard.evaluation.scorecard import (
    EVAL_SCHEMA_VERSION,
    publish_evaluation_run,
)
from learnguard.ml.preprocessing import _file_hash, build_preprocessing_artifact
from learnguard.ml.training import (
    LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION,
    model_contract_hash,
    train_logistic_baseline,
)
from tests.test_loading import _tables_all_presentations
from tests.test_features import write_canonical

CHECK_IDS_DATA = (
    "snapshot_schema", "ledger_consistency", "split_integrity",
    "label_balance", "missing_rate", "duplicate_keys", "drift_vs_baseline",
)
CHECK_IDS_ML = (
    "artifact_integrity", "validation_metrics", "alert_budget",
    "shap_additivity", "fairness_slices", "regression_vs_baseline",
)


# ---- fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_snapshot(tmp_path_factory):
    """Snapshot thật sinh từ pipeline (nhỏ) — ledger đầy đủ, 3 split."""
    tmp = tmp_path_factory.mktemp("evalgates-pipeline")
    processed = write_canonical(tmp, _tables_all_presentations())
    result = build_feature_snapshot(processed, tmp / "snapshot.parquet", cutoff_day=27)
    return tmp, result.snapshot_path


@pytest.fixture(scope="module")
def clean_snapshot(tmp_path_factory):
    """Snapshot synthetic sạch (300 dòng/split) — mọi data check đều pass."""
    tmp = tmp_path_factory.mktemp("evalgates-clean")
    return _write_synthetic_snapshot(tmp, _synthetic_snapshot_frame())


def _synthetic_snapshot_frame(n_per_split: int = 300, seed: int = 7) -> pd.DataFrame:
    """Snapshot synthetic đủ lớn cho PSI/fairness, label có tín hiệu với features."""
    rng = np.random.default_rng(seed)
    split_rules = {"train": "2013B", "validation": "2014B", "holdout": "2014J"}
    frames: list[pd.DataFrame] = []
    next_id = 1000
    for split, presentation in split_rules.items():
        ability = rng.normal(0.0, 1.0, n_per_split)
        clicks = np.clip(40 + 25 * ability + rng.normal(0, 8, n_per_split), 0, None)
        label = (ability + rng.normal(0, 0.6, n_per_split) > 0).astype(int)
        imd = np.where(rng.random(n_per_split) < 0.05, None, "10-20")
        frames.append(pd.DataFrame({
            "code_module": "AAA",
            "code_presentation": presentation,
            "id_student": np.arange(next_id, next_id + n_per_split),
            "gender": np.where(rng.random(n_per_split) < 0.5, "M", "F"),
            "region": "Wales",
            "highest_education": "A Level or Equivalent",
            "imd_band": imd,
            "age_band": "0-35",
            "num_of_prev_attempts": 0,
            "studied_credits": 60,
            "disability": "N",
            "imd_band_missing": pd.isna(imd),
            "module_presentation": f"AAA-{presentation}",
            "n_clicks_before_cutoff": clicks,
            "click_trend_w3_w4": rng.normal(0, 5, n_per_split),
            "n_submitted_before_cutoff": np.clip(3 + 2 * ability, 0, None),
            "avg_score_submitted": np.clip(70 + 15 * ability, 0, 100),
            "insufficient_data": False,
            "final_result": np.where(label == 1, "Fail", "Pass"),
            "label": label,
            "split": split,
        }))
        next_id += n_per_split
    return pd.concat(frames, ignore_index=True)


def _write_synthetic_snapshot(directory: Path, frame: pd.DataFrame,
                              name: str = "snapshot.parquet") -> Path:
    """Ghi parquet + ledger khớp hợp đồng (cutoff 27, feature_list_hash thật)."""
    from learnguard.data.features import _feature_list_hash

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_parquet(path, index=False)
    source_hash = "ab" * 32
    version = f"snapshot-v1:cutoff27:{source_hash[:8]}:{_feature_list_hash()[:8]}"
    (path.parent / f"{path.stem}.ledger.json").write_text(json.dumps({
        "cutoff_day": 27, "snapshot_version": version, "source_hash": source_hash,
        "feature_list_hash": _feature_list_hash(),
        "split_rules": dict(SPLIT_RULES),
    }))
    return path


@pytest.fixture(scope="module")
def ml_stack(tmp_path_factory):
    """Snapshot synthetic + preprocessing + model v2 train trên đó."""
    tmp = tmp_path_factory.mktemp("evalgates-ml")
    snapshot = _write_synthetic_snapshot(tmp, _synthetic_snapshot_frame())
    pre = build_preprocessing_artifact(snapshot, tmp / "artifacts", "pre")
    model = train_logistic_baseline(snapshot, pre.artifact_dir, tmp / "artifacts", "model")
    return tmp, snapshot, pre, model


@pytest.fixture(scope="module")
def v1_artifact(ml_stack, tmp_path_factory):
    """Bản copy legacy v1 hợp lệ của model v2 (đổi schema + contract hash)."""
    tmp, snapshot, pre, model = ml_stack
    dest = tmp_path_factory.mktemp("evalgates-v1") / "model-v1"
    shutil.copytree(model.artifact_dir, dest)
    (dest / "shap_background.npz").unlink()
    manifest = json.loads((dest / "model.manifest.json").read_text())
    manifest["schema_version"] = LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION
    manifest["model_contract_hash"] = model_contract_hash(
        LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION,
    )
    manifest.pop("explanation", None)
    (dest / "model.manifest.json").write_text(json.dumps(manifest, indent=2))
    marker_path = dest / "model.success.json"
    marker = json.loads(marker_path.read_text())
    marker["schema_version"] = LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION
    marker["manifest_sha256"] = _file_hash(dest / "model.manifest.json")
    marker.pop("background_sha256", None)
    marker_path.write_text(json.dumps(marker, indent=2))
    return dest


def _check_ids(card) -> list[str]:
    return [check.id for check in card.checks]


def _status_by_id(card) -> dict[str, str]:
    return {check.id: check.status for check in card.checks}


# ---- Gate 1: data gate -------------------------------------------------------

def test_data_gate_happy_path_schema_and_exit(clean_snapshot):
    snapshot = clean_snapshot
    card = run_data_gate(snapshot, None, "eval-data-test-1")
    assert _check_ids(card) == list(CHECK_IDS_DATA)
    statuses = _status_by_id(card)
    assert statuses["drift_vs_baseline"] == "skipped"
    assert all(status == "pass" for name, status in statuses.items()
               if name != "drift_vs_baseline")
    assert card.summary() == {"total": 7, "pass": 6, "warn": 0, "fail": 0, "skipped": 1}
    # schema JSON đúng hợp đồng eval-report.v1
    payload = card.to_dict()
    assert payload["schema_version"] == EVAL_SCHEMA_VERSION
    assert payload["subject"] == "data"
    assert payload["run_id"] == "eval-data-test-1"
    assert payload["inputs"] == {"snapshot": str(snapshot), "baseline_snapshot": None}
    for check in payload["checks"]:
        assert set(check) == {"id", "name", "status", "threshold", "observed", "details"}
    assert set(payload["summary"]) == {"total", "pass", "warn", "fail", "skipped"}


def test_data_gate_unknown_presentation_fails(pipeline_snapshot, tmp_path):
    tmp, snapshot = pipeline_snapshot
    frame = pd.read_parquet(snapshot)
    frame.loc[frame.index[0], "code_presentation"] = "2099X"
    broken = _write_synthetic_snapshot(tmp_path, frame, "unknown-pres.parquet")
    card = run_data_gate(broken, None, "eval-data-test-2")
    check = next(c for c in card.checks if c.id == "split_integrity")
    assert check.status == "fail"
    assert check.observed["unknown_presentations"] == ["2099X"]


def test_data_gate_label_imbalance_fails(pipeline_snapshot, tmp_path):
    tmp, snapshot = pipeline_snapshot
    frame = pd.read_parquet(snapshot)
    frame.loc[frame["split"] == "train", "label"] = 0
    frame.loc[frame["split"] == "validation", "label"] = 1
    broken = _write_synthetic_snapshot(tmp_path, frame, "imbalance.parquet")
    card = run_data_gate(broken, None, "eval-data-test-3")
    check = next(c for c in card.checks if c.id == "label_balance")
    assert check.status == "fail"
    assert check.observed["train_validation_delta"] > 0.20


def test_data_gate_duplicate_keys_fail(pipeline_snapshot, tmp_path):
    tmp, snapshot = pipeline_snapshot
    frame = pd.read_parquet(snapshot)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    broken = _write_synthetic_snapshot(tmp_path, frame, "dupes.parquet")
    card = run_data_gate(broken, None, "eval-data-test-4")
    check = next(c for c in card.checks if c.id == "duplicate_keys")
    assert check.status == "fail"
    assert check.observed["duplicate_key_rows"] == 2


def test_data_gate_drift_with_baseline(tmp_path):
    frame = _synthetic_snapshot_frame()
    current = _write_synthetic_snapshot(tmp_path / "cur", frame, "current.parquet")
    # baseline giống hệt → PSI ~ 0 → pass
    baseline_same = _write_synthetic_snapshot(tmp_path / "same", frame, "base.parquet")
    card = run_data_gate(current, baseline_same, "eval-data-test-5")
    check = next(c for c in card.checks if c.id == "drift_vs_baseline")
    assert check.status == "pass"
    assert check.observed["max_psi"] < 0.10
    # baseline phân phối n_clicks lệch hẳn → PSI vượt ngưỡng FAIL
    drifted = frame.copy()
    drifted["n_clicks_before_cutoff"] = drifted["n_clicks_before_cutoff"] * 5
    baseline_drift = _write_synthetic_snapshot(tmp_path / "drift", drifted, "base.parquet")
    card2 = run_data_gate(current, baseline_drift, "eval-data-test-6")
    check2 = next(c for c in card2.checks if c.id == "drift_vs_baseline")
    assert check2.status == "fail"
    assert check2.observed["max_psi"] > 0.20
    assert len(check2.observed["psi_by_feature_top5"]) == 5


# ---- Gate 2: ML gate ---------------------------------------------------------

def test_ml_gate_happy_path(ml_stack):
    tmp, snapshot, pre, model = ml_stack
    card = run_ml_gate(
        model.artifact_dir, pre.artifact_dir, snapshot, "eval-ml-test-1",
    )
    assert _check_ids(card) == list(CHECK_IDS_ML)
    statuses = _status_by_id(card)
    assert statuses["artifact_integrity"] == "pass"
    assert statuses["validation_metrics"] == "pass"
    assert statuses["alert_budget"] == "pass"
    assert statuses["shap_additivity"] == "pass"
    assert statuses["fairness_slices"] == "pass"
    assert statuses["regression_vs_baseline"] == "skipped"
    assert card.summary()["fail"] == 0
    assert card.subject == "ml"


def test_ml_gate_min_roc_auc_fail(ml_stack):
    tmp, snapshot, pre, model = ml_stack
    card = run_ml_gate(
        model.artifact_dir, pre.artifact_dir, snapshot, "eval-ml-test-2",
        min_roc_auc=0.99,
    )
    check = next(c for c in card.checks if c.id == "validation_metrics")
    assert check.status == "fail"


def test_ml_gate_v1_legacy_warn_and_shap_skipped(ml_stack, v1_artifact):
    tmp, snapshot, pre, model = ml_stack
    card = run_ml_gate(
        v1_artifact, pre.artifact_dir, snapshot, "eval-ml-test-3",
    )
    statuses = _status_by_id(card)
    assert statuses["artifact_integrity"] == "warn"
    assert statuses["shap_additivity"] == "skipped"
    # các check metrics vẫn chạy được trên model legacy
    assert statuses["validation_metrics"] in {"pass", "warn", "fail"}
    integrity = next(c for c in card.checks if c.id == "artifact_integrity")
    assert "coef×x" in integrity.details


def test_ml_gate_regression_vs_baseline(ml_stack):
    tmp, snapshot, pre, model = ml_stack
    card = run_ml_gate(
        model.artifact_dir, pre.artifact_dir, snapshot, "eval-ml-test-4",
        baseline_model_dir=model.artifact_dir,
    )
    check = next(c for c in card.checks if c.id == "regression_vs_baseline")
    assert check.status == "pass"
    assert check.observed["delta"] == pytest.approx(0.0, abs=1e-6)


# ---- publish ATOMIC ----------------------------------------------------------

def test_publish_run_id_conflict_raises_file_exists(clean_snapshot, tmp_path):
    artifacts = tmp_path / "artifacts"
    card = run_data_gate(clean_snapshot, None, "eval-data-conflict")
    publish_evaluation_run(artifacts, "eval-data-conflict", [card])
    with pytest.raises(FileExistsError, match="evaluation_run_id_exists"):
        publish_evaluation_run(artifacts, "eval-data-conflict", [card])


def test_publish_atomic_layout_and_success_hash(clean_snapshot, tmp_path):
    artifacts = tmp_path / "artifacts"
    card = run_data_gate(clean_snapshot, None, "eval-data-layout")
    run_dir = publish_evaluation_run(artifacts, "eval-data-layout", [card])
    assert run_dir == artifacts / "evaluations" / "eval-data-layout"
    scorecard = json.loads((run_dir / "eval_data.json").read_text())
    assert scorecard["schema_version"] == EVAL_SCHEMA_VERSION
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["subjects"] == ["data"]
    success = json.loads((run_dir / "success.json").read_text())
    assert success["manifest_sha256"] == _file_hash(run_dir / "manifest.json")
    assert manifest["scorecards"]["eval_data.json"] == _file_hash(run_dir / "eval_data.json")
    assert not any(path.name.startswith(".") for path in artifacts.iterdir())
