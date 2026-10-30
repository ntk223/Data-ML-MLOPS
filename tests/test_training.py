"""Logistic baseline tests that treat the GitHub data pipeline as read-only."""

import json
import shutil

import pytest

from learnguard.cli import build_parser, main
from learnguard.ml.preprocessing import build_preprocessing_artifact
from learnguard.ml.training import (
    LOGISTIC_BASELINE_PARAMS,
    load_logistic_model_artifact,
    model_contract_hash,
    train_logistic_baseline,
    verify_logistic_model_artifact,
)
from tests.test_preprocessing import preprocessing_run


@pytest.fixture(scope="module")
def model_run(preprocessing_run):
    root, snapshot, preprocessor = preprocessing_run
    result = train_logistic_baseline(
        snapshot, preprocessor.artifact_dir, root / "artifacts", "model"
    )
    return root, snapshot, preprocessor, result


def test_fixed_estimator_contract():
    assert LOGISTIC_BASELINE_PARAMS == {
        "penalty": "l2", "C": 1.0, "solver": "lbfgs", "class_weight": None,
        "max_iter": 2000, "tol": 1e-4, "fit_intercept": True,
        "random_state": 42, "n_jobs": None,
    }
    assert len(model_contract_hash()) == 64


def test_train_and_verify_round_trip(model_run):
    _, snapshot, preprocessor, result = model_run
    assert result.status == "ok"
    assert len(result.published_paths) == 4
    report = verify_logistic_model_artifact(
        result.artifact_dir, snapshot, preprocessor.artifact_dir
    )
    assert report.status == "ok"
    assert report.learned_classes == (0, 1)
    loaded = load_logistic_model_artifact(
        result.artifact_dir, snapshot, preprocessor.artifact_dir
    )
    assert not hasattr(loaded, "fit")


def test_manifest_has_no_evaluation_or_threshold(model_run):
    _, _, _, result = model_run
    manifest = json.loads((result.artifact_dir / "model.manifest.json").read_text())
    assert manifest["target_column"] == "label"
    assert manifest["snapshot_version"].startswith("snapshot-v1:cutoff27:")
    assert not ({"threshold", "validation_metrics", "holdout_metrics"} & set(manifest))


def test_identical_runs_are_semantically_equal(model_run):
    root, snapshot, preprocessor, first = model_run
    second = train_logistic_baseline(
        snapshot, preprocessor.artifact_dir, root / "artifacts", "model-2"
    )
    left = json.loads((first.artifact_dir / "model.manifest.json").read_text())
    right = json.loads((second.artifact_dir / "model.manifest.json").read_text())
    for field in (
        "model_semantic_version", "model_contract_hash",
        "training_matrix_semantic_hash", "training_target_semantic_hash",
        "coefficient_semantic_hash", "intercept_semantic_hash",
        "learned_state_hash", "train_probability_semantic_hash",
    ):
        assert left[field] == right[field]


@pytest.mark.parametrize("filename", ["model.joblib", "model.manifest.json", "model.training-report.json"])
def test_tampering_is_rejected(model_run, tmp_path, filename):
    _, snapshot, preprocessor, result = model_run
    copied = tmp_path / "tampered"
    shutil.copytree(result.artifact_dir, copied)
    with (copied / filename).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_logistic_model_artifact(copied, snapshot, preprocessor.artifact_dir)


@pytest.mark.parametrize("run_id", ["", "..", "../x", "nested/x", "/tmp/x"])
def test_unsafe_run_id_rejected(preprocessing_run, tmp_path, run_id):
    _, snapshot, preprocessor = preprocessing_run
    with pytest.raises(ValueError, match="invalid_run_id"):
        train_logistic_baseline(snapshot, preprocessor.artifact_dir, tmp_path, run_id)


def test_existing_run_is_not_overwritten(model_run):
    root, snapshot, preprocessor, _ = model_run
    with pytest.raises(FileExistsError):
        train_logistic_baseline(
            snapshot, preprocessor.artifact_dir, root / "artifacts", "model"
        )


def test_cli_model_commands_use_snapshot_not_data_run():
    parser = build_parser()
    args = parser.parse_args([
        "ml", "train-logistic", "--snapshot", "snapshot.parquet",
        "--preprocessing-artifact-dir", "pre", "--run-id", "model",
    ])
    assert args.command == "train-logistic"
    assert not hasattr(args, "data_run_dir")


def test_cli_failure_is_nonzero(tmp_path):
    assert main([
        "ml", "train-logistic", "--snapshot", str(tmp_path / "missing.parquet"),
        "--preprocessing-artifact-dir", str(tmp_path / "pre"),
        "--artifacts-dir", str(tmp_path / "artifacts"), "--run-id", "model",
    ]) == 2
