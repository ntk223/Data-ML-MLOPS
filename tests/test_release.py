"""Validation-only evaluation, immutable release, and release scoring tests."""

import json
import shutil
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from learnguard.cli import build_parser, main
from learnguard.ml.benchmark import run_benchmark
from learnguard.ml.preprocessing import _file_hash
from learnguard.ml.release import (
    build_model_release,
    build_validation_evaluation_artifact,
    load_and_verify_model_release,
)
from learnguard.ml.scoring import score_release_snapshot
from learnguard.ml.training import train_logistic_baseline
from tests.test_preprocessing import preprocessing_run


@pytest.fixture(scope="module")
def release_run(preprocessing_run):
    root, snapshot, preprocessor = preprocessing_run
    model = train_logistic_baseline(
        snapshot, preprocessor.artifact_dir, root / "release-artifacts", "model"
    )
    benchmark = run_benchmark(
        snapshot, preprocessor.artifact_dir, root / "release-artifacts", "validation",
        evaluation_scope="validation_only",
    )
    evaluation = build_validation_evaluation_artifact(
        snapshot, benchmark.artifact_dir, model.artifact_dir,
        preprocessor.artifact_dir, root / "release-artifacts", "evaluation",
    )
    release = build_model_release(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        evaluation.artifact_dir, root / "release-artifacts", "production",
    )
    return root, snapshot, preprocessor, model, benchmark, evaluation, release


def test_evaluation_artifact_has_confusion_matrices_and_no_holdout(release_run):
    _, _, _, _, _, evaluation, _ = release_run
    report = json.loads((evaluation.artifact_dir / "evaluation_report.json").read_text())
    policy = json.loads((evaluation.artifact_dir / "threshold_policy.json").read_text())
    assert report["evaluation_scope"] == "validation_only"
    assert report["holdout_accessed"] is False
    assert report["holdout_reused"] is False
    assert report["production_model"] == "logistic_baseline"
    assert policy["population"] == "all_finite_validation_scores"
    assert policy["target_fraction"] == pytest.approx(0.2)
    assert 0 <= policy["production_threshold"] <= 1
    assert policy["max_f1_comparator"]["threshold"] != policy["production_threshold"]
    csv_files = sorted((evaluation.artifact_dir / "confusion_matrices").glob("*.csv"))
    png_files = sorted((evaluation.artifact_dir / "plots").glob("confusion_matrix*.png"))
    assert len(csv_files) == 27
    assert len(png_files) == 27
    assert all(path.stat().st_size > 0 for path in png_files)
    for path in csv_files:
        matrix = pd.read_csv(path)
        assert set(matrix.columns) == {
            "candidate", "policy", "threshold", "actual_label",
            "predicted_label", "count",
        }
        assert len(matrix) == 4
        assert int(matrix["count"].sum()) == report["production_metrics"]["confusion_matrix"][0][0] + report["production_metrics"]["confusion_matrix"][0][1] + report["production_metrics"]["confusion_matrix"][1][0] + report["production_metrics"]["confusion_matrix"][1][1]
    names = [str(path.relative_to(evaluation.artifact_dir))
             for path in evaluation.artifact_dir.rglob("*") if path.is_file()]
    assert not any("holdout" in name or "id_student" in name for name in names)


def test_release_is_relocatable_verified_and_logistic_only(release_run, tmp_path):
    _, snapshot, _, model, _, _, release = release_run
    copied = tmp_path / "moved-release"
    shutil.copytree(release.artifact_dir, copied)
    loaded = load_and_verify_model_release(copied, snapshot)
    assert loaded.release_id == "production"
    assert loaded.model_version == model.model_semantic_version
    assert loaded.holdout_access_policy == "forbidden_prior_consumed"
    manifest = json.loads((copied / "release.manifest.json").read_text())
    assert manifest["production_release_eligible"] is True
    assert manifest["candidate_promotion_eligible"] is False
    assert manifest["schema_version"] == "model-release.v4"
    assert manifest["release_status"] == "production"
    assert manifest["database_write_allowed"] is True
    assert manifest["model_artifact_dir"] == "model"
    assert manifest["preprocessing_artifact_dir"] == "preprocessor"


def test_release_tampering_is_rejected(release_run, tmp_path):
    _, snapshot, _, _, _, _, release = release_run
    copied = tmp_path / "tampered-release"
    shutil.copytree(release.artifact_dir, copied)
    with (copied / "model/model.joblib").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact_payload_hash_mismatch"):
        load_and_verify_model_release(copied, snapshot)


@pytest.mark.parametrize("field,value,error", [
    ("production_threshold", 1.5, "model_release_threshold_invalid"),
    ("production_threshold", 0.7, "model_release_threshold_policy_mismatch"),
    ("holdout_access_policy", "allowed", "model_release_policy_invalid"),
    ("model_type", "sklearn.ensemble.RandomForestClassifier",
     "model_release_production_model_unsupported"),
])
def test_release_policy_tampering_is_rejected(
    release_run, tmp_path, field, value, error,
):
    _, snapshot, _, _, _, _, release = release_run
    copied = tmp_path / f"tampered-{field}"
    shutil.copytree(release.artifact_dir, copied)
    manifest_path = copied / "release.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    success_path = copied / "release.success.json"
    success = json.loads(success_path.read_text())
    success["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    success_path.write_text(json.dumps(success, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match=error):
        load_and_verify_model_release(copied, snapshot)


def test_release_scoring_uses_locked_threshold_and_sidecar(release_run):
    root, snapshot, _, _, _, _, release = release_run
    batch = root / "release-artifacts" / "prediction-batches" / "release-score"
    report = score_release_snapshot(
        snapshot, release.artifact_dir, batch / "predictions.parquet",
        "release-score", ("validation",),
    )
    loaded = load_and_verify_model_release(release.artifact_dir, snapshot)
    predictions = pd.read_parquet(report.output_path)
    assert report.release_id == release.release_id
    assert report.threshold == loaded.production_threshold
    assert set(predictions["threshold"]) == {loaded.production_threshold}
    labeled = predictions[predictions["predicted_label"].notna()]
    assert (
        labeled["predicted_label"].astype(int)
        == (labeled["risk_score"] >= loaded.production_threshold).astype(int)
    ).all()
    provenance = json.loads((batch / "release.provenance.json").read_text())
    success = json.loads((batch / "success.json").read_text())
    assert provenance["release_id"] == release.release_id
    assert success["release_provenance_sha256"] == _file_hash(
        batch / "release.provenance.json"
    )


def test_release_scoring_rejects_holdout(release_run):
    root, snapshot, _, _, _, _, release = release_run
    with pytest.raises(ValueError, match="scoring_release_holdout_forbidden"):
        score_release_snapshot(
            snapshot, release.artifact_dir,
            root / "release-artifacts/prediction-batches/forbidden/predictions.parquet",
            "forbidden", ("holdout",),
        )


def test_release_cli_contract_and_override_rejection(release_run, capsys):
    root, snapshot, _, _, _, _, release = release_run
    parser = build_parser()
    args = parser.parse_args([
        "ml", "score", "--snapshot", str(snapshot),
        "--release-artifact-dir", str(release.artifact_dir),
        "--run-id", "release-cli", "--split", "validation",
    ])
    assert args.release_artifact_dir == release.artifact_dir
    assert args.threshold is None
    code = main([
        "ml", "score", "--snapshot", str(snapshot),
        "--release-artifact-dir", str(release.artifact_dir),
        "--artifacts-dir", str(root / "release-artifacts"),
        "--run-id", "bad-release-cli", "--split", "validation",
        "--threshold", "0.5",
    ])
    assert code == 2
    assert "release_score_legacy_arguments_forbidden" in capsys.readouterr().err


def test_release_artifacts_are_not_overwritten(release_run):
    root, snapshot, preprocessor, model, benchmark, evaluation, release = release_run
    with pytest.raises(FileExistsError, match="evaluation_run_id_exists"):
        build_validation_evaluation_artifact(
            snapshot, benchmark.artifact_dir, model.artifact_dir,
            preprocessor.artifact_dir, root / "release-artifacts", "evaluation",
        )
    with pytest.raises(FileExistsError, match="model_release_id_exists"):
        build_model_release(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            evaluation.artifact_dir, root / "release-artifacts", release.release_id,
        )


@pytest.mark.parametrize("run_id", ["", "..", "../x", "nested/x", "/tmp/x"])
def test_release_commands_reject_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError, match="invalid_run_id"):
        build_validation_evaluation_artifact(
            "missing", "missing", "missing", "missing", tmp_path, run_id,
        )
    with pytest.raises(ValueError, match="invalid_run_id"):
        build_model_release(
            "missing", "missing", "missing", "missing", tmp_path, run_id,
        )


def test_release_evaluation_never_reads_snapshot_holdout(
    preprocessing_run, tmp_path, monkeypatch,
):
    root, snapshot, preprocessor = preprocessing_run
    model = train_logistic_baseline(snapshot, preprocessor.artifact_dir, tmp_path, "model")
    benchmark = run_benchmark(
        snapshot, preprocessor.artifact_dir, tmp_path, "benchmark",
        evaluation_scope="validation_only",
    )
    original = pd.read_parquet
    observed_snapshot_filters = []

    def spy(path, *args, **kwargs):
        if Path(path) == Path(snapshot):
            observed_snapshot_filters.append(kwargs.get("filters"))
            filters = kwargs.get("filters")
            assert filters == [("split", "in", ["train"])]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    build_validation_evaluation_artifact(
        snapshot, benchmark.artifact_dir, model.artifact_dir,
        preprocessor.artifact_dir, tmp_path, "evaluation",
    )
    assert observed_snapshot_filters
    assert all(item == [("split", "in", ["train"])]
               for item in observed_snapshot_filters)
