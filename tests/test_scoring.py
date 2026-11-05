"""M5 batch scoring tests: round-trip publish, pseudonymity, review routing."""

import json
import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learnguard.cli import build_parser, main
from learnguard.data.features import KEYS
from learnguard.data.loading import SPLITS, load_snapshot_splits
from learnguard.ml.scoring import STATS_FEATURE_COLUMNS
from learnguard.ml.contracts import load_prediction_batch_manifest
from learnguard.ml.preprocessing import _file_hash, _prepare_split
from learnguard.ml.scoring import (
    HIGH_UNCERTAINTY_THRESHOLD,
    PREDICTION_COLUMNS,
    BatchScorer,
    PredictionStatus,
    score_snapshot,
)
from learnguard.ml.model_interface import load_scoring_model
from learnguard.ml.training import load_logistic_model_artifact, train_logistic_baseline
from tests.test_preprocessing import preprocessing_run


@pytest.fixture(scope="module")
def model_run(preprocessing_run):
    root, snapshot, preprocessor = preprocessing_run
    result = train_logistic_baseline(
        snapshot, preprocessor.artifact_dir, root / "artifacts", "model"
    )
    return root, snapshot, preprocessor, result


@pytest.fixture(scope="module")
def score_run(model_run):
    root, snapshot, preprocessor, model = model_run
    batch_dir = root / "artifacts" / "prediction-batches" / "score"
    report = score_snapshot(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        batch_dir / "predictions.parquet", 0.5, "score",
    )
    return root, snapshot, preprocessor, model, report


def _expected_key(code_module: str, code_presentation: str, id_student: object) -> str:
    raw = f"learnguard:prediction-batch.v1:{code_module}:{code_presentation}:{id_student}"
    return sha256(raw.encode("utf-8")).hexdigest()[:16]


def test_score_round_trip_publishes_verified_batch(score_run):
    _, snapshot, _, model, report = score_run
    assert report.status == "ok"
    assert report.model_version == model.model_semantic_version
    assert report.threshold == 0.5
    assert report.input_rows == (
        report.scored_rows + report.manual_review_rows
        + report.insufficient_data_rows + report.no_prediction_rows
    )
    manifest_path = report.batch_dir / "manifest.json"
    manifest = load_prediction_batch_manifest(manifest_path)
    assert manifest.schema_version == "prediction-batch.v1"
    assert manifest.run_id == "score"
    assert manifest.model_version == report.model_version
    assert manifest.snapshot_version == report.snapshot_version
    assert manifest.cutoff_day == 27
    assert manifest.predictions_sha256 == _file_hash(report.output_path)
    assert manifest.predictions_path == report.output_path
    marker = json.loads((report.batch_dir / "success.json").read_text())
    assert marker["manifest_sha256"] == _file_hash(manifest_path)
    assert marker["predictions_sha256"] == manifest.predictions_sha256
    predictions = pd.read_parquet(report.output_path)
    splits = load_snapshot_splits(snapshot)
    expected_rows = sum(len(splits.frame(name)) for name in SPLITS)
    assert manifest.row_count == len(predictions) == expected_rows
    assert list(predictions.columns) == list(PREDICTION_COLUMNS)
    assert set(predictions["split"]) == set(SPLITS)


def test_features_json_carries_raw_stats_without_demographics(score_run):
    """features_json phải chứa đúng cột thống kê GỐC, không có demographics."""
    _, snapshot, _, _, report = score_run
    predictions = pd.read_parquet(report.output_path)
    assert "features_json" in predictions.columns
    splits = load_snapshot_splits(snapshot)
    holdout = splits.frame("holdout").sort_values(list(KEYS)).reset_index(drop=True)
    rows = predictions[predictions["split"] == "holdout"].sort_values(
        ["code_module", "code_presentation", "student_key"]
    ).reset_index(drop=True)
    for name in STATS_FEATURE_COLUMNS:
        features = json.loads(rows.loc[0, "features_json"])
        assert name in features
        if holdout.loc[0, name] is None or pd.isna(holdout.loc[0, name]):
            assert features[name] is None
        else:
            assert features[name] == float(holdout.loc[0, name])
    # Ranh giới PII: demographics không bao giờ rời snapshot.
    features = json.loads(rows.loc[0, "features_json"])
    for banned in ("gender", "region", "highest_education", "imd_band",
                   "age_band", "disability"):
        assert banned not in features


def test_pseudonymous_key_is_deterministic_and_non_identifying(score_run):
    _, snapshot, _, _, report = score_run
    predictions = pd.read_parquet(report.output_path)
    assert "id_student" not in predictions.columns
    assert predictions["student_key"].str.fullmatch(r"[0-9a-f]{16}").all()
    splits = load_snapshot_splits(snapshot)
    expected: list[str] = []
    for name in SPLITS:
        frame = splits.frame(name).sort_values(list(KEYS)).reset_index(drop=True)
        expected.extend(
            _expected_key(module, presentation, student)
            for module, presentation, student in zip(
                frame["code_module"], frame["code_presentation"], frame["id_student"]
            )
        )
    assert sorted(predictions["student_key"]) == sorted(expected)
    assert predictions["student_key"].nunique() == len(expected)
    # Same student in two presentations must yield two different keys.
    keys_101 = predictions[predictions["code_module"] == "BBB"]
    assert keys_101["student_key"].nunique() == keys_101["code_presentation"].nunique()


@pytest.mark.parametrize("risk,uncertainty,data_status,expected", [
    (0.9, 0.2, "ok", PredictionStatus.SCORED),
    (0.1, 0.2, "ok", PredictionStatus.SCORED),
    (0.5, 1.0, "ok", PredictionStatus.MANUAL_REVIEW),
    (0.97, HIGH_UNCERTAINTY_THRESHOLD, "ok", PredictionStatus.MANUAL_REVIEW),
    (0.9, HIGH_UNCERTAINTY_THRESHOLD - 0.1, "ok", PredictionStatus.SCORED),
    (0.9, 0.1, "insufficient_data", PredictionStatus.INSUFFICIENT_DATA),
])
def test_assign_review_state_routing(risk, uncertainty, data_status, expected):
    assert BatchScorer().assign_review_state(risk, 0.5, uncertainty, data_status) is expected


def test_insufficient_data_rows_keep_score_without_label(score_run):
    _, _, _, _, report = score_run
    predictions = pd.read_parquet(report.output_path)
    flagged = predictions[predictions["status"] == "insufficient_data"]
    assert not flagged.empty
    assert flagged["predicted_label"].isna().all()
    assert (flagged["review_reason"] == "insufficient_data").all()
    assert flagged["risk_score"].notna().all()
    assert flagged["uncertainty"].notna().all()
    assert report.insufficient_data_rows == len(flagged)
    uncertain = predictions[predictions["status"] == "manual_review"]
    assert (uncertain["review_reason"] == "high_uncertainty").all()
    assert (uncertain["uncertainty"] >= HIGH_UNCERTAINTY_THRESHOLD).all()
    scored = predictions[predictions["status"] == "scored"]
    assert scored["review_reason"].isna().all()
    assert report.manual_review_rows == len(uncertain)
    assert report.scored_rows == len(scored)


def test_explain_returns_sorted_top_signals(model_run):
    _, snapshot, preprocessor, model = model_run
    loaded = load_logistic_model_artifact(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    scoring_model = load_scoring_model(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    splits = load_snapshot_splits(snapshot)
    prepared = _prepare_split("train", splits.train)
    matrix = loaded._preprocessor.transform(prepared.X)
    signals = BatchScorer().explain(scoring_model, matrix[0])
    assert len(signals) == 5
    contributions = loaded._model.coef_[0] * matrix[0]
    expected_order = np.argsort(-np.abs(contributions), kind="stable")[:5]
    assert [signal.feature_name for signal in signals] == [loaded.feature_names[i] for i in expected_order]
    for signal, index in zip(signals, expected_order):
        assert signal.contribution == pytest.approx(float(contributions[index]))
        expected_direction = "increases_risk" if contributions[index] > 0 else "decreases_risk"
        assert signal.direction == expected_direction


def test_explain_rejects_wrong_space_row(model_run):
    _, snapshot, preprocessor, model = model_run
    loaded = load_logistic_model_artifact(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    scoring_model = load_scoring_model(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    scorer = BatchScorer()
    with pytest.raises(ValueError, match="scoring_explain_row_invalid"):
        scorer.explain(scoring_model, np.zeros(3))


def test_tampered_predictions_fail_checksum(score_run, tmp_path):
    _, _, _, _, report = score_run
    copied = tmp_path / "tampered"
    shutil.copytree(report.batch_dir, copied)
    with (copied / "predictions.parquet").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="prediction_batch_checksum_mismatch"):
        load_prediction_batch_manifest(copied / "manifest.json")


def test_threshold_boundary_labels_scored_row_positive(score_run, model_run):
    _, snapshot, preprocessor, model = model_run
    first = pd.read_parquet(score_run[4].output_path)
    target = float(first.loc[first["status"] == "scored", "risk_score"].max())
    batch_dir = model_run[0] / "artifacts" / "prediction-batches" / "score-boundary"
    report = score_snapshot(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        batch_dir / "predictions.parquet", target, "score-boundary",
    )
    assert report.threshold == target
    predictions = pd.read_parquet(report.output_path)
    boundary = predictions[predictions["risk_score"] == target]
    assert len(boundary) == 1
    assert int(boundary["predicted_label"].iloc[0]) == 1
    positive = predictions[
        (predictions["risk_score"] >= target) & (predictions["status"] != "insufficient_data")
    ]
    assert (positive["predicted_label"] == 1).all()


def test_existing_batch_is_not_overwritten(score_run):
    _, snapshot, preprocessor, model, report = score_run
    with pytest.raises(FileExistsError, match="prediction_batch_run_id_exists"):
        score_snapshot(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            report.output_path, 0.5, "score",
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_out_of_range_threshold_rejected(model_run, threshold):
    _, snapshot, preprocessor, model = model_run
    batch_dir = model_run[0] / "artifacts" / "prediction-batches" / "bad-threshold"
    with pytest.raises(ValueError, match="scoring_threshold_out_of_range"):
        score_snapshot(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            batch_dir / "predictions.parquet", threshold, "bad-threshold",
        )
    assert not batch_dir.parent.exists() or not batch_dir.exists()


def test_cli_score_command_parses_contract_args():
    parser = build_parser()
    args = parser.parse_args([
        "ml", "score", "--snapshot", "snapshot.parquet",
        "--model-artifact-dir", "model", "--preprocessing-artifact-dir", "pre",
        "--run-id", "score-1", "--threshold", "0.7",
    ])
    assert args.command == "score"
    assert args.model_artifact_dir == Path("model")
    assert args.preprocessing_artifact_dir == Path("pre")
    assert args.artifacts_dir == Path("artifacts")
    assert args.threshold == 0.7


def test_cli_score_end_to_end(score_run, capsys):
    _, snapshot, preprocessor, model, _ = score_run
    code = main([
        "ml", "score", "--snapshot", str(snapshot),
        "--model-artifact-dir", str(model.artifact_dir),
        "--preprocessing-artifact-dir", str(preprocessor.artifact_dir),
        "--artifacts-dir", str(score_run[0] / "artifacts"), "--run-id", "score-cli",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "ok"
    assert summary["input_rows"] == score_run[4].input_rows


def _subsample_run(model_run, run_id: str, max_students: int, seed: int = 42):
    """Score once with --max-students semantics into its own batch directory.

    Subsample requires exactly one split, so these runs score the validation
    split only (the intended production use case).
    """
    _, snapshot, preprocessor, model = model_run
    batch_dir = model_run[0] / "artifacts" / "prediction-batches" / run_id
    return score_snapshot(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        batch_dir / "predictions.parquet", 0.5, run_id,
        splits=("validation",),
        max_students=max_students, seed=seed,
    )


def test_subsample_same_seed_is_reproducible(model_run):
    """Same seed → cùng tập student_key qua 2 lần chạy độc lập."""
    first = _subsample_run(model_run, "sub-seed-a", 1, seed=42)
    second = _subsample_run(model_run, "sub-seed-b", 1, seed=42)
    keys_first = sorted(pd.read_parquet(first.output_path)["student_key"].unique())
    keys_second = sorted(pd.read_parquet(second.output_path)["student_key"].unique())
    assert keys_first == keys_second


def test_subsample_selects_exactly_n_or_all_students(model_run, score_run):
    """N < tổng → đúng N student unique; N > tổng → giữ nguyên toàn bộ."""
    full = pd.read_parquet(score_run[4].output_path)
    full = full[full["split"] == "validation"]
    total_students = int(full["student_key"].nunique())
    assert total_students > 1

    narrowed = _subsample_run(model_run, "sub-narrow", 1)
    narrowed_frame = pd.read_parquet(narrowed.output_path)
    assert narrowed_frame["student_key"].nunique() == 1
    assert narrowed_frame["student_key"].nunique() < total_students

    widened = _subsample_run(model_run, "sub-wide", total_students * 10)
    widened_frame = pd.read_parquet(widened.output_path)
    assert widened_frame["student_key"].nunique() == total_students
    assert len(widened_frame) == len(full)


def test_subsample_keeps_every_row_of_selected_students(model_run, score_run):
    """Mọi dòng prediction của student được chọn phải được giữ nguyên."""
    full = pd.read_parquet(score_run[4].output_path)
    full = full[full["split"] == "validation"]
    subsampled = pd.read_parquet(_subsample_run(model_run, "sub-rows", 1).output_path)
    full_counts = full.groupby("student_key").size()
    sub_counts = subsampled.groupby("student_key").size()
    assert set(sub_counts.index).issubset(set(full_counts.index))
    for student_key, count in sub_counts.items():
        assert count == full_counts[student_key]
    # Không có dòng nào ngoài các student được chọn.
    assert len(subsampled) == int(sub_counts.sum())


def test_subsample_manifest_records_audit_fields(model_run, score_run):
    """Manifest phải ghi lại đủ provenance subsample phục vụ audit."""
    full = pd.read_parquet(score_run[4].output_path)
    full = full[full["split"] == "validation"]
    report = _subsample_run(model_run, "sub-manifest", 1, seed=42)
    manifest = json.loads((report.batch_dir / "manifest.json").read_text())
    assert manifest["subsample"] == {
        "max_students": 1,
        "seed": 42,
        "n_students_before": int(full["student_key"].nunique()),
        "n_students_after": 1,
        "n_rows_after": int(len(pd.read_parquet(report.output_path))),
    }
    # Reader chuẩn vẫn đọc được manifest (bỏ qua trường lạ) và khớp payload.
    loaded = load_prediction_batch_manifest(report.batch_dir / "manifest.json")
    assert loaded.row_count == manifest["row_count"]


def test_without_max_students_manifest_has_no_subsample(score_run):
    """Không set --max-students → hành vi y như cũ (backward compat)."""
    report = score_run[4]
    manifest = json.loads((report.batch_dir / "manifest.json").read_text())
    assert "subsample" not in manifest


@pytest.mark.parametrize("splits", [None, ("train", "validation")])
def test_max_students_requires_single_split(model_run, splits):
    """--max-students chỉ hợp lệ với đúng 1 split (0 hoặc ≥2 split bị từ chối)."""
    _, snapshot, preprocessor, model = model_run
    batch_dir = model_run[0] / "artifacts" / "prediction-batches" / "sub-bad-split"
    with pytest.raises(ValueError, match="max_students_requires_single_split"):
        score_snapshot(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            batch_dir / "predictions.parquet", 0.5, "sub-bad-split",
            splits=splits, max_students=5,
        )
    assert not batch_dir.exists()


def test_max_students_must_be_positive(model_run):
    _, snapshot, preprocessor, model = model_run
    batch_dir = model_run[0] / "artifacts" / "prediction-batches" / "sub-zero"
    with pytest.raises(ValueError, match="max_students_invalid"):
        score_snapshot(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            batch_dir / "predictions.parquet", 0.5, "sub-zero",
            splits=("validation",), max_students=0,
        )
    assert not batch_dir.exists()


def test_max_students_forbidden_in_release_mode(model_run):
    """Release mode phải tái lập nguyên vẹn — subsample bị từ chối."""
    _, snapshot, preprocessor, model = model_run
    batch_dir = model_run[0] / "artifacts" / "prediction-batches" / "sub-release"
    with pytest.raises(ValueError, match="max_students_release_mode_forbidden"):
        BatchScorer().score(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            batch_dir / "predictions.parquet", 0.5, "sub-release",
            splits=("validation",),
            release_id="rel-1", release_manifest_sha256="a" * 64,
            max_students=5,
        )
    assert not batch_dir.exists()


def test_cli_rejects_max_students_in_release_mode(score_run):
    _, snapshot, _, _, _ = score_run
    code = main([
        "ml", "score", "--snapshot", str(snapshot),
        "--release-artifact-dir", str(score_run[0] / "artifacts" / "no-release"),
        "--artifacts-dir", str(score_run[0] / "artifacts"),
        "--run-id", "sub-release-cli",
        "--split", "validation", "--max-students", "5",
    ])
    assert code == 2


def test_cli_score_parses_subsample_args():
    parser = build_parser()
    args = parser.parse_args([
        "ml", "score", "--snapshot", "snapshot.parquet",
        "--model-artifact-dir", "model", "--preprocessing-artifact-dir", "pre",
        "--run-id", "score-2", "--split", "validation",
        "--max-students", "300", "--seed", "7",
    ])
    assert args.max_students == 300
    assert args.seed == 7


def test_cli_score_end_to_end_with_subsample(score_run, capsys):
    """CLI subsample: publish file đã subsample, exit 0, audit manifest."""
    _, snapshot, preprocessor, model, _ = score_run
    code = main([
        "ml", "score", "--snapshot", str(snapshot),
        "--model-artifact-dir", str(model.artifact_dir),
        "--preprocessing-artifact-dir", str(preprocessor.artifact_dir),
        "--artifacts-dir", str(score_run[0] / "artifacts"), "--run-id", "score-cli-sub",
        "--split", "validation", "--max-students", "1", "--seed", "42",
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "ok"
    batch_dir = score_run[0] / "artifacts" / "prediction-batches" / "score-cli-sub"
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    assert manifest["subsample"]["max_students"] == 1
    assert manifest["subsample"]["n_students_after"] == 1
    assert manifest["subsample"]["n_rows_after"] == manifest["row_count"]
    predictions = pd.read_parquet(batch_dir / "predictions.parquet")
    assert predictions["student_key"].nunique() == 1
