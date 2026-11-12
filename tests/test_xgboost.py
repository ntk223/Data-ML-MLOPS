"""End-to-end XGBoost candidate artifact, SHAP, gate, and release tests."""

import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learnguard.cli import _write_predictions_to_store, build_parser, main
from learnguard.evaluation.ml_gate import run_ml_gate
from learnguard.evaluation.scorecard import GateCheck, Scorecard, publish_evaluation_run
from learnguard.ml.adapters import SUPPORTED_MODEL_SCHEMAS, XGBoostScoringModel
from learnguard.ml.benchmark import run_benchmark
from learnguard.ml.contracts import load_prediction_batch_manifest
from learnguard.ml.explainability import build_shap_artifact
from learnguard.ml.model_interface import ScoringModel, load_scoring_model
from learnguard.ml.preprocessing import _prepare_split
from learnguard.ml.preprocessing import _file_hash
from learnguard.ml.model_comparison import publish_model_comparison
from learnguard.ml.release import (
    build_model_release,
    build_validation_evaluation_artifact,
    load_and_verify_model_release,
)
from learnguard.ml.scoring import score_release_snapshot, score_snapshot
from learnguard.ml.xgboost_training import (
    LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    XGBOOST_PRODUCTION_CANDIDATE,
    XGBOOST_PRODUCTION_PARAMS,
    XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    XGBOOST_MODEL_TYPE,
    load_xgboost_model_artifact,
    train_xgboost_model,
    verify_xgboost_model_artifact,
    xgboost_model_contract_hash,
)
from learnguard.data.loading import load_snapshot_partitions
from tests.test_preprocessing import preprocessing_run
from tests.test_model_interface import model_run


def _strict_json(path: Path):
    def reject_constant(token: str):
        raise ValueError(f"non_standard_json_constant:{token}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


@pytest.fixture(scope="module")
def xgboost_run(preprocessing_run):
    root, snapshot, preprocessor = preprocessing_run
    result = train_xgboost_model(
        snapshot, preprocessor.artifact_dir, root / "xgb-artifacts", "xgb-model",
    )
    return root, snapshot, preprocessor, result


def test_fixed_xgboost_contract_is_cpu_only():
    assert XGBOOST_PRODUCTION_PARAMS == {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "n_estimators": 400, "learning_rate": 0.03, "max_depth": 5,
        "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0.1, "reg_alpha": 0.05, "reg_lambda": 5.0,
        "scale_pos_weight": 1.0, "tree_method": "hist", "device": "cpu",
        "random_state": 42, "n_jobs": 1, "verbosity": 0,
    }
    assert len(xgboost_model_contract_hash()) == 64


def test_xgboost_train_verify_and_factory_round_trip(xgboost_run):
    _, snapshot, preprocessor, result = xgboost_run
    assert result.status == "ok"
    assert len(result.published_paths) == 5
    assert all(path.is_file() for path in result.published_paths)
    report = verify_xgboost_model_artifact(
        result.artifact_dir, snapshot, preprocessor.artifact_dir,
    )
    assert report.learned_classes == (0, 1)
    model = load_scoring_model(
        result.artifact_dir, snapshot, preprocessor.artifact_dir,
    )
    assert isinstance(model, XGBoostScoringModel)
    assert isinstance(model, ScoringModel)
    manifest = json.loads((result.artifact_dir / "model.manifest.json").read_text())
    assert manifest["schema_version"] == XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION
    assert manifest["model_type"] == XGBOOST_MODEL_TYPE
    assert manifest["class_weight_policy"] == "none_mild_imbalance"
    assert manifest["estimator_parameters"]["missing"] is None
    for path in result.artifact_dir.glob("*.json"):
        _strict_json(path)


def test_xgboost_tree_shap_additivity_and_signal_order(xgboost_run):
    _, snapshot, preprocessor, result = xgboost_run
    model = load_scoring_model(result.artifact_dir, snapshot, preprocessor.artifact_dir)
    selected = load_snapshot_partitions(snapshot, ("validation",))
    prepared = _prepare_split("validation", selected.frame("validation"))
    matrix = model.transform(prepared.X.head(12))
    explanation = model.explain_many(matrix)
    assert explanation.method == "tree-shap-interventional-v1"
    assert explanation.output_space == "log_odds"
    assert explanation.values.shape == matrix.shape
    np.testing.assert_allclose(
        explanation.expected_value + explanation.values.sum(axis=1),
        explanation.output_values, atol=1e-5, rtol=0,
    )
    signals = model.explain(matrix[0])
    assert [abs(signal.contribution) for signal in signals] == sorted(
        [abs(signal.contribution) for signal in signals], reverse=True,
    )
    assert {signal.direction for signal in signals} <= {
        "increases_risk", "decreases_risk", "neutral",
    }


@pytest.mark.parametrize("filename", [
    "model.ubj", "shap_background.npz", "model.manifest.json",
    "model.training-report.json", "model.success.json",
])
def test_xgboost_tampering_is_rejected(xgboost_run, tmp_path, filename):
    _, snapshot, preprocessor, result = xgboost_run
    copied = tmp_path / filename.replace(".", "-")
    shutil.copytree(result.artifact_dir, copied)
    with (copied / filename).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_xgboost_model_artifact(copied, snapshot, preprocessor.artifact_dir)


def test_xgboost_identical_runs_are_semantically_equal(xgboost_run):
    root, snapshot, preprocessor, first = xgboost_run
    second = train_xgboost_model(
        snapshot, preprocessor.artifact_dir, root / "xgb-artifacts", "xgb-model-2",
    )
    left = json.loads((first.artifact_dir / "model.manifest.json").read_text())
    right = json.loads((second.artifact_dir / "model.manifest.json").read_text())
    for field in (
        "model_semantic_version", "model_contract_hash", "learned_state_hash",
        "training_matrix_semantic_hash", "training_target_semantic_hash",
        "train_probability_semantic_hash",
    ):
        assert left[field] == right[field]


def test_xgboost_training_reads_train_partition_only(
    preprocessing_run, tmp_path, monkeypatch,
):
    _, snapshot, preprocessor = preprocessing_run
    original = pd.read_parquet
    observed: list[object] = []

    def spy(path, *args, **kwargs):
        if Path(path) == Path(snapshot):
            observed.append(kwargs.get("filters"))
            assert kwargs.get("filters") == [("split", "in", ["train"])]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    train_xgboost_model(snapshot, preprocessor.artifact_dir, tmp_path, "train-only")
    assert observed
    assert all(filters == [("split", "in", ["train"])] for filters in observed)


def test_xgboost_scoring_preserves_prediction_batch_v1(xgboost_run, tmp_path):
    _, snapshot, preprocessor, result = xgboost_run
    report = score_snapshot(
        snapshot, result.artifact_dir, preprocessor.artifact_dir,
        tmp_path / "prediction-batches/xgb/predictions.parquet",
        0.5, "xgb", splits=("validation",),
    )
    manifest = load_prediction_batch_manifest(report.batch_dir / "manifest.json")
    assert manifest.schema_version == "prediction-batch.v1"
    predictions = pd.read_parquet(report.output_path)
    assert "id_student" not in predictions
    assert predictions["model_version"].str.startswith("xgb-v2:").all()
    assert report.database_write_allowed is False
    with pytest.raises(ValueError, match="prediction_store_candidate_release_forbidden"):
        _write_predictions_to_store(report, Namespace(
            pg_dsn="postgresql://not-used", pg_table="ml_predictions", run_id="xgb",
        ))


def test_raw_xgboost_default_scores_validation_without_holdout(
    xgboost_run, tmp_path, monkeypatch, capsys,
):
    _, snapshot, preprocessor, result = xgboost_run
    original = pd.read_parquet
    observed: list[object] = []

    def spy(path, *args, **kwargs):
        if Path(path) == Path(snapshot):
            observed.append(kwargs.get("filters"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    code = main([
        "ml", "score", "--snapshot", str(snapshot),
        "--model-artifact-dir", str(result.artifact_dir),
        "--preprocessing-artifact-dir", str(preprocessor.artifact_dir),
        "--artifacts-dir", str(tmp_path), "--run-id", "xgb-default",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["input_rows"] > 0
    assert [("split", "in", ["validation"])] in observed
    assert all(
        filters is not None and not any(
            condition[0] == "split" and "holdout" in condition[2]
            for condition in filters
        )
        for filters in observed
    )
    predictions = original(payload["output_path"])
    assert set(predictions["split"]) == {"validation"}


def test_raw_xgboost_rejects_holdout_before_snapshot_read(
    xgboost_run, tmp_path, monkeypatch,
):
    _, snapshot, preprocessor, result = xgboost_run
    original = pd.read_parquet
    observed: list[object] = []

    def spy(path, *args, **kwargs):
        if Path(path) == Path(snapshot):
            observed.append(kwargs.get("filters"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    with pytest.raises(ValueError, match="scoring_raw_xgboost_holdout_forbidden"):
        score_snapshot(
            snapshot, result.artifact_dir, preprocessor.artifact_dir,
            tmp_path / "forbidden/predictions.parquet", 0.5, "xgb-holdout",
            splits=("holdout",),
        )
    assert observed == []


def test_legacy_xgboost_schema_is_registered_for_ml_gate():
    assert LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION in SUPPORTED_MODEL_SCHEMAS


def test_xgboost_explainability_artifact(xgboost_run, tmp_path):
    _, snapshot, preprocessor, result = xgboost_run
    report = build_shap_artifact(
        snapshot, result.artifact_dir, preprocessor.artifact_dir,
        tmp_path, "xgb-shap", splits=("validation",),
    )
    payload = json.loads((report.artifact_dir / "explanation_report.json").read_text())
    assert payload["schema_version"] == "shap-explanations.v2"
    assert payload["explanation_family"] == "tree_shap"
    assert payload["maximum_additivity_error"] <= 1e-5


def test_xgboost_explainability_defaults_to_validation_without_holdout(
    xgboost_run, tmp_path, monkeypatch,
):
    _, snapshot, preprocessor, result = xgboost_run
    original = pd.read_parquet
    observed: list[object] = []

    def spy(path, *args, **kwargs):
        if Path(path) == Path(snapshot):
            observed.append(kwargs.get("filters"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    report = build_shap_artifact(
        snapshot, result.artifact_dir, preprocessor.artifact_dir,
        tmp_path, "xgb-shap-default",
    )
    assert report.row_count > 0
    assert [("split", "in", ["validation"])] in observed
    assert all(
        filters is not None and not any(
            condition[0] == "split" and "holdout" in condition[2]
            for condition in filters
        )
        for filters in observed
    )


def test_xgboost_explainability_rejects_holdout_before_snapshot_read(
    xgboost_run, tmp_path, monkeypatch,
):
    _, snapshot, preprocessor, result = xgboost_run
    original = pd.read_parquet
    observed: list[object] = []

    def spy(path, *args, **kwargs):
        if Path(path) == Path(snapshot):
            observed.append(kwargs.get("filters"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    with pytest.raises(ValueError, match="explain_raw_xgboost_holdout_forbidden"):
        build_shap_artifact(
            snapshot, result.artifact_dir, preprocessor.artifact_dir,
            tmp_path, "xgb-shap-holdout", splits=("holdout",),
        )
    assert observed == []


def test_xgboost_ml_gate_and_regression_baseline(xgboost_run, model_run):
    _, snapshot, preprocessor, xgb = xgboost_run
    logistic = model_run[3]
    card = run_ml_gate(
        xgb.artifact_dir, preprocessor.artifact_dir, snapshot, "xgb-gate",
        baseline_model_dir=logistic.artifact_dir,
        baseline_preprocessing_artifact_dir=preprocessor.artifact_dir,
    )
    statuses = {check.id: check.status for check in card.checks}
    assert statuses["artifact_integrity"] == "pass"
    assert statuses["shap_additivity"] == "pass"
    assert statuses["regression_vs_baseline"] in {"pass", "fail"}


@pytest.fixture(scope="module")
def xgboost_candidate_release(xgboost_run, model_run):
    root, snapshot, preprocessor, model = xgboost_run
    benchmark = run_benchmark(
        snapshot, preprocessor.artifact_dir, root / "xgb-release", "validation",
        evaluation_scope="validation_only",
    )
    evaluation = build_validation_evaluation_artifact(
        snapshot, benchmark.artifact_dir, model.artifact_dir,
        preprocessor.artifact_dir, root / "xgb-release", "evaluation",
        candidate_name=XGBOOST_PRODUCTION_CANDIDATE,
    )
    logistic = model_run[3]
    rollback_evaluation = build_validation_evaluation_artifact(
        snapshot, benchmark.artifact_dir, logistic.artifact_dir,
        preprocessor.artifact_dir, root / "xgb-release", "rollback-evaluation",
        candidate_name="logistic_baseline",
    )
    rollback = build_model_release(
        snapshot, logistic.artifact_dir, preprocessor.artifact_dir,
        rollback_evaluation.artifact_dir, root / "xgb-release", "logistic-rollback",
    )
    release = build_model_release(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        evaluation.artifact_dir, root / "xgb-release", "candidate",
        release_status="validation_candidate",
    )
    return root, snapshot, preprocessor, model, benchmark, evaluation, release, rollback


def test_xgboost_candidate_release_policy(xgboost_candidate_release):
    _, snapshot, _, _, benchmark, _, release, _ = xgboost_candidate_release
    assert benchmark.candidate_count == 13
    loaded = load_and_verify_model_release(release.artifact_dir, snapshot)
    assert loaded.release_status == "validation_candidate"
    assert loaded.database_write_allowed is False
    assert loaded.allowed_score_splits == ("validation",)
    assert loaded.manifest["deployment_eligible"] is False
    configs = _strict_json(benchmark.artifact_dir / "candidate_configurations.json")
    assert configs["xgboost_regularized"]["parameters"]["missing"] is None
    for path in benchmark.artifact_dir.rglob("*.json"):
        _strict_json(path)


def test_xgboost_candidate_scores_validation_only(xgboost_candidate_release):
    root, snapshot, _, _, _, _, release, _ = xgboost_candidate_release
    report = score_release_snapshot(
        snapshot, release.artifact_dir,
        root / "xgb-release/prediction-batches/candidate/predictions.parquet",
        "candidate-score", ("validation",),
    )
    assert report.status == "ok"
    assert report.database_write_allowed is False
    with pytest.raises(ValueError, match="prediction_store_candidate_release_forbidden"):
        _write_predictions_to_store(report, Namespace(
            pg_dsn="postgresql://not-used", pg_table="ml_predictions",
            run_id="candidate-score",
        ))
    with pytest.raises(ValueError, match="scoring_release_split_forbidden:train"):
        score_release_snapshot(
            snapshot, release.artifact_dir,
            root / "xgb-release/prediction-batches/train/predictions.parquet",
            "candidate-train", ("train",),
        )
    with pytest.raises(ValueError, match="scoring_release_holdout_forbidden"):
        score_release_snapshot(
            snapshot, release.artifact_dir,
            root / "xgb-release/prediction-batches/holdout/predictions.parquet",
            "candidate-holdout", ("holdout",),
        )


def test_xgboost_production_promotion_requires_governance(xgboost_candidate_release, tmp_path):
    _, snapshot, preprocessor, model, _, evaluation, _, _ = xgboost_candidate_release
    with pytest.raises(ValueError, match="xgboost_production_promotion_decision_required"):
        build_model_release(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            evaluation.artifact_dir, tmp_path, "bad-production",
            release_status="production",
        )


def _ml_gate_artifact(path, snapshot, preprocessor, model, rollback, *, fail=False,
                      baseline_model_dir=None, baseline_preprocessor_dir=None):
    model_manifest = json.loads((model.artifact_dir / "model.manifest.json").read_text())
    loaded_rollback = load_and_verify_model_release(rollback.artifact_dir, snapshot)
    selected_baseline_model = (
        loaded_rollback.model_artifact_dir
        if baseline_model_dir is None else Path(baseline_model_dir)
    )
    selected_baseline_preprocessor = (
        loaded_rollback.preprocessing_artifact_dir
        if baseline_preprocessor_dir is None else Path(baseline_preprocessor_dir)
    )
    baseline_manifest_path = selected_baseline_model / "model.manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    card = Scorecard(
        run_id="xgb-promotion-gate",
        subject="ml",
        inputs={
            "snapshot": str(snapshot),
            "model": str(model.artifact_dir),
            "preprocessing": str(preprocessor.artifact_dir),
            "baseline_model": str(selected_baseline_model),
            "baseline_preprocessing": str(selected_baseline_preprocessor),
        },
    )
    for check_id in (
        "artifact_integrity", "validation_metrics", "alert_budget",
        "shap_additivity", "fairness_slices", "regression_vs_baseline",
    ):
        observed = {}
        if check_id == "artifact_integrity":
            observed = {"model_version": model_manifest["model_semantic_version"]}
        elif check_id == "regression_vs_baseline":
            observed = {
                "new_model_version": model_manifest["model_semantic_version"],
                "baseline_model_version": baseline_manifest["model_semantic_version"],
                "baseline_model_type": baseline_manifest["model_type"],
                "baseline_model_contract_hash": baseline_manifest["model_contract_hash"],
                "baseline_model_manifest_sha256": _file_hash(baseline_manifest_path),
                "baseline_model_artifact_sha256": baseline_manifest["model_artifact_sha256"],
            }
        status = "fail" if fail and check_id == "validation_metrics" else "pass"
        card.add(GateCheck(check_id, check_id, status, observed=observed))
    return publish_evaluation_run(path, card.run_id, [card])


def _promotion_decision(path, model, evaluation, ml_gate, rollback, snapshot):
    manifest = json.loads((model.artifact_dir / "model.manifest.json").read_text())
    report = json.loads((evaluation.artifact_dir / "evaluation_report.json").read_text())
    loaded_rollback = load_and_verify_model_release(rollback.artifact_dir, snapshot)
    rollback_model_manifest_path = loaded_rollback.model_artifact_dir / "model.manifest.json"
    rollback_model_manifest = json.loads(rollback_model_manifest_path.read_text())
    payload = {
        "schema_version": "promotion-decision.v2",
        "decision_id": "gov-xgb-primary-001",
        "approved_by": "ml-governance-test",
        "approved_at": "2026-11-09T00:00:00+00:00",
        "model_type": manifest["model_type"],
        "model_version": manifest["model_semantic_version"],
        "model_contract_hash": manifest["model_contract_hash"],
        "snapshot_version": manifest["snapshot_version"],
        "preprocessor_version": manifest["preprocessor_version"],
        "benchmark_manifest_sha256": report["validation_benchmark_manifest_sha256"],
        "evaluation_manifest_sha256": _file_hash(evaluation.artifact_dir / "manifest.json"),
        "ml_gate_manifest_sha256": _file_hash(ml_gate / "manifest.json"),
        "rollback_release_id": loaded_rollback.release_id,
        "rollback_release_manifest_sha256": _file_hash(
            rollback.artifact_dir / "release.manifest.json"
        ),
        "rollback_model_version": loaded_rollback.model_version,
        "rollback_model_manifest_sha256": _file_hash(rollback_model_manifest_path),
        "rollback_model_artifact_sha256": rollback_model_manifest["model_artifact_sha256"],
        "basis": "operational_primary_validation_only",
        "acknowledges_consumed_holdout": True,
        "scientific_superiority_claim": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def test_xgboost_governed_production_release_scores_and_keeps_holdout_guard(
    xgboost_candidate_release, tmp_path,
):
    _, snapshot, preprocessor, model, _, evaluation, _, rollback = xgboost_candidate_release
    ml_gate = _ml_gate_artifact(
        tmp_path / "gates", snapshot, preprocessor, model, rollback,
    )
    decision = _promotion_decision(
        tmp_path / "promotion.json", model, evaluation, ml_gate, rollback, snapshot,
    )
    release = build_model_release(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        evaluation.artifact_dir, tmp_path, "xgb-production",
        release_status="production", promotion_decision_path=decision,
        ml_gate_artifact_dir=ml_gate,
        rollback_release_artifact_dir=rollback.artifact_dir,
    )
    loaded = load_and_verify_model_release(release.artifact_dir, snapshot)
    assert loaded.release_status == "production"
    assert loaded.operational_role == "primary"
    assert loaded.evidence_status == "validation_only"
    assert loaded.database_write_allowed is True
    report = score_release_snapshot(
        snapshot, release.artifact_dir,
        tmp_path / "prediction-batches/production/predictions.parquet",
        "production-score", ("validation",),
    )
    assert report.database_write_allowed is True
    assert report.threshold == loaded.production_threshold
    with pytest.raises(ValueError, match="scoring_release_holdout_forbidden"):
        score_release_snapshot(
            snapshot, release.artifact_dir,
            tmp_path / "prediction-batches/holdout/predictions.parquet",
            "production-holdout", ("holdout",),
        )

    legacy_v3 = tmp_path / "legacy-v3-production"
    shutil.copytree(release.artifact_dir, legacy_v3)
    legacy_manifest_path = legacy_v3 / "release.manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    legacy_manifest["schema_version"] = "model-release.v3"
    legacy_manifest_path.write_text(json.dumps(legacy_manifest, indent=2, sort_keys=True))
    legacy_success_path = legacy_v3 / "release.success.json"
    legacy_success = json.loads(legacy_success_path.read_text())
    legacy_success["manifest_sha256"] = _file_hash(legacy_manifest_path)
    legacy_success_path.write_text(json.dumps(legacy_success, indent=2, sort_keys=True))
    with pytest.raises(ValueError, match="xgboost_production_release_upgrade_required"):
        load_and_verify_model_release(legacy_v3, snapshot)

    for index, (field, replacement) in enumerate((
        ("decision_id", None),
        ("approved_by", None),
        ("approved_at", None),
        ("decision_id", "different-decision-id"),
    )):
        copied = tmp_path / f"identity-tamper-{index}"
        shutil.copytree(release.artifact_dir, copied)
        decision_path = copied / "promotion_decision.json"
        tampered_decision = json.loads(decision_path.read_text())
        if replacement is None:
            tampered_decision.pop(field)
        else:
            tampered_decision[field] = replacement
        decision_path.write_text(json.dumps(tampered_decision, indent=2, sort_keys=True))

        manifest_path = copied / "release.manifest.json"
        release_manifest = json.loads(manifest_path.read_text())
        decision_hash = _file_hash(decision_path)
        release_manifest["promotion_decision_sha256"] = decision_hash
        release_manifest["files"]["promotion_decision.json"] = decision_hash
        manifest_path.write_text(json.dumps(release_manifest, indent=2, sort_keys=True))

        success_path = copied / "release.success.json"
        success = json.loads(success_path.read_text())
        success["manifest_sha256"] = _file_hash(manifest_path)
        success_path.write_text(json.dumps(success, indent=2, sort_keys=True))
        with pytest.raises(ValueError, match="xgboost_production_governance_invalid"):
            load_and_verify_model_release(copied, snapshot)


def test_xgboost_production_rejects_missing_or_failed_ml_gate(
    xgboost_candidate_release, tmp_path,
):
    _, snapshot, preprocessor, model, _, evaluation, _, rollback = xgboost_candidate_release
    passing = _ml_gate_artifact(
        tmp_path / "passing-gates", snapshot, preprocessor, model, rollback,
    )
    decision = _promotion_decision(
        tmp_path / "promotion-with-gate.json", model, evaluation, passing,
        rollback, snapshot,
    )
    with pytest.raises(ValueError, match="xgboost_production_rollback_release_required"):
        build_model_release(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            evaluation.artifact_dir, tmp_path, "missing-rollback",
            release_status="production", promotion_decision_path=decision,
            ml_gate_artifact_dir=passing,
        )
    with pytest.raises(ValueError, match="xgboost_production_ml_gate_required"):
        build_model_release(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            evaluation.artifact_dir, tmp_path, "missing-gate",
            release_status="production", promotion_decision_path=decision,
            rollback_release_artifact_dir=rollback.artifact_dir,
        )

    self_baseline_gate = _ml_gate_artifact(
        tmp_path / "self-baseline-gates", snapshot, preprocessor, model, rollback,
        baseline_model_dir=model.artifact_dir,
        baseline_preprocessor_dir=preprocessor.artifact_dir,
    )
    self_baseline_decision = _promotion_decision(
        tmp_path / "promotion-self-baseline.json", model, evaluation,
        self_baseline_gate, rollback, snapshot,
    )
    with pytest.raises(ValueError, match="ml_gate_logistic_baseline_mismatch"):
        build_model_release(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            evaluation.artifact_dir, tmp_path, "self-baseline",
            release_status="production", promotion_decision_path=self_baseline_decision,
            ml_gate_artifact_dir=self_baseline_gate,
            rollback_release_artifact_dir=rollback.artifact_dir,
        )
    failing = _ml_gate_artifact(
        tmp_path / "failing-gates", snapshot, preprocessor, model, rollback,
        fail=True,
    )
    failed_decision = _promotion_decision(
        tmp_path / "promotion-failed-gate.json", model, evaluation, failing,
        rollback, snapshot,
    )
    with pytest.raises(ValueError, match="ml_gate_artifact_invalid"):
        build_model_release(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            evaluation.artifact_dir, tmp_path, "failed-gate",
            release_status="production", promotion_decision_path=failed_decision,
            ml_gate_artifact_dir=failing,
            rollback_release_artifact_dir=rollback.artifact_dir,
        )


def test_model_comparison_is_deterministic(
    xgboost_candidate_release, model_run, tmp_path,
):
    _, snapshot, preprocessor, xgb, benchmark, _, _, _ = xgboost_candidate_release
    logistic = model_run[3]
    first = tmp_path / "comparison-1.md"
    second = tmp_path / "comparison-2.md"
    publish_model_comparison(
        benchmark.artifact_dir, xgb.artifact_dir, logistic.artifact_dir,
        snapshot, preprocessor.artifact_dir, first,
    )
    publish_model_comparison(
        benchmark.artifact_dir, xgb.artifact_dir, logistic.artifact_dir,
        snapshot, preprocessor.artifact_dir, second,
    )
    assert first.read_bytes() == second.read_bytes()
    text = first.read_text()
    assert all(name in text for name in (
        "xgboost_regularized", "logistic_baseline", "random_forest_b",
    ))
    assert "Holdout 2014J" in text


def test_model_comparison_rejects_metric_tampering(
    xgboost_candidate_release, model_run, tmp_path,
):
    _, snapshot, preprocessor, xgb, benchmark, _, _, _ = xgboost_candidate_release
    logistic = model_run[3]
    copied = tmp_path / "tampered-benchmark"
    shutil.copytree(benchmark.artifact_dir, copied)
    with (copied / "validation_metrics.csv").open("a") as handle:
        handle.write("tamper")
    with pytest.raises(ValueError, match="artifact_payload_hash_mismatch"):
        publish_model_comparison(
            copied, xgb.artifact_dir, logistic.artifact_dir,
            snapshot, preprocessor.artifact_dir, tmp_path / "bad.md",
        )


def test_model_comparison_derives_claims_from_verified_metrics(
    xgboost_candidate_release, model_run, tmp_path,
):
    _, snapshot, preprocessor, xgb, benchmark, _, _, _ = xgboost_candidate_release
    logistic = model_run[3]
    copied = tmp_path / "self-consistent-benchmark"
    shutil.copytree(benchmark.artifact_dir, copied)
    metrics_path = copied / "validation_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    logistic_pr = float(metrics.loc[
        metrics["candidate"] == "logistic_baseline", "pr_auc"
    ].iloc[0])
    metrics.loc[metrics["candidate"] == "xgboost_regularized", "pr_auc"] = logistic_pr - 0.1
    metrics.to_csv(metrics_path, index=False)
    configs_path = copied / "candidate_configurations.json"
    configs = json.loads(configs_path.read_text())
    configs["xgboost_regularized"]["validation_metrics"]["pr_auc"] = logistic_pr - 0.1
    configs_path.write_text(json.dumps(configs, indent=2, sort_keys=True))
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["validation_metrics.csv"] = _file_hash(metrics_path)
    manifest["files"]["candidate_configurations.json"] = _file_hash(configs_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    success_path = copied / "success.json"
    success = json.loads(success_path.read_text())
    success["manifest_sha256"] = _file_hash(manifest_path)
    success_path.write_text(json.dumps(success, indent=2, sort_keys=True))

    output = tmp_path / "derived.md"
    publish_model_comparison(
        copied, xgb.artifact_dir, logistic.artifact_dir,
        snapshot, preprocessor.artifact_dir, output,
    )
    text = output.read_text()
    assert "worse on PR-AUC" in text
    assert "does not verify that a production release has been approved" in text


def test_model_comparison_cli_parser():
    args = build_parser().parse_args([
        "ml", "publish-model-comparison", "--benchmark-artifact-dir", "bench",
        "--xgboost-model-artifact-dir", "xgb", "--logistic-model-artifact-dir", "log",
        "--snapshot", "snapshot.parquet", "--preprocessing-artifact-dir", "pre",
    ])
    assert args.command == "publish-model-comparison"
    assert args.output == Path("docs/MODEL_BENCHMARK_COMPARISON.md")


def test_xgboost_cli_contract_and_failure(tmp_path):
    args = build_parser().parse_args([
        "ml", "train-xgboost", "--snapshot", "snapshot.parquet",
        "--preprocessing-artifact-dir", "pre", "--run-id", "xgb",
    ])
    assert args.command == "train-xgboost"
    assert main([
        "ml", "train-xgboost", "--snapshot", str(tmp_path / "missing.parquet"),
        "--preprocessing-artifact-dir", str(tmp_path / "pre"),
        "--artifacts-dir", str(tmp_path), "--run-id", "xgb",
    ]) == 2


@pytest.mark.parametrize("run_id", ["", "..", "../x", "nested/x", "/tmp/x"])
def test_xgboost_unsafe_run_id_rejected(preprocessing_run, tmp_path, run_id):
    _, snapshot, preprocessor = preprocessing_run
    with pytest.raises(ValueError, match="invalid_run_id"):
        train_xgboost_model(snapshot, preprocessor.artifact_dir, tmp_path, run_id)

