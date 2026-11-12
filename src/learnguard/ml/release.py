"""Validation-only evaluation and immutable algorithm-aware model releases."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from learnguard.ml.evaluation import (
    select_alert_budget_threshold,
    threshold_metrics,
)
from learnguard.ml.preprocessing import _file_hash, _validate_run_id
from learnguard.ml.model_interface import load_scoring_model, verify_model_artifact
from learnguard.ml.adapters import LOGISTIC_MODEL_TYPE
from learnguard.ml.xgboost_training import XGBOOST_MODEL_TYPE, XGBOOST_PRODUCTION_CANDIDATE
from learnguard.evaluation.scorecard import EVAL_SCHEMA_VERSION

LEGACY_EVALUATION_RUN_SCHEMA_VERSION = "validation-release-evaluation.v1"
LEGACY_EVALUATION_RUN_SCHEMA_VERSION_V2 = "validation-release-evaluation.v2"
EVALUATION_RUN_SCHEMA_VERSION = "validation-release-evaluation.v3"
LEGACY_MODEL_RELEASE_SCHEMA_VERSION = "model-release.v1"
LEGACY_MODEL_RELEASE_SCHEMA_VERSION_V2 = "model-release.v2"
LEGACY_MODEL_RELEASE_SCHEMA_VERSION_V3 = "model-release.v3"
MODEL_RELEASE_SCHEMA_VERSION = "model-release.v4"
PROMOTION_DECISION_SCHEMA_VERSION = "promotion-decision.v2"
PRODUCTION_CANDIDATE = "logistic_baseline"
ALERT_BUDGET_FRACTION = 0.20
MAX_ALERT_BUDGET_DEVIATION = 0.01


@dataclass(frozen=True)
class EvaluationArtifactResult:
    status: str
    run_id: str
    artifact_dir: Path
    production_threshold: float
    realized_alert_rate: float
    candidate_count: int
    manifest_path: Path


@dataclass(frozen=True)
class ModelReleaseResult:
    status: str
    release_id: str
    artifact_dir: Path
    model_version: str
    preprocessor_version: str
    production_threshold: float
    manifest_path: Path


@dataclass(frozen=True)
class LoadedModelRelease:
    release_id: str
    artifact_dir: Path
    model_artifact_dir: Path
    preprocessing_artifact_dir: Path
    model_version: str
    snapshot_version: str
    production_threshold: float
    holdout_access_policy: str
    manifest: Mapping[str, object]
    release_status: str = "production"
    database_write_allowed: bool = True
    allowed_score_splits: tuple[str, ...] = ("train", "validation")
    operational_role: str = "primary"
    evidence_status: str = "validation_only"


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path.name}")
    return value


def _safe_relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"artifact_relative_path_invalid:{value}")
    path = root / relative
    if path.resolve().parent != root.resolve() and root.resolve() not in path.resolve().parents:
        raise ValueError(f"artifact_relative_path_invalid:{value}")
    return path


def _verify_file_map(root: Path, files: object) -> dict[str, str]:
    if not isinstance(files, dict) or not files:
        raise ValueError("artifact_file_map_invalid")
    checked: dict[str, str] = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("artifact_file_map_invalid")
        path = _safe_relative(root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"artifact_payload_missing:{relative}")
        if _file_hash(path) != expected:
            raise ValueError(f"artifact_payload_hash_mismatch:{relative}")
        checked[relative] = expected
    return checked


def _verify_benchmark_artifact(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = root / "manifest.json"
    success_path = root / "success.json"
    report_path = root / "benchmark_report.json"
    if any(not path.is_file() for path in (manifest_path, success_path, report_path)):
        raise FileNotFoundError("validation_benchmark_artifact_missing")
    manifest = _read_json(manifest_path)
    success = _read_json(success_path)
    if success.get("manifest_sha256") != _file_hash(manifest_path):
        raise ValueError("validation_benchmark_manifest_hash_mismatch")
    files = _verify_file_map(root, manifest.get("files"))
    selection = root / "validation_selection.json"
    if success.get("selection_sha256") != _file_hash(selection):
        raise ValueError("validation_benchmark_selection_hash_mismatch")
    if any("holdout" in name or "locked_config" in name for name in files):
        raise ValueError("validation_benchmark_contains_holdout_artifact")
    report = _read_json(report_path)
    if (
        report.get("evaluation_scope") != "validation_only"
        or report.get("holdout_accessed") is not False
        or report.get("production_release_eligible") is not False
    ):
        raise ValueError("validation_benchmark_scope_invalid")
    return report, manifest


def _probability_hash(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _feature_order_hash(names: object) -> str:
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("release_feature_names_invalid")
    return sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()


def _prediction_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1]).astype(int)
    return {
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def _assert_metric_match(actual: dict[str, object], expected: object) -> None:
    if not isinstance(expected, dict):
        raise ValueError("validation_source_metric_invalid")
    if actual["confusion_matrix"] != expected.get("confusion_matrix"):
        raise ValueError("validation_confusion_matrix_mismatch")
    for name in ("precision", "recall", "f1"):
        if not np.isclose(float(actual[name]), float(expected.get(name)), atol=1e-12, rtol=0):
            raise ValueError(f"validation_metric_mismatch:{name}")


def _write_confusion_artifacts(
    staging: Path,
    candidate: str,
    policy: str,
    threshold: float,
    matrix: list[list[int]],
) -> tuple[Path, Path]:
    matrix_dir = staging / "confusion_matrices"
    plot_dir = staging / "plots"
    matrix_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    stem = f"{candidate}__{policy}"
    rows = [
        {"candidate": candidate, "policy": policy, "threshold": threshold,
         "actual_label": actual, "predicted_label": predicted,
         "count": int(matrix[actual][predicted])}
        for actual in (0, 1) for predicted in (0, 1)
    ]
    csv_path = matrix_dir / f"{stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/learnguard-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.asarray(matrix, dtype=np.int64)
    figure, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(array, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(array[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    axis.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title(f"{candidate} — {policy}\nthreshold={threshold:.12g}; [[TN, FP], [FN, TP]]")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    png_path = plot_dir / f"confusion_matrix__{stem}.png"
    figure.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return csv_path, png_path


def build_validation_evaluation_artifact(
    snapshot_path: Path | str,
    benchmark_artifact_dir: Path | str,
    model_artifact_dir: Path | str,
    preprocessing_artifact_dir: Path | str,
    artifacts_dir: Path | str = Path("artifacts"),
    run_id: str = "validation-evaluation",
    candidate_name: str | None = None,
) -> EvaluationArtifactResult:
    """Build evaluation/reporting artifacts without reading snapshot validation/holdout."""
    run_id = _validate_run_id(run_id)
    snapshot_path = Path(snapshot_path)
    benchmark_root = Path(benchmark_artifact_dir)
    model_root = Path(model_artifact_dir)
    preprocessor_root = Path(preprocessing_artifact_dir)
    target = Path(artifacts_dir) / "evaluation-runs" / run_id
    if target.exists():
        raise FileExistsError(f"evaluation_run_id_exists:{run_id}")

    benchmark_report, _ = _verify_benchmark_artifact(benchmark_root)
    verify_model_artifact(model_root, snapshot_path, preprocessor_root)
    model_manifest = _read_json(model_root / "model.manifest.json")
    if candidate_name is None:
        candidate_name = (
            XGBOOST_PRODUCTION_CANDIDATE
            if model_manifest.get("model_type") == XGBOOST_MODEL_TYPE
            else PRODUCTION_CANDIDATE
        )
    preprocessor_manifest = _read_json(preprocessor_root / "preprocessor.manifest.json")
    configs = _read_json(benchmark_root / "candidate_configurations.json")
    candidate_config = configs.get(candidate_name)
    if not isinstance(candidate_config, dict):
        raise ValueError(f"validation_candidate_missing:{candidate_name}")
    if (
        benchmark_report.get("snapshot_version") != model_manifest.get("snapshot_version")
        or benchmark_report.get("preprocessor_version") != model_manifest.get("preprocessor_version")
        or preprocessor_manifest.get("preprocessor_version") != model_manifest.get("preprocessor_version")
    ):
        raise ValueError("validation_provenance_version_mismatch")
    candidate_class = candidate_config.get("class")
    expected_class = model_manifest.get("model_type")
    if (
        not isinstance(candidate_class, str)
        or not isinstance(expected_class, str)
        or candidate_class.rsplit(".", 1)[-1] != expected_class.rsplit(".", 1)[-1]
    ):
        raise ValueError("validation_candidate_model_type_mismatch")
    if candidate_config.get("parameters") != model_manifest.get("estimator_parameters"):
        raise ValueError("validation_production_parameters_mismatch")
    production_model = model_root / str(model_manifest.get("model_payload_filename", "model.joblib"))
    if not production_model.is_file():
        raise FileNotFoundError("validation_production_model_missing")
    candidate_model_path = benchmark_root / "candidates" / f"{candidate_name}.joblib"
    if not candidate_model_path.is_file():
        raise FileNotFoundError("validation_candidate_model_missing")
    if model_manifest.get("model_type") == LOGISTIC_MODEL_TYPE:
        if _file_hash(candidate_model_path) != _file_hash(production_model):
            raise ValueError("validation_production_model_mismatch")
    elif model_manifest.get("model_type") == XGBOOST_MODEL_TYPE:
        candidate_model = joblib.load(candidate_model_path)
        candidate_state = sha256(bytes(
            candidate_model.get_booster().save_raw(raw_format="ubj")
        )).hexdigest()
        if candidate_state != model_manifest.get("learned_state_hash"):
            raise ValueError("validation_production_model_mismatch")

    predictions = pd.read_parquet(benchmark_root / "validation_predictions.parquet")
    required = {
        "student_key", "code_module", "code_presentation", "split", "label",
        f"risk_score__{candidate_name}",
    }
    if not required <= set(predictions) or "id_student" in predictions:
        raise ValueError("validation_predictions_schema_invalid")
    if set(predictions["split"].astype(str)) != {"validation"}:
        raise ValueError("validation_predictions_scope_invalid")
    y_true = predictions["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(y_true), np.array([0, 1])):
        raise ValueError("validation_labels_invalid")
    candidate_scores = predictions[f"risk_score__{candidate_name}"].to_numpy(
        dtype=np.float64,
    )
    if _probability_hash(candidate_scores) != candidate_config.get("validation_probability_sha256"):
        raise ValueError("validation_probability_hash_mismatch")

    keys = predictions[["code_module", "code_presentation", "student_key"]]
    budget = select_alert_budget_threshold(candidate_scores, keys, ALERT_BUDGET_FRACTION)
    allowed_deviation = max(MAX_ALERT_BUDGET_DEVIATION, 1.0 / len(y_true))
    if budget.absolute_deviation > allowed_deviation:
        raise ValueError(
            f"alert_budget_scalar_deviation_exceeded:{budget.absolute_deviation}"
        )
    production_metrics = asdict(threshold_metrics(y_true, candidate_scores, budget.threshold))
    rank_labels = np.zeros(len(y_true), dtype=np.int64)
    rank_labels[budget.ranked_indices] = 1
    ranked_reference = _prediction_metrics(y_true, rank_labels)
    ranked_reference.update({
        "alert_count": budget.budget_count,
        "alert_rate": budget.budget_count / len(y_true),
    })

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        metric_rows: list[dict[str, object]] = []
        candidate_names = sorted(
            column.removeprefix("risk_score__")
            for column in predictions if column.startswith("risk_score__")
        )
        for candidate in candidate_names:
            config = configs.get(candidate)
            if not isinstance(config, dict):
                raise ValueError(f"validation_candidate_config_missing:{candidate}")
            scores = predictions[f"risk_score__{candidate}"].to_numpy(dtype=np.float64)
            if not np.isfinite(scores).all():
                raise ValueError(f"validation_candidate_probability_invalid:{candidate}")
            if _probability_hash(scores) != config.get("validation_probability_sha256"):
                raise ValueError(f"validation_candidate_probability_hash_mismatch:{candidate}")
            source_metrics = config.get("validation_metrics")
            if not isinstance(source_metrics, dict):
                raise ValueError(f"validation_candidate_metrics_missing:{candidate}")
            policies = (
                ("threshold-0_5", 0.5, source_metrics.get("threshold_0_5")),
                ("max-f1", float(source_metrics["selected_threshold"]["threshold"]),
                 source_metrics.get("selected_threshold")),
            )
            for policy, threshold, expected in policies:
                metrics = asdict(threshold_metrics(y_true, scores, threshold))
                _assert_metric_match(metrics, expected)
                _write_confusion_artifacts(
                    staging, candidate, policy, threshold,
                    metrics["confusion_matrix"],
                )
                metric_rows.append({"candidate": candidate, "policy": policy, **metrics})

        _write_confusion_artifacts(
            staging, candidate_name, "production-top20",
            budget.threshold, production_metrics["confusion_matrix"],
        )
        metric_rows.append({
            "candidate": candidate_name,
            "policy": "production-top20",
            **production_metrics,
        })
        pd.DataFrame(metric_rows).drop(columns=["confusion_matrix"]).to_csv(
            staging / "validation_metrics.csv", index=False,
        )
        max_f1 = candidate_config["validation_metrics"]["selected_threshold"]
        threshold_payload = {
            "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "policy": "validation_alert_budget_scalar_v1",
            "derivation_split": "validation",
            "population": "all_finite_validation_scores",
            "target_fraction": ALERT_BUDGET_FRACTION,
            "validation_sample_count": len(y_true),
            "budget_count": budget.budget_count,
            "boundary_score": budget.boundary_score,
            "boundary_tie_count": budget.boundary_tie_count,
            "tie_policy": budget.tie_policy,
            "production_threshold": budget.threshold,
            "realized_alert_count": budget.realized_alert_count,
            "realized_alert_rate": budget.realized_alert_rate,
            "absolute_budget_deviation": budget.absolute_deviation,
            "maximum_allowed_deviation": allowed_deviation,
            "production_threshold_metrics": production_metrics,
            "ranked_reference_metrics": ranked_reference,
            "max_f1_comparator": max_f1,
            "holdout_used": False,
        }
        (staging / "threshold_policy.json").write_text(json.dumps(
            threshold_payload, indent=2, sort_keys=True,
        ))
        report = {
            "status": "ok",
            "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_scope": "validation_only",
            "snapshot_version": model_manifest["snapshot_version"],
            "preprocessor_version": model_manifest["preprocessor_version"],
            "model_version": model_manifest["model_semantic_version"],
            "model_artifact_sha256": _file_hash(production_model),
            "validation_benchmark_run_id": benchmark_report["run_id"],
            "validation_benchmark_manifest_sha256": _file_hash(
                benchmark_root / "manifest.json"
            ),
            "validation_predictions_sha256": _file_hash(
                benchmark_root / "validation_predictions.parquet"
            ),
            "validation_probability_sha256": _probability_hash(candidate_scores),
            "candidate_count": len(candidate_names),
            "production_model": candidate_name,
            "model_type": model_manifest["model_type"],
            "model_contract_hash": model_manifest["model_contract_hash"],
            "validation_leader": benchmark_report["validation_leader"],
            "production_threshold": budget.threshold,
            "production_metrics": production_metrics,
            "holdout_accessed": False,
            "holdout_reused": False,
            "production_release_eligible": model_manifest["model_type"] == LOGISTIC_MODEL_TYPE,
            "governance_promotion_eligible": model_manifest["model_type"] == XGBOOST_MODEL_TYPE,
            "governance_approval_required": model_manifest["model_type"] == XGBOOST_MODEL_TYPE,
            "operational_role": (
                "fallback" if model_manifest["model_type"] == LOGISTIC_MODEL_TYPE
                else "primary"
            ),
            "evidence_status": "validation_only",
            "scientific_superiority_claim": False,
            "release_status": (
                "production" if model_manifest["model_type"] == LOGISTIC_MODEL_TYPE
                else "validation_candidate"
            ),
            "eligibility_basis": (
                "incumbent_logistic_validation_policy"
                if model_manifest["model_type"] == LOGISTIC_MODEL_TYPE
                else "validation_only_prior_holdout_consumed"
            ),
        }
        (staging / "evaluation_report.json").write_text(json.dumps(
            report, indent=2, sort_keys=True,
        ))
        files = [path for path in staging.rglob("*") if path.is_file()]
        manifest = {
            "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "files": {
                str(path.relative_to(staging)): _file_hash(path)
                for path in sorted(files)
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (staging / "success.json").write_text(json.dumps({
            "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "manifest_sha256": _file_hash(manifest_path),
        }, indent=2, sort_keys=True))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return EvaluationArtifactResult(
        "ok", run_id, target, budget.threshold, budget.realized_alert_rate,
        len(candidate_names), target / "manifest.json",
    )


def _verify_evaluation_artifact(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = root / "manifest.json"
    success_path = root / "success.json"
    report_path = root / "evaluation_report.json"
    policy_path = root / "threshold_policy.json"
    if any(not path.is_file() for path in (
        manifest_path, success_path, report_path, policy_path,
    )):
        raise FileNotFoundError("release_evaluation_artifact_missing")
    manifest = _read_json(manifest_path)
    success = _read_json(success_path)
    if manifest.get("schema_version") not in {
        LEGACY_EVALUATION_RUN_SCHEMA_VERSION,
        LEGACY_EVALUATION_RUN_SCHEMA_VERSION_V2,
        EVALUATION_RUN_SCHEMA_VERSION,
    }:
        raise ValueError("release_evaluation_schema_unsupported")
    if success.get("manifest_sha256") != _file_hash(manifest_path):
        raise ValueError("release_evaluation_manifest_hash_mismatch")
    _verify_file_map(root, manifest.get("files"))
    report = _read_json(report_path)
    policy = _read_json(policy_path)
    if (
        report.get("evaluation_scope") != "validation_only"
        or report.get("holdout_accessed") is not False
        or report.get("holdout_reused") is not False
        or policy.get("holdout_used") is not False
    ):
        raise ValueError("release_evaluation_holdout_policy_invalid")
    return report, policy


def _verify_promotion_decision(
    path: Path,
    model_manifest: Mapping[str, object],
    evaluation: Mapping[str, object],
    evaluation_manifest_sha256: str,
    ml_gate_manifest_sha256: str,
    rollback_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Verify the governance attestation used for validation-only promotion."""
    decision = _read_json(path)
    required_strings = (
        "decision_id", "approved_by", "approved_at", "model_type",
        "model_version", "model_contract_hash", "snapshot_version",
        "preprocessor_version", "benchmark_manifest_sha256",
        "evaluation_manifest_sha256",
        "ml_gate_manifest_sha256",
        "rollback_release_id", "rollback_release_manifest_sha256",
        "rollback_model_version", "rollback_model_manifest_sha256",
        "rollback_model_artifact_sha256",
    )
    if decision.get("schema_version") != PROMOTION_DECISION_SCHEMA_VERSION:
        raise ValueError("promotion_decision_schema_unsupported")
    if any(not isinstance(decision.get(key), str) or not decision[key] for key in required_strings):
        raise ValueError("promotion_decision_field_invalid")
    expected = {
        "model_type": model_manifest.get("model_type"),
        "model_version": model_manifest.get("model_semantic_version"),
        "model_contract_hash": model_manifest.get("model_contract_hash"),
        "snapshot_version": model_manifest.get("snapshot_version"),
        "preprocessor_version": model_manifest.get("preprocessor_version"),
        "benchmark_manifest_sha256": evaluation.get("validation_benchmark_manifest_sha256"),
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "ml_gate_manifest_sha256": ml_gate_manifest_sha256,
        **rollback_provenance,
    }
    if any(decision.get(key) != value for key, value in expected.items()):
        raise ValueError("promotion_decision_provenance_mismatch")
    if (
        decision.get("basis") != "operational_primary_validation_only"
        or decision.get("acknowledges_consumed_holdout") is not True
        or decision.get("scientific_superiority_claim") is not False
    ):
        raise ValueError("promotion_decision_policy_invalid")
    return decision


def _verify_ml_gate_artifact(
    root: Path,
    model_manifest: Mapping[str, object],
    *,
    snapshot_path: Path | None = None,
    model_artifact_dir: Path | None = None,
    preprocessing_artifact_dir: Path | None = None,
    baseline_model_manifest: Mapping[str, object] | None = None,
    baseline_model_manifest_sha256: str | None = None,
    baseline_model_artifact_dir: Path | None = None,
    baseline_preprocessing_artifact_dir: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Verify a published, non-failing ML gate bound to this model."""
    manifest_path = root / "manifest.json"
    success_path = root / "success.json"
    scorecard_path = root / "eval_ml.json"
    if any(not path.is_file() for path in (manifest_path, success_path, scorecard_path)):
        raise FileNotFoundError("ml_gate_artifact_missing")
    manifest = _read_json(manifest_path)
    success = _read_json(success_path)
    scorecard = _read_json(scorecard_path)
    manifest_hash = _file_hash(manifest_path)
    scorecards = manifest.get("scorecards")
    if (
        manifest.get("schema_version") != EVAL_SCHEMA_VERSION
        or not isinstance(manifest.get("subjects"), list)
        or "ml" not in manifest["subjects"]
        or not isinstance(scorecards, dict)
        or scorecards.get("eval_ml.json") != _file_hash(scorecard_path)
        or success.get("manifest_sha256") != manifest_hash
        or success.get("status") == "fail"
        or success.get("status") not in {"pass", "warn"}
        or scorecard.get("schema_version") != EVAL_SCHEMA_VERSION
        or scorecard.get("subject") != "ml"
        or scorecard.get("run_id") != manifest.get("run_id")
    ):
        raise ValueError("ml_gate_artifact_invalid")
    checks = scorecard.get("checks")
    required_ids = {
        "artifact_integrity", "validation_metrics", "alert_budget",
        "shap_additivity", "fairness_slices", "regression_vs_baseline",
    }
    if not isinstance(checks, list) or {
        check.get("id") for check in checks if isinstance(check, dict)
    } != required_ids:
        raise ValueError("ml_gate_checks_invalid")
    by_id = {str(check["id"]): check for check in checks if isinstance(check, dict)}
    if (
        any(check.get("status") == "fail" for check in by_id.values())
        or by_id["artifact_integrity"].get("status") != "pass"
        or by_id["regression_vs_baseline"].get("status") != "pass"
    ):
        raise ValueError("ml_gate_not_passing")
    integrity_observed = by_id["artifact_integrity"].get("observed")
    regression_observed = by_id["regression_vs_baseline"].get("observed")
    model_version = model_manifest.get("model_semantic_version", model_manifest.get("model_version"))
    if (
        not isinstance(integrity_observed, dict)
        or integrity_observed.get("model_version") != model_version
        or not isinstance(regression_observed, dict)
        or regression_observed.get("new_model_version") != model_version
        or not isinstance(regression_observed.get("baseline_model_version"), str)
        or not regression_observed["baseline_model_version"]
    ):
        raise ValueError("ml_gate_model_provenance_mismatch")
    if baseline_model_manifest is not None:
        baseline_payload_sha256 = baseline_model_manifest.get("model_artifact_sha256")
        expected_baseline = {
            "baseline_model_type": LOGISTIC_MODEL_TYPE,
            "baseline_model_version": baseline_model_manifest.get("model_semantic_version"),
            "baseline_model_contract_hash": baseline_model_manifest.get("model_contract_hash"),
            "baseline_model_manifest_sha256": baseline_model_manifest_sha256,
            "baseline_model_artifact_sha256": baseline_payload_sha256,
        }
        if (
            baseline_model_manifest.get("model_type") != LOGISTIC_MODEL_TYPE
            or any(regression_observed.get(key) != value for key, value in expected_baseline.items())
        ):
            raise ValueError("ml_gate_logistic_baseline_mismatch")
    if snapshot_path is not None:
        inputs = scorecard.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("ml_gate_inputs_invalid")
        expected_paths = {
            "snapshot": snapshot_path,
            "model": model_artifact_dir,
            "preprocessing": preprocessing_artifact_dir,
        }
        for key, expected in expected_paths.items():
            value = inputs.get(key)
            if expected is None or not isinstance(value, str) or Path(value).resolve() != expected.resolve():
                raise ValueError(f"ml_gate_input_mismatch:{key}")
        if not isinstance(inputs.get("baseline_model"), str):
            raise ValueError("ml_gate_baseline_required")
        baseline_paths = {
            "baseline_model": baseline_model_artifact_dir,
            "baseline_preprocessing": baseline_preprocessing_artifact_dir,
        }
        for key, expected in baseline_paths.items():
            value = inputs.get(key)
            if expected is None or not isinstance(value, str) or Path(value).resolve() != expected.resolve():
                raise ValueError(f"ml_gate_input_mismatch:{key}")
    return scorecard, manifest_hash


def _ledger_metadata(snapshot_path: Path) -> tuple[str, int, str]:
    ledger_path = snapshot_path.parent / f"{snapshot_path.stem}.ledger.json"
    ledger = _read_json(ledger_path)
    version, cutoff = ledger.get("snapshot_version"), ledger.get("cutoff_day")
    if not isinstance(version, str) or not isinstance(cutoff, int) or isinstance(cutoff, bool):
        raise ValueError("release_snapshot_ledger_invalid")
    return version, cutoff, _file_hash(ledger_path)


def build_model_release(
    snapshot_path: Path | str,
    model_artifact_dir: Path | str,
    preprocessing_artifact_dir: Path | str,
    evaluation_artifact_dir: Path | str,
    artifacts_dir: Path | str = Path("artifacts"),
    run_id: str = "logistic-production",
    release_status: str | None = None,
    promotion_decision_path: Path | str | None = None,
    ml_gate_artifact_dir: Path | str | None = None,
    rollback_release_artifact_dir: Path | str | None = None,
) -> ModelReleaseResult:
    """Copy verified payloads into a relocatable, immutable release tree."""
    release_id = _validate_run_id(run_id)
    snapshot_path = Path(snapshot_path)
    model_root = Path(model_artifact_dir)
    preprocessor_root = Path(preprocessing_artifact_dir)
    evaluation_root = Path(evaluation_artifact_dir)
    target = Path(artifacts_dir) / "model-releases" / release_id
    if target.exists():
        raise FileExistsError(f"model_release_id_exists:{release_id}")
    evaluation, policy = _verify_evaluation_artifact(evaluation_root)
    verify_model_artifact(model_root, snapshot_path, preprocessor_root)
    model_manifest = _read_json(model_root / "model.manifest.json")
    preprocessor_manifest = _read_json(preprocessor_root / "preprocessor.manifest.json")
    snapshot_version, cutoff_day, ledger_sha256 = _ledger_metadata(snapshot_path)
    model_payload = model_root / str(model_manifest.get("model_payload_filename", "model.joblib"))
    if (
        evaluation.get("model_version") != model_manifest.get("model_semantic_version")
        or evaluation.get("model_artifact_sha256") != _file_hash(model_payload)
        or evaluation.get("snapshot_version") != snapshot_version
        or evaluation.get("preprocessor_version") != preprocessor_manifest.get("preprocessor_version")
    ):
        raise ValueError("model_release_provenance_mismatch")
    threshold = policy.get("production_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
        raise ValueError("model_release_threshold_invalid")
    expected_status = str(evaluation.get("release_status", "production"))
    release_status = expected_status if release_status is None else release_status
    if release_status not in {"production", "validation_candidate"}:
        raise ValueError("model_release_status_invalid")
    is_xgboost = model_manifest.get("model_type") == XGBOOST_MODEL_TYPE
    governance_upgrade = (
        is_xgboost
        and expected_status == "validation_candidate"
        and release_status == "production"
        and evaluation.get("governance_promotion_eligible") is True
    )
    if release_status != expected_status and not governance_upgrade:
        raise ValueError("model_release_evaluation_status_mismatch")
    decision: dict[str, object] | None = None
    decision_path: Path | None = None
    ml_gate_root: Path | None = None
    ml_gate_manifest_hash: str | None = None
    rollback_root: Path | None = None
    rollback_provenance: dict[str, object] | None = None
    if governance_upgrade:
        if promotion_decision_path is None:
            raise ValueError("xgboost_production_promotion_decision_required")
        if ml_gate_artifact_dir is None:
            raise ValueError("xgboost_production_ml_gate_required")
        if rollback_release_artifact_dir is None:
            raise ValueError("xgboost_production_rollback_release_required")
        rollback_root = Path(rollback_release_artifact_dir)
        rollback = load_and_verify_model_release(rollback_root, snapshot_path)
        if (
            rollback.manifest.get("model_type") != LOGISTIC_MODEL_TYPE
            or rollback.release_status != "production"
            or rollback.database_write_allowed is not True
            or rollback.holdout_access_policy != "forbidden_prior_consumed"
        ):
            raise ValueError("xgboost_rollback_release_invalid")
        rollback_model_manifest_path = rollback.model_artifact_dir / "model.manifest.json"
        rollback_model_manifest = _read_json(rollback_model_manifest_path)
        rollback_manifest_hash = _file_hash(rollback_root / "release.manifest.json")
        rollback_model_manifest_hash = _file_hash(rollback_model_manifest_path)
        rollback_provenance = {
            "rollback_release_id": rollback.release_id,
            "rollback_release_manifest_sha256": rollback_manifest_hash,
            "rollback_model_version": rollback.model_version,
            "rollback_model_manifest_sha256": rollback_model_manifest_hash,
            "rollback_model_artifact_sha256": rollback_model_manifest["model_artifact_sha256"],
        }
        ml_gate_root = Path(ml_gate_artifact_dir)
        _, ml_gate_manifest_hash = _verify_ml_gate_artifact(
            ml_gate_root,
            model_manifest,
            snapshot_path=snapshot_path,
            model_artifact_dir=model_root,
            preprocessing_artifact_dir=preprocessor_root,
            baseline_model_manifest=rollback_model_manifest,
            baseline_model_manifest_sha256=rollback_model_manifest_hash,
            baseline_model_artifact_dir=rollback.model_artifact_dir,
            baseline_preprocessing_artifact_dir=rollback.preprocessing_artifact_dir,
        )
        decision_path = Path(promotion_decision_path)
        decision = _verify_promotion_decision(
            decision_path,
            model_manifest,
            evaluation,
            _file_hash(evaluation_root / "manifest.json"),
            ml_gate_manifest_hash,
            rollback_provenance,
        )
    elif (
        promotion_decision_path is not None
        or ml_gate_artifact_dir is not None
        or rollback_release_artifact_dir is not None
    ):
        raise ValueError("promotion_evidence_not_applicable")
    deployment_eligible = release_status == "production"
    allowed_score_splits = ["train", "validation"] if deployment_eligible else ["validation"]

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{release_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        shutil.copytree(model_root, staging / "model")
        shutil.copytree(preprocessor_root, staging / "preprocessor")
        shutil.copy2(evaluation_root / "evaluation_report.json", staging / "validation_evaluation.json")
        shutil.copy2(evaluation_root / "threshold_policy.json", staging / "threshold_policy.json")
        shutil.copytree(evaluation_root / "confusion_matrices", staging / "confusion_matrices")
        shutil.copytree(evaluation_root / "plots", staging / "plots")
        if decision_path is not None:
            shutil.copy2(decision_path, staging / "promotion_decision.json")
        if ml_gate_root is not None:
            shutil.copytree(ml_gate_root, staging / "ml_gate")
        if rollback_root is not None:
            shutil.copytree(rollback_root, staging / "rollback_release")
        model_label = str(model_manifest["model_type"])
        model_card = (
            f"# LearnGuard {release_status.replace('_', ' ')} release\n\n"
            f"Release: `{release_id}`  \n"
            f"Model type: `{model_label}`  \n"
            f"Model: `{model_manifest['model_semantic_version']}`  \n"
            f"Production threshold: `{float(threshold):.12g}`  \n"
            "Policy: scalar threshold derived from a validation Top-20% alert budget.  \n"
            + ("This candidate is validation-only and cannot write the production database.  \n"
               if not deployment_eligible else
               "This is the operational production model.  \n") +
            ("Promotion basis: governance exception over validation evidence; this is not a scientific superiority claim.  \n"
             if governance_upgrade else "") +
            (f"Verified Logistic rollback release: `{rollback_provenance['rollback_release_id']}`.  \n"
             if rollback_provenance is not None else "") +
            "Holdout 2014J was not reused for this release.  \n"
            "SHAP explanations are associations, not causal effects.\n"
        )
        (staging / "model_card.md").write_text(model_card)
        payload_files = [path for path in staging.rglob("*") if path.is_file()]
        release_manifest = {
            "schema_version": MODEL_RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_status": release_status,
            "production_release_eligible": deployment_eligible,
            "deployment_eligible": deployment_eligible,
            "database_write_allowed": deployment_eligible,
            "allowed_score_splits": allowed_score_splits,
            "eligibility_basis": evaluation["eligibility_basis"],
            "operational_role": (
                "primary" if is_xgboost and deployment_eligible else
                "fallback" if model_manifest.get("model_type") == LOGISTIC_MODEL_TYPE else
                "challenger"
            ),
            "evidence_status": "validation_only",
            "promotion_basis": (
                "governance_exception" if governance_upgrade else
                "incumbent_legacy" if deployment_eligible else
                "validation_only"
            ),
            "scientific_superiority_claim": False,
            "promotion_decision_sha256": (
                _file_hash(staging / "promotion_decision.json")
                if decision is not None else None
            ),
            "promotion_decision_id": decision.get("decision_id") if decision else None,
            "source_evaluation_manifest_sha256": _file_hash(
                evaluation_root / "manifest.json"
            ),
            "source_benchmark_manifest_sha256": evaluation.get(
                "validation_benchmark_manifest_sha256"
            ),
            "source_ml_gate_manifest_sha256": ml_gate_manifest_hash,
            **(rollback_provenance or {
                "rollback_release_id": None,
                "rollback_release_manifest_sha256": None,
                "rollback_model_version": None,
                "rollback_model_manifest_sha256": None,
                "rollback_model_artifact_sha256": None,
            }),
            "scientific_evaluation_scope": "validation_only",
            "model_type": model_manifest["model_type"],
            "model_version": model_manifest["model_semantic_version"],
            "model_contract_hash": model_manifest["model_contract_hash"],
            "model_artifact_dir": "model",
            "model_artifact_sha256": model_manifest["model_artifact_sha256"],
            "preprocessing_artifact_dir": "preprocessor",
            "preprocessor_version": preprocessor_manifest["preprocessor_version"],
            "preprocessor_artifact_sha256": preprocessor_manifest["artifact_sha256"],
            "snapshot_version": snapshot_version,
            "snapshot_ledger_sha256": ledger_sha256,
            "cutoff_day": cutoff_day,
            "target_column": model_manifest["target_column"],
            "positive_class": model_manifest["positive_class"],
            "negative_class": model_manifest["negative_class"],
            "output_feature_count": model_manifest["output_feature_count"],
            "output_feature_order_sha256": _feature_order_hash(
                model_manifest["output_feature_names"]
            ),
            "production_threshold": float(threshold),
            "threshold_policy": "validation_alert_budget_scalar_v1",
            "threshold_derivation_split": "validation",
            "target_alert_fraction": ALERT_BUDGET_FRACTION,
            "realized_validation_alert_rate": policy["realized_alert_rate"],
            "maximum_f1_comparator_threshold": policy["max_f1_comparator"]["threshold"],
            "validation_metrics": evaluation["production_metrics"],
            "random_seed": 42,
            "packages": model_manifest["packages"],
            "explanation": model_manifest["explanation"],
            "production_model_status": (
                "operational_primary" if is_xgboost and deployment_eligible else
                "verified_fallback" if deployment_eligible else
                "validation_release_candidate"
            ),
            "candidate_promotion_eligible": False,
            "holdout_reused": False,
            "holdout_access_policy": "forbidden_prior_consumed",
            "prior_holdout_exposure": (
                "2014J was consumed by a prior benchmark and is not evidence for this release"
            ),
            "files": {
                str(path.relative_to(staging)): _file_hash(path)
                for path in sorted(payload_files)
            },
        }
        manifest_path = staging / "release.manifest.json"
        manifest_path.write_text(json.dumps(release_manifest, indent=2, sort_keys=True))
        (staging / "release.success.json").write_text(json.dumps({
            "schema_version": MODEL_RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "manifest_sha256": _file_hash(manifest_path),
            "payload_file_count": len(payload_files),
        }, indent=2, sort_keys=True))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ModelReleaseResult(
        "ok", release_id, target, str(model_manifest["model_semantic_version"]),
        str(preprocessor_manifest["preprocessor_version"]), float(threshold),
        target / "release.manifest.json",
    )


def load_and_verify_model_release(
    release_artifact_dir: Path | str,
    snapshot_path: Path | str,
) -> LoadedModelRelease:
    root = Path(release_artifact_dir)
    snapshot_path = Path(snapshot_path)
    manifest_path = root / "release.manifest.json"
    success_path = root / "release.success.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError("model_release_artifact_missing")
    manifest = _read_json(manifest_path)
    success = _read_json(success_path)
    schema = manifest.get("schema_version")
    if schema not in {
        LEGACY_MODEL_RELEASE_SCHEMA_VERSION,
        LEGACY_MODEL_RELEASE_SCHEMA_VERSION_V2,
        LEGACY_MODEL_RELEASE_SCHEMA_VERSION_V3,
        MODEL_RELEASE_SCHEMA_VERSION,
    }:
        raise ValueError("model_release_schema_unsupported")
    if success.get("manifest_sha256") != _file_hash(manifest_path):
        raise ValueError("model_release_manifest_hash_mismatch")
    files = _verify_file_map(root, manifest.get("files"))
    if int(success.get("payload_file_count", -1)) != len(files):
        raise ValueError("model_release_payload_count_mismatch")
    if schema == LEGACY_MODEL_RELEASE_SCHEMA_VERSION:
        if manifest.get("model_type") != LOGISTIC_MODEL_TYPE:
            raise ValueError("model_release_production_model_unsupported")
        if (
            manifest.get("production_release_eligible") is not True
            or manifest.get("random_forest_b_status") != "validation_release_candidate"
            or manifest.get("candidate_promotion_eligible") is not False
            or manifest.get("holdout_reused") is not False
            or manifest.get("holdout_access_policy") != "forbidden_prior_consumed"
        ):
            raise ValueError("model_release_policy_invalid")
        release_status = "production"
        database_write_allowed = True
        allowed_score_splits = ("train", "validation")
        operational_role = "primary"
        evidence_status = "legacy_validation_policy"
    else:
        release_status = str(manifest.get("release_status"))
        if manifest.get("model_type") not in {LOGISTIC_MODEL_TYPE, XGBOOST_MODEL_TYPE}:
            raise ValueError("model_release_production_model_unsupported")
        deployment_eligible = manifest.get("deployment_eligible")
        database_write_allowed = manifest.get("database_write_allowed")
        raw_allowed = manifest.get("allowed_score_splits")
        if (
            release_status not in {"production", "validation_candidate"}
            or not isinstance(deployment_eligible, bool)
            or not isinstance(database_write_allowed, bool)
            or not isinstance(raw_allowed, list)
            or not raw_allowed
            or not all(value in {"train", "validation"} for value in raw_allowed)
            or manifest.get("candidate_promotion_eligible") is not False
            or manifest.get("holdout_reused") is not False
            or manifest.get("holdout_access_policy") != "forbidden_prior_consumed"
        ):
            raise ValueError("model_release_policy_invalid")
        if release_status == "validation_candidate" and (
            deployment_eligible is not False
            or database_write_allowed is not False
            or raw_allowed != ["validation"]
        ):
            raise ValueError("model_release_candidate_policy_invalid")
        if release_status == "production" and (
            deployment_eligible is not True or database_write_allowed is not True
        ):
            raise ValueError("model_release_production_policy_invalid")
        if (
            schema == LEGACY_MODEL_RELEASE_SCHEMA_VERSION_V2
            and manifest.get("model_type") == XGBOOST_MODEL_TYPE
            and release_status != "validation_candidate"
        ):
            raise ValueError("xgboost_production_promotion_forbidden")
        if (
            schema == LEGACY_MODEL_RELEASE_SCHEMA_VERSION_V3
            and manifest.get("model_type") == XGBOOST_MODEL_TYPE
            and release_status == "production"
        ):
            raise ValueError("xgboost_production_release_upgrade_required")
        if schema in {LEGACY_MODEL_RELEASE_SCHEMA_VERSION_V3, MODEL_RELEASE_SCHEMA_VERSION}:
            operational_role = manifest.get("operational_role")
            evidence_status = manifest.get("evidence_status")
            if (
                operational_role not in {"primary", "fallback", "challenger"}
                or evidence_status != "validation_only"
                or manifest.get("scientific_superiority_claim") is not False
            ):
                raise ValueError("model_release_evidence_policy_invalid")
            if (
                schema == MODEL_RELEASE_SCHEMA_VERSION
                and manifest.get("model_type") == XGBOOST_MODEL_TYPE
                and release_status == "production"
            ):
                decision_path = root / "promotion_decision.json"
                if (
                    manifest.get("promotion_basis") != "governance_exception"
                    or not decision_path.is_file()
                    or manifest.get("promotion_decision_sha256") != _file_hash(decision_path)
                ):
                    raise ValueError("xgboost_production_governance_invalid")
                decision = _read_json(decision_path)
                embedded_evaluation = _read_json(root / "validation_evaluation.json")
                rollback_root = root / "rollback_release"
                rollback_manifest_path = rollback_root / "release.manifest.json"
                if not rollback_manifest_path.is_file():
                    raise ValueError("xgboost_rollback_release_missing")
                raw_rollback_manifest = _read_json(rollback_manifest_path)
                if raw_rollback_manifest.get("model_type") != LOGISTIC_MODEL_TYPE:
                    raise ValueError("xgboost_rollback_release_invalid")
                rollback = load_and_verify_model_release(rollback_root, snapshot_path)
                if (
                    rollback.release_status != "production"
                    or rollback.database_write_allowed is not True
                    or rollback.holdout_access_policy != "forbidden_prior_consumed"
                ):
                    raise ValueError("xgboost_rollback_release_invalid")
                rollback_model_manifest_path = (
                    rollback.model_artifact_dir / "model.manifest.json"
                )
                rollback_model_manifest = _read_json(rollback_model_manifest_path)
                rollback_provenance = {
                    "rollback_release_id": rollback.release_id,
                    "rollback_release_manifest_sha256": _file_hash(rollback_manifest_path),
                    "rollback_model_version": rollback.model_version,
                    "rollback_model_manifest_sha256": _file_hash(
                        rollback_model_manifest_path
                    ),
                    "rollback_model_artifact_sha256": rollback_model_manifest.get(
                        "model_artifact_sha256"
                    ),
                }
                _, embedded_gate_hash = _verify_ml_gate_artifact(
                    root / "ml_gate", manifest,
                    baseline_model_manifest=rollback_model_manifest,
                    baseline_model_manifest_sha256=str(
                        rollback_provenance["rollback_model_manifest_sha256"]
                    ),
                )
                approval_identity_fields = ("decision_id", "approved_by", "approved_at")
                if (
                    decision.get("schema_version") != PROMOTION_DECISION_SCHEMA_VERSION
                    or any(
                        not isinstance(decision.get(field), str) or not decision[field]
                        for field in approval_identity_fields
                    )
                    or decision.get("decision_id") != manifest.get("promotion_decision_id")
                    or decision.get("model_type") != manifest.get("model_type")
                    or decision.get("model_version") != manifest.get("model_version")
                    or decision.get("model_contract_hash") != manifest.get("model_contract_hash")
                    or decision.get("snapshot_version") != manifest.get("snapshot_version")
                    or decision.get("preprocessor_version") != manifest.get("preprocessor_version")
                    or decision.get("benchmark_manifest_sha256")
                    != manifest.get("source_benchmark_manifest_sha256")
                    or decision.get("benchmark_manifest_sha256")
                    != embedded_evaluation.get("validation_benchmark_manifest_sha256")
                    or decision.get("evaluation_manifest_sha256")
                    != manifest.get("source_evaluation_manifest_sha256")
                    or decision.get("ml_gate_manifest_sha256")
                    != manifest.get("source_ml_gate_manifest_sha256")
                    or embedded_gate_hash != manifest.get("source_ml_gate_manifest_sha256")
                    or any(
                        manifest.get(key) != value
                        or decision.get(key) != value
                        for key, value in rollback_provenance.items()
                    )
                    or decision.get("basis") != "operational_primary_validation_only"
                    or decision.get("acknowledges_consumed_holdout") is not True
                    or decision.get("scientific_superiority_claim") is not False
                ):
                    raise ValueError("xgboost_production_governance_invalid")
        else:
            operational_role = (
                "primary" if release_status == "production" else "challenger"
            )
            evidence_status = "validation_only"
        allowed_score_splits = tuple(str(value) for value in raw_allowed)
    threshold = manifest.get("production_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
        raise ValueError("model_release_threshold_invalid")
    threshold_policy = _read_json(root / "threshold_policy.json")
    validation_evaluation = _read_json(root / "validation_evaluation.json")
    policy_threshold = threshold_policy.get("production_threshold")
    evaluation_threshold = validation_evaluation.get("production_threshold")
    if (
        threshold_policy.get("policy") != "validation_alert_budget_scalar_v1"
        or threshold_policy.get("derivation_split") != "validation"
        or threshold_policy.get("holdout_used") is not False
        or validation_evaluation.get("evaluation_scope") != "validation_only"
        or validation_evaluation.get("holdout_accessed") is not False
        or validation_evaluation.get("holdout_reused") is not False
    ):
        raise ValueError("model_release_embedded_policy_invalid")
    if (
        isinstance(policy_threshold, bool)
        or not isinstance(policy_threshold, (int, float))
        or isinstance(evaluation_threshold, bool)
        or not isinstance(evaluation_threshold, (int, float))
        or float(threshold) != float(policy_threshold)
        or float(threshold) != float(evaluation_threshold)
    ):
        raise ValueError("model_release_threshold_policy_mismatch")
    if (
        manifest.get("threshold_policy") != threshold_policy.get("policy")
        or manifest.get("threshold_derivation_split") != threshold_policy.get("derivation_split")
        or manifest.get("target_alert_fraction") != threshold_policy.get("target_fraction")
        or manifest.get("realized_validation_alert_rate") != threshold_policy.get("realized_alert_rate")
    ):
        raise ValueError("model_release_threshold_metadata_mismatch")
    snapshot_version, cutoff_day, ledger_sha256 = _ledger_metadata(snapshot_path)
    if (
        manifest.get("snapshot_version") != snapshot_version
        or manifest.get("cutoff_day") != cutoff_day
        or manifest.get("snapshot_ledger_sha256") != ledger_sha256
    ):
        raise ValueError("model_release_snapshot_mismatch")
    model_dir = _safe_relative(root, str(manifest.get("model_artifact_dir")))
    preprocessor_dir = _safe_relative(root, str(manifest.get("preprocessing_artifact_dir")))
    loaded = load_scoring_model(model_dir, snapshot_path, preprocessor_dir)
    embedded_manifest = _read_json(model_dir / "model.manifest.json")
    if (
        embedded_manifest.get("model_semantic_version") != manifest.get("model_version")
        or embedded_manifest.get("model_type") != manifest.get("model_type")
        or embedded_manifest.get("preprocessor_version") != manifest.get("preprocessor_version")
        or _feature_order_hash(list(loaded.feature_names)) != manifest.get("output_feature_order_sha256")
    ):
        raise ValueError("model_release_embedded_artifact_mismatch")
    return LoadedModelRelease(
        release_id=str(manifest["release_id"]),
        artifact_dir=root,
        model_artifact_dir=model_dir,
        preprocessing_artifact_dir=preprocessor_dir,
        model_version=str(manifest["model_version"]),
        snapshot_version=snapshot_version,
        production_threshold=float(threshold),
        holdout_access_policy=str(manifest["holdout_access_policy"]),
        manifest=manifest,
        release_status=release_status,
        database_write_allowed=bool(database_write_allowed),
        allowed_score_splits=allowed_score_splits,
        operational_role=str(operational_role),
        evidence_status=str(evidence_status),
    )
