"""Validation-only evaluation and immutable Logistic production releases."""

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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from learnguard.ml.evaluation import (
    select_alert_budget_threshold,
    threshold_metrics,
)
from learnguard.ml.preprocessing import _file_hash, _validate_run_id
from learnguard.ml.training import load_logistic_model_artifact

EVALUATION_RUN_SCHEMA_VERSION = "validation-release-evaluation.v1"
MODEL_RELEASE_SCHEMA_VERSION = "model-release.v1"
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
    loaded = load_logistic_model_artifact(model_root, snapshot_path, preprocessor_root)
    model_manifest = dict(loaded.manifest)
    preprocessor_manifest = _read_json(preprocessor_root / "preprocessor.manifest.json")
    configs = _read_json(benchmark_root / "candidate_configurations.json")
    logistic_config = configs.get(PRODUCTION_CANDIDATE)
    if not isinstance(logistic_config, dict):
        raise ValueError("validation_logistic_candidate_missing")
    if (
        benchmark_report.get("snapshot_version") != model_manifest.get("snapshot_version")
        or benchmark_report.get("preprocessor_version") != model_manifest.get("preprocessor_version")
        or preprocessor_manifest.get("preprocessor_version") != model_manifest.get("preprocessor_version")
    ):
        raise ValueError("validation_provenance_version_mismatch")
    candidate_model = benchmark_root / "candidates" / f"{PRODUCTION_CANDIDATE}.joblib"
    production_model = model_root / "model.joblib"
    if _file_hash(candidate_model) != _file_hash(production_model):
        raise ValueError("validation_production_model_mismatch")
    if logistic_config.get("parameters") != model_manifest.get("estimator_parameters"):
        raise ValueError("validation_production_parameters_mismatch")

    predictions = pd.read_parquet(benchmark_root / "validation_predictions.parquet")
    required = {
        "student_key", "code_module", "code_presentation", "split", "label",
        f"risk_score__{PRODUCTION_CANDIDATE}",
    }
    if not required <= set(predictions) or "id_student" in predictions:
        raise ValueError("validation_predictions_schema_invalid")
    if set(predictions["split"].astype(str)) != {"validation"}:
        raise ValueError("validation_predictions_scope_invalid")
    y_true = predictions["label"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(y_true), np.array([0, 1])):
        raise ValueError("validation_labels_invalid")
    logistic_scores = predictions[f"risk_score__{PRODUCTION_CANDIDATE}"].to_numpy(
        dtype=np.float64,
    )
    if _probability_hash(logistic_scores) != logistic_config.get("validation_probability_sha256"):
        raise ValueError("validation_probability_hash_mismatch")

    keys = predictions[["code_module", "code_presentation", "student_key"]]
    budget = select_alert_budget_threshold(logistic_scores, keys, ALERT_BUDGET_FRACTION)
    allowed_deviation = max(MAX_ALERT_BUDGET_DEVIATION, 1.0 / len(y_true))
    if budget.absolute_deviation > allowed_deviation:
        raise ValueError(
            f"alert_budget_scalar_deviation_exceeded:{budget.absolute_deviation}"
        )
    production_metrics = asdict(threshold_metrics(y_true, logistic_scores, budget.threshold))
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
            staging, PRODUCTION_CANDIDATE, "production-top20",
            budget.threshold, production_metrics["confusion_matrix"],
        )
        metric_rows.append({
            "candidate": PRODUCTION_CANDIDATE,
            "policy": "production-top20",
            **production_metrics,
        })
        pd.DataFrame(metric_rows).drop(columns=["confusion_matrix"]).to_csv(
            staging / "validation_metrics.csv", index=False,
        )
        max_f1 = logistic_config["validation_metrics"]["selected_threshold"]
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
            "validation_predictions_sha256": _file_hash(
                benchmark_root / "validation_predictions.parquet"
            ),
            "validation_probability_sha256": _probability_hash(logistic_scores),
            "candidate_count": len(candidate_names),
            "production_model": PRODUCTION_CANDIDATE,
            "validation_leader": benchmark_report["validation_leader"],
            "random_forest_b_status": "validation_release_candidate",
            "production_threshold": budget.threshold,
            "production_metrics": production_metrics,
            "holdout_accessed": False,
            "holdout_reused": False,
            "production_release_eligible": True,
            "eligibility_basis": "incumbent_logistic_validation_policy",
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
    if manifest.get("schema_version") != EVALUATION_RUN_SCHEMA_VERSION:
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
    loaded = load_logistic_model_artifact(model_root, snapshot_path, preprocessor_root)
    model_manifest = dict(loaded.manifest)
    preprocessor_manifest = _read_json(preprocessor_root / "preprocessor.manifest.json")
    snapshot_version, cutoff_day, ledger_sha256 = _ledger_metadata(snapshot_path)
    if (
        evaluation.get("model_version") != model_manifest.get("model_semantic_version")
        or evaluation.get("model_artifact_sha256") != _file_hash(model_root / "model.joblib")
        or evaluation.get("snapshot_version") != snapshot_version
        or evaluation.get("preprocessor_version") != preprocessor_manifest.get("preprocessor_version")
    ):
        raise ValueError("model_release_provenance_mismatch")
    threshold = policy.get("production_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
        raise ValueError("model_release_threshold_invalid")

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
        (staging / "model_card.md").write_text(
            "# LearnGuard Logistic production release\n\n"
            f"Release: `{release_id}`  \n"
            f"Model: `{model_manifest['model_semantic_version']}`  \n"
            f"Production threshold: `{float(threshold):.12g}`  \n"
            "Policy: scalar threshold derived from a validation Top-20% alert budget.  \n"
            "Logistic Regression remains the incumbent production model.  \n"
            "Random Forest B is a validation-only release candidate and was not promoted.  \n"
            "Holdout 2014J was not reused for this release.  \n"
            "SHAP explanations are associations, not causal effects.\n"
        )
        payload_files = [path for path in staging.rglob("*") if path.is_file()]
        release_manifest = {
            "schema_version": MODEL_RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_status": "production",
            "production_release_eligible": True,
            "eligibility_basis": "incumbent_logistic_validation_policy",
            "scientific_evaluation_scope": "validation_only",
            "model_type": model_manifest["model_type"],
            "model_version": model_manifest["model_semantic_version"],
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
            "production_model_status": "incumbent_production",
            "random_forest_b_status": "validation_release_candidate",
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
    if manifest.get("schema_version") != MODEL_RELEASE_SCHEMA_VERSION:
        raise ValueError("model_release_schema_unsupported")
    if success.get("manifest_sha256") != _file_hash(manifest_path):
        raise ValueError("model_release_manifest_hash_mismatch")
    files = _verify_file_map(root, manifest.get("files"))
    if int(success.get("payload_file_count", -1)) != len(files):
        raise ValueError("model_release_payload_count_mismatch")
    if manifest.get("model_type") != "sklearn.linear_model.LogisticRegression":
        raise ValueError("model_release_production_model_unsupported")
    if (
        manifest.get("production_release_eligible") is not True
        or manifest.get("random_forest_b_status") != "validation_release_candidate"
        or manifest.get("candidate_promotion_eligible") is not False
        or manifest.get("holdout_reused") is not False
        or manifest.get("holdout_access_policy") != "forbidden_prior_consumed"
    ):
        raise ValueError("model_release_policy_invalid")
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
    loaded = load_logistic_model_artifact(model_dir, snapshot_path, preprocessor_dir)
    if (
        loaded.manifest.get("model_semantic_version") != manifest.get("model_version")
        or loaded.manifest.get("preprocessor_version") != manifest.get("preprocessor_version")
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
    )
