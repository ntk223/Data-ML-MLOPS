"""Deterministically publish the three-model validation comparison."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from learnguard.data.loading import load_snapshot_partitions
from learnguard.ml.model_interface import load_scoring_model, verify_model_artifact
from learnguard.ml.preprocessing import _file_hash, _prepare_split
from learnguard.ml.release import _read_json, _verify_benchmark_artifact
from learnguard.ml.xgboost_training import XGBOOST_PRODUCTION_CANDIDATE

REQUIRED_CANDIDATES = (
    XGBOOST_PRODUCTION_CANDIDATE,
    "logistic_baseline",
    "random_forest_b",
)


@dataclass(frozen=True)
class ModelComparisonResult:
    status: str
    output_path: Path
    benchmark_run_id: str
    benchmark_manifest_sha256: str
    candidate_count: int


def _probability_hash(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _format(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _validate_metric_consistency(
    benchmark_root: Path,
    selected: pd.DataFrame,
    configs: dict[str, object],
) -> None:
    """Bind rendered CSV metrics to configs, probabilities, and payload sizes."""
    direct = (
        "sample_count", "roc_auc", "pr_auc", "recall_at_top_10_pct",
        "precision_at_top_10_pct", "recall_at_top_20_pct",
        "precision_at_top_20_pct", "brier_score", "ece",
    )
    rows = selected.set_index("candidate")
    for name in REQUIRED_CANDIDATES:
        config = configs[name]
        assert isinstance(config, dict)
        source = config.get("validation_metrics")
        if not isinstance(source, dict):
            raise ValueError(f"comparison_source_metrics_missing:{name}")
        row = rows.loc[name]
        for key in direct:
            if not np.isclose(float(row[key]), float(source[key]), atol=1e-9, rtol=0):
                raise ValueError(f"comparison_metric_source_mismatch:{name}:{key}")
        selected_threshold = source.get("selected_threshold")
        if not isinstance(selected_threshold, dict):
            raise ValueError(f"comparison_source_threshold_missing:{name}")
        threshold_map = {
            "threshold": "threshold", "precision_at_selected": "precision",
            "recall_at_selected": "recall", "f1_at_selected": "f1",
        }
        for column, key in threshold_map.items():
            if not np.isclose(
                float(row[column]), float(selected_threshold[key]), atol=1e-9, rtol=0,
            ):
                raise ValueError(f"comparison_metric_source_mismatch:{name}:{column}")
        payload = benchmark_root / "candidates" / f"{name}.joblib"
        if int(row["artifact_bytes"]) != payload.stat().st_size:
            raise ValueError(f"comparison_artifact_size_mismatch:{name}")
        if not np.isclose(
            float(row["inference_rows_per_second"]),
            float(row["sample_count"]) / float(row["inference_seconds"]),
            rtol=1e-9, atol=1e-9,
        ):
            raise ValueError(f"comparison_throughput_mismatch:{name}")


def _metric_summary(rows: pd.DataFrame, left: str, right: str) -> str:
    comparisons = (
        ("PR-AUC", "pr_auc", True),
        ("ROC-AUC", "roc_auc", True),
        ("Top-20% precision", "precision_at_top_20_pct", True),
        ("Top-20% recall", "recall_at_top_20_pct", True),
        ("Brier score", "brier_score", False),
        ("ECE", "ece", False),
    )
    wins, losses, ties = [], [], []
    for label, key, higher_is_better in comparisons:
        lvalue, rvalue = float(rows.loc[left, key]), float(rows.loc[right, key])
        if np.isclose(lvalue, rvalue, atol=1e-12, rtol=0):
            ties.append(label)
        elif (lvalue > rvalue) == higher_is_better:
            wins.append(label)
        else:
            losses.append(label)
    parts = []
    if wins:
        parts.append("better on " + ", ".join(wins))
    if losses:
        parts.append("worse on " + ", ".join(losses))
    if ties:
        parts.append("tied on " + ", ".join(ties))
    return "; ".join(parts)


def _verify_candidate_model(
    name: str,
    root: Path,
    snapshot_path: Path,
    preprocessing_root: Path,
    report: dict[str, object],
    config: dict[str, object],
    validation_frame: pd.DataFrame,
) -> dict[str, object]:
    verify_model_artifact(root, snapshot_path, preprocessing_root)
    manifest = _read_json(root / "model.manifest.json")
    if (
        manifest.get("snapshot_version") != report.get("snapshot_version")
        or manifest.get("preprocessor_version") != report.get("preprocessor_version")
        or manifest.get("estimator_parameters") != config.get("parameters")
    ):
        raise ValueError(f"comparison_model_provenance_mismatch:{name}")
    candidate_class = str(config.get("class"))
    if candidate_class.rsplit(".", 1)[-1] != str(manifest.get("model_type")).rsplit(".", 1)[-1]:
        raise ValueError(f"comparison_model_type_mismatch:{name}")
    prepared = _prepare_split("validation", validation_frame)
    loaded = load_scoring_model(root, snapshot_path, preprocessing_root)
    probabilities = loaded.predict_positive_proba(prepared.X)
    if _probability_hash(probabilities) != config.get("validation_probability_sha256"):
        raise ValueError(f"comparison_probability_hash_mismatch:{name}")
    return manifest


def publish_model_comparison(
    benchmark_artifact_dir: Path | str,
    xgboost_model_artifact_dir: Path | str,
    logistic_model_artifact_dir: Path | str,
    snapshot_path: Path | str,
    preprocessing_artifact_dir: Path | str,
    output_path: Path | str = Path("docs/MODEL_BENCHMARK_COMPARISON.md"),
) -> ModelComparisonResult:
    benchmark_root = Path(benchmark_artifact_dir)
    snapshot_path = Path(snapshot_path)
    preprocessing_root = Path(preprocessing_artifact_dir)
    output_path = Path(output_path)
    report, benchmark_manifest = _verify_benchmark_artifact(benchmark_root)
    if report.get("xgboost_candidates_included") is not True:
        raise ValueError("comparison_benchmark_missing_xgboost_scope")
    required_sources = {
        "candidate_configurations.json", "validation_metrics.csv",
        "validation_slice_metrics.csv", "validation_predictions.parquet",
        "validation_selection.json", "benchmark_report.json",
    }
    if not required_sources <= set(benchmark_manifest.get("files", {})):
        raise ValueError("comparison_benchmark_source_map_incomplete")
    configs = _read_json(benchmark_root / "candidate_configurations.json")
    missing = [name for name in REQUIRED_CANDIDATES if not isinstance(configs.get(name), dict)]
    if missing:
        raise ValueError(f"comparison_candidate_missing:{','.join(missing)}")

    metrics = pd.read_csv(benchmark_root / "validation_metrics.csv")
    slices = pd.read_csv(benchmark_root / "validation_slice_metrics.csv")
    if not set(REQUIRED_CANDIDATES) <= set(metrics["candidate"].astype(str)):
        raise ValueError("comparison_metrics_candidate_missing")
    if not set(REQUIRED_CANDIDATES) <= set(slices["candidate"].astype(str)):
        raise ValueError("comparison_slices_candidate_missing")
    selected = metrics[metrics["candidate"].isin(REQUIRED_CANDIDATES)].copy()
    if len(selected) != len(REQUIRED_CANDIDATES) or len(set(selected["sample_count"])) != 1:
        raise ValueError("comparison_metrics_shape_invalid")
    _validate_metric_consistency(benchmark_root, selected, configs)

    validation = load_snapshot_partitions(snapshot_path, ("validation",))
    if validation.snapshot_version != report.get("snapshot_version"):
        raise ValueError("comparison_snapshot_version_mismatch")
    validation_frame = validation.frame("validation")
    manifests = {
        XGBOOST_PRODUCTION_CANDIDATE: _verify_candidate_model(
            XGBOOST_PRODUCTION_CANDIDATE, Path(xgboost_model_artifact_dir),
            snapshot_path, preprocessing_root, report,
            configs[XGBOOST_PRODUCTION_CANDIDATE], validation_frame,
        ),
        "logistic_baseline": _verify_candidate_model(
            "logistic_baseline", Path(logistic_model_artifact_dir),
            snapshot_path, preprocessing_root, report,
            configs["logistic_baseline"], validation_frame,
        ),
    }
    selection = _read_json(benchmark_root / "validation_selection.json")
    generated_at = str(selection.get("selected_at"))
    manifest_hash = _file_hash(benchmark_root / "manifest.json")
    file_hashes = {
        name: _file_hash(benchmark_root / name)
        for name in (
            "candidate_configurations.json", "validation_metrics.csv",
            "validation_slice_metrics.csv", "validation_predictions.parquet",
        )
    }
    explanations = {
        XGBOOST_PRODUCTION_CANDIDATE: manifests[XGBOOST_PRODUCTION_CANDIDATE]["explanation"]["method"],
        "logistic_baseline": manifests["logistic_baseline"]["explanation"]["method"],
        "random_forest_b": "benchmark-only; no production explainer",
    }
    statuses = {
        XGBOOST_PRODUCTION_CANDIDATE: "selected production contract; release approval not evaluated",
        "logistic_baseline": "fallback contract; release activation not evaluated",
        "random_forest_b": "benchmark challenger only",
    }
    rows = selected.set_index("candidate")
    fnr_gaps = slices[slices["candidate"].isin(REQUIRED_CANDIDATES)].groupby(
        "candidate"
    )["eligible_fnr_gap"].max()
    xgb_vs_logistic = _metric_summary(
        rows, XGBOOST_PRODUCTION_CANDIDATE, "logistic_baseline",
    )
    xgb_vs_rf = _metric_summary(
        rows, XGBOOST_PRODUCTION_CANDIDATE, "random_forest_b",
    )
    size_relation = (
        "smaller" if rows.loc[XGBOOST_PRODUCTION_CANDIDATE, "artifact_bytes"]
        < rows.loc["random_forest_b", "artifact_bytes"] else "not smaller"
    )
    throughput_relation = (
        "higher" if rows.loc[XGBOOST_PRODUCTION_CANDIDATE, "inference_rows_per_second"]
        > rows.loc["random_forest_b", "inference_rows_per_second"] else "not higher"
    )
    fairness_relation = (
        "narrower" if fnr_gaps[XGBOOST_PRODUCTION_CANDIDATE]
        < fnr_gaps["random_forest_b"] else "not narrower"
    )
    columns = (
        ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
        ("Precision@10%", "precision_at_top_10_pct"),
        ("Recall@10%", "recall_at_top_10_pct"),
        ("Precision@20%", "precision_at_top_20_pct"),
        ("Recall@20%", "recall_at_top_20_pct"),
        ("Brier", "brier_score"), ("ECE", "ece"),
        ("Max-F1 threshold", "threshold"),
        ("Precision@threshold", "precision_at_selected"),
        ("Recall@threshold", "recall_at_selected"),
        ("F1@threshold", "f1_at_selected"),
        ("Train seconds", "train_seconds"),
        ("Inference seconds", "inference_seconds"),
        ("Rows/second", "inference_rows_per_second"),
        ("Artifact bytes", "artifact_bytes"),
    )
    lines = [
        "# LearnGuard model benchmark comparison", "",
        f"Generated from immutable validation artifact at `{generated_at}`.", "",
        f"- Benchmark run: `{report['run_id']}`",
        f"- Snapshot: `{report['snapshot_version']}`",
        f"- Preprocessing: `{report['preprocessor_version']}`",
        "- Split: `validation` only",
        f"- Rows: `{int(selected.iloc[0]['sample_count'])}`",
        f"- Benchmark manifest SHA-256: `{manifest_hash}`",
        f"- Package versions: `{json.dumps(report['package_versions'], sort_keys=True)}`",
        "",
        "> Holdout 2014J was previously consumed and was not read by this run. "
        "This document does not claim superiority on an untouched holdout.", "",
        "## Metrics", "",
        "| Model | " + " | ".join(label for label, _ in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for name in REQUIRED_CANDIDATES:
        row = rows.loc[name]
        lines.append("| `" + name + "` | " + " | ".join(
            _format(row[key]) for _, key in columns
        ) + " |")
    lines += ["", "## Fairness slices", "",
              "| Model | Slice | Group | N | Positives | Precision | Recall | FNR | Eligible FNR gap |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    fairness = slices[slices["candidate"].isin(REQUIRED_CANDIDATES)].sort_values(
        ["candidate", "slice", "group"], kind="mergesort",
    )
    for row in fairness.itertuples(index=False):
        lines.append(
            f"| `{row.candidate}` | {row.slice} | {row.group} | {row.sample_count} | "
            f"{row.positive_count} | {_format(row.precision)} | {_format(row.recall)} | "
            f"{_format(row.false_negative_rate)} | {_format(row.eligible_fnr_gap)} |"
        )
    lines += ["", "## Contracts and operational status", "",
              "| Model | Hyperparameters | Explanation | Status |",
              "|---|---|---|---|"]
    for name in REQUIRED_CANDIDATES:
        params = json.dumps(configs[name]["parameters"], sort_keys=True, separators=(",", ":"), default=str)
        lines.append(f"| `{name}` | `{params}` | {explanations[name]} | {statuses[name]} |")
    lines += [
        "", "## Decision", "",
        "The repository selects `xgboost_regularized` as its production training contract; "
        "this comparison does not verify that a production release has been approved or activated.", "",
        f"Against Logistic on this validation artifact, XGBoost is {xgb_vs_logistic}. "
        "Logistic remains the rollback contract.", "",
        f"Against Random Forest B, XGBoost is {xgb_vs_rf}. Its benchmark payload is "
        f"{size_relation}, its measured throughput is {throughput_relation}, and its maximum "
        f"eligible FNR gap is {fairness_relation}. Random Forest remains benchmark-only because "
        "no production adapter/release contract is supplied to this command.", "",
        "Promotion is a governance exception over validation evidence, not scientific proof of "
        "superiority. Activation requires a separately verified promotion-decision artifact; a "
        "new untouched temporal cohort is required to change the scientific evidence status.",
        "", "## Source hashes", "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in sorted(file_hashes.items()))
    content = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(content)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return ModelComparisonResult(
        "ok", output_path, str(report["run_id"]), manifest_hash, len(REQUIRED_CANDIDATES),
    )
