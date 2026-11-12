"""Bounded, leakage-safe model benchmark with one-shot holdout evaluation."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from threadpoolctl import threadpool_limits

from learnguard.data.loading import load_snapshot_partitions
from learnguard.ml.evaluation import binary_metrics, fairness_slices, select_f1_threshold
from learnguard.ml.preprocessing import (
    _file_hash, _prepare_split, _validate_run_id, load_preprocessing_artifact,
)
from learnguard.ml.scoring import _pseudonymous_key
from learnguard.ml.training import LOGISTIC_BASELINE_PARAMS

BENCHMARK_SCHEMA_VERSION = "model-benchmark.v2"


@dataclass(frozen=True)
class BenchmarkReport:
    status: str
    run_id: str
    artifact_dir: Path
    winner: str | None
    validation_leader: str
    selected_threshold: float
    candidate_count: int
    evaluation_scope: str
    selection_path: Path
    locked_config_path: Path | None
    holdout_report_path: Path | None


def _candidate_models() -> list[tuple[str, object]]:
    return [
        ("dummy_prior", DummyClassifier(strategy="prior", random_state=42)),
        ("logistic_baseline", LogisticRegression(**LOGISTIC_BASELINE_PARAMS)),
        ("extra_trees_a", ExtraTreesClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=1,
            class_weight=None, random_state=42, n_jobs=1,
        )),
        ("extra_trees_b", ExtraTreesClassifier(
            n_estimators=400, max_features=0.7, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=1,
        )),
        ("hist_gradient_boosting_a", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        )),
        ("hist_gradient_boosting_b", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42,
        )),
        ("hist_gradient_boosting_c", HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=200, max_leaf_nodes=15,
            l2_regularization=5.0, random_state=42,
        )),
        ("random_forest_a", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", class_weight=None, bootstrap=True,
            random_state=42, n_jobs=1,
        )),
        ("random_forest_b", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", class_weight="balanced_subsample", bootstrap=True,
            random_state=42, n_jobs=1,
        )),
        ("random_forest_c", RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=10,
            max_features=0.7, class_weight="balanced_subsample", bootstrap=True,
            random_state=42, n_jobs=1,
        )),
    ]


def _scores(model: object, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if values.shape != (len(matrix), 2) or not np.isfinite(values).all():
        raise ValueError("benchmark_probability_invalid")
    return values[:, 1]


def _flat_row(name: str, metrics: dict[str, object], threshold: float,
              train_seconds: float, inference_seconds: float,
              artifact_bytes: int) -> dict[str, object]:
    at_half = metrics["threshold_0_5"]
    at_selected = metrics["selected_threshold"]
    assert isinstance(at_half, dict) and isinstance(at_selected, dict)
    return {
        "candidate": name, "sample_count": metrics["sample_count"],
        "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
        "recall_at_top_10_pct": metrics["recall_at_top_10_pct"],
        "precision_at_top_10_pct": metrics["precision_at_top_10_pct"],
        "recall_at_top_20_pct": metrics["recall_at_top_20_pct"],
        "precision_at_top_20_pct": metrics["precision_at_top_20_pct"],
        "brier_score": metrics["brier_score"], "ece": metrics["ece"],
        "threshold": threshold, "precision_at_0_5": at_half["precision"],
        "recall_at_0_5": at_half["recall"], "f1_at_0_5": at_half["f1"],
        "precision_at_selected": at_selected["precision"],
        "recall_at_selected": at_selected["recall"],
        "f1_at_selected": at_selected["f1"],
        "train_seconds": train_seconds, "inference_seconds": inference_seconds,
        "inference_rows_per_second": len(range(int(metrics["sample_count"]))) / max(inference_seconds, 1e-12),
        "artifact_bytes": artifact_bytes,
    }


def _selection_key(row: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        float(row["pr_auc"]),
        float(row["recall_at_top_20_pct"]),
        -float(row["ece"]),
        -float(row["inference_seconds"]),
    )


def _validation_plots(plot_dir: Path, validation_y: np.ndarray,
                      validation_scores: np.ndarray,
                      validation_table: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/learnguard-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RocCurveDisplay.from_predictions(validation_y, validation_scores)
    plt.savefig(plot_dir / "validation_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    PrecisionRecallDisplay.from_predictions(validation_y, validation_scores)
    plt.savefig(plot_dir / "validation_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    CalibrationDisplay.from_predictions(
        validation_y, validation_scores, n_bins=10, strategy="uniform",
    )
    plt.savefig(plot_dir / "validation_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    chart = validation_table.sort_values("pr_auc", ascending=True)
    plt.barh(chart["candidate"], chart["pr_auc"])
    plt.xlabel("Validation PR-AUC")
    plt.tight_layout()
    plt.savefig(plot_dir / "validation_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def _holdout_plots(plot_dir: Path, holdout_y: np.ndarray,
                   holdout_scores: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    RocCurveDisplay.from_predictions(holdout_y, holdout_scores)
    plt.savefig(plot_dir / "holdout_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    PrecisionRecallDisplay.from_predictions(holdout_y, holdout_scores)
    plt.savefig(plot_dir / "holdout_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    CalibrationDisplay.from_predictions(
        holdout_y, holdout_scores, n_bins=10, strategy="uniform",
    )
    plt.savefig(plot_dir / "holdout_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()


def _probability_hash(values: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _config_hash(model: object) -> str:
    payload = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": model.get_params(deep=False),
    }
    return sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def run_benchmark(snapshot_path: Path | str, preprocessing_artifact_dir: Path | str,
                  artifacts_dir: Path | str = Path("artifacts"),
                  run_id: str = "benchmark", *,
                  evaluation_scope: Literal["full", "validation_only"] = "full") -> BenchmarkReport:
    snapshot_path = Path(snapshot_path)
    run_id = _validate_run_id(run_id)
    if evaluation_scope not in {"full", "validation_only"}:
        raise ValueError(f"benchmark_scope_invalid:{evaluation_scope}")
    target = Path(artifacts_dir) / "benchmark-runs" / run_id
    if target.exists():
        raise FileExistsError(f"benchmark_run_id_exists:{run_id}")
    selected = load_snapshot_partitions(snapshot_path, ("train", "validation"))
    snapshot_version = selected.snapshot_version
    source_frames = {
        name: selected.frame(name) for name in ("train", "validation")
    }
    prepared = {
        name: _prepare_split(name, frame) for name, frame in source_frames.items()
    }
    preprocessor = load_preprocessing_artifact(preprocessing_artifact_dir, snapshot_path)
    matrices = {name: preprocessor.transform(item.X) for name, item in prepared.items()}
    targets = {name: item.y.to_numpy(dtype=np.int64) for name, item in prepared.items()}

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        candidates_dir = staging / "candidates"
        plots_dir = staging / "plots"
        candidates_dir.mkdir()
        plots_dir.mkdir()
        validation_rows: list[dict[str, object]] = []
        validation_slices: list[pd.DataFrame] = []
        candidate_configs: dict[str, object] = {}
        models: dict[str, object] = {}
        validation_scores_by_name: dict[str, np.ndarray] = {}
        validation_predictions = prepared["validation"].keys.copy()
        validation_predictions.insert(0, "student_key", [
            _pseudonymous_key(row.code_module, row.code_presentation, row.id_student)
            for row in validation_predictions.itertuples(index=False)
        ])
        validation_predictions = validation_predictions.drop(columns=["id_student"])
        validation_predictions["split"] = "validation"
        validation_predictions["label"] = targets["validation"]
        for name, model in _candidate_models():
            started = time.perf_counter()
            with threadpool_limits(limits=1):
                model.fit(matrices["train"], targets["train"])
            train_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with threadpool_limits(limits=1):
                validation_scores = _scores(model, matrices["validation"])
            inference_seconds = time.perf_counter() - started
            threshold = select_f1_threshold(targets["validation"], validation_scores)
            metrics = binary_metrics(
                targets["validation"], validation_scores,
                prepared["validation"].keys, threshold,
            )
            artifact_path = candidates_dir / f"{name}.joblib"
            joblib.dump(model, artifact_path, compress=3)
            reloaded_scores = _scores(joblib.load(artifact_path), matrices["validation"])
            if not np.array_equal(validation_scores, reloaded_scores):
                raise ValueError(f"benchmark_reload_probability_mismatch:{name}")
            row = _flat_row(
                name, metrics, threshold, train_seconds, inference_seconds,
                artifact_path.stat().st_size,
            )
            validation_rows.append(row)
            validation_predictions[f"risk_score__{name}"] = validation_scores
            slices = fairness_slices(
                source_frames["validation"].sort_values(
                    list(prepared["validation"].keys.columns), kind="mergesort",
                ).reset_index(drop=True),
                targets["validation"], validation_scores, threshold,
            )
            slices.insert(0, "candidate", name)
            validation_slices.append(slices)
            candidate_configs[name] = {
                "class": f"{type(model).__module__}.{type(model).__name__}",
                "parameters": model.get_params(deep=False),
                "config_sha256": _config_hash(model),
                "validation_probability_sha256": _probability_hash(validation_scores),
                "validation_metrics": metrics,
            }
            models[name] = model
            validation_scores_by_name[name] = validation_scores
        validation_table = pd.DataFrame(validation_rows)
        winner_row = max(validation_rows, key=_selection_key)
        winner = str(winner_row["candidate"])
        threshold = float(winner_row["threshold"])
        selection = {
            "schema_version": BENCHMARK_SCHEMA_VERSION, "run_id": run_id,
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_scope": evaluation_scope,
            "validation_leader": winner,
            "winner": winner if evaluation_scope == "full" else None,
            "parameters": models[winner].get_params(deep=False),
            "selected_threshold": threshold,
            "selection_policy": ["validation_pr_auc", "validation_recall_at_top_20_pct", "lower_ece", "lower_inference_latency"],
            "threshold_policy": "maximum_validation_f1_tie_higher_threshold",
            "snapshot_version": snapshot_version,
            "preprocessor_version": preprocessor.manifest["preprocessor_version"],
            "prior_holdout_exposure": "prediction scores existed before this benchmark; no label metrics were previously computed",
            "holdout_used_for_selection": False,
            # These describe state at lock time. The final benchmark report records
            # the successful post-lock holdout evaluation separately.
            "holdout_accessed": False,
            "production_release_eligible": False,
        }
        validation_table.to_csv(staging / "validation_metrics.csv", index=False)
        pd.concat(validation_slices, ignore_index=True).to_csv(
            staging / "validation_slice_metrics.csv", index=False,
        )
        validation_predictions.to_parquet(
            staging / "validation_predictions.parquet", index=False,
        )
        (staging / "candidate_configurations.json").write_text(
            json.dumps(candidate_configs, indent=2, sort_keys=True, default=str)
        )
        if evaluation_scope == "validation_only":
            selection_path = staging / "validation_selection.json"
            locked_path: Path | None = None
            holdout_report_path: Path | None = None
            selection_path.write_text(json.dumps(
                selection, indent=2, sort_keys=True, default=str,
            ))
        else:
            locked_path = staging / "locked_config.json"
            selection_path = locked_path
            locked_path.write_text(json.dumps(
                selection, indent=2, sort_keys=True, default=str,
            ))
            locked_hash = _file_hash(locked_path)
            # Do not even load or transform holdout until the selection contract is
            # durably persisted and hashed.
            holdout_partition = load_snapshot_partitions(
                snapshot_path, ("holdout",),
            )
            if holdout_partition.snapshot_version != snapshot_version:
                raise ValueError("benchmark_holdout_snapshot_version_mismatch")
            holdout_source = holdout_partition.frame("holdout")
            holdout_prepared = _prepare_split("holdout", holdout_source)
            holdout_matrix = preprocessor.transform(holdout_prepared.X)
            holdout_target = holdout_prepared.y.to_numpy(dtype=np.int64)
            # The only holdout prediction call. All reports reuse this vector.
            holdout_started = time.perf_counter()
            with threadpool_limits(limits=1):
                holdout_scores = _scores(models[winner], holdout_matrix)
            holdout_seconds = time.perf_counter() - holdout_started
            holdout_metrics = binary_metrics(
                holdout_target, holdout_scores,
                holdout_prepared.keys, threshold,
            )
            holdout_frame = holdout_prepared.keys.copy()
            holdout_frame.insert(0, "student_key", [
                _pseudonymous_key(row.code_module, row.code_presentation, row.id_student)
                for row in holdout_frame.itertuples(index=False)
            ])
            holdout_frame = holdout_frame.drop(columns=["id_student"])
            holdout_frame["risk_score"] = holdout_scores
            holdout_frame["label"] = holdout_target
            holdout_frame.to_parquet(
                staging / "holdout_predictions.parquet", index=False,
            )
            holdout_slices = fairness_slices(
                holdout_source.sort_values(
                    list(holdout_prepared.keys.columns), kind="mergesort",
                ).reset_index(drop=True),
                holdout_target, holdout_scores, threshold,
            )
            holdout_slices.to_csv(
                staging / "holdout_slice_metrics.csv", index=False,
            )
            holdout_report = {
                "status": "ok", "winner": winner,
                "selected_threshold": threshold,
                "locked_config_sha256": locked_hash, "prediction_calls": 1,
                "inference_seconds": holdout_seconds, "metrics": holdout_metrics,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
            holdout_report_path = staging / "holdout_final_report.json"
            holdout_report_path.write_text(json.dumps(
                holdout_report, indent=2, sort_keys=True,
            ))
        report = {
            "status": "ok", "schema_version": BENCHMARK_SCHEMA_VERSION,
            "run_id": run_id, "snapshot_version": snapshot_version,
            "preprocessor_version": preprocessor.manifest["preprocessor_version"],
            "candidate_count": len(validation_rows),
            "winner": winner if evaluation_scope == "full" else None,
            "validation_leader": winner,
            "selected_threshold": threshold,
            "evaluation_scope": evaluation_scope,
            "holdout_accessed": evaluation_scope == "full",
            "production_release_eligible": evaluation_scope == "full",
            "holdout_prediction_calls": 1 if evaluation_scope == "full" else 0,
            "random_seed": 42, "native_threads": 1,
            "package_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        }
        (staging / "benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        (staging / "model_card.md").write_text(
            f"# Validation leader: {winner}\n\n"
            f"Snapshot: `{snapshot_version}`  \n"
            f"Selected threshold: `{threshold:.12g}`  \n"
            f"Evaluation scope: `{evaluation_scope}`. "
            + ("Holdout was not loaded or evaluated; this run is not eligible for production release. "
               if evaluation_scope == "validation_only" else
               "Selected on validation; holdout was evaluated once after locking. ")
            +
            "Predictions and explanations are associations, not causal effects.\n"
        )
        _validation_plots(
            plots_dir, targets["validation"],
            validation_scores_by_name[winner], validation_table,
        )
        if evaluation_scope == "full":
            _holdout_plots(plots_dir, holdout_target, holdout_scores)
        files = [p for p in staging.rglob("*") if p.is_file()]
        manifest = {
            "schema_version": BENCHMARK_SCHEMA_VERSION, "run_id": run_id,
            "files": {str(p.relative_to(staging)): _file_hash(p) for p in sorted(files)},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (staging / "success.json").write_text(json.dumps({
            "schema_version": BENCHMARK_SCHEMA_VERSION, "run_id": run_id,
            "manifest_sha256": _file_hash(staging / "manifest.json"),
            "selection_sha256": _file_hash(selection_path),
        }, indent=2, sort_keys=True))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return BenchmarkReport(
        "ok", run_id, target,
        winner if evaluation_scope == "full" else None, winner,
        threshold, len(validation_rows),
        evaluation_scope, target / selection_path.name,
        target / "locked_config.json" if evaluation_scope == "full" else None,
        target / "holdout_final_report.json" if evaluation_scope == "full" else None,
    )
