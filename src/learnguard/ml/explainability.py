"""Auditable Linear SHAP artifacts for the published Logistic model."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from learnguard.data.features import KEYS
from learnguard.data.loading import SPLITS, load_snapshot_partitions
from learnguard.ml.model_interface import load_scoring_model
from learnguard.ml.preprocessing import _file_hash, _prepare_split, _validate_run_id
from learnguard.ml.scoring import _pseudonymous_key, _validate_splits

EXPLANATION_SCHEMA_VERSION = "linear-shap-explanations.v1"


@dataclass(frozen=True)
class ExplainabilityReport:
    status: str
    run_id: str
    artifact_dir: Path
    row_count: int
    feature_count: int
    method: str
    output_space: str
    maximum_additivity_error: float
    manifest_path: Path


def _business_feature(name: str) -> str:
    rest = name.split("__", 1)[-1]
    for prefix in ("module_presentation_", "region_", "gender_", "disability_"):
        if rest.startswith(prefix):
            return prefix[:-1]
    if rest == "missingindicator_avg_score_submitted":
        return "avg_score_submitted_missing"
    return rest


def _local_frames(keys: pd.DataFrame, split: str, names: tuple[str, ...],
                  values: np.ndarray, expected: float,
                  outputs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = pd.DataFrame({
        "student_key": [
            _pseudonymous_key(row.code_module, row.code_presentation, row.id_student)
            for row in keys.itertuples(index=False)
        ],
        "code_module": keys["code_module"].astype(str).to_numpy(),
        "code_presentation": keys["code_presentation"].astype(str).to_numpy(),
        "split": split,
        "expected_value": expected,
        "model_output_log_odds": outputs,
    })
    transformed = pd.concat([
        identity,
        pd.DataFrame(values, columns=[f"shap__{name}" for name in names]),
    ], axis=1)
    business_names = tuple(dict.fromkeys(_business_feature(name) for name in names))
    business_values = np.column_stack([
        values[:, [i for i, name in enumerate(names) if _business_feature(name) == business]].sum(axis=1)
        for business in business_names
    ])
    business = pd.concat([
        identity,
        pd.DataFrame(business_values, columns=[f"shap__{name}" for name in business_names]),
    ], axis=1)
    return transformed, business


def _plots(plot_dir: Path, matrix: np.ndarray, values: np.ndarray,
           names: tuple[str, ...], expected: float, outputs: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/learnguard-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import shap
    except ModuleNotFoundError:
        order = np.argsort(np.abs(values).mean(axis=0))[-20:]
        for position, index in enumerate(order):
            sample = np.linspace(0, len(values) - 1, min(500, len(values)), dtype=int)
            plt.scatter(values[sample, index], np.full(len(sample), position),
                        c=matrix[sample, index], cmap="coolwarm", s=8, alpha=0.6)
        plt.yticks(range(len(order)), [names[i] for i in order])
        plt.xlabel("SHAP value (log-odds)")
        plt.tight_layout()
        plt.savefig(plot_dir / "summary_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        order = order[::-1]
        plt.barh(range(len(order)), np.abs(values[:, order]).mean(axis=0))
        plt.yticks(range(len(order)), [names[i] for i in order])
        plt.xlabel("mean(|SHAP value|)")
        plt.tight_layout()
        plt.savefig(plot_dir / "global_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        explanation = None
    else:
        explanation = shap.Explanation(
            values=values, base_values=np.full(len(values), expected),
            data=matrix, feature_names=list(names),
        )
        shap.summary_plot(explanation, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(plot_dir / "summary_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        shap.plots.bar(explanation, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(plot_dir / "global_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
    representatives = {
        "highest_risk": int(np.argmax(outputs)),
        "lowest_risk": int(np.argmin(outputs)),
        "most_uncertain": int(np.argmin(np.abs(outputs))),
    }
    for label, index in representatives.items():
        if explanation is not None:
            shap.plots.waterfall(explanation[index], max_display=15, show=False)
        else:
            order = np.argsort(np.abs(values[index]))[-15:]
            colors = ["#d62728" if values[index, i] > 0 else "#1f77b4" for i in order]
            plt.barh(range(len(order)), values[index, order], color=colors)
            plt.yticks(range(len(order)), [names[i] for i in order])
            plt.xlabel("SHAP value (log-odds)")
        plt.tight_layout()
        plt.savefig(plot_dir / f"waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()


def build_shap_artifact(snapshot_path: Path | str, model_artifact_dir: Path | str,
                        preprocessing_artifact_dir: Path | str,
                        artifacts_dir: Path | str = Path("artifacts"),
                        run_id: str = "shap", splits: Sequence[str] = SPLITS) -> ExplainabilityReport:
    """Publish vectorized, train-background Linear SHAP explanations."""
    snapshot_path = Path(snapshot_path)
    run_id = _validate_run_id(run_id)
    requested = _validate_splits(splits)
    target = Path(artifacts_dir) / "explainability-runs" / run_id
    if target.exists():
        raise FileExistsError(f"explainability_run_id_exists:{run_id}")
    model = load_scoring_model(model_artifact_dir, snapshot_path, preprocessing_artifact_dir)
    snapshot = load_snapshot_partitions(snapshot_path, requested)
    local_parts: list[pd.DataFrame] = []
    business_parts: list[pd.DataFrame] = []
    matrices: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    all_outputs: list[np.ndarray] = []
    expected: float | None = None
    method = output_space = ""
    for split in requested:
        prepared = _prepare_split(split, snapshot.frame(split))
        matrix = model.transform(prepared.X)
        explanation = model.explain_many(matrix)
        if not explanation.method.startswith("linear-shap"):
            raise ValueError("shap_requires_logistic_model_v2")
        transformed, business = _local_frames(
            prepared.keys, split, explanation.feature_names, explanation.values,
            explanation.expected_value, explanation.output_values,
        )
        local_parts.append(transformed)
        business_parts.append(business)
        matrices.append(matrix)
        all_values.append(explanation.values)
        all_outputs.append(explanation.output_values)
        expected = explanation.expected_value
        method, output_space = explanation.method, explanation.output_space
    assert expected is not None
    matrix = np.vstack(matrices)
    values = np.vstack(all_values)
    outputs = np.concatenate(all_outputs)
    if not np.isfinite(values).all():
        raise ValueError("shap_values_non_finite")
    errors = np.abs(expected + values.sum(axis=1) - outputs)
    max_error = float(errors.max(initial=0.0))
    if max_error > 1e-8:
        raise ValueError(f"shap_additivity_failed:{max_error}")
    local = pd.concat(local_parts, ignore_index=True)
    business = pd.concat(business_parts, ignore_index=True)
    global_rows = []
    for index, name in enumerate(model.feature_names):
        global_rows.append({
            "feature_name": name, "business_feature": _business_feature(name),
            "mean_abs_shap": float(np.abs(values[:, index]).mean()),
            "mean_signed_shap": float(values[:, index].mean()),
        })
    global_frame = pd.DataFrame(global_rows).sort_values(
        "mean_abs_shap", ascending=False, kind="stable",
    ).reset_index(drop=True)
    global_frame.insert(0, "rank", np.arange(1, len(global_frame) + 1))
    business_shap_columns = [column for column in business.columns if column.startswith("shap__")]
    business_global = pd.DataFrame({
        "business_feature": [column.removeprefix("shap__") for column in business_shap_columns],
        "mean_abs_shap": [float(business[column].abs().mean()) for column in business_shap_columns],
        "mean_signed_shap": [float(business[column].mean()) for column in business_shap_columns],
    }).sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)
    business_global.insert(0, "rank", np.arange(1, len(business_global) + 1))

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        plot_dir = staging / "plots"
        plot_dir.mkdir()
        local.to_parquet(staging / "shap_values.parquet", index=False)
        business.to_parquet(staging / "business_shap_values.parquet", index=False)
        global_frame.to_csv(staging / "global_importance.csv", index=False)
        business_global.to_csv(staging / "business_global_importance.csv", index=False)
        _plots(plot_dir, matrix, values, model.feature_names, expected, outputs)
        report = {
            "status": "ok", "schema_version": EXPLANATION_SCHEMA_VERSION,
            "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
            "model_version": model.version, "snapshot_version": snapshot.snapshot_version,
            "preprocessing_artifact_dir": str(preprocessing_artifact_dir),
            "splits": list(requested), "row_count": len(local),
            "feature_count": len(model.feature_names), "method": method,
            "output_space": output_space, "background_split": "train",
            "maximum_additivity_error": max_error,
            "top_positive_signals": global_frame.sort_values(
                "mean_signed_shap", ascending=False, kind="stable",
            ).head(10).to_dict(orient="records"),
            "top_negative_signals": global_frame.sort_values(
                "mean_signed_shap", ascending=True, kind="stable",
            ).head(10).to_dict(orient="records"),
            "associations_not_causal": True,
        }
        (staging / "explanation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        files = [p for p in staging.rglob("*") if p.is_file()]
        manifest = {
            "schema_version": EXPLANATION_SCHEMA_VERSION, "run_id": run_id,
            "files": {str(p.relative_to(staging)): _file_hash(p) for p in sorted(files)},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (staging / "success.json").write_text(json.dumps({
            "schema_version": EXPLANATION_SCHEMA_VERSION, "run_id": run_id,
            "manifest_sha256": _file_hash(staging / "manifest.json"),
        }, indent=2, sort_keys=True))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ExplainabilityReport(
        "ok", run_id, target, len(local), len(model.feature_names), method,
        output_space, max_error, target / "manifest.json",
    )
