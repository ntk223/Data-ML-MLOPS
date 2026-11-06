"""Gate 2 — `learnguard evaluate ml`: đánh giá model artifact trên validation.

6 check với id cố định: artifact_integrity, validation_metrics, alert_budget,
shap_additivity, fairness_slices, regression_vs_baseline. Tái dùng tối đa
`evaluation.py` (binary_metrics, select_alert_budget_threshold,
fairness_slices), `model_interface.load_scoring_model`, `preprocessing`
(_prepare_split, _file_hash) và `scoring._pseudonymous_key`. CHỈ chạm split
validation — holdout không bao giờ được load trong gate này.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from learnguard.data.loading import load_snapshot_partitions
from learnguard.evaluation.scorecard import GateCheck, Scorecard
from learnguard.ml.evaluation import (
    binary_metrics,
    fairness_slices,
    select_alert_budget_threshold,
)
from learnguard.ml.model_interface import (
    MODEL_MANIFEST_FILENAME,
    ScoringModel,
    load_scoring_model,
)
from learnguard.ml.preprocessing import _file_hash, _prepare_split, _validate_run_id
from learnguard.ml.scoring import _pseudonymous_key
from learnguard.ml.training import (
    LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    SHAP_BACKGROUND_FILENAME,
)

ALERT_BUDGET_FRACTION = 0.20
FAIRNESS_COLUMN = "gender"
DECISION_THRESHOLD = 0.5


@dataclass(frozen=True)
class MLGateThresholds:
    """Ngưỡng warn/fail của ML gate — chốt theo hợp đồng scorecard v1.0."""

    roc_auc_warn: float = 0.75
    alert_budget_warn: float = 0.50
    alert_budget_fail: float = 0.35
    shap_max_error: float = 1e-8
    fairness_gap_warn: float = 0.10
    regression_tolerance: float = 0.01


THRESHOLDS = MLGateThresholds()


def run_ml_gate(
    model_artifact_dir: Path | str,
    preprocessing_artifact_dir: Path | str,
    snapshot_path: Path | str,
    run_id: str,
    min_roc_auc: float = 0.70,
    baseline_model_dir: Path | str | None = None,
    baseline_preprocessing_artifact_dir: Path | str | None = None,
) -> Scorecard:
    """Chạy 6 check ML gate và trả về `Scorecard` subject="ml"."""
    run_id = _validate_run_id(run_id)
    model_artifact_dir = Path(model_artifact_dir)
    preprocessing_artifact_dir = Path(preprocessing_artifact_dir)
    snapshot_path = Path(snapshot_path)
    card = Scorecard(
        run_id=run_id,
        subject="ml",
        inputs={
            "snapshot": str(snapshot_path),
            "model": str(model_artifact_dir),
            "preprocessing": str(preprocessing_artifact_dir),
            "baseline_model": (
                str(Path(baseline_model_dir)) if baseline_model_dir is not None else None
            ),
            "baseline_preprocessing": (
                str(Path(baseline_preprocessing_artifact_dir))
                if baseline_preprocessing_artifact_dir is not None else None
            ),
        },
    )
    integrity = _check_artifact_integrity(model_artifact_dir)
    card.add(integrity)
    schema_version = str(integrity.observed.get("schema_version", ""))

    context = _score_validation(
        model_artifact_dir, preprocessing_artifact_dir, snapshot_path,
    )
    if context is None:
        skip_reason = (
            f"model_load_failed — artifact_integrity đã báo lỗi, các check phụ "
            f"thuộc model bị bỏ qua (schema={schema_version or 'unknown'})"
        )
        card.add(_skipped("validation_metrics", skip_reason))
        card.add(_skipped("alert_budget", skip_reason))
        card.add(_skipped("shap_additivity", skip_reason))
        card.add(_skipped("fairness_slices", skip_reason))
        card.add(_skipped("regression_vs_baseline", skip_reason))
        return card

    card.add(_check_validation_metrics(context, min_roc_auc))
    card.add(_check_alert_budget(context))
    card.add(_check_shap_additivity(context, schema_version))
    card.add(_check_fairness_slices(context))
    card.add(_check_regression_vs_baseline(
        context, baseline_model_dir,
        preprocessing_artifact_dir
        if baseline_preprocessing_artifact_dir is None
        else Path(baseline_preprocessing_artifact_dir),
        snapshot_path,
    ))
    return card


# ---- context: score validation split một lần, dùng chung cho mọi check --------

@dataclass(frozen=True)
class ValidationContext:
    """Kết quả score TOÀN BỘ split validation (không bao giờ đụng holdout)."""

    model: ScoringModel
    model_version: str
    schema_version: str
    y_true: np.ndarray
    scores: np.ndarray
    keys: pd.DataFrame
    raw_frame: pd.DataFrame


def _score_validation(
    model_artifact_dir: Path, preprocessing_artifact_dir: Path, snapshot_path: Path,
) -> ValidationContext | None:
    """Score validation; trả về None khi artifact không tải/score được.

    Việc load dùng `load_scoring_model` — tự verify toàn bộ hash/provenance,
    nên lỗi ở đây đồng nghĩa artifact hỏng (artifact_integrity đã FAIL trước).
    """
    try:
        model = load_scoring_model(
            model_artifact_dir, snapshot_path, preprocessing_artifact_dir,
        )
        selected = load_snapshot_partitions(snapshot_path, ("validation",))
        prepared = _prepare_split("validation", selected.frame("validation"))
        scores = np.asarray(
            model.predict_positive_proba(prepared.X), dtype=np.float64,
        )
        if scores.shape != (len(prepared.X),) or not np.isfinite(scores).all():
            raise ValueError("evaluation_scores_invalid")
    except Exception:  # noqa: BLE001 — mọi lỗi artifact → các check phụ skipped
        return None
    keys = prepared.keys.copy()
    keys["student_key"] = [
        _pseudonymous_key(str(module), str(presentation), student)
        for module, presentation, student in zip(
            keys["code_module"], keys["code_presentation"], keys["id_student"],
            strict=True,
        )
    ]
    manifest_schema = ""
    try:
        manifest = json.loads(
            (Path(model_artifact_dir) / MODEL_MANIFEST_FILENAME).read_text()
        )
        manifest_schema = str(manifest.get("schema_version", ""))
    except (OSError, json.JSONDecodeError):
        manifest_schema = ""
    return ValidationContext(
        model=model,
        model_version=str(model.version),
        schema_version=manifest_schema,
        y_true=prepared.y.to_numpy(dtype=np.int64),
        scores=scores,
        keys=keys,
        raw_frame=prepared.X,
    )


# ---- 1. artifact_integrity ---------------------------------------------------

def _check_artifact_integrity(model_artifact_dir: Path) -> GateCheck:
    manifest_path = model_artifact_dir / MODEL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return GateCheck(
            "artifact_integrity", "Model manifest tồn tại, schema + hash khớp",
            "fail", {"manifest_path": str(manifest_path)}, None,
            f"Thiếu model manifest: {manifest_path}",
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return GateCheck(
            "artifact_integrity", "Model manifest tồn tại, schema + hash khớp",
            "fail", {"manifest_path": str(manifest_path)}, None,
            f"Manifest không parse được JSON: {exc}",
        )
    schema = str(manifest.get("schema_version", ""))
    observed: dict[str, object] = {
        "schema_version": schema,
        "model_version": manifest.get("model_semantic_version"),
    }
    if schema not in {LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION, MODEL_ARTIFACT_SCHEMA_VERSION}:
        observed["supported"] = False
        return GateCheck(
            "artifact_integrity", "Model manifest tồn tại, schema + hash khớp",
            "fail", observed, None,
            f"Schema model không hỗ trợ: {schema}",
        )
    observed["supported"] = True
    # Hash manifest phải do model.success.json bảo kê.
    success_path = model_artifact_dir / "model.success.json"
    hash_ok = False
    if success_path.is_file():
        recorded = json.loads(success_path.read_text()).get("manifest_sha256")
        hash_ok = recorded == _file_hash(manifest_path)
    observed["manifest_sha256_ok"] = hash_ok
    if not hash_ok:
        return GateCheck(
            "artifact_integrity", "Model manifest tồn tại, schema + hash khớp",
            "fail", observed, None,
            "model.success.json thiếu hoặc manifest_sha256 lệch hash manifest.json",
        )
    if schema == MODEL_ARTIFACT_SCHEMA_VERSION:
        background = model_artifact_dir / SHAP_BACKGROUND_FILENAME
        observed["shap_background_exists"] = background.is_file()
        if not background.is_file():
            return GateCheck(
                "artifact_integrity", "Model manifest tồn tại, schema + hash khớp",
                "fail", observed,
                f"schema {MODEL_ARTIFACT_SCHEMA_VERSION} yêu cầu {SHAP_BACKGROUND_FILENAME}",
                f"Model v2 thiếu {SHAP_BACKGROUND_FILENAME} — không giải thích được",
            )
        return GateCheck(
            "artifact_integrity", "Model manifest tồn tại, schema + hash khớp",
            "pass", observed, None,
            f"Model {schema} hợp lệ, manifest hash khớp, SHAP background sẵn sàng",
        )
    # Legacy v1: dùng được nhưng phải cảnh báo rõ semantics giải thích.
    return GateCheck(
        "artifact_integrity", "Model manifest tồn tại, schema + hash khớp",
        "warn", observed, None,
        f"Model legacy {schema} (không có SHAP): top_signals là coef×x, "
        "KHÔNG phải SHAP interventional — chỉ dùng được cho giải thích tuyến tính thô",
    )


# ---- 2. validation_metrics ---------------------------------------------------

def _check_validation_metrics(context: ValidationContext, min_roc_auc: float) -> GateCheck:
    metrics = binary_metrics(
        context.y_true, context.scores, context.keys, DECISION_THRESHOLD,
    )
    roc_auc = float(metrics["roc_auc"])
    observed: dict[str, object] = {"roc_auc": round(roc_auc, 4), "metrics": metrics}
    threshold_desc = f"roc_auc ≥ {min_roc_auc} (warn < {THRESHOLDS.roc_auc_warn})"
    if roc_auc < min_roc_auc:
        return GateCheck(
            "validation_metrics", "Metrics phân loại trên split validation",
            "fail", observed, threshold_desc,
            f"roc_auc={roc_auc:.4f} < ngưỡng tối thiểu {min_roc_auc}",
        )
    if roc_auc < THRESHOLDS.roc_auc_warn:
        return GateCheck(
            "validation_metrics", "Metrics phân loại trên split validation",
            "warn", observed, threshold_desc,
            f"roc_auc={roc_auc:.4f} < {THRESHOLDS.roc_auc_warn} — chất lượng méo",
        )
    return GateCheck(
        "validation_metrics", "Metrics phân loại trên split validation",
        "pass", observed, threshold_desc,
        f"roc_auc={roc_auc:.4f}, pr_auc={float(metrics['pr_auc']):.4f}, "
        f"brier={float(metrics['brier_score']):.4f}",
    )


# ---- 3. alert_budget ---------------------------------------------------------

def _check_alert_budget(context: ValidationContext) -> GateCheck:
    selection = select_alert_budget_threshold(
        context.scores, context.keys, fraction=ALERT_BUDGET_FRACTION,
    )
    budget_count = selection.budget_count
    hits = int(context.y_true[selection.ranked_indices].sum())
    precision = hits / budget_count if budget_count else 0.0
    observed: dict[str, object] = {
        "target_fraction": ALERT_BUDGET_FRACTION,
        "budget_count": budget_count,
        "true_positives_in_budget": hits,
        "precision_at_budget": round(precision, 4),
        "selected_threshold": round(selection.threshold, 6),
        "tie_policy": selection.tie_policy,
    }
    if precision < THRESHOLDS.alert_budget_fail:
        return GateCheck(
            "alert_budget", f"Precision tại alert budget top-{int(ALERT_BUDGET_FRACTION * 100)}%",
            "fail", observed,
            f"precision@budget ≥ {THRESHOLDS.alert_budget_fail}",
            f"precision@budget={precision:.3f} < {THRESHOLDS.alert_budget_fail} "
            "— ngân sách cảnh báo quá nhiều nhiễu",
        )
    if precision < THRESHOLDS.alert_budget_warn:
        return GateCheck(
            "alert_budget", f"Precision tại alert budget top-{int(ALERT_BUDGET_FRACTION * 100)}%",
            "warn", observed,
            f"precision@budget ≥ {THRESHOLDS.alert_budget_warn}",
            f"precision@budget={precision:.3f} < {THRESHOLDS.alert_budget_warn}",
        )
    return GateCheck(
        "alert_budget", f"Precision tại alert budget top-{int(ALERT_BUDGET_FRACTION * 100)}%",
        "pass", observed, f"precision@budget ≥ {THRESHOLDS.alert_budget_warn}",
        f"precision@budget={precision:.3f} trên {budget_count} cảnh báo",
    )


# ---- 4. shap_additivity ------------------------------------------------------

def _check_shap_additivity(context: ValidationContext, schema_version: str) -> GateCheck:
    if schema_version == LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION:
        return GateCheck(
            "shap_additivity", "expected_value + ΣSHAP == decision_function",
            "skipped", {"reason": "legacy_model_v1"}, None,
            f"Model legacy {schema_version} không có SHAP — bỏ qua check additivity",
        )
    matrix = context.model.transform(context.raw_frame)
    batch = context.model.explain_many(matrix)
    reconstructed = batch.expected_value + batch.values.sum(axis=1)
    errors = np.abs(reconstructed - batch.output_values)
    max_error = float(errors.max()) if errors.size else 0.0
    observed: dict[str, object] = {
        "method": batch.method,
        "rows": int(matrix.shape[0]),
        "max_abs_error": max_error,
        "mean_abs_error": float(errors.mean()) if errors.size else 0.0,
    }
    if max_error > THRESHOLDS.shap_max_error:
        return GateCheck(
            "shap_additivity", "expected_value + ΣSHAP == decision_function",
            "fail", observed, f"max error ≤ {THRESHOLDS.shap_max_error:g}",
            f"SHAP không cộng đúng: max error={max_error:.3e} > "
            f"{THRESHOLDS.shap_max_error:g} trên {matrix.shape[0]} dòng validation",
        )
    return GateCheck(
        "shap_additivity", "expected_value + ΣSHAP == decision_function",
        "pass", observed, f"max error ≤ {THRESHOLDS.shap_max_error:g}",
        f"SHAP cộng đúng trên {matrix.shape[0]} dòng (max error={max_error:.3e}, "
        f"method={batch.method})",
    )


# ---- 5. fairness_slices ------------------------------------------------------

def _check_fairness_slices(context: ValidationContext) -> GateCheck:
    rows = fairness_slices(
        context.raw_frame, context.y_true, context.scores, DECISION_THRESHOLD,
        columns=(FAIRNESS_COLUMN,),
    )
    slice_rows = rows[rows["slice"] == FAIRNESS_COLUMN]
    eligible = slice_rows[~slice_rows["small_sample"]]
    groups = [
        {
            "group": str(row["group"]),
            "sample_count": int(row["sample_count"]),
            "positive_count": int(row["positive_count"]),
            "false_negative_rate": round(float(row["false_negative_rate"]), 4),
            "positive_prediction_rate": round(float(row["positive_prediction_rate"]), 4),
            "small_sample": bool(row["small_sample"]),
        }
        for _, row in slice_rows.iterrows()
    ]
    observed: dict[str, object] = {"column": FAIRNESS_COLUMN, "groups": groups}
    if eligible.empty:
        observed["reason"] = "small_sample_all_groups"
        return GateCheck(
            "fairness_slices", f"FNR gap giữa nhóm {FAIRNESS_COLUMN}",
            "skipped", observed, None,
            "Mọi nhóm quá nhỏ để đo công bằng (dưới MIN_SLICE_ROWS/MIN_SLICE_POSITIVES)",
        )
    max_gap = float(eligible["eligible_fnr_gap"].max())
    observed["max_fnr_gap"] = round(max_gap, 4)
    if max_gap > THRESHOLDS.fairness_gap_warn:
        return GateCheck(
            "fairness_slices", f"FNR gap giữa nhóm {FAIRNESS_COLUMN}",
            "warn", observed, f"gap ≤ {THRESHOLDS.fairness_gap_warn}",
            f"FNR gap giữa nhóm {FAIRNESS_COLUMN} = {max_gap:.3f} > "
            f"{THRESHOLDS.fairness_gap_warn} — rà soát công bằng theo nhóm",
        )
    return GateCheck(
        "fairness_slices", f"FNR gap giữa nhóm {FAIRNESS_COLUMN}",
        "pass", observed, f"gap ≤ {THRESHOLDS.fairness_gap_warn}",
        f"FNR gap giữa nhóm {FAIRNESS_COLUMN} = {max_gap:.3f}, trong ngưỡng",
    )


# ---- 6. regression_vs_baseline ----------------------------------------------

def _check_regression_vs_baseline(
    context: ValidationContext,
    baseline_model_dir: Path | str | None,
    baseline_preprocessing_dir: Path,
    snapshot_path: Path,
) -> GateCheck:
    if baseline_model_dir is None:
        return GateCheck(
            "regression_vs_baseline", "roc_auc mới không tụt so với baseline model",
            "skipped", {"reason": "no_baseline_model"}, None,
            "Không truyền --baseline-model-dir → bỏ qua check hồi quy",
        )
    baseline_model = load_scoring_model(
        Path(baseline_model_dir), snapshot_path, baseline_preprocessing_dir,
    )
    selected = load_snapshot_partitions(snapshot_path, ("validation",))
    prepared = _prepare_split("validation", selected.frame("validation"))
    baseline_scores = np.asarray(
        baseline_model.predict_positive_proba(prepared.X), dtype=np.float64,
    )
    new_auc = float(roc_auc_score(context.y_true, context.scores))
    old_auc = float(roc_auc_score(prepared.y.to_numpy(dtype=np.int64), baseline_scores))
    delta = new_auc - old_auc
    observed: dict[str, object] = {
        "new_roc_auc": round(new_auc, 4),
        "baseline_roc_auc": round(old_auc, 4),
        "delta": round(delta, 4),
        "new_model_version": context.model_version,
        "baseline_model_version": str(baseline_model.version),
    }
    if delta < -THRESHOLDS.regression_tolerance:
        return GateCheck(
            "regression_vs_baseline", "roc_auc mới không tụt so với baseline model",
            "fail", observed,
            f"new ≥ baseline − {THRESHOLDS.regression_tolerance}",
            f"roc_auc tụt {abs(delta):.4f} so với baseline "
            f"({observed['baseline_model_version']} → {context.model_version})",
        )
    return GateCheck(
        "regression_vs_baseline", "roc_auc mới không tụt so với baseline model",
        "pass", observed, f"new ≥ baseline − {THRESHOLDS.regression_tolerance}",
        f"roc_auc mới {new_auc:.4f} vs baseline {old_auc:.4f} (delta {delta:+.4f})",
    )


# ---- helpers -----------------------------------------------------------------

def _skipped(check_id: str, reason: str) -> GateCheck:
    """Check bị bỏ qua vì model artifact không tải được (lỗi đã báo ở integrity)."""
    names = {
        "validation_metrics": "Metrics phân loại trên split validation",
        "alert_budget": f"Precision tại alert budget top-{int(ALERT_BUDGET_FRACTION * 100)}%",
        "shap_additivity": "expected_value + ΣSHAP == decision_function",
        "fairness_slices": f"FNR gap giữa nhóm {FAIRNESS_COLUMN}",
        "regression_vs_baseline": "roc_auc mới không tụt so với baseline model",
    }
    return GateCheck(
        check_id, names[check_id], "skipped", {"reason": "model_load_failed"}, None,
        reason,
    )
