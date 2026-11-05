"""Deterministic binary-classification metrics and fairness slices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

ECE_BINS = 10
MIN_SLICE_ROWS = 50
MIN_SLICE_POSITIVES = 20


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_negatives: int
    false_negative_rate: float
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class AlertBudgetSelection:
    """Deterministic scalar threshold derived from a ranked alert budget."""

    target_fraction: float
    budget_count: int
    boundary_score: float
    boundary_tie_count: int
    threshold: float
    realized_alert_count: int
    realized_alert_rate: float
    absolute_deviation: float
    tie_policy: str
    ranked_indices: np.ndarray


def expected_calibration_error(y_true: np.ndarray, scores: np.ndarray,
                               bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(scores, edges[1:-1], right=False), bins - 1)
    result = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            result += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(scores[mask].mean()))
    return result if len(y_true) else float("nan")


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Maximize validation F1; ties select the higher threshold."""
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    best = (-1.0, -1.0)
    selected = 0.5
    for threshold in candidates:
        value = f1_score(y_true, scores >= threshold, zero_division=0)
        key = (float(value), float(threshold))
        if key > best:
            best, selected = key, float(threshold)
    return selected


def threshold_metrics(y_true: np.ndarray, scores: np.ndarray,
                      threshold: float) -> ThresholdMetrics:
    predicted = (scores >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    false_negatives = int(matrix[1, 0])
    positives = int(matrix[1].sum())
    return ThresholdMetrics(
        float(threshold), float(precision_score(y_true, predicted, zero_division=0)),
        float(recall_score(y_true, predicted, zero_division=0)),
        float(f1_score(y_true, predicted, zero_division=0)), false_negatives,
        false_negatives / positives if positives else 0.0,
        matrix.astype(int).tolist(),
    )


def select_alert_budget_threshold(
    scores: np.ndarray,
    keys: pd.DataFrame,
    fraction: float = 0.20,
) -> AlertBudgetSelection:
    """Choose a reproducible scalar threshold for a ranked alert budget.

    Ranking is only the reference policy.  The deployed labels continue to use
    ``score >= threshold``.  If equal scores cross the budget boundary, choose
    the inclusive or exclusive scalar with the smallest count deviation; an
    exact tie prefers staying under budget.
    """
    probability = np.asarray(scores, dtype=np.float64)
    required = ("code_module", "code_presentation", "student_key")
    if (
        probability.ndim != 1 or not len(probability)
        or not np.isfinite(probability).all()
        or ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise ValueError("alert_budget_scores_invalid")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("alert_budget_fraction_invalid")
    if len(keys) != len(probability) or any(name not in keys for name in required):
        raise ValueError("alert_budget_keys_invalid")

    ordered = keys.loc[:, required].reset_index(drop=True).copy()
    ordered["_score"] = probability
    ranked_indices = ordered.sort_values(
        ["_score", *required],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).index.to_numpy(dtype=np.int64)
    budget_count = int(np.ceil(len(probability) * float(fraction)))
    boundary = float(probability[ranked_indices[budget_count - 1]])
    tie_count = int(np.count_nonzero(probability == boundary))

    candidates = [(boundary, int(np.count_nonzero(probability >= boundary)), "include_boundary_ties")]
    exclusive = float(np.nextafter(boundary, np.inf))
    if exclusive <= 1.0:
        candidates.append((
            exclusive,
            int(np.count_nonzero(probability >= exclusive)),
            "exclude_boundary_ties",
        ))
    threshold, realized, policy = min(
        candidates,
        key=lambda item: (abs(item[1] - budget_count), item[1] > budget_count, item[1]),
    )
    realized_rate = realized / len(probability)
    return AlertBudgetSelection(
        target_fraction=float(fraction),
        budget_count=budget_count,
        boundary_score=boundary,
        boundary_tie_count=tie_count,
        threshold=float(threshold),
        realized_alert_count=realized,
        realized_alert_rate=realized_rate,
        absolute_deviation=abs(realized_rate - float(fraction)),
        tie_policy=policy,
        ranked_indices=ranked_indices[:budget_count],
    )


def _top_k(y_true: np.ndarray, scores: np.ndarray, keys: pd.DataFrame,
           fraction: float) -> tuple[float, float, int]:
    count = int(np.ceil(len(scores) * fraction))
    ordered_frame = keys.reset_index(drop=True).copy()
    ordered_frame["_score"] = scores
    ordered = ordered_frame.sort_values(
        ["_score", "code_module", "code_presentation", "id_student"],
        ascending=[False, True, True, True], kind="mergesort",
    ).index.to_numpy()[:count]
    found = int(y_true[ordered].sum())
    positives = int(y_true.sum())
    return found / positives if positives else 0.0, found / count if count else 0.0, count


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, keys: pd.DataFrame,
                   selected_threshold: float) -> dict[str, object]:
    y = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(scores, dtype=np.float64)
    if y.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("evaluation_inputs_invalid")
    result: dict[str, object] = {
        "sample_count": len(y), "positive_count": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "ece": expected_calibration_error(y, probability), "ece_bins": ECE_BINS,
    }
    for percent in (10, 20):
        recall, precision, count = _top_k(y, probability, keys, percent / 100)
        result[f"recall_at_top_{percent}_pct"] = recall
        result[f"precision_at_top_{percent}_pct"] = precision
        result[f"top_{percent}_pct_count"] = count
    result["threshold_0_5"] = threshold_metrics(y, probability, 0.5).__dict__
    result["selected_threshold"] = threshold_metrics(
        y, probability, selected_threshold,
    ).__dict__
    return result


def fairness_slices(frame: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray,
                    threshold: float,
                    columns: tuple[str, ...] = ("gender", "region", "disability", "age_band", "code_module")) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        values = frame[column].astype("string").fillna("__MISSING__")
        for group in sorted(values.unique()):
            mask = values.eq(group).to_numpy()
            y = y_true[mask]
            predicted = scores[mask] >= threshold
            positive_count = int(y.sum())
            fn = int(np.count_nonzero((y == 1) & ~predicted))
            rows.append({
                "slice": column, "group": str(group), "sample_count": int(mask.sum()),
                "positive_count": positive_count,
                "precision": float(precision_score(y, predicted, zero_division=0)),
                "recall": float(recall_score(y, predicted, zero_division=0)),
                "false_negative_rate": fn / positive_count if positive_count else 0.0,
                "positive_prediction_rate": float(predicted.mean()),
                "small_sample": bool(mask.sum() < MIN_SLICE_ROWS or positive_count < MIN_SLICE_POSITIVES),
            })
    result = pd.DataFrame(rows)
    eligible = result[~result["small_sample"]]
    gaps = eligible.groupby("slice")["false_negative_rate"].agg(
        lambda x: float(x.max() - x.min())
    )
    result["eligible_fnr_gap"] = result["slice"].map(gaps)
    return result
