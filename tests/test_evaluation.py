import numpy as np
import pandas as pd
import pytest

from learnguard.ml.evaluation import (
    binary_metrics, expected_calibration_error, fairness_slices,
    select_alert_budget_threshold, select_f1_threshold,
)


def test_metric_definitions_and_top_k_are_deterministic():
    y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.55])
    keys = pd.DataFrame({
        "code_module": ["A"] * 10, "code_presentation": ["P"] * 10,
        "id_student": range(10),
    })
    threshold = select_f1_threshold(y, scores)
    result = binary_metrics(y, scores, keys, threshold)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["recall_at_top_20_pct"] == pytest.approx(0.4)
    assert result["precision_at_top_20_pct"] == pytest.approx(1.0)
    assert 0 <= expected_calibration_error(y, scores) <= 1


def test_threshold_tie_prefers_higher_value():
    y = np.array([0, 1])
    scores = np.array([0.2, 0.8])
    assert select_f1_threshold(y, scores) == pytest.approx(0.8)


def test_fairness_marks_small_groups():
    frame = pd.DataFrame({
        "gender": ["F"] * 60 + ["M"] * 10,
        "region": ["R"] * 70, "disability": ["N"] * 70,
        "age_band": ["0-35"] * 70, "code_module": ["AAA"] * 70,
    })
    y = np.array(([0, 1] * 35), dtype=np.int64)
    scores = np.linspace(0, 1, 70)
    result = fairness_slices(frame, y, scores, 0.5)
    gender = result[result["slice"] == "gender"].set_index("group")
    assert not bool(gender.loc["F", "small_sample"])
    assert bool(gender.loc["M", "small_sample"])


def test_alert_budget_threshold_is_deterministic_and_exact_without_boundary_tie():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    keys = pd.DataFrame({
        "code_module": ["A"] * 5,
        "code_presentation": ["P"] * 5,
        "student_key": ["e", "d", "c", "b", "a"],
    })
    result = select_alert_budget_threshold(scores, keys, 0.4)
    assert result.budget_count == 2
    assert result.threshold == pytest.approx(0.8)
    assert result.realized_alert_count == 2
    assert result.realized_alert_rate == pytest.approx(0.4)
    np.testing.assert_array_equal(result.ranked_indices, [0, 1])


def test_alert_budget_boundary_tie_prefers_under_budget_on_equal_deviation():
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    keys = pd.DataFrame({
        "code_module": ["A"] * 4,
        "code_presentation": ["P"] * 4,
        "student_key": ["d", "c", "b", "a"],
    })
    result = select_alert_budget_threshold(scores, keys, 0.5)
    assert result.budget_count == 2
    assert result.boundary_tie_count == 2
    assert result.tie_policy == "exclude_boundary_ties"
    assert result.realized_alert_count == 1
    assert result.threshold > 0.8
    # Ranked reference is still exactly the budget and tie-breaks by student_key.
    np.testing.assert_array_equal(result.ranked_indices, [0, 2])
