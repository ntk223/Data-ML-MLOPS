"""Future evaluation report contracts; scoring logic is outside Phase 5."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvaluationMetrics:
    sample_count: int
    roc_auc: float | None = None
    pr_auc: float | None = None
    recall_at_top_k: float | None = None
    expected_calibration_error: float | None = None
    fnr_gap: float | None = None
    by_slice: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    split_name: str
    metrics: EvaluationMetrics
    confusion_matrix: list[list[int]] = field(default_factory=list)
    slice_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    report_path: Path | None = None


def evaluate_model(model_path: Path, snapshot_path: Path,
                   split_name: str) -> EvaluationReport:
    """TODO: produce a reproducible evaluation report for one temporal split."""
    raise NotImplementedError
