"""Evaluation report contracts; scoring logic is TODO."""

from dataclasses import dataclass, field
from pathlib import Path

from learnguard.ml.training import EvaluationMetrics


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
