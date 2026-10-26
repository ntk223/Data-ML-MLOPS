"""M4 contracts for model training, evaluation, and registry metadata."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    target_column: str = "label_at_risk"
    validation_year: str = "2013"
    holdout_year: str = "2014"
    review_fraction: float = 0.20
    random_seed: int = 42


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
class TrainingResult:
    model_version: str
    model_path: Path
    model_card_path: Path
    threshold: float
    validation_metrics: EvaluationMetrics
    holdout_metrics: EvaluationMetrics
    baseline_metrics: EvaluationMetrics | None = None


class ModelEvaluator:
    def evaluate(self, labels: object, scores: object, groups: object | None = None,
                 review_fraction: float = 0.20) -> EvaluationMetrics:
        """TODO: calculate ranking, calibration, workload, and slice metrics."""
        raise NotImplementedError

    def select_threshold(self, labels: object, scores: object,
                         review_fraction: float) -> float:
        """TODO: choose an operating threshold using validation data only."""
        raise NotImplementedError


class ModelTrainer:
    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()

    def train(self, snapshot_path: Path, model_dir: Path) -> TrainingResult:
        """TODO: temporal split, baseline/model fit, evaluation, and model card."""
        raise NotImplementedError

    def register(self, result: TrainingResult, registry_uri: str) -> str:
        """TODO: register only artifacts that pass required audit gates."""
        raise NotImplementedError


def train_and_evaluate(snapshot_path: Path, model_dir: Path) -> TrainingResult:
    """TODO wrapper retained as the M4 pipeline entry point."""
    raise NotImplementedError
