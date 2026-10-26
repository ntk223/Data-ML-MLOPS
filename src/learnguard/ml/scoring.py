"""M5 contracts for scoring, explainability, and manual-review states."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PredictionStatus(str, Enum):
    SCORED = "scored"
    INSUFFICIENT_DATA = "insufficient_data"
    MANUAL_REVIEW = "manual_review"
    NO_PREDICTION = "no_prediction"


@dataclass(frozen=True)
class PredictionSignal:
    feature_name: str
    contribution: float
    direction: str


@dataclass
class PredictionRecord:
    student_key: str
    code_module: str
    code_presentation: str
    risk_score: float | None
    predicted_label: int | None
    threshold: float
    uncertainty: float | None
    status: PredictionStatus
    model_version: str
    snapshot_version: str
    cutoff_day: int
    top_signals: list[PredictionSignal] = field(default_factory=list)
    review_reason: str | None = None


@dataclass
class ScoringReport:
    model_version: str
    snapshot_version: str
    input_rows: int
    scored_rows: int
    manual_review_rows: int
    insufficient_data_rows: int
    output_path: Path


class BatchScorer:
    def score(self, snapshot_path: Path, model_path: Path,
              output_path: Path) -> ScoringReport:
        """TODO: score eligible rows and persist every outcome with provenance."""
        raise NotImplementedError

    def explain(self, model: object, feature_row: object) -> list[PredictionSignal]:
        """TODO: calculate top signals; treat contributions as associations."""
        raise NotImplementedError

    def assign_review_state(self, risk_score: float, threshold: float,
                            uncertainty: float, data_status: str) -> PredictionStatus:
        """TODO: route uncertain and insufficient-data rows to manual review."""
        raise NotImplementedError


def score_snapshot(snapshot_path: Path, model_path: Path,
                   output_path: Path) -> ScoringReport:
    """TODO wrapper retained as the M5 pipeline entry point."""
    raise NotImplementedError
