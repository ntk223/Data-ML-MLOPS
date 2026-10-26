"""Top-level pipeline contracts; stage implementations remain TODO."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from learnguard.settings import CUTOFF_DAY


PIPELINE_STAGES = (
    "ingestion",
    "feature_snapshot",
    "leakage_audit",
    "quality_gate",
    "training_evaluation",
    "batch_scoring",
    "prediction_batch_export",
)


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    artifacts_dir: Path
    cutoff_day: int = CUTOFF_DAY


@dataclass
class StageResult:
    stage: str
    status: str
    output_paths: list[Path] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineReport:
    run_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    stages: list[StageResult] = field(default_factory=list)
    report_path: Path | None = None


class LearnGuardPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> PipelineReport:
        """TODO: execute stages in order, stop on failed gates, write report."""
        raise NotImplementedError

    def run_stage(self, stage_name: str) -> StageResult:
        """TODO: dispatch one named stage to its owning module."""
        raise NotImplementedError


def run_pipeline(data_dir: Path, artifacts_dir: Path,
                 cutoff_day: int = CUTOFF_DAY) -> PipelineReport:
    """TODO wrapper used by the CLI and scheduled batch job."""
    raise NotImplementedError
