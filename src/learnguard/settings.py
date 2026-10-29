"""Application configuration loaded from explicit LearnGuard environment keys."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from learnguard.data.snapshot_contract import CUTOFF_DAY, validate_cutoff


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_run_id(run_id: str) -> str:
    """Return a safe single-component artifact run ID or fail before path use."""
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "invalid_run_id:use 1-128 ASCII letters, digits, dots, underscores or hyphens; "
            "the first character must be alphanumeric"
        )
    return run_id


@dataclass(frozen=True)
class Settings:
    cutoff_day: int = CUTOFF_DAY
    data_dir: Path = Path("data/archive.zip")
    artifacts_dir: Path = Path("artifacts")
    baseline_snapshot: Path | None = None
    mlflow_tracking_uri: str = ""

    @classmethod
    def from_environment(cls) -> "Settings":
        """Read only LEARNGUARD_* variables so unrelated process state is ignored."""
        raw_cutoff = os.getenv("LEARNGUARD_CUTOFF_DAY", str(CUTOFF_DAY))
        try:
            cutoff = int(raw_cutoff)
        except ValueError as exc:
            raise ValueError("LEARNGUARD_CUTOFF_DAY must be an integer") from exc
        validate_cutoff(cutoff)
        baseline = os.getenv("LEARNGUARD_BASELINE_SNAPSHOT")
        return cls(
            cutoff_day=cutoff,
            data_dir=Path(os.getenv("LEARNGUARD_SOURCE", "data/archive.zip")),
            artifacts_dir=Path(os.getenv("LEARNGUARD_ARTIFACTS_DIR", "artifacts")),
            baseline_snapshot=Path(baseline) if baseline else None,
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", ""),
        )
