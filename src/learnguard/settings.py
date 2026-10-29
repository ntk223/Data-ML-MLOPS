"""Application configuration contract (runtime loading remains TODO)."""

from dataclasses import dataclass
from pathlib import Path


CUTOFF_DAY = 27  # End of week 4, inclusive.


@dataclass(frozen=True)
class Settings:
    cutoff_day: int = CUTOFF_DAY
    data_dir: Path = Path("data/oulad")
    artifacts_dir: Path = Path("artifacts")
    mlflow_tracking_uri: str = ""

    @classmethod
    def from_environment(cls) -> "Settings":
        """TODO: read environment variables and validate required settings."""
        raise NotImplementedError
