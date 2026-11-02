"""Application configuration, loaded from environment variables.

Everything is optional with a working default — no credential fallbacks, no
secrets in code. `pg_dsn` empty means "not configured": the shared prediction
store is skipped and the batch pipeline stays file-only.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


CUTOFF_DAY = 27  # End of week 4, inclusive.

_PG_DSN_VAR = "LEARNGUARD_PG_DSN"
_ARTIFACTS_DIR_VAR = "LEARNGUARD_ARTIFACTS_DIR"
_DATA_DIR_VAR = "LEARNGUARD_DATA_DIR"
_MLFLOW_URI_VAR = "LEARNGUARD_MLFLOW_TRACKING_URI"
_CUTOFF_DAY_VAR = "LEARNGUARD_CUTOFF_DAY"


@dataclass(frozen=True)
class Settings:
    cutoff_day: int = CUTOFF_DAY
    data_dir: Path = Path("data/oulad")
    artifacts_dir: Path = Path("artifacts")
    mlflow_tracking_uri: str = ""
    pg_dsn: str = ""

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables (all optional).

        Raises ValueError only for values that are present but invalid (e.g. a
        non-integer cutoff day) — absent values keep their defaults.
        """
        env = os.environ if environ is None else environ
        cutoff_raw = env.get(_CUTOFF_DAY_VAR, str(CUTOFF_DAY))
        try:
            cutoff_day = int(cutoff_raw)
        except ValueError as exc:
            raise ValueError(f"settings_cutoff_day_invalid:{cutoff_raw}") from exc
        return cls(
            cutoff_day=cutoff_day,
            data_dir=Path(env.get(_DATA_DIR_VAR, "data/oulad")),
            artifacts_dir=Path(env.get(_ARTIFACTS_DIR_VAR, "artifacts")),
            mlflow_tracking_uri=env.get(_MLFLOW_URI_VAR, ""),
            pg_dsn=env.get(_PG_DSN_VAR, ""),
        )
