"""Data quality gate contracts for schema, missingness, volume, and drift."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class QualityThresholds:
    max_null_fraction: float = 0.70
    max_volume_ratio: float = 10.0
    min_volume_ratio: float = 0.10
    max_psi: float = 0.20


@dataclass
class QualityReport:
    status: str
    row_count: int
    null_fraction_by_feature: dict[str, float] = field(default_factory=dict)
    volume_ratio: float | None = None
    psi_by_feature: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class DataQualityGate:
    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def validate_snapshot(self, snapshot_path: Path,
                          previous_snapshot_path: Path | None = None) -> QualityReport:
        """TODO: evaluate schema, nulls, cohort volume, and feature PSI."""
        raise NotImplementedError


def validate_snapshot(snapshot_path: Path,
                      previous_snapshot_path: Path | None = None) -> QualityReport:
    """TODO wrapper for the snapshot quality gate."""
    raise NotImplementedError
