"""M3 contracts for catalog and point-in-time leakage audits."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LeakageViolation:
    feature_name: str
    rule: str
    details: str


@dataclass
class LeakageAuditReport:
    status: str
    cutoff_day: int
    checked_features: int = 0
    source_checks: int = 0
    violations: list[LeakageViolation] = field(default_factory=list)


class LeakageAuditor:
    def audit_catalog(self) -> LeakageAuditReport:
        """TODO: reject unknown rules and target/identity columns in catalog."""
        raise NotImplementedError

    def audit_snapshot(self, snapshot_path: Path, processed_dir: Path,
                       cutoff_day: int) -> LeakageAuditReport:
        """TODO: compare snapshot aggregates with eligible source records."""
        raise NotImplementedError


def audit_snapshot(snapshot_path: Path, processed_dir: Path,
                   cutoff_day: int) -> LeakageAuditReport:
    """TODO wrapper for the M3 audit service."""
    raise NotImplementedError
