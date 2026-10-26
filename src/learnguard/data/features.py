"""M2 contracts for catalogued, point-in-time feature snapshots."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from learnguard.settings import CUTOFF_DAY

FeatureKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    source: str
    aggregation: str
    as_of_rule: str
    kind: FeatureKind


FEATURE_CATALOG: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("module_presentation", "courses", "course/presentation code", "static_before_start", "categorical"),
    # Quyết định EDA #2/#6: tính cả click TRƯỚC ngày khai giảng (date âm, min −25) —
    # hành vi "xem trước tài liệu" là signal chăm học; mọi thứ có trước cutoff đều hợp lệ.
    FeatureDefinition("n_clicks_before_cutoff", "studentVle", "sum_click where date <= cutoff", "date_at_or_before_cutoff", "numeric"),
    FeatureDefinition("click_trend_w3_w4", "studentVle", "week 3-4 clicks minus week 1-2 clicks", "date_at_or_before_cutoff", "numeric"),
    FeatureDefinition("n_submitted_before_cutoff", "studentAssessment+assessments", "count eligible submissions", "submission_and_assessment_at_or_before_cutoff", "numeric"),
    FeatureDefinition("avg_score_submitted", "studentAssessment+assessments", "mean eligible score", "submission_and_assessment_at_or_before_cutoff", "numeric"),
)


@dataclass
class AsOfLedger:
    cutoff_day: int
    snapshot_version: str
    source_hash: str
    feature_rules: dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureSnapshotResult:
    snapshot_path: Path
    row_count: int
    snapshot_version: str
    ledger_path: Path
    insufficient_data_count: int = 0


class PointInTimeFeatureBuilder:
    """Builds a cohort and features available at a course-day cutoff."""

    def build(self, processed_dir: Path, output_path: Path,
              cutoff_day: int = CUTOFF_DAY) -> FeatureSnapshotResult:
        """TODO: apply catalog/cohort rules and persist snapshot plus ledger."""
        raise NotImplementedError

    def build_cohort(self, tables: object, cutoff_day: int) -> object:
        """TODO: retain students registered at cutoff; exclude prior withdrawals."""
        raise NotImplementedError

    def build_feature(self, feature: FeatureDefinition, tables: object,
                      cutoff_day: int) -> object:
        """TODO: compute one catalog feature using its as-of rule."""
        raise NotImplementedError


def build_feature_snapshot(processed_dir: Path, output_path: Path,
                           cutoff_day: int = CUTOFF_DAY) -> FeatureSnapshotResult:
    """TODO wrapper retained as the M2 pipeline entry point."""
    raise NotImplementedError
