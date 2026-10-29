"""M2 — Point-in-time feature snapshot tới cutoff day-27.

Nguyên tắc cốt lõi: một feature chỉ được dùng dữ liệu **có trước thời điểm dự đoán**
(`date <= cutoff`). Mọi rule đều truy vết về EDA 2026-10-26 và catalog ở dưới.
`AsOfLedger` ghi lại bằng chứng (source hash + rule) để M3 audit và M5 xuất batch.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pandas as pd

from learnguard.data.schema import TARGET_LABELS
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
    # Quyết định EDA #4: Exam không ngày (date null) KHÔNG được tính vào feature trước cutoff.
    FeatureDefinition("n_submitted_before_cutoff", "studentAssessment+assessments", "count eligible submissions", "submission_and_assessment_at_or_before_cutoff", "numeric"),
    FeatureDefinition("avg_score_submitted", "studentAssessment+assessments", "mean eligible score", "submission_and_assessment_at_or_before_cutoff", "numeric"),
)

STATIC_FEATURES: tuple[str, ...] = (
    "gender",
    "region",
    "highest_education",
    "imd_band",  # null hợp lệ (EDA #2) — kèm cờ imd_band_missing
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
)

# Temporal split (quyết định M4, chốt trong IMPLEMENTATION_TODO): chia theo KỲ HỌC,
# không random — mô phỏng đúng kịch bản production (train kỳ cũ → dự đoán kỳ mới).
# Data engineer cấp hợp đồng này qua cột `split`; ML engineer chỉ đọc và tuân theo.
SPLIT_RULES: dict[str, str] = {
    "2013B": "train",
    "2013J": "train",       # gộp cả 2 kỳ 2013 — ưu tiên dữ liệu train nhiều (quyết định chủ project)
    "2014B": "validation",  # tuning threshold/hyperparameter CHỈ ở đây
    "2014J": "holdout",     # kỳ duy nhất chưa bị tiêu hao — chạm 1 lần khi báo cáo
}

KEYS = ("code_module", "code_presentation", "id_student")


@dataclass
class AsOfLedger:
    cutoff_day: int
    snapshot_version: str
    source_hash: str
    feature_rules: dict[str, str] = field(default_factory=dict)
    cohort_rules: dict[str, str] = field(default_factory=dict)
    split_rules: dict[str, str] = field(default_factory=dict)
    input_hashes: dict[str, str] = field(default_factory=dict)
    feature_list_hash: str = ""
    created_at: str = ""


@dataclass
class FeatureSnapshotResult:
    snapshot_path: Path
    row_count: int
    snapshot_version: str
    ledger_path: Path
    insufficient_data_count: int = 0
    cohort_exclusions: dict[str, int] = field(default_factory=dict)


class PointInTimeFeatureBuilder:
    """Builds a cohort and features available at a course-day cutoff."""

    # ---- cohort -----------------------------------------------------------

    def build_cohort(self, tables: object, cutoff_day: int) -> pd.DataFrame:
        """Cohort = SV ĐANG học tại cutoff.

        - Giữ: date_registration <= cutoff (hoặc null — 45 dòng, đếm minh bạch trong ledger)
        - Loại: đã rút lui tính tới cutoff (date_unregistration <= cutoff) —
          họ đã "có đáp án" trước thời điểm dự đoán → giữ lại là leakage.
        """
        sr: pd.DataFrame = tables["studentRegistration"].copy()
        unknown_reg = int(sr["date_registration"].isna().sum())
        registered = sr[sr["date_registration"].isna() | (sr["date_registration"] <= cutoff_day)]
        withdrew = registered["date_unregistration"].notna() & (registered["date_unregistration"] <= cutoff_day)
        excluded_withdrawn = int(withdrew.sum())
        cohort = registered[~withdrew].loc[:, KEYS].copy()
        self._cohort_exclusions = {
            "withdrawn_by_cutoff": excluded_withdrawn,
            "kept_unknown_registration_date": unknown_reg,
        }
        self._cohort_rules = {
            "include": "date_registration <= cutoff OR date_registration is null",
            "exclude": "date_unregistration <= cutoff (rút lui tính tới cutoff là thông tin đã biết)",
        }
        return cohort

    # ---- từng feature -----------------------------------------------------

    def build_feature(
        self, feature: FeatureDefinition, tables: object, cutoff_day: int
    ) -> pd.DataFrame:
        """Tính MỘT feature theo đúng as-of rule trong catalog."""
        if feature.name == "module_presentation":
            courses: pd.DataFrame = tables["courses"]
            mp = courses.assign(
                module_presentation=courses["code_module"] + "-" + courses["code_presentation"]
            )
            return mp[["code_module", "code_presentation", "module_presentation"]]

        if feature.name == "n_clicks_before_cutoff":
            sv: pd.DataFrame = tables["studentVle"]
            eligible = sv[sv["date"] <= cutoff_day]  # gồm date âm (EDA #6)
            agg = eligible.groupby(list(KEYS), as_index=False)["sum_click"].sum()
            agg = agg.rename(columns={"sum_click": feature.name})
            return agg

        if feature.name == "click_trend_w3_w4":
            sv = tables["studentVle"]
            key_list = list(KEYS)
            w12 = (
                sv[sv["date"].between(0, 13)].groupby(key_list)["sum_click"].sum()
            )
            w34 = (
                sv[sv["date"].between(14, cutoff_day)].groupby(key_list)["sum_click"].sum()
            )
            trend = (w34 - w12).rename(feature.name).reset_index()
            return trend

        if feature.name in {"n_submitted_before_cutoff", "avg_score_submitted"}:
            sa: pd.DataFrame = tables["studentAssessment"]
            asm: pd.DataFrame = tables["assessments"]
            joined = sa.merge(
                asm[["id_assessment", "code_module", "code_presentation", "date"]],
                on="id_assessment",
                how="left",
            )
            # Eligible: đã nộp trước cutoff VÀ bài đánh giá diễn ra trước cutoff.
            # Exam không ngày (date null) không đủ căn cứ → loại (EDA #4).
            eligible = (
                joined["date_submitted"].le(cutoff_day)
                & joined["date"].notna()
                & joined["date"].le(cutoff_day)
            )
            eligible_rows = joined[eligible]
            if feature.name == "n_submitted_before_cutoff":
                counts = eligible_rows.groupby(list(KEYS)).size().rename(feature.name)
                return counts.reset_index()
            scored = eligible_rows[eligible_rows["score"].notna()]
            means = scored.groupby(list(KEYS))["score"].mean().rename(feature.name)
            return means.reset_index()

        raise KeyError(f"feature_not_in_catalog:{feature.name}")

    # ---- temporal split ------------------------------------------------------

    def assign_split(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        """Gán cột `split` theo SPLIT_RULES; kỳ lạ → fail lớn thay vì gán im lặng."""
        snapshot = snapshot.copy()
        snapshot["split"] = snapshot["code_presentation"].map(SPLIT_RULES)
        if snapshot["split"].isna().any():
            unknown = sorted(snapshot.loc[snapshot["split"].isna(), "code_presentation"].unique())
            raise ValueError(f"unknown_presentation_for_split:{','.join(unknown)}")
        return snapshot

    # ---- tổng hợp ---------------------------------------------------------

    def build(self, processed_dir: Path, output_path: Path,
              cutoff_day: int = CUTOFF_DAY) -> FeatureSnapshotResult:
        """Cohort + static features + catalog features → snapshot Parquet + ledger JSON."""
        processed_dir = Path(processed_dir)
        canonical = processed_dir / "canonical"
        parquet_dir = canonical if canonical.is_dir() else processed_dir
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tables: dict[str, pd.DataFrame] = {
            name: pd.read_parquet(parquet_dir / f"{name}.parquet")
            for name in (
                "courses", "assessments", "studentInfo",
                "studentRegistration", "studentAssessment", "studentVle",
            )
        }
        input_hashes = {
            name: _sha256_file(parquet_dir / f"{name}.parquet") for name in tables
        }

        cohort = self.build_cohort(tables, cutoff_day)

        snapshot = cohort.merge(
            tables["studentInfo"][list(KEYS) + list(STATIC_FEATURES) + ["final_result"]],
            on=list(KEYS),
            how="left",
            validate="many_to_one",
        )
        snapshot["imd_band_missing"] = snapshot["imd_band"].isna()

        for feature in FEATURE_CATALOG:
            feature_df = self.build_feature(feature, tables, cutoff_day)
            # module_presentation là feature khóa-học (2 cột khóa), còn lại là per-student
            on = ["code_module", "code_presentation"] if feature.name == "module_presentation" else list(KEYS)
            snapshot = snapshot.merge(feature_df, on=on, how="left", validate="many_to_one")

        # Feature dạng count/trend: không có tương tác = 0 (không phải thiếu dữ liệu);
        # thiếu dữ liệu thật sự = không click VÀ không nộp bài nào trước cutoff.
        for col in ("n_clicks_before_cutoff", "click_trend_w3_w4", "n_submitted_before_cutoff"):
            snapshot[col] = snapshot[col].fillna(0)
        snapshot["insufficient_data"] = (
            (snapshot["n_clicks_before_cutoff"] == 0)
            & (snapshot["n_submitted_before_cutoff"] == 0)
        )
        snapshot["label"] = snapshot["final_result"].map(TARGET_LABELS)

        # Temporal split — hợp đồng cho M4
        snapshot = self.assign_split(snapshot)

        snapshot = snapshot.sort_values(list(KEYS)).reset_index(drop=True)
        snapshot.to_parquet(output_path, index=False)

        feature_list_hash = _feature_list_hash()
        source_hash = _combined_source_hash(input_hashes)
        # Version = dữ liệu (source_hash) + hợp đồng rule (feature_list_hash gồm split rules)
        version = f"snapshot-v1:cutoff{cutoff_day}:{source_hash[:8]}:{feature_list_hash[:8]}"
        ledger = AsOfLedger(
            cutoff_day=cutoff_day,
            snapshot_version=version,
            source_hash=source_hash,
            feature_rules={f.name: f"{f.aggregation} [{f.as_of_rule}]" for f in FEATURE_CATALOG},
            cohort_rules=getattr(self, "_cohort_rules", {}),
            split_rules=dict(SPLIT_RULES),
            input_hashes=input_hashes,
            feature_list_hash=feature_list_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        ledger_path = output_path.parent / (output_path.stem + ".ledger.json")
        ledger_path.write_text(json.dumps(asdict(ledger), indent=2))

        insufficient = int(snapshot["insufficient_data"].sum())
        return FeatureSnapshotResult(
            snapshot_path=output_path,
            row_count=len(snapshot),
            snapshot_version=version,
            ledger_path=ledger_path,
            insufficient_data_count=insufficient,
            cohort_exclusions=getattr(self, "_cohort_exclusions", {}),
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_source_hash(input_hashes: dict[str, str]) -> str:
    combined = "\n".join(f"{name}:{h}" for name, h in sorted(input_hashes.items()))
    return sha256(combined.encode()).hexdigest()


def _feature_list_hash() -> str:
    lines = [f"{f.name}:{f.aggregation}:{f.as_of_rule}" for f in FEATURE_CATALOG]
    # Split rule cũng là một "rule" của snapshot — đổi rule thì version phải đổi
    lines += [f"split:{k}:{v}" for k, v in sorted(SPLIT_RULES.items())]
    return sha256("\n".join(lines).encode()).hexdigest()


def build_feature_snapshot(processed_dir: Path, output_path: Path,
                           cutoff_day: int = CUTOFF_DAY) -> FeatureSnapshotResult:
    """M2 pipeline entry point."""
    return PointInTimeFeatureBuilder().build(processed_dir, output_path, cutoff_day)
