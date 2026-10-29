"""M2 — deterministic point-in-time feature snapshot at course day 27."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from learnguard.data.schema import TARGET_LABELS
from learnguard.data.ingestion import _dataframe_content_hash
from learnguard.data.snapshot_contract import (
    COHORT_RULES,
    DERIVED_RULES,
    FEATURE_CATALOG,
    HASH_ALGORITHM,
    KEYS,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_SCHEMA_VERSION,
    SPLIT_RULES,
    STATIC_FEATURES,
    FeatureDefinition,
    contract_hash,
    expected_snapshot_version,
    feature_rules,
    snapshot_content_hash as _snapshot_content_hash,
    success_path,
    validate_cutoff,
)
from learnguard.settings import CUTOFF_DAY


@dataclass
class AsOfLedger:
    cutoff_day: int
    snapshot_version: str
    source_hash: str
    feature_rules: dict[str, str] = field(default_factory=dict)
    cohort_rules: dict[str, str] = field(default_factory=dict)
    split_rules: dict[str, str] = field(default_factory=dict)
    derived_rules: dict[str, str] = field(default_factory=dict)
    input_hashes: dict[str, str] = field(default_factory=dict)
    input_content_hashes: dict[str, str] = field(default_factory=dict)
    feature_list_hash: str = ""
    contract_hash: str = ""
    schema_version: str = SNAPSHOT_SCHEMA_VERSION
    hash_algorithm: str = HASH_ALGORITHM
    output_schema: list[str] = field(default_factory=list)
    row_count: int = 0
    snapshot_content_hash: str = ""
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
    """Build a cohort and features using only information available by day 27."""

    def build_cohort(self, tables: object, cutoff_day: int) -> pd.DataFrame:
        validate_cutoff(cutoff_day)
        registrations: pd.DataFrame = tables["studentRegistration"].copy()
        unknown_registration = int(registrations["date_registration"].isna().sum())
        registered = registrations[
            registrations["date_registration"].isna()
            | registrations["date_registration"].le(cutoff_day)
        ]
        withdrew = (
            registered["date_unregistration"].notna()
            & registered["date_unregistration"].le(cutoff_day)
        )
        cohort = registered[~withdrew].loc[:, KEYS].copy()
        self._cohort_exclusions = {
            "withdrawn_by_cutoff": int(withdrew.sum()),
            "kept_unknown_registration_date": unknown_registration,
        }
        self._cohort_rules = dict(COHORT_RULES)
        return cohort

    def build_feature(
        self, feature: FeatureDefinition, tables: object, cutoff_day: int
    ) -> pd.DataFrame:
        validate_cutoff(cutoff_day)
        if feature.name == "module_presentation":
            courses: pd.DataFrame = tables["courses"]
            values = courses.assign(
                module_presentation=(
                    courses["code_module"] + "-" + courses["code_presentation"]
                )
            )
            return values[["code_module", "code_presentation", feature.name]]

        if feature.name == "n_clicks_before_cutoff":
            events: pd.DataFrame = tables["studentVle"]
            eligible = events[events["date"].le(cutoff_day)]
            values = eligible.groupby(list(KEYS), as_index=False)["sum_click"].sum()
            return values.rename(columns={"sum_click": feature.name})

        if feature.name == "active_days_before_cutoff":
            events = tables["studentVle"]
            eligible = events[events["date"].le(cutoff_day)]
            values = eligible.groupby(list(KEYS))["date"].nunique().rename(feature.name)
            return values.reset_index()

        if feature.name == "click_trend_w3_w4":
            events = tables["studentVle"]
            keys = list(KEYS)
            weeks_1_2 = events[events["date"].between(0, 13)].groupby(keys)["sum_click"].sum()
            weeks_3_4 = events[events["date"].between(14, cutoff_day)].groupby(keys)["sum_click"].sum()
            return (weeks_3_4 - weeks_1_2).rename(feature.name).reset_index()

        if feature.name in {"n_submitted_before_cutoff", "avg_score_submitted"}:
            submissions: pd.DataFrame = tables["studentAssessment"]
            assessments: pd.DataFrame = tables["assessments"]
            joined = submissions.merge(
                assessments[["id_assessment", "code_module", "code_presentation", "date"]],
                on="id_assessment", how="left",
            )
            eligible = (
                joined["date_submitted"].le(cutoff_day)
                & joined["date"].notna()
                & joined["date"].le(cutoff_day)
            )
            eligible_rows = joined[eligible]
            if feature.name == "n_submitted_before_cutoff":
                return eligible_rows.groupby(list(KEYS)).size().rename(feature.name).reset_index()
            scored = eligible_rows[eligible_rows["score"].notna()]
            return scored.groupby(list(KEYS))["score"].mean().rename(feature.name).reset_index()

        raise KeyError(f"feature_not_in_catalog:{feature.name}")

    def assign_split(self, snapshot: pd.DataFrame) -> pd.DataFrame:
        snapshot = snapshot.copy()
        snapshot["split"] = snapshot["code_presentation"].map(SPLIT_RULES)
        if snapshot["split"].isna().any():
            unknown = sorted(snapshot.loc[snapshot["split"].isna(), "code_presentation"].unique())
            raise ValueError(f"unknown_presentation_for_split:{','.join(unknown)}")
        return snapshot

    def build_frame(self, tables: dict[str, pd.DataFrame], cutoff_day: int) -> pd.DataFrame:
        validate_cutoff(cutoff_day)
        cohort = self.build_cohort(tables, cutoff_day)
        snapshot = cohort.merge(
            tables["studentInfo"][list(KEYS) + list(STATIC_FEATURES) + ["final_result"]],
            on=list(KEYS), how="left", validate="many_to_one",
        )
        snapshot["imd_band_missing"] = snapshot["imd_band"].isna()

        for feature in FEATURE_CATALOG:
            feature_frame = self.build_feature(feature, tables, cutoff_day)
            merge_keys = (
                ["code_module", "code_presentation"]
                if feature.name == "module_presentation" else list(KEYS)
            )
            snapshot = snapshot.merge(
                feature_frame, on=merge_keys, how="left", validate="many_to_one"
            )

        for column in (
            "n_clicks_before_cutoff", "active_days_before_cutoff",
            "click_trend_w3_w4", "n_submitted_before_cutoff",
        ):
            snapshot[column] = snapshot[column].fillna(0)
        snapshot["insufficient_data"] = (
            snapshot["n_clicks_before_cutoff"].eq(0)
            & snapshot["n_submitted_before_cutoff"].eq(0)
        )
        snapshot["label"] = snapshot["final_result"].map(TARGET_LABELS)
        snapshot = self.assign_split(snapshot)
        snapshot = snapshot.sort_values(list(KEYS)).reset_index(drop=True)
        return snapshot[list(SNAPSHOT_COLUMNS)]

    def build(self, processed_dir: Path, output_path: Path,
              cutoff_day: int = CUTOFF_DAY) -> FeatureSnapshotResult:
        validate_cutoff(cutoff_day)
        processed_dir, output_path = Path(processed_dir), Path(output_path)
        canonical_dir = processed_dir / "canonical"
        report_path = processed_dir / "ingestion_report.json"
        success_marker = processed_dir / "ingestion.success.json"
        if not canonical_dir.is_dir() or not report_path.is_file() or not success_marker.is_file():
            raise ValueError("canonical_not_published:run ingestion before snapshot")
        ingestion_report = json.loads(report_path.read_text())
        if ingestion_report.get("status") != "ok" or not ingestion_report.get("canonical_published"):
            raise ValueError("canonical_ingestion_failed")

        table_names = (
            "courses", "assessments", "studentInfo", "studentRegistration",
            "studentAssessment", "studentVle",
        )
        tables = {
            name: pd.read_parquet(canonical_dir / f"{name}.parquet")
            for name in table_names
        }
        input_hashes = {
            name: _sha256_file(canonical_dir / f"{name}.parquet") for name in table_names
        }
        reported_hashes = ingestion_report.get("canonical_content_hashes", {})
        input_content_hashes = {name: reported_hashes.get(name, "") for name in table_names}
        if any(not value for value in input_content_hashes.values()):
            raise ValueError("canonical_content_hash_missing")
        actual_content_hashes = {
            name: _dataframe_content_hash(frame) for name, frame in tables.items()
        }
        if input_content_hashes != actual_content_hashes:
            raise ValueError("canonical_content_hash_mismatch")

        snapshot = self.build_frame(tables, cutoff_day)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_snapshot = output_path.parent / f".{output_path.name}.tmp"
        snapshot.to_parquet(temp_snapshot, index=False)

        full_contract_hash = contract_hash()
        source_hash = _combined_source_hash(input_content_hashes)
        content_hash = snapshot_content_hash(snapshot)
        version = expected_snapshot_version(source_hash)
        ledger = AsOfLedger(
            cutoff_day=cutoff_day,
            snapshot_version=version,
            source_hash=source_hash,
            feature_rules=feature_rules(),
            cohort_rules=dict(COHORT_RULES),
            split_rules=dict(SPLIT_RULES),
            derived_rules=dict(DERIVED_RULES),
            input_hashes=input_hashes,
            input_content_hashes=input_content_hashes,
            feature_list_hash=full_contract_hash,
            contract_hash=full_contract_hash,
            output_schema=list(SNAPSHOT_COLUMNS),
            row_count=len(snapshot),
            snapshot_content_hash=content_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        ledger_path = output_path.parent / f"{output_path.stem}.ledger.json"
        temp_ledger = output_path.parent / f".{ledger_path.name}.tmp"
        temp_ledger.write_text(json.dumps(asdict(ledger), indent=2))
        temp_snapshot.replace(output_path)
        temp_ledger.replace(ledger_path)
        success_path(output_path).write_text(json.dumps({
            "snapshot_version": version,
            "snapshot_content_hash": content_hash,
        }, indent=2))

        return FeatureSnapshotResult(
            snapshot_path=output_path,
            row_count=len(snapshot),
            snapshot_version=version,
            ledger_path=ledger_path,
            insufficient_data_count=int(snapshot["insufficient_data"].sum()),
            cohort_exclusions=getattr(self, "_cohort_exclusions", {}),
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_source_hash(input_hashes: dict[str, str]) -> str:
    combined = "\n".join(f"{name}:{value}" for name, value in sorted(input_hashes.items()))
    return sha256(combined.encode()).hexdigest()


def _feature_list_hash(cutoff_day: int = CUTOFF_DAY) -> str:
    validate_cutoff(cutoff_day)
    return contract_hash()


def snapshot_content_hash(snapshot: pd.DataFrame) -> str:
    return _snapshot_content_hash(snapshot)


def build_feature_snapshot(processed_dir: Path, output_path: Path,
                           cutoff_day: int = CUTOFF_DAY) -> FeatureSnapshotResult:
    return PointInTimeFeatureBuilder().build(processed_dir, output_path, cutoff_day)
