"""Single source of truth for the day-27 OULAD snapshot contract."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pandas as pd

from learnguard.data.schema import TARGET_LABELS

CUTOFF_DAY = 27

SNAPSHOT_SCHEMA_VERSION = "snapshot-v3"
HASH_ALGORITHM = "pandas-hash-v1"
KEYS = ("code_module", "code_presentation", "id_student")
SPLITS = ("train", "validation", "holdout")
SPLIT_RULES: dict[str, str] = {
    "2013B": "train",
    "2013J": "train",
    "2014B": "validation",
    "2014J": "holdout",
}
STATIC_FEATURES = (
    "gender", "region", "highest_education", "imd_band", "age_band",
    "num_of_prev_attempts", "studied_credits", "disability",
)
COHORT_RULES = {
    "include": "date_registration <= 27 OR date_registration is null",
    "exclude": "date_unregistration <= 27",
}
DERIVED_RULES = {
    "imd_band_missing": "imd_band is null",
    "insufficient_data": "n_clicks_before_cutoff == 0 AND n_submitted_before_cutoff == 0",
    "label": "Fail/Withdrawn=1; Pass/Distinction=0",
    "fill_zero": (
        "n_clicks_before_cutoff,active_days_before_cutoff,"
        "click_trend_w3_w4,n_submitted_before_cutoff"
    ),
}

FeatureKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    source: str
    aggregation: str
    as_of_rule: str
    kind: FeatureKind


FEATURE_CATALOG: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("module_presentation", "courses", "code_module + '-' + code_presentation", "static_before_start", "categorical"),
    FeatureDefinition("n_clicks_before_cutoff", "studentVle", "sum(sum_click) where date <= 27", "date_at_or_before_cutoff", "numeric"),
    FeatureDefinition("active_days_before_cutoff", "studentVle", "nunique(date) where date <= 27; negative dates included", "date_at_or_before_cutoff", "numeric"),
    FeatureDefinition("click_trend_w3_w4", "studentVle", "sum(clicks date 14..27) - sum(clicks date 0..13)", "date_at_or_before_cutoff", "numeric"),
    FeatureDefinition("n_submitted_before_cutoff", "studentAssessment+assessments", "count where submission_date <= 27 AND assessment_date <= 27", "submission_and_assessment_at_or_before_cutoff", "numeric"),
    FeatureDefinition("avg_score_submitted", "studentAssessment+assessments", "mean non-null score for eligible submissions", "submission_and_assessment_at_or_before_cutoff", "numeric"),
)

BEHAVIORAL_FEATURES = tuple(
    feature.name for feature in FEATURE_CATALOG if feature.name != "module_presentation"
)
MODEL_FEATURE_COLUMNS = (
    *STATIC_FEATURES,
    "module_presentation",
    *BEHAVIORAL_FEATURES,
    "imd_band_missing",
    "insufficient_data",
)
SNAPSHOT_COLUMNS = (
    *KEYS,
    *STATIC_FEATURES,
    "final_result",
    "imd_band_missing",
    "module_presentation",
    *BEHAVIORAL_FEATURES,
    "insufficient_data",
    "label",
    "split",
)
NULLABLE_COLUMNS = {"imd_band", "avg_score_submitted"}
NUMERIC_FEATURES = (
    "n_clicks_before_cutoff", "active_days_before_cutoff", "click_trend_w3_w4",
    "n_submitted_before_cutoff", "avg_score_submitted", "num_of_prev_attempts",
    "studied_credits",
)
CATEGORICAL_FEATURES = (
    "gender", "region", "highest_education", "imd_band", "age_band",
    "disability", "module_presentation",
)


def validate_cutoff(cutoff_day: int) -> None:
    if cutoff_day != CUTOFF_DAY:
        raise ValueError(f"unsupported_cutoff_day:{cutoff_day}:required={CUTOFF_DAY}")


def feature_rules() -> dict[str, str]:
    return {
        feature.name: f"{feature.aggregation} [{feature.as_of_rule}]"
        for feature in FEATURE_CATALOG
    }


def contract_hash() -> str:
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "cutoff_day": CUTOFF_DAY,
        "keys": KEYS,
        "snapshot_columns": SNAPSHOT_COLUMNS,
        "nullable_columns": sorted(NULLABLE_COLUMNS),
        "static_features": STATIC_FEATURES,
        "feature_catalog": [asdict(feature) for feature in FEATURE_CATALOG],
        "cohort_rules": COHORT_RULES,
        "derived_rules": DERIVED_RULES,
        "split_rules": SPLIT_RULES,
        "target_labels": TARGET_LABELS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def snapshot_content_hash(snapshot: pd.DataFrame) -> str:
    ordered = snapshot.sort_values(list(KEYS)).reset_index(drop=True)
    schema = "\n".join(f"{column}:{ordered[column].dtype}" for column in ordered.columns)
    values = pd.util.hash_pandas_object(ordered, index=False, categorize=True).to_numpy()
    digest = sha256(schema.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def expected_snapshot_version(source_hash: str) -> str:
    return (
        f"{SNAPSHOT_SCHEMA_VERSION}:cutoff{CUTOFF_DAY}:"
        f"{source_hash[:8]}:{contract_hash()[:8]}"
    )


def ledger_path(snapshot_path: Path) -> Path:
    return snapshot_path.parent / f"{snapshot_path.stem}.ledger.json"


def success_path(snapshot_path: Path) -> Path:
    return snapshot_path.parent / f"{snapshot_path.stem}.success.json"


def validate_ledger_metadata(ledger: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if ledger.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if ledger.get("cutoff_day") != CUTOFF_DAY:
        errors.append("cutoff_mismatch")
    expected_contract = contract_hash()
    if ledger.get("contract_hash") != expected_contract:
        errors.append("contract_hash_mismatch")
    if ledger.get("feature_list_hash") != expected_contract:
        errors.append("feature_list_hash_mismatch")
    if ledger.get("feature_rules") != feature_rules():
        errors.append("feature_rules_mismatch")
    if ledger.get("cohort_rules") != COHORT_RULES:
        errors.append("cohort_rules_mismatch")
    if ledger.get("derived_rules") != DERIVED_RULES:
        errors.append("derived_rules_mismatch")
    if ledger.get("split_rules") != SPLIT_RULES:
        errors.append("split_rules_mismatch")
    if ledger.get("hash_algorithm") != HASH_ALGORITHM:
        errors.append("hash_algorithm_mismatch")
    source_hash = ledger.get("source_hash")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        errors.append("source_hash_invalid")
    elif ledger.get("snapshot_version") != expected_snapshot_version(source_hash):
        errors.append("snapshot_version_invalid")
    return errors
