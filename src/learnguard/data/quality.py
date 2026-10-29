"""M3 snapshot schema, missingness, volume, and PSI quality gates."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from learnguard.data.loading import validate_snapshot_artifact
from learnguard.data.schema import (
    AGE_BANDS,
    CODE_MODULES,
    CODE_PRESENTATIONS,
    HIGHEST_EDUCATION,
    IMD_BANDS,
    REGIONS,
    TARGET_LABELS,
)
from learnguard.data.snapshot_contract import (
    CATEGORICAL_FEATURES,
    KEYS,
    NULLABLE_COLUMNS,
    NUMERIC_FEATURES,
    SNAPSHOT_COLUMNS,
    SPLITS,
    SPLIT_RULES,
)

_PSI_EPSILON = 1e-6
MIN_PSI_SAMPLE = 200


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
    gate_status: dict[str, str] = field(default_factory=dict)
    baseline_snapshot_version: str | None = None
    errors: list[str] = field(default_factory=list)


class DataQualityGate:
    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def validate_snapshot(self, snapshot_path: Path,
                          previous_snapshot_path: Path | None = None) -> QualityReport:
        snapshot_path = Path(snapshot_path)
        snapshot = pd.read_parquet(snapshot_path)
        report = QualityReport(
            "ok", len(snapshot),
            gate_status={
                "schema": "passed", "missingness": "passed",
                "volume": "not_applicable", "psi": "not_applicable",
            },
        )
        self._validate_schema(snapshot, report)
        self._validate_missingness(snapshot, report)

        previous: pd.DataFrame | None = None
        if previous_snapshot_path is not None:
            baseline_path = Path(previous_snapshot_path)
            if not baseline_path.is_file():
                report.errors.append(f"baseline_missing:{baseline_path}")
            else:
                try:
                    previous, ledger = validate_snapshot_artifact(baseline_path)
                    report.baseline_snapshot_version = str(ledger["snapshot_version"])
                    if previous.empty:
                        report.errors.append("baseline_empty")
                except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    report.errors.append(f"baseline_invalid:{type(exc).__name__}:{exc}")

        if previous is not None and not previous.empty:
            report.gate_status["volume"] = "passed"
            ratio = len(snapshot) / len(previous)
            report.volume_ratio = round(ratio, 4)
            if ratio < self.thresholds.min_volume_ratio or ratio > self.thresholds.max_volume_ratio:
                report.errors.append(
                    f"volume_out_of_bounds:{ratio:.3f}:allowed="
                    f"[{self.thresholds.min_volume_ratio},{self.thresholds.max_volume_ratio}]"
                )
            schema_failed = any(
                error.startswith("schema_") for error in report.errors
            )
            if (
                not schema_failed
                and len(previous) >= MIN_PSI_SAMPLE
                and len(snapshot) >= MIN_PSI_SAMPLE
            ):
                report.gate_status["psi"] = "passed"
                for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
                    psi = (
                        _psi_numeric(previous[column], snapshot[column])
                        if column in NUMERIC_FEATURES
                        else _psi_categorical(previous[column], snapshot[column])
                    )
                    report.psi_by_feature[column] = round(psi, 4)
                    if psi > self.thresholds.max_psi:
                        report.errors.append(f"psi_exceeded:{column}:{psi:.3f}")

        return self._finish(report)

    def _validate_schema(self, snapshot: pd.DataFrame, report: QualityReport) -> None:
        missing = [column for column in SNAPSHOT_COLUMNS if column not in snapshot.columns]
        extra = [column for column in snapshot.columns if column not in SNAPSHOT_COLUMNS]
        if missing:
            report.errors.append(f"schema_missing_columns:{','.join(missing)}")
        if extra:
            report.errors.append(f"schema_unexpected_columns:{','.join(extra)}")
        if missing or extra:
            return
        if list(snapshot.columns) != list(SNAPSHOT_COLUMNS):
            report.errors.append("schema_column_order_mismatch")
            return
        if snapshot.empty:
            report.errors.append("schema_empty_snapshot")
            return

        string_columns = (
            "code_module", "code_presentation", "gender", "region",
            "highest_education", "imd_band", "age_band", "disability",
            "final_result", "module_presentation", "split",
        )
        integer_columns = ("id_student", "num_of_prev_attempts", "studied_credits", "label")
        for column in string_columns:
            if not is_string_dtype(snapshot[column].dtype):
                report.errors.append(f"schema_dtype:{column}:expected=string")
        for column in integer_columns:
            if not is_integer_dtype(snapshot[column].dtype):
                report.errors.append(f"schema_dtype:{column}:expected=integer")
        for column in NUMERIC_FEATURES:
            if not is_numeric_dtype(snapshot[column].dtype):
                report.errors.append(f"schema_dtype:{column}:expected=numeric")
        for column in ("imd_band_missing", "insufficient_data"):
            if not is_bool_dtype(snapshot[column].dtype):
                report.errors.append(f"schema_dtype:{column}:expected=bool")

        domains = {
            "code_module": set(CODE_MODULES),
            "code_presentation": set(CODE_PRESENTATIONS),
            "gender": {"M", "F"},
            "region": set(REGIONS),
            "highest_education": set(HIGHEST_EDUCATION),
            "imd_band": set(IMD_BANDS),
            "age_band": set(AGE_BANDS),
            "disability": {"Y", "N"},
            "final_result": set(TARGET_LABELS),
            "label": {0, 1},
            "split": set(SPLITS),
        }
        for column, allowed in domains.items():
            bad = set(snapshot[column].dropna().unique()) - allowed
            if bad:
                report.errors.append(f"schema_domain:{column}:{','.join(map(str, sorted(bad)))}")
        if not snapshot["label"].isin([0, 1]).all():
            report.errors.append("schema_label_out_of_domain")

        duplicates = snapshot.duplicated(subset=list(KEYS), keep=False)
        if duplicates.any():
            report.errors.append(f"schema_duplicate_keys:{int(duplicates.sum())}")
        if not set(SPLITS) <= set(snapshot["split"].dropna().unique()):
            report.errors.append("schema_split_empty")
        expected_split = snapshot["code_presentation"].map(SPLIT_RULES)
        if expected_split.isna().any() or not expected_split.equals(snapshot["split"]):
            report.errors.append("schema_split_rule_mismatch")
        if not snapshot["final_result"].map(TARGET_LABELS).equals(snapshot["label"]):
            report.errors.append("schema_label_final_result_mismatch")
        expected_module = snapshot["code_module"] + "-" + snapshot["code_presentation"]
        if not expected_module.equals(snapshot["module_presentation"]):
            report.errors.append("schema_module_presentation_mismatch")
        if not snapshot["imd_band"].isna().equals(snapshot["imd_band_missing"]):
            report.errors.append("schema_imd_missing_flag_mismatch")
        expected_insufficient = (
            snapshot["n_clicks_before_cutoff"].eq(0)
            & snapshot["n_submitted_before_cutoff"].eq(0)
        )
        if not expected_insufficient.equals(snapshot["insufficient_data"]):
            report.errors.append("schema_insufficient_data_flag_mismatch")

        for column in (
            "n_clicks_before_cutoff", "active_days_before_cutoff",
            "n_submitted_before_cutoff", "num_of_prev_attempts", "studied_credits",
        ):
            values = snapshot[column].dropna()
            if (values < 0).any():
                report.errors.append(f"schema_negative_value:{column}")
            if column in {"n_clicks_before_cutoff", "active_days_before_cutoff", "n_submitted_before_cutoff"} and not np.allclose(values, np.round(values)):
                report.errors.append(f"schema_non_integer_count:{column}")
        scores = snapshot["avg_score_submitted"].dropna()
        if ((scores < 0) | (scores > 100)).any():
            report.errors.append("schema_avg_score_out_of_range")

    def _validate_missingness(self, snapshot: pd.DataFrame, report: QualityReport) -> None:
        for column in SNAPSHOT_COLUMNS:
            if column not in snapshot.columns:
                continue
            fraction = float(snapshot[column].isna().mean())
            report.null_fraction_by_feature[column] = round(fraction, 4)
            if column not in NULLABLE_COLUMNS and fraction > 0:
                report.errors.append(f"missingness_not_allowed:{column}:{fraction:.3f}")
            elif column in NULLABLE_COLUMNS and fraction > self.thresholds.max_null_fraction:
                report.errors.append(f"missingness_exceeded:{column}:{fraction:.3f}")

    @staticmethod
    def _finish(report: QualityReport) -> QualityReport:
        if report.errors:
            report.status = "failed"
            prefixes = {
                "schema": ("schema_",),
                "missingness": ("missingness_",),
                "volume": ("volume_", "baseline_"),
                "psi": ("psi_", "baseline_"),
            }
            for gate, values in prefixes.items():
                if any(error.startswith(values) for error in report.errors):
                    report.gate_status[gate] = "failed"
        return report


def _psi_numeric(base: pd.Series, current: pd.Series, bins: int = 10) -> float:
    base_values = pd.to_numeric(base, errors="coerce").dropna()
    current_values = pd.to_numeric(current, errors="coerce").dropna()
    if base_values.empty or current_values.empty or base_values.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(base_values.to_numpy(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    base_prop = _proportions(np.histogram(base_values.to_numpy(), bins=edges)[0], len(base_values))
    current_prop = _proportions(np.histogram(current_values.to_numpy(), bins=edges)[0], len(current_values))
    return float(np.sum((current_prop - base_prop) * np.log(current_prop / base_prop)))


def _psi_categorical(base: pd.Series, current: pd.Series) -> float:
    base_prop = base.fillna("__NA__").value_counts(normalize=True)
    current_prop = current.fillna("__NA__").value_counts(normalize=True)
    psi = 0.0
    for category in set(base_prop.index) | set(current_prop.index):
        base_value = max(float(base_prop.get(category, 0.0)), _PSI_EPSILON)
        current_value = max(float(current_prop.get(category, 0.0)), _PSI_EPSILON)
        psi += (current_value - base_value) * np.log(current_value / base_value)
    return float(psi)


def _proportions(counts: np.ndarray, total: int) -> np.ndarray:
    return np.clip(counts / max(total, 1), _PSI_EPSILON, None)


def validate_snapshot(snapshot_path: Path,
                      previous_snapshot_path: Path | None = None) -> QualityReport:
    return DataQualityGate().validate_snapshot(snapshot_path, previous_snapshot_path)
