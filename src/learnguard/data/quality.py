"""M3 — Data quality gates: schema, missingness, volume, PSI (drift).

Gate FAIL → chặn pipeline, KHÔNG xuất batch (theo TODO mục Orchestration).
PSI đo trôi phân bố feature so với snapshot baseline — quan trọng khi chuyển
từ OULAD sang dữ liệu production thật.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from learnguard.data.features import FEATURE_CATALOG, KEYS, STATIC_FEATURES

NUMERIC_FEATURES = (
    "n_clicks_before_cutoff",
    "click_trend_w3_w4",
    "n_submitted_before_cutoff",
    "avg_score_submitted",
    "num_of_prev_attempts",
    "studied_credits",
)
CATEGORICAL_FEATURES = (
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "module_presentation",
)
EXPECTED_COLUMNS = tuple(KEYS) + tuple(STATIC_FEATURES) + (
    "imd_band_missing",
    "module_presentation",
    "n_clicks_before_cutoff",
    "click_trend_w3_w4",
    "n_submitted_before_cutoff",
    "avg_score_submitted",
    "insufficient_data",
    "final_result",
    "label",
)

_PSI_EPSILON = 1e-6
# PSI không đáng tin với mẫu nhỏ (nhiễu lấy mẫu lấn át tín hiệu trôi — đã chứng minh
# bằng thực nghiệm: 2 mẫu CÙNG phân bố n≈100 cho PSI 1.2). Chỉ tính khi đủ mẫu.
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
    errors: list[str] = field(default_factory=list)


class DataQualityGate:
    def __init__(self, thresholds: QualityThresholds | None = None) -> None:
        self.thresholds = thresholds or QualityThresholds()

    def validate_snapshot(self, snapshot_path: Path,
                          previous_snapshot_path: Path | None = None) -> QualityReport:
        """Đánh giá 4 gates: schema, missingness, volume, PSI."""
        snapshot_path = Path(snapshot_path)
        snap = pd.read_parquet(snapshot_path)
        report = QualityReport(status="ok", row_count=len(snap))

        # --- Gate 1: schema ----------------------------------------------------
        missing = [c for c in EXPECTED_COLUMNS if c not in snap.columns]
        if missing:
            report.errors.append(f"schema_missing_columns:{','.join(missing)}")
        else:
            if not snap["label"].isin([0, 1]).all():
                report.errors.append("schema_label_out_of_domain")
            dup = snap.duplicated(subset=list(KEYS), keep=False)
            if dup.any():
                report.errors.append(f"schema_duplicate_keys:{int(dup.sum())}")

        # --- Gate 2: missingness ------------------------------------------------
        feature_cols = [c for c in EXPECTED_COLUMNS if c in snap.columns and c != "label"]
        for col in feature_cols:
            frac = float(snap[col].isna().mean())
            report.null_fraction_by_feature[col] = round(frac, 4)
            if frac > self.thresholds.max_null_fraction:
                report.errors.append(f"missingness_exceeded:{col}:{frac:.3f}")

        # --- Gate 3 + 4: volume + PSI (cần snapshot baseline) -------------------
        has_previous = previous_snapshot_path is not None and Path(previous_snapshot_path).is_file()
        previous = pd.read_parquet(previous_snapshot_path) if has_previous else None

        if has_previous and previous is not None and len(previous) > 0:
            ratio = len(snap) / len(previous)
            report.volume_ratio = round(ratio, 4)
            if ratio < self.thresholds.min_volume_ratio or ratio > self.thresholds.max_volume_ratio:
                report.errors.append(
                    f"volume_out_of_bounds:{ratio:.3f}"
                    f":allowed=[{self.thresholds.min_volume_ratio},{self.thresholds.max_volume_ratio}]"
                )

        # PSI: mẫu quá nhỏ → bỏ qua (nhiễu > tín hiệu); kích thước do volume gate quản.
        if has_previous and len(previous) >= MIN_PSI_SAMPLE and len(snap) >= MIN_PSI_SAMPLE:
            for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
                if col not in snap.columns or col not in previous.columns:
                    continue
                if col in NUMERIC_FEATURES:
                    psi = _psi_numeric(previous[col], snap[col])
                else:
                    psi = _psi_categorical(previous[col], snap[col])
                report.psi_by_feature[col] = round(psi, 4)
                if psi > self.thresholds.max_psi:
                    report.errors.append(f"psi_exceeded:{col}:{psi:.3f}")

        return self._finish(report)

    def _finish(self, report: QualityReport) -> QualityReport:
        if report.errors:
            report.status = "failed"
        return report


def _psi_numeric(base: pd.Series, current: pd.Series, bins: int = 10) -> float:
    base = pd.to_numeric(base, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if base.empty or current.empty or base.nunique() < 2:
        return 0.0
    quantiles = np.unique(np.quantile(base.to_numpy(), np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 2:
        return 0.0
    edges = quantiles.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    base_prop = _proportions(np.histogram(base.to_numpy(), bins=edges)[0], len(base))
    cur_prop = _proportions(np.histogram(current.to_numpy(), bins=edges)[0], len(current))
    return float(np.sum((cur_prop - base_prop) * np.log(cur_prop / base_prop)))


def _psi_categorical(base: pd.Series, current: pd.Series) -> float:
    base_prop = base.fillna("__NA__").value_counts(normalize=True)
    cur_prop = current.fillna("__NA__").value_counts(normalize=True)
    categories = set(base_prop.index) | set(cur_prop.index)
    psi = 0.0
    for cat in categories:
        b = max(float(base_prop.get(cat, 0.0)), _PSI_EPSILON)
        c = max(float(cur_prop.get(cat, 0.0)), _PSI_EPSILON)
        psi += (c - b) * np.log(c / b)
    return float(psi)


def _proportions(counts: np.ndarray, total: int) -> np.ndarray:
    prop = counts / max(total, 1)
    return np.clip(prop, _PSI_EPSILON, None)


def validate_snapshot(snapshot_path: Path,
                      previous_snapshot_path: Path | None = None) -> QualityReport:
    """M3 quality gate entry point."""
    return DataQualityGate().validate_snapshot(snapshot_path, previous_snapshot_path)
