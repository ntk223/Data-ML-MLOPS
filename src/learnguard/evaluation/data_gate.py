"""Gate 1 — `learnguard evaluate data`: đánh giá chất lượng feature snapshot.

7 check với id cố định: snapshot_schema, ledger_consistency, split_integrity,
label_balance, missing_rate, duplicate_keys, drift_vs_baseline. Tái dùng tối đa
`quality.py` (EXPECTED_COLUMNS, _psi_numeric/_psi_categorical, MIN_PSI_SAMPLE),
`features.py` (SPLIT_RULES, KEYS, _feature_list_hash) và `settings.CUTOFF_DAY`.
Kết quả: một `Scorecard` subject="data"; WARN không fail, FAIL do CLI đổi exit code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from learnguard.data.features import (
    KEYS,
    STATIC_FEATURES,
    SPLIT_RULES,
    _feature_list_hash,
)
from learnguard.data.loading import BEHAVIORAL_FEATURES, SPLITS
from learnguard.data.quality import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    MIN_PSI_SAMPLE,
    NUMERIC_FEATURES,
    _psi_categorical,
    _psi_numeric,
)
from learnguard.evaluation.scorecard import GateCheck, Scorecard
from learnguard.ml.preprocessing import _validate_run_id
from learnguard.settings import CUTOFF_DAY

# Feature GỐC (không tính cột *_missing indicator) — phạm vi check missing_rate.
ORIGINAL_FEATURES: tuple[str, ...] = (
    tuple(STATIC_FEATURES) + ("module_presentation",) + tuple(BEHAVIORAL_FEATURES)
)

_LEDGER_VERSION_PARTS = 4
_HEX_DIGITS = set("0123456789abcdef")


@dataclass(frozen=True)
class DataGateThresholds:
    """Ngưỡng warn/fail của data gate — chốt theo hợp đồng scorecard v1.0."""

    label_balance_warn: float = 0.10
    label_balance_fail: float = 0.20
    missing_warn: float = 0.30
    missing_fail: float = 0.60
    psi_warn: float = 0.10
    psi_fail: float = 0.20


THRESHOLDS = DataGateThresholds()


def run_data_gate(
    snapshot_path: Path | str,
    baseline_snapshot_path: Path | str | None,
    run_id: str,
) -> Scorecard:
    """Chạy 7 check data gate và trả về `Scorecard` subject="data"."""
    run_id = _validate_run_id(run_id)
    snapshot_path = Path(snapshot_path)
    frame = pd.read_parquet(snapshot_path)
    card = Scorecard(
        run_id=run_id,
        subject="data",
        inputs={
            "snapshot": str(snapshot_path),
            "baseline_snapshot": (
                str(Path(baseline_snapshot_path))
                if baseline_snapshot_path is not None
                else None
            ),
        },
    )
    card.add(_check_snapshot_schema(frame))
    card.add(_check_ledger_consistency(snapshot_path))
    card.add(_check_split_integrity(frame))
    card.add(_check_label_balance(frame))
    card.add(_check_missing_rate(frame))
    card.add(_check_duplicate_keys(frame))
    card.add(_check_drift_vs_baseline(frame, baseline_snapshot_path))
    return card


# ---- 1. snapshot_schema ------------------------------------------------------

def _check_snapshot_schema(frame: pd.DataFrame) -> GateCheck:
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    observed: dict[str, object] = {
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "missing_columns": missing,
    }
    if missing:
        return GateCheck(
            "snapshot_schema",
            "Cột bắt buộc của snapshot tồn tại + dtypes hợp lý",
            "fail",
            observed,
            f"đủ {len(EXPECTED_COLUMNS)} cột theo EXPECTED_COLUMNS",
            f"Snapshot thiếu {len(missing)} cột bắt buộc: {', '.join(missing)}",
        )
    bad_dtypes = [
        column for column in NUMERIC_FEATURES
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    label_ok = frame["label"].isin([0, 1]).all()
    observed["non_numeric_feature_columns"] = bad_dtypes
    observed["label_domain_ok"] = bool(label_ok)
    if bad_dtypes or not label_ok:
        problems = []
        if bad_dtypes:
            problems.append(f"cột numeric sai dtype: {', '.join(bad_dtypes)}")
        if not label_ok:
            problems.append("label ngoài miền {0,1}")
        return GateCheck(
            "snapshot_schema", "Cột bắt buộc của snapshot tồn tại + dtypes hợp lý",
            "fail", observed, "numeric feature có dtype số; label ∈ {0,1}",
            f"Snapshot lệch schema: {'; '.join(problems)}",
        )
    return GateCheck(
        "snapshot_schema", "Cột bắt buộc của snapshot tồn tại + dtypes hợp lý",
        "pass", observed, "đủ cột + dtypes hợp lý",
        f"Đủ {len(EXPECTED_COLUMNS)} cột bắt buộc, {len(frame)} dòng, dtypes hợp lý",
    )


# ---- 2. ledger_consistency ---------------------------------------------------

def _check_ledger_consistency(snapshot_path: Path) -> GateCheck:
    ledger_path = snapshot_path.parent / (snapshot_path.stem + ".ledger.json")
    if not ledger_path.is_file():
        return GateCheck(
            "ledger_consistency", "Ledger khớp cutoff_day + snapshot_version",
            "fail", {"ledger_path": str(ledger_path)}, f"cutoff_day == {CUTOFF_DAY}",
            f"Thiếu ledger snapshot: {ledger_path}",
        )
    ledger = json.loads(ledger_path.read_text())
    cutoff = ledger.get("cutoff_day")
    observed: dict[str, object] = {
        "ledger_cutoff_day": cutoff, "snapshot_version": ledger.get("snapshot_version"),
    }
    if cutoff != CUTOFF_DAY:
        return GateCheck(
            "ledger_consistency", "Ledger khớp cutoff_day + snapshot_version",
            "fail", observed, f"cutoff_day == {CUTOFF_DAY}",
            f"Ledger cutoff_day={cutoff} lệch settings.CUTOFF_DAY={CUTOFF_DAY}",
        )
    version = str(ledger.get("snapshot_version", ""))
    source_hash = str(ledger.get("source_hash", ""))
    feature_hash = str(ledger.get("feature_list_hash", ""))
    recomputed_feature_hash = _feature_list_hash()
    parts = version.split(":")
    format_ok = (
        len(parts) == _LEDGER_VERSION_PARTS
        and parts[0] == "snapshot-v1"
        and parts[1] == f"cutoff{CUTOFF_DAY}"
        and all(len(part) == 8 and set(part) <= _HEX_DIGITS for part in parts[2:])
    )
    source_ok = len(source_hash) == 64 and parts[2:3] == [source_hash[:8]]
    # feature_list_hash tính lại được từ FEATURE_CATALOG + SPLIT_RULES hiện hành.
    feature_ok = feature_hash == recomputed_feature_hash and parts[3:4] == [feature_hash[:8]]
    split_rules_ok = ledger.get("split_rules") == dict(SPLIT_RULES)
    observed.update({
        "version_format_ok": format_ok,
        "source_hash_segment_ok": source_ok,
        "feature_list_hash_recomputed_ok": feature_ok,
        "split_rules_match": split_rules_ok,
    })
    if not (format_ok and source_ok and feature_ok and split_rules_ok):
        return GateCheck(
            "ledger_consistency", "Ledger khớp cutoff_day + snapshot_version",
            "fail", observed,
            f"cutoff_day == {CUTOFF_DAY}; snapshot_version tái lập được",
            "Ledger không khớp hợp đồng: kiểm tra chi tiết trong observed "
            "(format version / source_hash / feature_list_hash / split_rules)",
        )
    return GateCheck(
        "ledger_consistency", "Ledger khớp cutoff_day + snapshot_version",
        "pass", observed, f"cutoff_day == {CUTOFF_DAY}",
        f"cutoff_day={CUTOFF_DAY}, snapshot_version={version} khớp hợp đồng",
    )


# ---- 3. split_integrity ------------------------------------------------------

def _check_split_integrity(frame: pd.DataFrame) -> GateCheck:
    presentations = frame["code_presentation"].astype(str)
    split_values = frame["split"].astype(str)
    unknown = sorted(set(presentations.unique()) - set(SPLIT_RULES))
    invalid_splits = sorted(set(split_values.unique()) - set(SPLITS))
    expected = presentations.map(SPLIT_RULES)
    mismatch_rows = int((split_values != expected).sum()) if "split" in frame.columns else -1
    counts = {name: int((split_values == name).sum()) for name in SPLITS}
    observed: dict[str, object] = {
        "rows_per_split": counts,
        "presentations_by_split": {
            name: sorted(presentations[split_values == name].unique().tolist())
            for name in sorted(set(split_values.unique()))
        },
        "unknown_presentations": unknown,
        "invalid_split_values": invalid_splits,
        "split_rule_mismatch_rows": mismatch_rows,
    }
    if unknown:
        return GateCheck(
            "split_integrity", "Split khớp SPLIT_RULES theo kỳ học",
            "fail", observed, "mọi code_presentation ∈ SPLIT_RULES",
            f"unknown_presentation: {', '.join(unknown)} — không có rule split",
        )
    if invalid_splits:
        return GateCheck(
            "split_integrity", "Split khớp SPLIT_RULES theo kỳ học",
            "fail", observed, f"split ∈ {{{', '.join(SPLITS)}}}",
            f"Cột split chứa giá trị ngoài domain: {', '.join(invalid_splits)}",
        )
    if mismatch_rows:
        return GateCheck(
            "split_integrity", "Split khớp SPLIT_RULES theo kỳ học",
            "fail", observed, "split == SPLIT_RULES[code_presentation] trên mọi dòng",
            f"{mismatch_rows} dòng có split lệch SPLIT_RULES theo kỳ học",
        )
    return GateCheck(
        "split_integrity", "Split khớp SPLIT_RULES theo kỳ học",
        "pass", observed, "mọi code_presentation ∈ SPLIT_RULES, split khớp 1-1",
        f"Split khớp SPLIT_RULES — train={counts['train']}, "
        f"validation={counts['validation']}, holdout={counts['holdout']}",
    )


# ---- 4. label_balance --------------------------------------------------------

def _check_label_balance(frame: pd.DataFrame) -> GateCheck:
    rates = {
        name: float(frame.loc[frame["split"] == name, "label"].mean())
        for name in SPLITS if (frame["split"] == name).any()
    }
    observed: dict[str, object] = {"label_rate_by_split": _round_dict(rates)}
    if "train" not in rates or "validation" not in rates:
        return GateCheck(
            "label_balance", "Label rate train vs validation không lệch quá xa",
            "fail", observed, f"|rate_train − rate_validation| ≤ {THRESHOLDS.label_balance_warn}",
            "Thiếu split train hoặc validation — không đo được chênh lệch label rate",
        )
    delta = abs(rates["train"] - rates["validation"])
    observed["train_validation_delta"] = round(delta, 4)
    if delta > THRESHOLDS.label_balance_fail:
        return GateCheck(
            "label_balance", "Label rate train vs validation không lệch quá xa",
            "fail", observed,
            f"|rate_train − rate_validation| ≤ {THRESHOLDS.label_balance_fail}",
            f"Label rate lệch {delta:.3f} > {THRESHOLDS.label_balance_fail} "
            f"(train={rates['train']:.3f}, validation={rates['validation']:.3f})",
        )
    if delta > THRESHOLDS.label_balance_warn:
        return GateCheck(
            "label_balance", "Label rate train vs validation không lệch quá xa",
            "warn", observed,
            f"|rate_train − rate_validation| ≤ {THRESHOLDS.label_balance_warn}",
            f"Label rate lệch {delta:.3f} > {THRESHOLDS.label_balance_warn} "
            f"(train={rates['train']:.3f}, validation={rates['validation']:.3f})",
        )
    return GateCheck(
        "label_balance", "Label rate train vs validation không lệch quá xa",
        "pass", observed, f"|rate_train − rate_validation| ≤ {THRESHOLDS.label_balance_warn}",
        f"Label rate cân bằng: train={rates['train']:.3f}, validation={rates['validation']:.3f}",
    )


# ---- 5. missing_rate ---------------------------------------------------------

def _check_missing_rate(frame: pd.DataFrame) -> GateCheck:
    ratios = {
        column: float(frame[column].isna().mean())
        for column in ORIGINAL_FEATURES if column in frame.columns
    }
    warn_features = sorted(
        name for name, ratio in ratios.items()
        if THRESHOLDS.missing_warn < ratio <= THRESHOLDS.missing_fail
    )
    fail_features = sorted(
        name for name, ratio in ratios.items() if ratio > THRESHOLDS.missing_fail
    )
    observed: dict[str, object] = {
        "null_ratio_by_feature": _round_dict(ratios),
        "warn_features": warn_features,
        "fail_features": fail_features,
    }
    if fail_features:
        return GateCheck(
            "missing_rate", "Tỉ lệ null feature gốc trong ngưỡng cho phép",
            "fail", observed, f"null ratio ≤ {THRESHOLDS.missing_fail} mỗi feature",
            f"Feature null quá nặng (> {THRESHOLDS.missing_fail}): "
            f"{_format_pairs(fail_features, ratios)}",
        )
    if warn_features:
        return GateCheck(
            "missing_rate", "Tỉ lệ null feature gốc trong ngưỡng cho phép",
            "warn", observed, f"null ratio ≤ {THRESHOLDS.missing_warn} mỗi feature",
            f"Feature null cao (> {THRESHOLDS.missing_warn}): "
            f"{_format_pairs(warn_features, ratios)}",
        )
    worst = max(ratios.items(), key=lambda item: item[1]) if ratios else ("", 0.0)
    return GateCheck(
        "missing_rate", "Tỉ lệ null feature gốc trong ngưỡng cho phép",
        "pass", observed, f"null ratio ≤ {THRESHOLDS.missing_warn} mỗi feature",
        f"Không feature nào vượt ngưỡng null (cao nhất: {worst[0]}={worst[1]:.3f})",
    )


# ---- 6. duplicate_keys -------------------------------------------------------

def _check_duplicate_keys(frame: pd.DataFrame) -> GateCheck:
    duplicate_mask = frame.duplicated(subset=list(KEYS), keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    observed = {"duplicate_key_rows": duplicate_rows}
    if duplicate_rows:
        return GateCheck(
            "duplicate_keys", "(code_module, code_presentation, id_student) unique",
            "fail", observed, "0 dòng trùng khóa composite",
            f"Phát hiện {duplicate_rows} dòng trùng khóa composite — snapshot không đếm được duy nhất",
        )
    return GateCheck(
        "duplicate_keys", "(code_module, code_presentation, id_student) unique",
        "pass", observed, "0 dòng trùng khóa composite",
        "Khóa composite unique trên toàn bộ snapshot",
    )


# ---- 7. drift_vs_baseline ----------------------------------------------------

def _check_drift_vs_baseline(
    frame: pd.DataFrame, baseline_snapshot_path: Path | str | None,
) -> GateCheck:
    if baseline_snapshot_path is None:
        return GateCheck(
            "drift_vs_baseline", "PSI so với snapshot baseline (cùng split validation)",
            "skipped", {"reason": "no_baseline_snapshot"}, None,
            "Không truyền --baseline-snapshot → bỏ qua check drift",
        )
    baseline = pd.read_parquet(Path(baseline_snapshot_path))
    current_validation = frame[frame["split"] == "validation"]
    baseline_validation = baseline[baseline["split"] == "validation"]
    observed: dict[str, object] = {
        "baseline_snapshot": str(baseline_snapshot_path),
        "current_validation_rows": int(len(current_validation)),
        "baseline_validation_rows": int(len(baseline_validation)),
    }
    if len(current_validation) < MIN_PSI_SAMPLE or len(baseline_validation) < MIN_PSI_SAMPLE:
        observed["reason"] = "insufficient_sample"
        return GateCheck(
            "drift_vs_baseline", "PSI so với snapshot baseline (cùng split validation)",
            "skipped", observed, None,
            f"PSI không đáng tin với mẫu nhỏ (< {MIN_PSI_SAMPLE} dòng validation) — "
            "bỏ qua theo MIN_PSI_SAMPLE của quality.py",
        )
    psi_by_feature: dict[str, float] = {}
    for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if column not in frame.columns or column not in baseline.columns:
            continue
        if column in NUMERIC_FEATURES:
            psi = _psi_numeric(baseline_validation[column], current_validation[column])
        else:
            psi = _psi_categorical(baseline_validation[column], current_validation[column])
        psi_by_feature[column] = round(float(psi), 4)
    top5 = dict(sorted(psi_by_feature.items(), key=lambda item: -item[1])[:5])
    max_psi = max(psi_by_feature.values(), default=0.0)
    worst_feature = max(psi_by_feature.items(), key=lambda item: item[1])[0] if psi_by_feature else ""
    observed["psi_by_feature_top5"] = top5
    observed["max_psi"] = max_psi
    observed["max_psi_feature"] = worst_feature
    if max_psi > THRESHOLDS.psi_fail:
        return GateCheck(
            "drift_vs_baseline", "PSI so với snapshot baseline (cùng split validation)",
            "fail", observed, f"PSI ≤ {THRESHOLDS.psi_fail} mỗi feature",
            f"Drift nặng: PSI={max_psi:.3f} > {THRESHOLDS.psi_fail} tại '{worst_feature}'",
        )
    if max_psi > THRESHOLDS.psi_warn:
        return GateCheck(
            "drift_vs_baseline", "PSI so với snapshot baseline (cùng split validation)",
            "warn", observed, f"PSI ≤ {THRESHOLDS.psi_warn} mỗi feature",
            f"Drift đáng chú ý: PSI={max_psi:.3f} > {THRESHOLDS.psi_warn} tại '{worst_feature}'",
        )
    return GateCheck(
        "drift_vs_baseline", "PSI so với snapshot baseline (cùng split validation)",
        "pass", observed, f"PSI ≤ {THRESHOLDS.psi_warn} mỗi feature",
        f"Không drift đáng kể (PSI cao nhất {max_psi:.3f} tại '{worst_feature}')",
    )


# ---- helpers -----------------------------------------------------------------

def _round_dict(values: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 4) for name, value in values.items()}


def _format_pairs(names: list[str], values: dict[str, float]) -> str:
    return ", ".join(f"{name}={values[name]:.3f}" for name in names)
