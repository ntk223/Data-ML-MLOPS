"""M3 leakage audit: provenance, full recompute, and future-event probes."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pandas as pd

from learnguard.data.features import (
    PointInTimeFeatureBuilder,
    _combined_source_hash,
    _sha256_file,
    snapshot_content_hash,
)
from learnguard.data.ingestion import _dataframe_content_hash
from learnguard.data.snapshot_contract import (
    FEATURE_CATALOG,
    KEYS,
    success_path,
    validate_cutoff,
    validate_ledger_metadata,
)

INJECTED_ID_OFFSET = 999_000
MAX_VALUE = 999_999
ALLOWED_AS_OF_RULES = {
    "static_before_start",
    "date_at_or_before_cutoff",
    "submission_and_assessment_at_or_before_cutoff",
}
FORBIDDEN_TERMS = ("final_result", "label", "target", "future")


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
    probe_students: int = 0
    violations: list[LeakageViolation] = field(default_factory=list)


class LeakageAuditor:
    def audit_catalog(self) -> LeakageAuditReport:
        report = LeakageAuditReport("ok", 27, checked_features=len(FEATURE_CATALOG))
        for feature in FEATURE_CATALOG:
            if feature.as_of_rule not in ALLOWED_AS_OF_RULES:
                report.violations.append(
                    LeakageViolation(feature.name, "unknown_as_of_rule", feature.as_of_rule)
                )
            lowered = f"{feature.name} {feature.aggregation}".lower()
            for term in FORBIDDEN_TERMS:
                if term in lowered:
                    report.violations.append(
                        LeakageViolation(feature.name, "target_or_identity_reference", term)
                    )
        if report.violations:
            report.status = "violated"
        return report

    def audit_snapshot(self, snapshot_path: Path, processed_dir: Path,
                       cutoff_day: int) -> LeakageAuditReport:
        validate_cutoff(cutoff_day)
        snapshot_path, processed_dir = Path(snapshot_path), Path(processed_dir)
        report = LeakageAuditReport("ok", cutoff_day, checked_features=len(FEATURE_CATALOG))
        report.violations.extend(self.audit_catalog().violations)
        snapshot = pd.read_parquet(snapshot_path)

        canonical_dir = processed_dir / "canonical"
        table_names = (
            "courses", "assessments", "studentInfo", "studentRegistration",
            "studentAssessment", "studentVle",
        )
        tables = {
            name: pd.read_parquet(canonical_dir / f"{name}.parquet")
            for name in table_names
        }
        artifact_hashes = {
            name: _sha256_file(canonical_dir / f"{name}.parquet") for name in table_names
        }
        content_hashes = {
            name: _dataframe_content_hash(frame) for name, frame in tables.items()
        }
        actual_source_hash = _combined_source_hash(content_hashes)

        ledger_path = snapshot_path.parent / f"{snapshot_path.stem}.ledger.json"
        marker_path = success_path(snapshot_path)
        ledger: dict[str, object] = {}
        if not ledger_path.is_file():
            report.violations.append(
                LeakageViolation("ledger", "missing_ledger", str(ledger_path))
            )
        else:
            ledger = json.loads(ledger_path.read_text())
            for error in validate_ledger_metadata(ledger):
                report.violations.append(
                    LeakageViolation("ledger", error, "ledger metadata differs from contract")
                )
            if ledger.get("source_hash") != actual_source_hash:
                report.violations.append(
                    LeakageViolation("ledger", "source_hash_mismatch", "canonical content changed")
                )
            if ledger.get("input_hashes") != artifact_hashes:
                report.violations.append(
                    LeakageViolation("ledger", "input_hashes_mismatch", "Parquet artifact changed")
                )
            if ledger.get("input_content_hashes") != content_hashes:
                report.violations.append(
                    LeakageViolation("ledger", "input_content_hashes_mismatch", "canonical values changed")
                )
            if ledger.get("row_count") != len(snapshot):
                report.violations.append(
                    LeakageViolation("ledger", "row_count_mismatch", str(len(snapshot)))
                )
            if ledger.get("snapshot_content_hash") != snapshot_content_hash(snapshot):
                report.violations.append(
                    LeakageViolation("ledger", "snapshot_content_hash_mismatch", "snapshot differs from ledger")
                )

        if not marker_path.is_file():
            report.violations.append(
                LeakageViolation("snapshot", "missing_success_marker", str(marker_path))
            )
        elif ledger:
            marker = json.loads(marker_path.read_text())
            if marker.get("snapshot_version") != ledger.get("snapshot_version") or marker.get(
                "snapshot_content_hash"
            ) != ledger.get("snapshot_content_hash"):
                report.violations.append(
                    LeakageViolation("snapshot", "success_marker_mismatch", "marker differs from ledger")
                )

        builder = PointInTimeFeatureBuilder()
        recomputed = builder.build_frame(tables, cutoff_day)
        if list(snapshot.columns) != list(recomputed.columns):
            report.violations.append(
                LeakageViolation("snapshot", "snapshot_schema_mismatch", "column order/schema differs")
            )
        elif snapshot_content_hash(snapshot) != snapshot_content_hash(recomputed):
            report.violations.append(
                LeakageViolation("snapshot", "snapshot_recompute_mismatch", "content differs from canonical")
            )

        probe_students = self._pick_probe_students(snapshot)
        report.probe_students = len(probe_students)
        for key in probe_students:
            for probe_name, injected_tables in self._future_probes(tables, key, cutoff_day):
                report.source_checks += 1
                after = builder.build_frame(injected_tables, cutoff_day)
                if snapshot_content_hash(recomputed) != snapshot_content_hash(after):
                    report.violations.append(
                        LeakageViolation(
                            "snapshot", f"future_event_leak:{probe_name}",
                            "snapshot changed after an event beyond cutoff",
                        )
                    )

        if report.violations:
            report.status = "violated"
        return report

    @staticmethod
    def _pick_probe_students(snapshot: pd.DataFrame, n: int = 4) -> list[dict]:
        if snapshot.empty:
            return []
        columns = list(KEYS)
        ranked = snapshot.sort_values(
            ["code_presentation", "n_clicks_before_cutoff", "code_module", "id_student"],
            ascending=[True, False, True, True],
        )
        picked = ranked.groupby("code_presentation", sort=True).head(1).head(n)
        return picked[columns].to_dict("records")

    @staticmethod
    def _future_probes(
        tables: dict[str, pd.DataFrame], key: dict, cutoff_day: int
    ) -> Iterator[tuple[str, dict[str, pd.DataFrame]]]:
        module, presentation, student = (
            key["code_module"], key["code_presentation"], key["id_student"]
        )
        future_day = cutoff_day + 30

        def clone() -> dict[str, pd.DataFrame]:
            return {name: frame.copy() for name, frame in tables.items()}

        injected = clone()
        injected["studentVle"] = pd.concat([
            injected["studentVle"],
            pd.DataFrame({
                "code_module": [module], "code_presentation": [presentation],
                "id_student": [student], "id_site": [INJECTED_ID_OFFSET],
                "date": [future_day], "sum_click": [MAX_VALUE],
            }),
        ], ignore_index=True)
        yield "vle_future_click", injected

        injected = clone()
        injected["assessments"] = pd.concat([
            injected["assessments"],
            pd.DataFrame({
                "id_assessment": [INJECTED_ID_OFFSET + 1], "code_module": [module],
                "code_presentation": [presentation], "assessment_type": ["TMA"],
                "date": [float(future_day)], "weight": [50.0],
            }),
        ], ignore_index=True)
        injected["studentAssessment"] = pd.concat([
            injected["studentAssessment"],
            pd.DataFrame({
                "id_assessment": [INJECTED_ID_OFFSET + 1], "id_student": [student],
                "date_submitted": [10], "is_banked": [0], "score": [100.0],
            }),
        ], ignore_index=True)
        yield "future_assessment_date", injected

        # Always create an eligible assessment in the probe student's offering;
        # using an arbitrary existing assessment could make this probe ineffective.
        injected = clone()
        injected["assessments"] = pd.concat([
            injected["assessments"],
            pd.DataFrame({
                "id_assessment": [INJECTED_ID_OFFSET + 2], "code_module": [module],
                "code_presentation": [presentation], "assessment_type": ["TMA"],
                "date": [float(cutoff_day - 7)], "weight": [50.0],
            }),
        ], ignore_index=True)
        injected["studentAssessment"] = pd.concat([
            injected["studentAssessment"],
            pd.DataFrame({
                "id_assessment": [INJECTED_ID_OFFSET + 2], "id_student": [student],
                "date_submitted": [future_day], "is_banked": [0], "score": [100.0],
            }),
        ], ignore_index=True)
        yield "future_submission", injected

        injected = clone()
        injected["assessments"] = pd.concat([
            injected["assessments"],
            pd.DataFrame({
                "id_assessment": [INJECTED_ID_OFFSET + 3], "code_module": [module],
                "code_presentation": [presentation], "assessment_type": ["Exam"],
                "date": [float("nan")], "weight": [100.0],
            }),
        ], ignore_index=True)
        injected["studentAssessment"] = pd.concat([
            injected["studentAssessment"],
            pd.DataFrame({
                "id_assessment": [INJECTED_ID_OFFSET + 3], "id_student": [student],
                "date_submitted": [10], "is_banked": [0], "score": [100.0],
            }),
        ], ignore_index=True)
        yield "dateless_exam", injected


def audit_snapshot(snapshot_path: Path, processed_dir: Path,
                   cutoff_day: int) -> LeakageAuditReport:
    return LeakageAuditor().audit_snapshot(snapshot_path, processed_dir, cutoff_day)
