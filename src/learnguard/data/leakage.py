"""M3 — Leakage audit: chứng minh snapshot KHÔNG chứa thông tin sau cutoff.

Chiến lược 3 lớp (mỗi lớp bắt một kiểu lỗi khác nhau):
  1. `audit_catalog`      — từ chối rule lạ / cột target-identity trong catalog
  2. Recompute check      — tính lại feature từ dữ liệu nguồn, đối chiếu từng dòng
                            với snapshot trên đĩa (bắt snapshot được sửa tay / lệch nguồn)
  3. Injection probes     — cắm sự kiện tương lai (date > cutoff, Exam không ngày,
                            nộp-sớm/thi-muộn) vào nguồn rồi tính lại: kết quả PHẢI
                            bất biến. Bất biến = không kênh nào cho tương lai lọt vào.
"""

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import pandas as pd

from learnguard.data.features import (
    FEATURE_CATALOG,
    PointInTimeFeatureBuilder,
    _combined_source_hash,
)

INJECTED_ID_OFFSET = 999_000  # id cấy chủ đích — không đụng id thật
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
    violations: list[LeakageViolation] = field(default_factory=list)


class LeakageAuditor:
    def audit_catalog(self) -> LeakageAuditReport:
        """Từ chối rule lạ và cột target/identity trong catalog."""
        report = LeakageAuditReport(
            status="ok", cutoff_day=0, checked_features=len(FEATURE_CATALOG)
        )
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
            # Ràng buộc cutoff của feature không-tĩnh được đảm bảo bởi ALLOWED_AS_OF_RULES;
            # việc aggregation có thực sự tôn trọng rule được kiểm bằng injection probe (lớp 2).
        if report.violations:
            report.status = "violated"
        return report

    def audit_snapshot(self, snapshot_path: Path, processed_dir: Path,
                       cutoff_day: int) -> LeakageAuditReport:
        """Audit snapshot trên đĩa: recompute + injection probes + ledger hash."""
        snapshot_path, processed_dir = Path(snapshot_path), Path(processed_dir)
        report = LeakageAuditReport(status="ok", cutoff_day=cutoff_day,
                                    checked_features=len(FEATURE_CATALOG))
        snapshot = pd.read_parquet(snapshot_path)
        canonical = processed_dir / "canonical"
        parquet_dir = canonical if canonical.is_dir() else processed_dir

        tables: dict[str, pd.DataFrame] = {
            name: pd.read_parquet(parquet_dir / f"{name}.parquet")
            for name in ("courses", "assessments", "studentInfo",
                         "studentRegistration", "studentAssessment", "studentVle")
        }

        # --- Lớp 0: ledger source_hash phải khớp dữ liệu nguồn hiện tại --------
        ledger_path = snapshot_path.parent / (snapshot_path.stem + ".ledger.json")
        if ledger_path.is_file():
            ledger = json.loads(ledger_path.read_text())
            actual = _combined_source_hash(
                {name: _sha256_file(parquet_dir / f"{name}.parquet") for name in tables}
            )
            if ledger.get("source_hash") != actual:
                report.violations.append(
                    LeakageViolation("ledger", "source_hash_mismatch",
                                     "snapshot sinh từ dữ liệu nguồn khác processed_dir")
                )
        else:
            report.violations.append(
                LeakageViolation("ledger", "missing_ledger", str(ledger_path))
            )

        # --- Lớp 1: snapshot phải khớp recompute từ nguồn (bắt sửa tay) --------
        # Feature đếm/trend có ngữ nghĩa "không tương tác = 0" (M2 fillna) — áp cùng
        # ngữ nghĩa trước khi so sánh, nếu không SV 0-click bị báo nhầm là lệch.
        count_like = {"n_clicks_before_cutoff", "click_trend_w3_w4", "n_submitted_before_cutoff"}
        clean_features = self._compute_all(tables, cutoff_day)
        for name, feature_df in clean_features.items():
            merged = snapshot.merge(feature_df, on=self._merge_keys(name), how="left",
                                    suffixes=("", "_recomputed"))
            recomputed_col = f"{name}_recomputed"
            if recomputed_col not in merged.columns:
                report.violations.append(
                    LeakageViolation(name, "feature_column_missing", "snapshot thiếu cột catalog")
                )
                continue
            if name in count_like:
                merged[name] = merged[name].fillna(0)
                merged[recomputed_col] = merged[recomputed_col].fillna(0)
            both = merged[merged[name].notna() | merged[recomputed_col].notna()]
            mismatch = ~_values_equal(both[name], both[recomputed_col])
            if mismatch.any():
                report.violations.append(
                    LeakageViolation(name, "snapshot_recompute_mismatch",
                                     f"{int(mismatch.sum())} dòng lệch so với recompute")
                )

        # --- Lớp 2: injection probes — tương lai phải vô hình ------------------
        source_checks = 0
        for key in self._pick_probe_students(snapshot):
            for probe_name, injected_tables in self._future_probes(tables, key, cutoff_day):
                source_checks += 1
                after = self._compute_all(injected_tables, cutoff_day)
                for fname in clean_features:
                    if not _frames_equal(clean_features[fname], after[fname]):
                        report.violations.append(
                            LeakageViolation(
                                fname, f"future_event_leak:{probe_name}",
                                "feature thay đổi khi cắm sự kiện sau cutoff",
                            )
                        )

        report.source_checks = source_checks
        if report.violations:
            report.status = "violated"
        return report

    # ---- nội bộ -------------------------------------------------------------

    def _compute_all(
        self, tables: dict[str, pd.DataFrame], cutoff_day: int
    ) -> dict[str, pd.DataFrame]:
        builder = PointInTimeFeatureBuilder()
        return {f.name: builder.build_feature(f, tables, cutoff_day) for f in FEATURE_CATALOG}

    @staticmethod
    def _merge_keys(feature_name: str) -> list[str]:
        # module_presentation là feature khóa-học (2 cột), còn lại per-student (3 cột)
        if feature_name == "module_presentation":
            return ["code_module", "code_presentation"]
        return ["code_module", "code_presentation", "id_student"]

    @staticmethod
    def _pick_probe_students(snapshot: pd.DataFrame, n: int = 3) -> list[dict]:
        """Chọn SV có nhiều click (nhạy với leakage) + SV đầu danh sách."""
        cols = ["code_module", "code_presentation", "id_student"]
        if snapshot.empty:
            return []
        picked = snapshot.sort_values("n_clicks_before_cutoff", ascending=False).head(n)[cols]
        picked = pd.concat([picked, snapshot[cols].head(1)]).drop_duplicates()
        return picked.to_dict("records")

    @staticmethod
    def _future_probes(
        tables: dict[str, pd.DataFrame], key: dict, cutoff_day: int
    ) -> list[tuple[str, dict[str, pd.DataFrame]]]:
        """Cắm 4 loại sự kiện 'tương lai/mô hồ' — tất cả PHẢI vô hình với feature."""
        module, pres, student = key["code_module"], key["code_presentation"], key["id_student"]
        future_day = cutoff_day + 30
        vle, asm, sa = tables["studentVle"], tables["assessments"], tables["studentAssessment"]

        def clone() -> dict[str, pd.DataFrame]:
            return {name: df.copy() for name, df in tables.items()}

        # P1: click khổng lồ SAU cutoff
        p1 = clone()
        p1["studentVle"] = pd.concat([vle, pd.DataFrame({
            "code_module": [module], "code_presentation": [pres], "id_student": [student],
            "id_site": [INJECTED_ID_OFFSET], "date": [future_day], "sum_click": [MAX_VALUE],
        })], ignore_index=True)

        # P2: bài đánh giá diễn ra SAU cutoff nhưng nộp TRƯỚC — phải bị loại
        p2 = clone()
        p2["assessments"] = pd.concat([asm, pd.DataFrame({
            "id_assessment": [INJECTED_ID_OFFSET + 1], "code_module": [module],
            "code_presentation": [pres], "assessment_type": ["TMA"],
            "date": [float(future_day)], "weight": [50.0],
        })], ignore_index=True)
        p2["studentAssessment"] = pd.concat([sa, pd.DataFrame({
            "id_assessment": [INJECTED_ID_OFFSET + 1], "id_student": [student],
            "date_submitted": [10], "is_banked": [0], "score": [100.0],
        })], ignore_index=True)

        # P3: nộp SAU cutoff (bài diễn ra trước) — phải bị loại
        p3 = clone()
        p3["studentAssessment"] = pd.concat([sa, pd.DataFrame({
            "id_assessment": [int(asm["id_assessment"].iloc[0])], "id_student": [student],
            "date_submitted": [future_day], "is_banked": [0], "score": [100.0],
        })], ignore_index=True)

        # P4: Exam KHÔNG NGÀY + có điểm + nộp trước cutoff — không đủ căn cứ, phải loại
        p4 = clone()
        p4["assessments"] = pd.concat([asm, pd.DataFrame({
            "id_assessment": [INJECTED_ID_OFFSET + 2], "code_module": [module],
            "code_presentation": [pres], "assessment_type": ["Exam"],
            "date": [float("nan")], "weight": [100.0],
        })], ignore_index=True)
        p4["studentAssessment"] = pd.concat([sa, pd.DataFrame({
            "id_assessment": [INJECTED_ID_OFFSET + 2], "id_student": [student],
            "date_submitted": [10], "is_banked": [0], "score": [100.0],
        })], ignore_index=True)

        return [
            ("vle_future_click", p1),
            ("future_assessment_date", p2),
            ("future_submission", p3),
            ("dateless_exam", p4),
        ]


def _values_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    return (left.fillna(-1) - right.fillna(-1)).abs().le(1e-6)


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Đẳng thức multiset: cùng cột, cùng số dòng, mỗi dòng của a có mặt trong b."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    if a.empty:
        return True
    joined = a.merge(b, on=list(a.columns), how="outer", indicator=True)
    if (joined["_merge"] != "both").any():
        return False
    return len(joined) == len(a)  # không sinh dòng lai từ cặp không khớp


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_snapshot(snapshot_path: Path, processed_dir: Path,
                   cutoff_day: int) -> LeakageAuditReport:
    """M3 audit entry point."""
    return LeakageAuditor().audit_snapshot(snapshot_path, processed_dir, cutoff_day)
