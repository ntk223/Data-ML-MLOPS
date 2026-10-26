"""M1 — OULAD ingestion: validate, quarantine, referential integrity, Parquet output.

Quy trình `ingest()` cho từng bảng:
  1. Đọc CSV + tính SHA-256 của file nguồn (provenance).
  2. Kiểm contract cột (thiếu/thừa cột → lỗi nghiêm trọng, bỏ bảng).
  3. Kiểm contract dòng (dtype cột, null policy, enum, range, PK) —
     dòng vi phạm bị tách vào quarantine kèm lý do, KHÔNG xóa im lặng.
  4. Kiểm referential integrity giữa các bảng — dòng orphan bị quarantine.
  5. Ghi Parquet chuẩn hóa + file quarantine + `ingestion_report.json`.

Mọi rule đều truy vết về EDA: notebooks/01_oulad_eda.ipynb.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from learnguard.data.schema import OULAD_TABLE_SCHEMAS, TableSchema

_QUARANTINE_REASON_COL = "_quarantine_reason"


@dataclass(frozen=True)
class QuarantineRecord:
    source_table: str
    reason: str
    row_count: int
    output_path: Path | None = None


@dataclass
class IngestionReport:
    status: str
    source_hashes: dict[str, str] = field(default_factory=dict)
    # row_counts = số dòng SẠCH ghi vào Parquet; raw/quarantined để đối chiếu
    row_counts: dict[str, int] = field(default_factory=dict)
    raw_row_counts: dict[str, int] = field(default_factory=dict)
    quarantined_row_counts: dict[str, int] = field(default_factory=dict)
    # Bản sao Y HỆT toàn cột (artifact) bị loại ở M1 — không ghi quarantine file
    # vì không mất thông tin; đếm minh bạch tại đây. (Verify trên OULAD thật:
    # studentVle có 1,429,783 dòng exact-duplicate.)
    deduped_row_counts: dict[str, int] = field(default_factory=dict)
    quarantine: list[QuarantineRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype_matches(actual: str, expected: str) -> bool:
    if expected == "str":
        return actual in {"str", "object", "string"}
    if expected == "int64":
        return actual in {"int64", "Int64"}
    if expected == "float64":
        return actual in {"float64", "Float64"}
    return actual == expected


def _row_rule_violations(schema: TableSchema, df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Trả về [(lý do, mask dòng vi phạm)] — đánh giá vector hóa cho 10.6M dòng."""
    rules: list[tuple[str, pd.Series]] = []
    for col, spec in schema.columns.items():
        series = df[col]
        # Null policy (EDA #2, #3): null chỉ hợp lệ khi nullable=True
        if not spec.nullable:
            rules.append((f"null_not_allowed:{col}", series.isna()))
        # Enum (EDA section 2): giá trị ngoài miền → quarantine
        if spec.allowed_values is not None:
            allowed = set(spec.allowed_values)
            rules.append((f"value_out_of_enum:{col}", series.notna() & ~series.isin(allowed)))
        # Range (EDA section 3): chỉ áp cho số, bỏ qua null đã xử lý ở trên
        if spec.min_value is not None:
            rules.append(
                (f"below_min:{col}", series.notna() & (series < spec.min_value))
            )
        if spec.max_value is not None:
            rules.append(
                (f"above_max:{col}", series.notna() & (series > spec.max_value))
            )
    return rules


# Quan hệ FK phải bảo toàn — 9 quan hệ đã xác minh 0 orphan trong EDA (section 5)
FK_RULES: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    ("assessments", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("vle", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("studentRegistration", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("studentVle", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("studentRegistration", ("id_student",), "studentInfo", ("id_student",)),
    ("studentAssessment", ("id_student",), "studentInfo", ("id_student",)),
    ("studentVle", ("id_student",), "studentInfo", ("id_student",)),
    ("studentAssessment", ("id_assessment",), "assessments", ("id_assessment",)),
    ("studentVle", ("id_site",), "vle", ("id_site",)),
)


class OuladIngestionService:
    """Coordinates raw table loading and canonical output creation."""

    def ingest(self, data_dir: Path, output_dir: Path) -> IngestionReport:
        """Load seven tables, validate, quarantine, and write canonical outputs."""
        data_dir, output_dir = Path(data_dir), Path(output_dir)
        canonical_dir = output_dir / "canonical"
        quarantine_dir = output_dir / "quarantine"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        report = IngestionReport(status="ok")
        tables: dict[str, pd.DataFrame] = {}

        # --- Pha 1: nạp + kiểm contract từng bảng ---------------------------
        for name, schema in OULAD_TABLE_SCHEMAS.items():
            src = data_dir / f"{name}.csv"
            if not src.is_file():
                report.status = "failed"
                report.errors.append(f"missing_source_file:{name}")
                continue
            report.source_hashes[name] = _sha256_file(src)
            df = pd.read_csv(src)
            report.raw_row_counts[name] = len(df)

            missing = [c for c in schema.required_columns if c not in df.columns]
            if missing:
                report.status = "failed"
                report.errors.append(f"missing_columns:{name}:{','.join(missing)}")
                continue
            extra = [c for c in df.columns if c not in schema.columns]
            if extra:
                report.status = "failed"
                report.errors.append(f"unexpected_columns:{name}:{','.join(extra)}")
                continue

            dtype_errors = [
                f"dtype_mismatch:{name}:{col}:expected={spec.dtype}:actual={df[col].dtype}"
                for col, spec in schema.columns.items()
                if not _dtype_matches(str(df[col].dtype), spec.dtype)
            ]
            if dtype_errors:
                report.status = "failed"
                report.errors.extend(dtype_errors)
                continue

            clean, quarantined, msgs = self.validate_table(name, df)
            report.errors.extend(msgs)
            report.quarantined_row_counts[name] = len(quarantined)

            # Exact-duplicate (y hệt toàn cột) là artifact xuất trùng — loại ở M1,
            # giữ dòng đầu; phần trùng key nhưng sum_click khác là sự kiện riêng,
            # được giữ nguyên để M2 SUM (EDA #5 bản sửa 2026-10-26).
            if not clean.empty:
                exact_dup = clean.duplicated(keep="first")
                if exact_dup.any():
                    n = int(exact_dup.sum())
                    report.deduped_row_counts[name] = n
                    report.errors.append(f"exact_duplicate_rows:{name}:{n}")
                    clean = clean[~exact_dup]

            tables[name] = clean  # chỉ bảng HỢP LỆ mới tham gia pipeline + output
            if not quarantined.empty:
                qpath = quarantine_dir / f"{name}.csv"
                quarantined.to_csv(qpath, index=False)
                report.quarantine.append(
                    QuarantineRecord(name, "row_contract_violation", len(quarantined), qpath)
                )
            tables[name] = clean

        # --- Pha 2: referential integrity trên toàn bộ bảng -----------------
        orphan_reasons = self._orphan_masks(tables)
        for child, (reason, mask) in orphan_reasons.items():
            if not mask.any():
                continue
            orphans = tables[child][mask].copy()
            orphans[_QUARANTINE_REASON_COL] = reason
            qpath = quarantine_dir / f"{child}__orphan.csv"
            orphans.to_csv(qpath, index=False)
            prev = report.quarantined_row_counts.get(child, 0)
            report.quarantined_row_counts[child] = prev + int(mask.sum())
            report.quarantine.append(
                QuarantineRecord(child, reason, int(mask.sum()), qpath)
            )
            tables[child] = tables[child][~mask]

        # --- Pha 3: ghi Parquet + report ------------------------------------
        for name, df in tables.items():
            out = canonical_dir / f"{name}.parquet"
            df.to_parquet(out, index=False)
            report.row_counts[name] = len(df)

        report_path = output_dir / "ingestion_report.json"
        report_path.write_text(json.dumps(asdict(report), indent=2, default=str))

        if any(e.startswith(("missing_source_file", "missing_columns", "unexpected_columns", "dtype_mismatch")) for e in report.errors):
            report.status = "failed"
        return report

    def validate_table(
        self, table_name: str, rows: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Validate dòng theo contract; trả về (clean, quarantined, messages).

        Dòng vi phạm bị tách ra với cột `_quarantine_reason` — không xóa im lặng.
        """
        schema = OULAD_TABLE_SCHEMAS[table_name]
        messages: list[str] = []
        violations = _row_rule_violations(schema, rows)

        combined = pd.Series(False, index=rows.index)
        for _, mask in violations:
            combined |= mask

        quarantined = rows[combined].copy()
        if combined.any():
            labels: list[str] = []
            for idx in quarantined.index:
                row_reasons = [label for label, mask in violations if bool(mask.at[idx])]
                labels.append(";".join(row_reasons))
            quarantined[_QUARANTINE_REASON_COL] = labels

        # PK trùng: toàn bộ các bản sao bị quarantine (không giữ bản nào tùy tiện).
        # Chỉ áp khi bảng có key nghiêm ngặt — studentVle là event log, không PK (EDA #5).
        if not schema.key_columns:
            if quarantined.empty:
                return rows, quarantined, messages
            clean = rows[~rows.index.isin(quarantined.index)]
            messages.append(f"quarantined_rows:{table_name}:{len(quarantined)}")
            return clean, quarantined, messages

        dup_mask = rows.duplicated(subset=schema.key_columns, keep=False)
        if dup_mask.any():
            dup_rows = rows[dup_mask].copy()
            dup_rows[_QUARANTINE_REASON_COL] = "duplicate_primary_key:" + "|".join(schema.key_columns)
            existing = set(quarantined.index)
            extra = dup_rows[~dup_rows.index.isin(existing)]
            quarantined = pd.concat([quarantined, extra])
            messages.append(f"duplicate_primary_key:{table_name}:{len(dup_rows)}")

        if quarantined.empty:
            return rows, quarantined, messages

        clean = rows[~rows.index.isin(quarantined.index)]
        messages.append(f"quarantined_rows:{table_name}:{len(quarantined)}")
        return clean, quarantined, messages

    def check_referential_integrity(
        self, tables: Mapping[str, pd.DataFrame]
    ) -> list[QuarantineRecord]:
        """Kiểm 9 quan hệ FK; trả về record tổng hợp cho mỗi quan hệ có orphan."""
        records: list[QuarantineRecord] = []
        for child, (reason, mask) in self._orphan_masks(tables).items():
            if mask.any():
                records.append(QuarantineRecord(child, reason, int(mask.sum())))
        return records

    # ---- nội bộ -----------------------------------------------------------

    def _orphan_masks(
        self, tables: Mapping[str, pd.DataFrame]
    ) -> dict[str, tuple[str, pd.Series]]:
        result: dict[str, tuple[str, pd.Series]] = {}
        for child, child_keys, parent, parent_keys in FK_RULES:
            if child not in tables or parent not in tables:
                continue
            mask = self._orphan_mask(tables[child], child_keys, tables[parent], parent_keys)
            reason = f"orphan_fk:{child}->{parent}"
            if child in result:
                prev_reason, prev_mask = result[child]
                result[child] = (f"{prev_reason};{reason}", prev_mask | mask)
            else:
                result[child] = (reason, mask)
        return result

    @staticmethod
    def _orphan_mask(
        child: pd.DataFrame,
        child_keys: tuple[str, ...],
        parent: pd.DataFrame,
        parent_keys: tuple[str, ...],
    ) -> pd.Series:
        # Merge tạo RangeIndex mới → mask phải được gán lại về index gốc của child
        merged = child.reset_index(drop=True).merge(
            parent[list(parent_keys)].drop_duplicates(),
            left_on=list(child_keys),
            right_on=list(parent_keys),
            how="left",
            indicator=True,
        )
        mask = merged["_merge"].eq("left_only")
        return pd.Series(mask.to_numpy(), index=child.index)


def ingest_oulad(data_dir: Path, output_dir: Path) -> IngestionReport:
    """M1 pipeline entry point."""
    return OuladIngestionService().ingest(data_dir, output_dir)
