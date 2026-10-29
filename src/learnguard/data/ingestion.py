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
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO, Mapping

import pandas as pd

from learnguard.data.schema import OULAD_TABLE_SCHEMAS, TableSchema
from learnguard.data.snapshot_contract import HASH_ALGORITHM

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
    # studentVle có 787,170 dòng exact-duplicate.)
    deduped_row_counts: dict[str, int] = field(default_factory=dict)
    canonical_content_hashes: dict[str, str] = field(default_factory=dict)
    hash_algorithm: str = HASH_ALGORITHM
    canonical_published: bool = False
    quarantine: list[QuarantineRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return _sha256_stream(fh)


def _sha256_stream(fh: BinaryIO) -> str:
    digest = hashlib.sha256()
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
    ("studentInfo", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("assessments", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("vle", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("studentRegistration", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("studentVle", ("code_module", "code_presentation"), "courses", ("code_module", "code_presentation")),
    ("studentRegistration", ("code_module", "code_presentation", "id_student"),
     "studentInfo", ("code_module", "code_presentation", "id_student")),
    ("studentVle", ("code_module", "code_presentation", "id_student"),
     "studentInfo", ("code_module", "code_presentation", "id_student")),
    ("studentAssessment", ("id_assessment",), "assessments", ("id_assessment",)),
    ("studentVle", ("code_module", "code_presentation", "id_site"),
     "vle", ("code_module", "code_presentation", "id_site")),
)


class OuladIngestionService:
    """Coordinates raw table loading and canonical output creation."""

    def ingest(self, data_dir: Path, output_dir: Path) -> IngestionReport:
        """Load seven tables from a directory or ZIP, validate, and write outputs."""
        data_dir, output_dir = Path(data_dir), Path(output_dir)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"ingestion_output_not_empty:{output_dir}")
        canonical_dir = output_dir / "canonical"
        staging_dir = output_dir / "canonical.staging"
        quarantine_dir = output_dir / "quarantine"
        output_dir.mkdir(parents=True, exist_ok=True)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        report = IngestionReport(status="ok")
        tables: dict[str, pd.DataFrame] = {}

        zip_file: zipfile.ZipFile | None = None
        zip_members: dict[str, zipfile.ZipInfo] = {}
        if data_dir.is_file() and data_dir.suffix.lower() == ".zip":
            try:
                zip_file = zipfile.ZipFile(data_dir)
                csv_infos = [i for i in zip_file.infolist() if not i.is_dir() and i.filename.endswith(".csv")]
                duplicates = sorted({i.filename for i in csv_infos if sum(j.filename == i.filename for j in csv_infos) > 1})
                expected = {f"{name}.csv" for name in OULAD_TABLE_SCHEMAS}
                actual = {i.filename for i in csv_infos}
                missing, unexpected = sorted(expected - actual), sorted(actual - expected)
                if missing or unexpected or duplicates:
                    report.status = "failed"
                    report.errors.append(
                        "zip_members_invalid:"
                        f"missing={','.join(missing) or '-'}:"
                        f"unexpected={','.join(unexpected) or '-'}:"
                        f"duplicates={','.join(duplicates) or '-'}"
                    )
                zip_members = {i.filename: i for i in csv_infos}
            except zipfile.BadZipFile:
                report.status = "failed"
                report.errors.append("invalid_zip_archive")
        elif not data_dir.is_dir():
            report.status = "failed"
            report.errors.append(f"source_not_found:{data_dir}")

        # --- Pha 1: nạp + kiểm contract từng bảng ---------------------------
        for name, schema in OULAD_TABLE_SCHEMAS.items():
            filename = f"{name}.csv"
            try:
                if zip_file is not None:
                    if filename not in zip_members:
                        continue
                    with zip_file.open(zip_members[filename]) as fh:
                        report.source_hashes[name] = _sha256_stream(fh)
                    with zip_file.open(zip_members[filename]) as fh:
                        df = pd.read_csv(fh)
                else:
                    src = data_dir / filename
                    if not src.is_file():
                        report.status = "failed"
                        report.errors.append(f"missing_source_file:{name}")
                        continue
                    report.source_hashes[name] = _sha256_file(src)
                    df = pd.read_csv(src)
            except (OSError, RuntimeError, UnicodeDecodeError, pd.errors.ParserError,
                    zipfile.BadZipFile) as exc:
                report.status = "failed"
                report.errors.append(f"source_read_error:{name}:{type(exc).__name__}")
                continue
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
            report.warnings.extend(msgs)
            report.quarantined_row_counts[name] = len(quarantined)

            # Exact-duplicate (y hệt toàn cột) là artifact xuất trùng — loại ở M1,
            # giữ dòng đầu; phần trùng key nhưng sum_click khác là sự kiện riêng,
            # được giữ nguyên để M2 SUM (EDA #5 bản sửa 2026-10-26).
            if not clean.empty:
                exact_dup = clean.duplicated(keep="first")
                if exact_dup.any():
                    n = int(exact_dup.sum())
                    report.deduped_row_counts[name] = n
                    report.warnings.append(f"exact_duplicate_rows:{name}:{n}")
                    clean = clean[~exact_dup]

            tables[name] = clean  # chỉ bảng HỢP LỆ mới tham gia pipeline + output
            if not quarantined.empty:
                qpath = quarantine_dir / f"{name}.csv"
                quarantined.to_csv(qpath, index=False)
                report.quarantine.append(
                    QuarantineRecord(name, "row_contract_violation", len(quarantined), qpath)
                )
            tables[name] = clean

        if zip_file is not None:
            zip_file.close()

        # --- Pha 2: referential integrity trên toàn bộ bảng -----------------
        orphan_rules = self._orphan_masks(tables)
        for child, rules in orphan_rules.items():
            combined = pd.Series(False, index=tables[child].index)
            reasons = pd.Series("", index=tables[child].index, dtype="object")
            for reason, mask in rules:
                combined |= mask
                reasons.loc[mask] = reasons.loc[mask].map(
                    lambda current: f"{current};{reason}" if current else reason
                )
            if not combined.any():
                continue
            orphans = tables[child][combined].copy()
            orphans[_QUARANTINE_REASON_COL] = reasons.loc[combined]
            qpath = quarantine_dir / f"{child}__orphan.csv"
            orphans.to_csv(qpath, index=False)
            prev = report.quarantined_row_counts.get(child, 0)
            report.quarantined_row_counts[child] = prev + int(combined.sum())
            record_reason = ";".join(reason for reason, mask in rules if mask.any())
            report.quarantine.append(
                QuarantineRecord(child, record_reason, int(combined.sum()), qpath)
            )
            report.warnings.append(f"orphan_rows:{child}:{int(combined.sum())}")
            tables[child] = tables[child][~combined]

        # --- Pha 3: ghi Parquet + report ------------------------------------
        if report.status == "ok":
            staging_dir.mkdir(parents=True, exist_ok=False)
            for name, df in tables.items():
                out = staging_dir / f"{name}.parquet"
                df.to_parquet(out, index=False)
                report.row_counts[name] = len(df)
                report.canonical_content_hashes[name] = _dataframe_content_hash(df)
            staging_dir.rename(canonical_dir)
            report.canonical_published = True
            (output_dir / "ingestion.success.json").write_text(json.dumps({
                "status": "ok",
                "hash_algorithm": HASH_ALGORITHM,
                "canonical_content_hashes": report.canonical_content_hashes,
            }, indent=2))
        elif staging_dir.exists():
            shutil.rmtree(staging_dir)

        report_path = output_dir / "ingestion_report.json"
        report_path.write_text(json.dumps(asdict(report), indent=2, default=str))

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
        for child, rules in self._orphan_masks(tables).items():
            for reason, mask in rules:
                if mask.any():
                    records.append(QuarantineRecord(child, reason, int(mask.sum())))
        return records

    # ---- nội bộ -----------------------------------------------------------

    def _orphan_masks(
        self, tables: Mapping[str, pd.DataFrame]
    ) -> dict[str, list[tuple[str, pd.Series]]]:
        result: dict[str, list[tuple[str, pd.Series]]] = {}
        for child, child_keys, parent, parent_keys in FK_RULES:
            if child not in tables or parent not in tables:
                continue
            mask = self._orphan_mask(tables[child], child_keys, tables[parent], parent_keys)
            reason = f"orphan_fk:{child}->{parent}"
            result.setdefault(child, []).append((reason, mask))

        # studentAssessment không chứa module/presentation. Suy ra offering qua
        # assessment rồi kiểm enrollment composite trong studentInfo; kiểm chỉ
        # theo id_student sẽ bỏ lọt SV thuộc một offering khác.
        needed = {"studentAssessment", "assessments", "studentInfo"}
        if needed <= set(tables):
            sa = tables["studentAssessment"]
            asm = tables["assessments"]
            info = tables["studentInfo"]
            enriched = sa.reset_index(names="_source_index").merge(
                asm[["id_assessment", "code_module", "code_presentation"]],
                on="id_assessment", how="left",
            )
            enrollment = info[["code_module", "code_presentation", "id_student"]].drop_duplicates()
            checked = enriched.merge(
                enrollment,
                on=["code_module", "code_presentation", "id_student"],
                how="left", indicator=True,
            )
            mask = checked["_merge"].eq("left_only")
            mask = pd.Series(mask.to_numpy(), index=sa.index)
            reason = "orphan_fk:studentAssessment->studentInfo_offering"
            result.setdefault("studentAssessment", []).append((reason, mask))
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


def _dataframe_content_hash(frame: pd.DataFrame) -> str:
    """Semantic hash for reproducibility, independent from Parquet metadata."""
    schema = "\n".join(f"{col}:{frame[col].dtype}" for col in frame.columns)
    values = pd.util.hash_pandas_object(frame.reset_index(drop=True), index=False,
                                        categorize=True).to_numpy()
    digest = hashlib.sha256(schema.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()
