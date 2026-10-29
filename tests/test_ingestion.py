"""Test ingestion service — quarantine, FK, source hash, Parquet, report."""

import hashlib
import json

import pandas as pd

from learnguard.data.ingestion import OuladIngestionService, ingest_oulad
from tests.conftest import mutate, write_csvs

TABLES = {
    "courses",
    "assessments",
    "vle",
    "studentInfo",
    "studentRegistration",
    "studentAssessment",
    "studentVle",
}


def test_clean_fixture_ingests_without_quarantine(clean_data_dir, tmp_path):
    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.status == "ok", report.errors
    assert report.quarantine == []
    assert report.errors == []
    assert set(report.row_counts) == TABLES
    assert report.row_counts["studentInfo"] == 3
    assert report.row_counts["studentVle"] == 2
    # Parquet được ghi cho đủ 7 bảng
    for name in TABLES:
        assert (tmp_path / "out" / "canonical" / f"{name}.parquet").is_file(), name


def test_source_hash_is_sha256_and_tracks_changes(clean_data_dir, tmp_path):
    service = OuladIngestionService()
    out = tmp_path / "out"
    first = service.ingest(clean_data_dir, out)

    assert set(first.source_hashes) == TABLES
    for h in first.source_hashes.values():
        assert len(h) == 64 and int(h, 16) >= 0  # hex SHA-256

    # Sửa 1 file → hash của ĐÚNG bảng đó đổi, bảng khác giữ nguyên
    frames = pd.read_csv(clean_data_dir / "assessments.csv")
    frames.loc[len(frames)] = [3, "AAA", "2013B", "TMA", 30.0, 50.0]
    frames.to_csv(clean_data_dir / "assessments.csv", index=False)
    second = service.ingest(clean_data_dir, tmp_path / "out2")

    assert second.source_hashes["assessments"] != first.source_hashes["assessments"]
    assert second.source_hashes["courses"] == first.source_hashes["courses"]


def test_invalid_enum_value_quarantined_with_reason(clean_data_dir, tmp_path):
    frames = pd.read_csv(clean_data_dir / "studentInfo.csv")
    frames.loc[0, "gender"] = "X"
    frames.to_csv(clean_data_dir / "studentInfo.csv", index=False)

    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.status == "ok"  # quarantine không phá vỡ run
    assert report.row_counts["studentInfo"] == 2
    assert report.quarantined_row_counts["studentInfo"] == 1
    qfile = tmp_path / "out" / "quarantine" / "studentInfo.csv"
    assert qfile.is_file()
    quarantined = pd.read_csv(qfile)
    assert "value_out_of_enum:gender" in quarantined.loc[0, "_quarantine_reason"]


def test_score_out_of_range_quarantined_but_null_score_kept(clean_data_dir, tmp_path):
    frames = pd.read_csv(clean_data_dir / "studentAssessment.csv")
    frames.loc[0, "score"] = 150.0  # ngoài [0,100]
    frames.loc[1, "score"] = float("nan")  # đã null sẵn — phải ĐƯỢC giữ
    frames.to_csv(clean_data_dir / "studentAssessment.csv", index=False)

    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.status == "ok"
    assert report.quarantined_row_counts["studentAssessment"] == 1
    assert report.row_counts["studentAssessment"] == 1  # dòng NaN score vẫn sạch
    q = pd.read_csv(tmp_path / "out" / "quarantine" / "studentAssessment.csv")
    assert "above_max:score" in q.loc[0, "_quarantine_reason"]


def test_duplicate_primary_key_all_copies_quarantined(clean_data_dir, tmp_path):
    frames = pd.read_csv(clean_data_dir / "studentInfo.csv")
    # Nhân đôi dòng SV 101 kỳ AAA/2013B → cả bản gốc lẫn bản sao phải bị quarantine
    dup = pd.concat([frames, frames.iloc[[0]]], ignore_index=True)
    dup.to_csv(clean_data_dir / "studentInfo.csv", index=False)

    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.quarantined_row_counts["studentInfo"] == 2
    q = pd.read_csv(tmp_path / "out" / "quarantine" / "studentInfo.csv")
    assert (q["_quarantine_reason"].str.startswith("duplicate_primary_key")).all()
    assert report.row_counts["studentInfo"] == 2  # còn SV 102 và 101@BBB


def test_orphan_fk_quarantined_and_removed_from_clean(clean_data_dir, tmp_path):
    frames = pd.read_csv(clean_data_dir / "studentAssessment.csv")
    frames.loc[0, "id_assessment"] = 999  # không tồn tại trong assessments
    frames.to_csv(clean_data_dir / "studentAssessment.csv", index=False)

    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.status == "ok"
    assert report.row_counts["studentAssessment"] == 1
    reasons = [r.reason for r in report.quarantine]
    assert any("orphan_fk:studentAssessment->assessments" in r for r in reasons)
    qfile = tmp_path / "out" / "quarantine" / "studentAssessment__orphan.csv"
    assert qfile.is_file()


def test_referential_integrity_clean_data_has_zero_orphans(base_tables, tmp_path):
    # Đối chiếu với EDA section 5: dữ liệu sạch → 0 orphan trên cả 9 quan hệ
    service = OuladIngestionService()
    assert service.check_referential_integrity(base_tables) == []


def test_exact_duplicates_deduped_but_distinct_events_kept(tmp_path):
    """EDA #5 (bản sửa): bản sao y hệt → M1 dedupe; trùng key khác sum_click → giữ."""
    from tests.conftest import write_csvs

    frames = {
        "courses": pd.DataFrame(
            {"code_module": ["AAA"], "code_presentation": ["2013B"], "module_presentation_length": [240]}
        ),
        "vle": pd.DataFrame(
            {"id_site": [10], "code_module": ["AAA"], "code_presentation": ["2013B"], "activity_type": ["resource"], "week_from": [1.0], "week_to": [30.0]}
        ),
        "studentInfo": pd.DataFrame(
            {
                "code_module": ["AAA"], "code_presentation": ["2013B"], "id_student": [101],
                "gender": ["M"], "region": ["Scotland"],
                "highest_education": ["A Level or Equivalent"], "imd_band": ["20-30%"],
                "age_band": ["0-35"], "num_of_prev_attempts": [0], "studied_credits": [60],
                "disability": ["N"], "final_result": ["Pass"],
            }
        ),
        "studentRegistration": pd.DataFrame(
            {
                "code_module": ["AAA"], "code_presentation": ["2013B"], "id_student": [101],
                "date_registration": [-5.0], "date_unregistration": [float("nan")],
            }
        ),
        "assessments": pd.DataFrame(
            {
                "id_assessment": [1], "code_module": ["AAA"], "code_presentation": ["2013B"],
                "assessment_type": ["TMA"], "date": [20.0], "weight": [50.0],
            }
        ),
        "studentAssessment": pd.DataFrame(
            {
                "id_assessment": [1], "id_student": [101], "date_submitted": [15],
                "is_banked": [0], "score": [80.0],
            }
        ),
        "studentVle": pd.DataFrame(
            {
                "code_module": ["AAA"] * 4,
                "code_presentation": ["2013B"] * 4,
                "id_student": [101] * 4,
                "id_site": [10, 10, 10, 10],
                "date": [3, 3, 3, 5],
                "sum_click": [10, 10, 25, 7],  # 2 dòng đầu là exact-dup; dòng 3 là sự kiện riêng
            }
        ),
    }
    data_dir = write_csvs(tmp_path / "src", frames)
    report = ingest_oulad(data_dir, tmp_path / "out")

    assert report.status == "ok", report.errors
    assert report.deduped_row_counts["studentVle"] == 1  # bỏ 1 bản sao y hệt
    assert report.row_counts["studentVle"] == 3  # sự kiện date=3 sum=25 vẫn được giữ
    assert report.quarantined_row_counts["studentVle"] == 0


def test_missing_column_is_fatal(clean_data_dir, tmp_path):
    frames = pd.read_csv(clean_data_dir / "assessments.csv").drop(columns=["weight"])
    frames.to_csv(clean_data_dir / "assessments.csv", index=False)

    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.status == "failed"
    assert any(e.startswith("missing_columns:assessments") for e in report.errors)
    assert not (tmp_path / "out" / "canonical" / "assessments.parquet").exists()
    # Bảng khác vẫn xử lý bình thường
    assert (tmp_path / "out" / "canonical" / "courses.parquet").exists()


def test_missing_source_file_is_fatal(clean_data_dir, tmp_path):
    (clean_data_dir / "courses.csv").unlink()

    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.status == "failed"
    assert "missing_source_file:courses" in report.errors


def test_report_json_written_and_parseable(clean_data_dir, tmp_path):
    ingest_oulad(clean_data_dir, tmp_path / "out")

    report_path = tmp_path / "out" / "ingestion_report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text())
    assert payload["status"] == "ok"
    assert set(payload["source_hashes"]) == TABLES
    assert payload["row_counts"]["studentInfo"] == 3


def test_unexpected_extra_column_is_fatal(clean_data_dir, tmp_path):
    frames = pd.read_csv(clean_data_dir / "courses.csv")
    frames["suspicious_extra"] = 1
    frames.to_csv(clean_data_dir / "courses.csv", index=False)

    report = ingest_oulad(clean_data_dir, tmp_path / "out")

    assert report.status == "failed"
    assert any(e.startswith("unexpected_columns:courses") for e in report.errors)


def test_hash_helper_matches_hashlib(clean_data_dir):
    from learnguard.data.ingestion import _sha256_file

    path = clean_data_dir / "courses.csv"
    assert _sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
