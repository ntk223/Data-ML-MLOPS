"""Test M2 — point-in-time feature builder: cutoff boundary, leakage, cohort, ledger."""

import json

import pandas as pd
import pytest

from learnguard.data.features import (
    FEATURE_CATALOG,
    PointInTimeFeatureBuilder,
    build_feature_snapshot,
)

KEYS = ["code_module", "code_presentation", "id_student"]


def base_tables() -> dict[str, pd.DataFrame]:
    """Bộ bảng nhỏ sạch (tương thích M1 canonical), 2 SV × 1 khóa."""
    return {
        "courses": pd.DataFrame(
            {"code_module": ["AAA"], "code_presentation": ["2013B"], "module_presentation_length": [240]}
        ),
        "assessments": pd.DataFrame(
            {
                "id_assessment": [1, 2, 3],
                "code_module": ["AAA"] * 3,
                "code_presentation": ["2013B"] * 3,
                "assessment_type": ["TMA", "TMA", "Exam"],
                "date": [20.0, 40.0, float("nan")],  # TMA d20 OK; TMA d40 sau cutoff; Exam không ngày
                "weight": [50.0, 50.0, 100.0],
            }
        ),
        "vle": pd.DataFrame(
            {"id_site": [10], "code_module": ["AAA"], "code_presentation": ["2013B"], "activity_type": ["resource"], "week_from": [1.0], "week_to": [30.0]}
        ),
        "studentInfo": pd.DataFrame(
            {
                "code_module": ["AAA", "AAA", "AAA"],
                "code_presentation": ["2013B"] * 3,
                "id_student": [101, 102, 103],
                "gender": ["M", "F", "M"],
                "region": ["Scotland", "London Region", "Wales"],
                "highest_education": ["A Level or Equivalent"] * 3,
                "imd_band": ["20-30%", "40-50%", None],  # SV 103 missing → cờ
                "age_band": ["0-35", "35-55", "0-35"],
                "num_of_prev_attempts": [0, 1, 0],
                "studied_credits": [60, 90, 60],
                "disability": ["N", "N", "Y"],
                "final_result": ["Pass", "Fail", "Withdrawn"],
            }
        ),
        "studentRegistration": pd.DataFrame(
            {
                "code_module": ["AAA"] * 3,
                "code_presentation": ["2013B"] * 3,
                "id_student": [101, 102, 103],
                "date_registration": [-5.0, 3.0, 1.0],
                "date_unregistration": [float("nan"), 30.0, 25.0],  # 102 rút lui d30 (giữ), 103 d25 (loại)
            }
        ),
        "studentAssessment": pd.DataFrame(
            {
                "id_assessment": [1, 2, 3, 1],
                "id_student": [101, 101, 101, 102],
                "date_submitted": [18, 21, 15, 19],
                "is_banked": [0, 0, 0, 0],
                "score": [80.0, 70.0, 60.0, None],  # SV102 nộp TMA d20 nhưng score null
            }
        ),
        "studentVle": pd.DataFrame(
            {
                "code_module": ["AAA"] * 5,
                "code_presentation": ["2013B"] * 5,
                "id_student": [101] * 5,
                "id_site": [10] * 5,
                "date": [-5, 3, 14, 27, 28],  # 27 được tính, 28 KHÔNG
                "sum_click": [10, 20, 30, 40, 999],
            }
        ),
    }


def build_from_tables(tables, cutoff=27, tmp_path=None):
    builder = PointInTimeFeatureBuilder()
    cohort = builder.build_cohort(tables, cutoff)
    snap = cohort.merge(
        tables["studentInfo"][KEYS + ["gender", "region", "highest_education", "imd_band",
                                      "age_band", "num_of_prev_attempts", "studied_credits",
                                      "disability", "final_result"]],
        on=KEYS, how="left",
    )
    snap["imd_band_missing"] = snap["imd_band"].isna()
    for f in FEATURE_CATALOG:
        fdf = builder.build_feature(f, tables, cutoff)
        on = ["code_module", "code_presentation"] if f.name == "module_presentation" else KEYS
        snap = snap.merge(fdf, on=on, how="left")
    snap = builder.assign_split(snap)
    for col in ("n_clicks_before_cutoff", "click_trend_w3_w4", "n_submitted_before_cutoff"):
        snap[col] = snap[col].fillna(0)
    snap["insufficient_data"] = (snap["n_clicks_before_cutoff"] == 0) & (snap["n_submitted_before_cutoff"] == 0)
    return snap, builder


# ---- cohort ---------------------------------------------------------------

def test_cohort_excludes_withdrawn_before_cutoff_keeps_later():
    snap, _ = build_from_tables(base_tables())
    kept = set(snap["id_student"])
    assert 101 in kept  # chưa rút lui
    assert 102 in kept  # rút lui d30 — SAU cutoff 27 → vẫn dự đoán được
    assert 103 not in kept  # rút lui d25 — TRƯỚC cutoff → loại (leakage)


def test_cohort_counts_exclusions():
    builder = PointInTimeFeatureBuilder()
    builder.build_cohort(base_tables(), 27)
    assert builder._cohort_exclusions["withdrawn_by_cutoff"] == 1


# ---- cutoff boundary --------------------------------------------------------

def test_click_cutoff_boundary_inclusive():
    # SV101: clicks d(-5,3,14,27,28) = 10+20+30+40=100; d28=999 phải bị loại
    snap, _ = build_from_tables(base_tables())
    row = snap[snap["id_student"] == 101].iloc[0]
    assert row["n_clicks_before_cutoff"] == 100


def test_prestart_clicks_included():
    # Quyết định EDA #6: date=-5 (10 clicks) PHẢI được tính
    snap, _ = build_from_tables(base_tables())
    row = snap[snap["id_student"] == 101].iloc[0]
    assert row["n_clicks_before_cutoff"] >= 10


def test_future_event_injection_does_not_change_snapshot():
    """Test leakage cốt lõi: cắm sự kiện tương lai (date=60) → snapshot KHÔNG đổi."""
    tables = base_tables()
    snap_before, _ = build_from_tables(tables)

    injected = tables.copy()
    extra = pd.DataFrame(
        {
            "code_module": ["AAA"], "code_presentation": ["2013B"], "id_student": [101],
            "id_site": [10], "date": [60], "sum_click": [5000],
        }
    )
    injected["studentVle"] = pd.concat([injected["studentVle"], extra], ignore_index=True)
    snap_after, _ = build_from_tables(injected)

    feature_cols = ["n_clicks_before_cutoff", "click_trend_w3_w4",
                    "n_submitted_before_cutoff", "avg_score_submitted"]
    pd.testing.assert_frame_equal(
        snap_before[KEYS + feature_cols], snap_after[KEYS + feature_cols]
    )


def test_click_trend_w3_w4():
    # SV101: w3-w4 (d14..27) = 30+40=70; w1-w2 (d0..13) = 20; trend = 50
    snap, _ = build_from_tables(base_tables())
    row = snap[snap["id_student"] == 101].iloc[0]
    assert row["click_trend_w3_w4"] == 50


# ---- submissions ------------------------------------------------------------

def test_exam_without_date_excluded_and_future_assessment_excluded():
    # SV101 nộp: TMA1 (d20 nộp d18 → eligible, score 80), TMA2 (d40 → sau cutoff, loại),
    # Exam (không ngày → loại). → n_submitted = 1, avg = 80.0
    snap, _ = build_from_tables(base_tables())
    row = snap[snap["id_student"] == 101].iloc[0]
    assert row["n_submitted_before_cutoff"] == 1
    assert row["avg_score_submitted"] == 80.0


def test_null_score_not_counted_in_avg_but_in_count():
    # SV102 nộp TMA1 d19 (eligible) nhưng score null → count=1, avg=NaN
    snap, _ = build_from_tables(base_tables())
    row = snap[snap["id_student"] == 102].iloc[0]
    assert row["n_submitted_before_cutoff"] == 1
    assert pd.isna(row["avg_score_submitted"])


def test_zero_activity_sv_gets_zeros_and_insufficient_flag():
    # SV103 không có click/nộp bài nào — feature = 0, cờ insufficient_data = True
    tables = base_tables()
    # giữ SV103 trong cohort bằng cách đổi ngày rút lui ra sau cutoff
    tables["studentRegistration"].loc[2, "date_unregistration"] = 40.0
    snap, _ = build_from_tables(tables)
    row = snap[snap["id_student"] == 103].iloc[0]
    assert row["n_clicks_before_cutoff"] == 0
    assert bool(row["insufficient_data"]) is True
    assert bool(snap[snap["id_student"] == 101].iloc[0]["insufficient_data"]) is False


def test_imd_band_missing_flag():
    tables = base_tables()
    # SV103 rút lui d25 sẽ bị cohort loại — đưa ngày rút lui ra sau cutoff để test cờ
    tables["studentRegistration"].loc[2, "date_unregistration"] = 40.0
    snap, _ = build_from_tables(tables)
    assert bool(snap[snap["id_student"] == 103].iloc[0]["imd_band_missing"]) is True
    assert bool(snap[snap["id_student"] == 101].iloc[0]["imd_band_missing"]) is False


# ---- temporal split ---------------------------------------------------------

def test_split_assignment_follows_presentation_year():
    """Quyết định M4: chia theo KỲ HỌC, không random."""
    tables = base_tables()
    # Thêm SV ở 2013J (validation) và 2014J (holdout)
    extra_si = pd.DataFrame({
        "code_module": ["AAA", "AAA"], "code_presentation": ["2013J", "2014J"],
        "id_student": [201, 202], "gender": ["F", "M"], "region": ["Wales", "Wales"],
        "highest_education": ["HE Qualification"] * 2, "imd_band": ["10-20", "10-20"],
        "age_band": ["0-35"] * 2, "num_of_prev_attempts": [0, 0],
        "studied_credits": [60, 60], "disability": ["N", "N"],
        "final_result": ["Pass", "Fail"],
    })
    extra_sr = pd.DataFrame({
        "code_module": ["AAA", "AAA"], "code_presentation": ["2013J", "2014J"],
        "id_student": [201, 202], "date_registration": [-5.0, -5.0],
        "date_unregistration": [float("nan")] * 2,
    })
    tables["studentInfo"] = pd.concat([tables["studentInfo"], extra_si], ignore_index=True)
    tables["studentRegistration"] = pd.concat([tables["studentRegistration"], extra_sr], ignore_index=True)
    tables["courses"] = pd.concat([tables["courses"], pd.DataFrame(
        {"code_module": ["AAA", "AAA"], "code_presentation": ["2013J", "2014J"],
         "module_presentation_length": [240, 240]})], ignore_index=True)

    snap, _ = build_from_tables(tables)
    mapping = snap.set_index("id_student")["split"].to_dict()
    assert mapping[101] == "train"        # 2013B
    assert mapping[201] == "validation"   # 2013J
    assert mapping[202] == "holdout"      # 2014J


def test_split_unknown_presentation_fails_loudly():
    """Kỳ không có rule → fail lớn, KHÔNG gán im lặng."""
    tables = base_tables()
    tables["studentRegistration"] = pd.concat([tables["studentRegistration"], pd.DataFrame(
        {"code_module": ["AAA"], "code_presentation": ["2015B"], "id_student": [999],
         "date_registration": [-5.0], "date_unregistration": [float("nan")]})], ignore_index=True)
    builder = PointInTimeFeatureBuilder()
    cohort = builder.build_cohort(tables, 27)
    with pytest.raises(ValueError, match="unknown_presentation_for_split"):
        builder.assign_split(cohort)


def test_split_rules_hashed_into_version():
    """Đổi split rule → snapshot_version phải đổi (rule là một phần của hợp đồng)."""
    from learnguard.data.features import _feature_list_hash
    import learnguard.data.features as feat_mod
    before = _feature_list_hash()
    old = dict(feat_mod.SPLIT_RULES)
    try:
        feat_mod.SPLIT_RULES["2014B"] = "train"  # type: ignore[index]
        assert _feature_list_hash() != before
    finally:
        feat_mod.SPLIT_RULES.clear()
        feat_mod.SPLIT_RULES.update(old)


# ---- end-to-end + ledger ----------------------------------------------------

def write_canonical(tmp_path, tables):
    canonical = tmp_path / "canonical"
    canonical.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(canonical / f"{name}.parquet", index=False)
    return tmp_path


def test_end_to_end_build_writes_snapshot_and_ledger(tmp_path):
    processed = write_canonical(tmp_path, base_tables())
    result = build_feature_snapshot(processed, tmp_path / "snapshot.parquet", cutoff_day=27)

    assert result.row_count == 2  # SV101 + SV102 (103 bị loại cohort)
    assert result.insufficient_data_count == 0
    assert result.snapshot_path.is_file()
    snap = pd.read_parquet(result.snapshot_path)
    assert "label" in snap.columns and "final_result" in snap.columns
    assert "final_result" not in [f.name for f in FEATURE_CATALOG]  # target không nằm trong catalog
    assert "split" in snap.columns  # temporal split — hợp đồng cho M4
    assert set(snap["split"].unique()) == {"train"}  # fixture chỉ có 2013B

    ledger = json.loads(result.ledger_path.read_text())
    assert ledger["cutoff_day"] == 27
    assert ledger["source_hash"] and len(ledger["source_hash"]) == 64
    assert ledger["split_rules"] == {"2013B": "train", "2013J": "validation",
                                     "2014B": "holdout", "2014J": "holdout"}
    assert set(ledger["input_hashes"]) == {
        "courses", "assessments", "studentInfo", "studentRegistration", "studentAssessment", "studentVle"
    }
    assert "n_clicks_before_cutoff" in ledger["feature_rules"]
    assert result.snapshot_version.startswith("snapshot-v1:cutoff27:")


def test_snapshot_changes_when_input_changes(tmp_path):
    """Cùng rule, khác dữ liệu → source_hash khác (ledger phải nói thật)."""
    tables = base_tables()
    result1 = build_feature_snapshot(write_canonical(tmp_path / "a", tables),
                                     tmp_path / "a.parquet")
    tables["studentVle"] = tables["studentVle"].iloc[:-1]  # bỏ click d28
    result2 = build_feature_snapshot(write_canonical(tmp_path / "b", tables),
                                     tmp_path / "b.parquet")
    assert result1.snapshot_version != result2.snapshot_version


def test_builder_deterministic_for_same_input(tmp_path):
    processed = write_canonical(tmp_path, base_tables())
    r1 = build_feature_snapshot(processed, tmp_path / "s1.parquet")
    r2 = build_feature_snapshot(processed, tmp_path / "s2.parquet")
    assert r1.snapshot_version == r2.snapshot_version
    assert r1.row_count == r2.row_count
