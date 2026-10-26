"""Test schema contracts — mỗi test bám một quyết định EDA (section 7 notebook)."""

from learnguard.data.schema import OULAD_TABLE_SCHEMAS, ColumnSpec


def test_all_seven_tables_have_contracts():
    assert set(OULAD_TABLE_SCHEMAS) == {
        "courses",
        "assessments",
        "vle",
        "studentInfo",
        "studentRegistration",
        "studentAssessment",
        "studentVle",
    }


def test_student_info_pk_is_composite():
    # EDA quyết định #1: id_student KHÔNG unique — 3.538 SV học nhiều kỳ
    assert OULAD_TABLE_SCHEMAS["studentInfo"].key_columns == (
        "code_module",
        "code_presentation",
        "id_student",
    )


def test_score_allows_null_but_bounded():
    # EDA #3: score null (rút lui) hợp lệ; giá trị trong [0, 100]
    spec = OULAD_TABLE_SCHEMAS["studentAssessment"].columns["score"]
    assert spec.nullable is True
    assert spec.min_value == 0
    assert spec.max_value == 100


def test_registration_dates_nullable():
    # EDA #3: date_unregistration null 69% = chưa rút lui — hợp lệ
    cols = OULAD_TABLE_SCHEMAS["studentRegistration"].columns
    assert cols["date_unregistration"].nullable is True
    assert cols["date_registration"].nullable is True


def test_student_vle_date_allows_negative():
    # EDA #6: click trước khai giảng (min −25) là dữ liệu hợp lệ
    spec = OULAD_TABLE_SCHEMAS["studentVle"].columns["date"]
    assert spec.min_value == -25
    assert spec.nullable is False


def test_assessment_date_nullable_for_exams():
    # EDA #4: 11 Exam không công bố ngày — null hợp lệ
    spec = OULAD_TABLE_SCHEMAS["assessments"].columns["date"]
    assert spec.nullable is True


def test_categorical_enums_match_eda():
    assert OULAD_TABLE_SCHEMAS["studentInfo"].columns["gender"].allowed_values == ("M", "F")
    assert OULAD_TABLE_SCHEMAS["studentInfo"].columns["final_result"].allowed_values == (
        "Pass",
        "Fail",
        "Withdrawn",
        "Distinction",
    )
    assert OULAD_TABLE_SCHEMAS["assessments"].columns["assessment_type"].allowed_values == (
        "TMA",
        "CMA",
        "Exam",
    )


def test_student_vle_has_no_strict_pk_event_log():
    # EDA #5 (bản sửa 2026-10-26): studentVle là event log — trùng key là sự kiện
    # click riêng biệt, KHÔNG được quarantine theo PK; chỉ exact-duplicate mới dedupe.
    assert OULAD_TABLE_SCHEMAS["studentVle"].key_columns == ()


def test_required_columns_property_derives_from_contract():
    schema = OULAD_TABLE_SCHEMAS["studentVle"]
    assert schema.required_columns == (
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    )


def test_every_column_has_a_spec():
    for name, schema in OULAD_TABLE_SCHEMAS.items():
        for col, spec in schema.columns.items():
            assert isinstance(spec, ColumnSpec), f"{name}.{col}"
            assert spec.dtype in {"str", "int64", "float64"}, f"{name}.{col}"
