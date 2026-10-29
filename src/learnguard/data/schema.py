"""Static OULAD table contracts (M1).

Mọi rule dưới đây đều truy vết được về EDA run 2026-10-26
(notebooks/01_oulad_eda.ipynb) — không rule nào là đoán:

- dtype          : kiểu pandas khi nạp CSV gốc
- nullable       : null policy từ EDA (null hợp lệ nghiệp vụ thì không quarantine)
- allowed_values : miền giá trị quan sát thực tế (section 2 của EDA)
- min/max        : biên quan sát thực tế (section 3 của EDA)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ColumnSpec:
    """Contract cho một cột: kiểu, null policy, miền giá trị."""

    dtype: str  # "str" | "int64" | "float64" (dtype pandas khi nạp CSV)
    nullable: bool = False
    allowed_values: tuple[str | int, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class TableSchema:
    name: str
    key_columns: tuple[str, ...]
    columns: dict[str, ColumnSpec] = field(default_factory=dict)

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(self.columns)


# ---- Miền giá trị dùng chung (EDA section 2) -------------------------------

CODE_MODULES = ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")
CODE_PRESENTATIONS = ("2013B", "2013J", "2014B", "2014J")

REGIONS = (
    "Scotland", "East Anglian Region", "London Region", "South Region",
    "North Western Region", "West Midlands Region", "South West Region",
    "East Midlands Region", "South East Region", "Wales", "Yorkshire Region",
    "North Region", "Ireland",
)
HIGHEST_EDUCATION = (
    "A Level or Equivalent", "Lower Than A Level", "HE Qualification",
    "No Formal quals", "Post Graduate Qualification",
)
IMD_BANDS = (
    "0-10%", "10-20", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
)
AGE_BANDS = ("0-35", "35-55", "55<=")
ACTIVITY_TYPES = (
    "resource", "subpage", "oucontent", "url", "forumng", "quiz", "page",
    "oucollaborate", "questionnaire", "ouwiki", "dataplus", "externalquiz",
    "homepage", "glossary", "ouelluminate", "dualpane", "repeatactivity",
    "htmlactivity", "sharedsubpage", "folder",
)
ASSESSMENT_TYPES = ("TMA", "CMA", "Exam")

_INT = ColumnSpec(dtype="int64", min_value=1)
_STR_CODE_MODULE = ColumnSpec(dtype="str", allowed_values=CODE_MODULES)
_STR_CODE_PRES = ColumnSpec(dtype="str", allowed_values=CODE_PRESENTATIONS)

OULAD_TABLE_SCHEMAS: dict[str, TableSchema] = {
    "courses": TableSchema(
        "courses",
        ("code_module", "code_presentation"),
        {
            "code_module": _STR_CODE_MODULE,
            "code_presentation": _STR_CODE_PRES,
            "module_presentation_length": ColumnSpec(dtype="int64", min_value=234, max_value=269),
        },
    ),
    "assessments": TableSchema(
        "assessments",
        ("id_assessment",),
        {
            "id_assessment": _INT,
            "code_module": _STR_CODE_MODULE,
            "code_presentation": _STR_CODE_PRES,
            "assessment_type": ColumnSpec(dtype="str", allowed_values=ASSESSMENT_TYPES),
            # Null 11 dòng = Exam không công bố ngày — HỢP LỆ (EDA quyết định #4)
            "date": ColumnSpec(dtype="float64", nullable=True, min_value=0, max_value=269),
            "weight": ColumnSpec(dtype="float64", min_value=0, max_value=100),
        },
    ),
    "vle": TableSchema(
        "vle",
        ("id_site",),
        {
            "id_site": _INT,
            "code_module": _STR_CODE_MODULE,
            "code_presentation": _STR_CODE_PRES,
            "activity_type": ColumnSpec(dtype="str", allowed_values=ACTIVITY_TYPES),
            # Null 82% là cấu trúc (MNAR) — hợp lệ; loại khỏi feature catalog (EDA #8)
            "week_from": ColumnSpec(dtype="float64", nullable=True),
            "week_to": ColumnSpec(dtype="float64", nullable=True),
        },
    ),
    "studentInfo": TableSchema(
        # Quyết định EDA #1: id_student KHÔNG unique — PK composite 3 cột
        "studentInfo",
        ("code_module", "code_presentation", "id_student"),
        {
            "code_module": _STR_CODE_MODULE,
            "code_presentation": _STR_CODE_PRES,
            "id_student": _INT,
            "gender": ColumnSpec(dtype="str", allowed_values=("M", "F")),
            "region": ColumnSpec(dtype="str", allowed_values=REGIONS),
            "highest_education": ColumnSpec(dtype="str", allowed_values=HIGHEST_EDUCATION),
            # Null 3.41% — giữ NA + cờ imd_band_missing ở M2, không drop (EDA #2)
            "imd_band": ColumnSpec(dtype="str", nullable=True, allowed_values=IMD_BANDS),
            "age_band": ColumnSpec(dtype="str", allowed_values=AGE_BANDS),
            "num_of_prev_attempts": ColumnSpec(dtype="int64", min_value=0, max_value=6),
            "studied_credits": ColumnSpec(dtype="int64", min_value=30, max_value=655),
            "disability": ColumnSpec(dtype="str", allowed_values=("Y", "N")),
            "final_result": ColumnSpec(dtype="str", allowed_values=("Pass", "Fail", "Withdrawn", "Distinction")),
        },
    ),
    "studentRegistration": TableSchema(
        "studentRegistration",
        ("code_module", "code_presentation", "id_student"),
        {
            "code_module": _STR_CODE_MODULE,
            "code_presentation": _STR_CODE_PRES,
            "id_student": _INT,
            # Null 45 dòng; date có thể âm (đăng ký trước khai giảng)
            "date_registration": ColumnSpec(dtype="float64", nullable=True, min_value=-322, max_value=167),
            # Null 69% = chưa rút lui — HỢP LỆ (EDA #3)
            "date_unregistration": ColumnSpec(dtype="float64", nullable=True, min_value=-365, max_value=444),
        },
    ),
    "studentAssessment": TableSchema(
        "studentAssessment",
        ("id_student", "id_assessment"),
        {
            "id_assessment": _INT,
            "id_student": _INT,
            "date_submitted": ColumnSpec(dtype="int64", min_value=-11, max_value=608),
            "is_banked": ColumnSpec(dtype="int64", allowed_values=(0, 1)),
            # Null 173 dòng = rút lui trước khi có điểm — hợp lệ, chú ý leakage M3 (EDA #3)
            "score": ColumnSpec(dtype="float64", nullable=True, min_value=0, max_value=100),
        },
    ),
    "studentVle": TableSchema(
        # Event log — KHÔNG có PK nghiêm ngặt: trùng (module,pres,student,site,date)
        # là hợp lệ (sự kiện click riêng). Bản sao y hệt toàn cột được M1 dedupe
        # (verify lại 2026-10-29: 787,170 dòng exact-duplicate), phần sự kiện SUM ở M2.
        "studentVle",
        (),
        {
            "code_module": _STR_CODE_MODULE,
            "code_presentation": _STR_CODE_PRES,
            "id_student": _INT,
            "id_site": _INT,
            # Date âm hợp lệ (min −25 = xem trước tài liệu); trùng key → SUM ở M2 (EDA #5, #6)
            "date": ColumnSpec(dtype="int64", min_value=-25, max_value=269),
            "sum_click": ColumnSpec(dtype="int64", min_value=1, max_value=6977),
        },
    ),
}

TARGET_LABELS: dict[str, int] = {"Pass": 0, "Distinction": 0, "Fail": 1, "Withdrawn": 1}
