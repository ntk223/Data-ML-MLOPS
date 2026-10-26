"""Fixture chung: bộ 7 bảng OULAD thu nhỏ, sạch — test sẽ cấy lỗi chủ đích.

Nguyên tắc (theo kế hoạch M1): KHÔNG test trên OULAD thật 10.6M dòng;
fixture nhỏ + deterministic, lỗi được cấy có kiểm soát.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


def base_frames() -> dict[str, pd.DataFrame]:
    """Bộ dữ liệu nhỏ SẠCH, đúng toàn bộ contract schema (43 cột / 7 bảng)."""
    return {
        "courses": pd.DataFrame(
            {
                "code_module": ["AAA", "BBB"],
                "code_presentation": ["2013B", "2013J"],
                "module_presentation_length": [240, 250],
            }
        ),
        "assessments": pd.DataFrame(
            {
                "id_assessment": [1, 2],
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013B", "2013B"],
                "assessment_type": ["TMA", "Exam"],
                "date": [20.0, float("nan")],  # Exam không có ngày — hợp lệ
                "weight": [50.0, 100.0],
            }
        ),
        "vle": pd.DataFrame(
            {
                "id_site": [10, 11],
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013B", "2013B"],
                "activity_type": ["resource", "forumng"],
                "week_from": [1.0, float("nan")],  # null 82% là cấu trúc — hợp lệ
                "week_to": [30.0, float("nan")],
            }
        ),
        "studentInfo": pd.DataFrame(
            {
                "code_module": ["AAA", "AAA", "BBB"],
                "code_presentation": ["2013B", "2013B", "2013J"],
                "id_student": [101, 102, 101],  # 101 học 2 kỳ — id KHÔNG unique
                "gender": ["M", "F", "M"],
                "region": ["Scotland", "London Region", "Wales"],
                "highest_education": [
                    "A Level or Equivalent",
                    "Lower Than A Level",
                    "HE Qualification",
                ],
                "imd_band": ["20-30%", "40-50%", None],  # null hợp lệ (EDA #2)
                "age_band": ["0-35", "35-55", "0-35"],
                "num_of_prev_attempts": [0, 1, 0],
                "studied_credits": [60, 90, 60],
                "disability": ["N", "N", "Y"],
                "final_result": ["Pass", "Fail", "Withdrawn"],
            }
        ),
        "studentRegistration": pd.DataFrame(
            {
                "code_module": ["AAA", "AAA", "BBB"],
                "code_presentation": ["2013B", "2013B", "2013J"],
                "id_student": [101, 102, 101],
                "date_registration": [-5.0, 3.0, 1.0],
                "date_unregistration": [float("nan")] * 3,  # null = chưa rút lui — hợp lệ
            }
        ),
        "studentAssessment": pd.DataFrame(
            {
                "id_assessment": [1, 1],
                "id_student": [101, 102],
                "date_submitted": [15, 16],
                "is_banked": [0, 0],
                "score": [80.0, None],  # null = rút lui trước khi chấm — hợp lệ
            }
        ),
        "studentVle": pd.DataFrame(
            {
                "code_module": ["AAA", "AAA"],
                "code_presentation": ["2013B", "2013B"],
                "id_student": [101, 101],
                "id_site": [10, 10],
                "date": [-5, 3],  # date âm = xem trước khai giảng — hợp lệ (EDA #6)
                "sum_click": [10, 20],
            }
        ),
    }


def write_csvs(data_dir: Path, frames: dict[str, pd.DataFrame]) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(data_dir / f"{name}.csv", index=False)
    return data_dir


@pytest.fixture()
def clean_data_dir(tmp_path: Path) -> Path:
    """Thư mục CSV sạch cho 7 bảng."""
    return write_csvs(tmp_path / "src", base_frames())


@pytest.fixture()
def base_tables() -> dict[str, pd.DataFrame]:
    return base_frames()


def mutate(frames: dict[str, pd.DataFrame], table: str, **columns: Any) -> dict[str, pd.DataFrame]:
    """Trả về bản sao frames với cột của một bảng được thay thế."""
    copies = {name: df.copy() for name, df in frames.items()}
    for col, values in columns.items():
        copies[table][col] = values
    return copies
