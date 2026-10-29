"""API tiêu thụ snapshot cho ML owner — KHÔNG tự lọc tay, KHÔNG copy code.

Cách dùng duy nhất được cấp phép:

    from learnguard.data.loading import load_snapshot_splits

    splits = load_snapshot_splits("artifacts/feature_snapshot_m2/snapshot.parquet")
    X_tr, y_tr = splits.x_y("train")
    print(splits.snapshot_version)  # bắt buộc ghi vào model card

Loader tự kiểm tra: cột `split` tồn tại + đúng domain, ledger đi kèm. Bất kỳ
vi phạm nào fail lớn thay vì trả về dữ liệu lờ mờ.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from learnguard.data.features import (
    FEATURE_CATALOG,
    STATIC_FEATURES,
)

SPLITS: tuple[str, ...] = ("train", "validation", "holdout")
TARGET_COLUMN = "label"

# Cột đầu vào model: static + module_presentation + behavioral (từ catalog) + cờ hệ thống
BEHAVIORAL_FEATURES: tuple[str, ...] = tuple(
    f.name for f in FEATURE_CATALOG if f.name != "module_presentation"
)
MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    tuple(STATIC_FEATURES)
    + ("module_presentation",)
    + BEHAVIORAL_FEATURES
    + ("imd_band_missing", "insufficient_data")
)


@dataclass(frozen=True)
class SnapshotSplits:
    """Bộ 3 tập + phiên bản snapshot — kết quả duy nhất ML owner được dùng."""

    snapshot_version: str
    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame

    def frame(self, split: str) -> pd.DataFrame:
        if split not in SPLITS:
            raise ValueError(f"unknown_split:{split}:allowed={','.join(SPLITS)}")
        return getattr(self, split)

    def x_y(self, split: str) -> tuple[pd.DataFrame, pd.Series]:
        """(X, y) theo đúng MODEL_FEATURE_COLUMNS — cột được cam kết, đúng thứ tự."""
        df = self.frame(split)
        missing = [c for c in MODEL_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"feature_columns_missing:{','.join(missing)}")
        return df[list(MODEL_FEATURE_COLUMNS)].copy(), df[TARGET_COLUMN].copy()


def load_snapshot_splits(snapshot_path: Path | str) -> SnapshotSplits:
    """Đọc snapshot + ledger, kiểm tra tính hợp lệ, trả về bộ 3 tập."""
    snapshot_path = Path(snapshot_path)
    snap = pd.read_parquet(snapshot_path)

    if "split" not in snap.columns:
        raise ValueError(f"split_column_missing:{snapshot_path}")
    present = set(snap["split"].unique())
    invalid = present - set(SPLITS)
    if invalid:
        raise ValueError(f"split_out_of_domain:{','.join(sorted(invalid))}")
    empty = [s for s in SPLITS if s not in present]
    if empty:
        raise ValueError(f"split_empty:{','.join(empty)}")

    ledger_path = snapshot_path.parent / (snapshot_path.stem + ".ledger.json")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"missing_ledger:{ledger_path} — rebuild snapshot bằng pipeline")
    version = json.loads(ledger_path.read_text()).get("snapshot_version", "")

    return SnapshotSplits(
        snapshot_version=version,
        train=snap[snap["split"] == "train"].copy(),
        validation=snap[snap["split"] == "validation"].copy(),
        holdout=snap[snap["split"] == "holdout"].copy(),
    )
