"""ML-owned train-only preprocessing for the snapshot contract in PIPELINE_GUIDE."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from learnguard.data.loading import MODEL_FEATURE_COLUMNS, TARGET_COLUMN, load_snapshot_splits
from learnguard.data.features import KEYS, SPLIT_RULES

PREPROCESSOR_SCHEMA_VERSION = "preprocessor-v1"
PREPROCESSING_INPUT_COLUMNS = tuple(MODEL_FEATURE_COLUMNS)
FORBIDDEN_OUTPUT_COLUMNS = {
    TARGET_COLUMN, "final_result", "id_student", "code_presentation", "split"
}
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    keys: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class PreprocessingBuildResult:
    status: str
    run_id: str
    artifact_dir: Path
    preprocessor_version: str
    output_feature_count: int
    manifest_path: Path
    verification_path: Path
    success_path: Path


@dataclass
class PreprocessingVerificationReport:
    status: str
    preprocessor_version: str
    output_feature_count: int
    output_feature_names: list[str]
    split_results: dict[str, dict[str, object]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FittedPreprocessor:
    _pipeline: ColumnTransformer
    manifest: dict[str, object]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.manifest["output_feature_names"])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        _validate_input(X)
        values = np.ascontiguousarray(self._pipeline.transform(X), dtype=np.float64)
        if values.shape[1] != len(self.feature_names) or not np.isfinite(values).all():
            raise ValueError("preprocessor_transform_invalid")
        return values


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid_run_id")
    return run_id


def _json_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_hash(keys: pd.DataFrame, frame: pd.DataFrame) -> str:
    combined = pd.concat([keys.reset_index(drop=True), frame.reset_index(drop=True)], axis=1)
    digest = sha256("\n".join(f"{c}:{combined[c].dtype}" for c in combined).encode())
    digest.update(pd.util.hash_pandas_object(combined, index=False, categorize=True).to_numpy().tobytes())
    return digest.hexdigest()


def _matrix_hash(keys: pd.DataFrame, values: np.ndarray, names: list[str]) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = sha256(json.dumps(names, separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(keys, index=False, categorize=True).to_numpy().tobytes())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _series_hash(keys: pd.DataFrame, values: pd.Series) -> str:
    return _frame_hash(keys, values.rename(values.name or "value").to_frame())


def _snapshot_version(snapshot_path: Path) -> str:
    ledger = snapshot_path.parent / f"{snapshot_path.stem}.ledger.json"
    if not ledger.is_file():
        raise FileNotFoundError(f"missing_ledger:{ledger}")
    value = json.loads(ledger.read_text()).get("snapshot_version")
    if not isinstance(value, str) or not value:
        raise ValueError("snapshot_version_missing")
    return value


def _prepare_split(name: str, frame: pd.DataFrame) -> PreparedSplit:
    ordered = frame.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    keys = ordered[list(KEYS)].copy()
    X = ordered[list(PREPROCESSING_INPUT_COLUMNS)].copy()
    _validate_input(X)
    if keys.duplicated().any():
        raise ValueError(f"preprocessing_duplicate_keys:{name}")
    return PreparedSplit(name, keys, X, ordered[TARGET_COLUMN].copy())


def _prepare_splits(snapshot_path: Path) -> dict[str, PreparedSplit]:
    splits = load_snapshot_splits(snapshot_path)
    return {name: _prepare_split(name, splits.frame(name)) for name in ("train", "validation", "holdout")}


def _validate_input(X: pd.DataFrame) -> None:
    if list(X.columns) != list(PREPROCESSING_INPUT_COLUMNS):
        raise ValueError("preprocessing_input_schema_or_order_mismatch")
    if X.empty:
        raise ValueError("preprocessing_input_empty")


def _as_float64(values: object) -> np.ndarray:
    """Convert binary passthrough columns without using an unpicklable lambda."""
    return np.asarray(values, dtype=np.float64)


def _build_transformer() -> ColumnTransformer:
    nominal = ["module_presentation", "gender", "region", "disability"]
    ordinal = ["highest_education", "age_band"]
    numeric = ["num_of_prev_attempts", "studied_credits", "click_trend_w3_w4", "n_submitted_before_cutoff"]
    transformer = ColumnTransformer([
        ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64), nominal),
        ("ordinal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64)),
            ("scale", StandardScaler()),
        ]), ordinal),
        ("imd", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64)),
            ("scale", StandardScaler()),
        ]), ["imd_band"]),
        ("numeric", StandardScaler(), numeric),
        ("clicks", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), ["n_clicks_before_cutoff"]),
        ("score", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), ["avg_score_submitted"]),
        ("binary", FunctionTransformer(_as_float64, feature_names_out="one-to-one"), ["imd_band_missing", "insufficient_data"]),
    ], remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=True)
    return transformer


def preprocessing_contract_hash() -> str:
    return _json_hash({
        "schema": PREPROCESSOR_SCHEMA_VERSION,
        "inputs": PREPROCESSING_INPUT_COLUMNS,
        "output": "dense-float64",
        "sklearn": sklearn.__version__,
    })


def _verification(transformer: ColumnTransformer, version: str, prepared: dict[str, PreparedSplit]) -> PreprocessingVerificationReport:
    names = [str(x) for x in transformer.get_feature_names_out()]
    report = PreprocessingVerificationReport("ok", version, len(names), names)
    for name, split in prepared.items():
        values = np.asarray(transformer.transform(split.X), dtype=np.float64)
        if values.shape != (len(split.X), len(names)) or not np.isfinite(values).all():
            report.errors.append(f"invalid_transform:{name}")
        report.split_results[name] = {
            "row_count": len(split.X),
            "target_hash": _series_hash(split.keys, split.y),
            "transformed_semantic_hash": _matrix_hash(split.keys, values, names),
            "nan_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
        }
    if report.errors:
        report.status = "failed"
    return report


def build_preprocessing_artifact(snapshot_path: Path | str, artifacts_dir: Path | str = Path("artifacts"), run_id: str = "preprocessing") -> PreprocessingBuildResult:
    snapshot_path = Path(snapshot_path)
    run_id = _validate_run_id(run_id)
    target = Path(artifacts_dir) / "preprocessing-runs" / run_id
    if target.exists():
        raise FileExistsError(f"preprocessing_run_id_exists:{run_id}")
    prepared = _prepare_splits(snapshot_path)
    transformer = _build_transformer()
    transformer.fit(prepared["train"].X)
    names = [str(x) for x in transformer.get_feature_names_out()]
    train_hash = _frame_hash(prepared["train"].keys, prepared["train"].X)
    contract_hash = preprocessing_contract_hash()
    state_hash = _json_hash({"feature_names": names, "params": transformer.get_params(deep=False)})
    version = f"{PREPROCESSOR_SCHEMA_VERSION}:{contract_hash[:8]}:{train_hash[:8]}:{state_hash[:8]}"
    report = _verification(transformer, version, prepared)
    if report.status != "ok":
        raise ValueError("preprocessing_verification_failed")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        artifact = staging / "preprocessor.joblib"
        verification = staging / "preprocessor.verification.json"
        manifest_path = staging / "preprocessor.manifest.json"
        marker = staging / "preprocessor.success.json"
        joblib.dump(transformer, artifact, compress=3)
        verification.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))
        manifest = {
            "schema_version": PREPROCESSOR_SCHEMA_VERSION,
            "preprocessor_version": version,
            "preprocessing_contract_hash": contract_hash,
            "snapshot_version": _snapshot_version(snapshot_path),
            "train_row_count": len(prepared["train"].X),
            "train_feature_semantic_hash": train_hash,
            "learned_state_hash": state_hash,
            "input_columns": list(PREPROCESSING_INPUT_COLUMNS),
            "key_columns": list(KEYS),
            "target_column": TARGET_COLUMN,
            "output_feature_names": names,
            "output_feature_count": len(names),
            "artifact_sha256": _file_hash(artifact),
            "verification_sha256": _file_hash(verification),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        marker.write_text(json.dumps({
            "schema_version": PREPROCESSOR_SCHEMA_VERSION,
            "preprocessor_version": version,
            "manifest_sha256": _file_hash(manifest_path),
            "artifact_sha256": manifest["artifact_sha256"],
            "verification_sha256": manifest["verification_sha256"],
        }, indent=2, sort_keys=True))
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PreprocessingBuildResult("ok", run_id, target, version, len(names), target / manifest_path.name, target / verification.name, target / marker.name)


def load_preprocessing_artifact(artifact_dir: Path | str, snapshot_path: Path | str) -> FittedPreprocessor:
    artifact_dir, snapshot_path = Path(artifact_dir), Path(snapshot_path)
    paths = {name: artifact_dir / filename for name, filename in {
        "artifact": "preprocessor.joblib", "manifest": "preprocessor.manifest.json",
        "verification": "preprocessor.verification.json", "marker": "preprocessor.success.json",
    }.items()}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("preprocessing_artifact_missing")
    marker = json.loads(paths["marker"].read_text())
    if marker.get("manifest_sha256") != _file_hash(paths["manifest"]):
        raise ValueError("preprocessor_manifest_hash_mismatch")
    manifest = json.loads(paths["manifest"].read_text())
    if manifest.get("schema_version") != PREPROCESSOR_SCHEMA_VERSION or manifest.get("preprocessing_contract_hash") != preprocessing_contract_hash():
        raise ValueError("preprocessor_contract_mismatch")
    if manifest.get("snapshot_version") != _snapshot_version(snapshot_path):
        raise ValueError("preprocessor_snapshot_version_mismatch")
    if manifest.get("artifact_sha256") != _file_hash(paths["artifact"]) or marker.get("artifact_sha256") != _file_hash(paths["artifact"]):
        raise ValueError("preprocessor_artifact_hash_mismatch")
    if manifest.get("verification_sha256") != _file_hash(paths["verification"]) or marker.get("verification_sha256") != _file_hash(paths["verification"]):
        raise ValueError("preprocessor_verification_hash_mismatch")
    prepared = _prepare_splits(snapshot_path)
    if manifest.get("train_feature_semantic_hash") != _frame_hash(prepared["train"].keys, prepared["train"].X):
        raise ValueError("preprocessor_train_feature_hash_mismatch")
    transformer = joblib.load(paths["artifact"])
    names = [str(x) for x in transformer.get_feature_names_out()]
    if names != manifest.get("output_feature_names"):
        raise ValueError("preprocessor_feature_names_mismatch")
    return FittedPreprocessor(transformer, manifest)


def verify_preprocessing_artifact(artifact_dir: Path | str, snapshot_path: Path | str) -> PreprocessingVerificationReport:
    bundle = load_preprocessing_artifact(artifact_dir, snapshot_path)
    return _verification(bundle._pipeline, str(bundle.manifest["preprocessor_version"]), _prepare_splits(Path(snapshot_path)))
