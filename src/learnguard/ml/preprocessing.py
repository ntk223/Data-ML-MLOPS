"""Train-only preprocessing contract for the day-27 OULAD snapshot."""

from __future__ import annotations

import json
import os
import platform
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from learnguard.data.leakage import audit_snapshot
from learnguard.data.loading import load_snapshot_splits, validate_snapshot_artifact
from learnguard.data.quality import validate_snapshot
from learnguard.data.schema import AGE_BANDS, IMD_BANDS
from learnguard.data.snapshot_contract import CUTOFF_DAY, KEYS, SPLITS, SPLIT_RULES
from learnguard.settings import validate_run_id

PREPROCESSOR_SCHEMA_VERSION = "preprocessor-v1"

NOMINAL_FEATURES = ("code_module", "gender", "region", "disability")
ORDINAL_FEATURES = ("highest_education", "age_band")
IMD_FEATURE = ("imd_band",)
LOG_NUMERIC_FEATURES = ("n_clicks_before_cutoff",)
NUMERIC_FEATURES = (
    "num_of_prev_attempts",
    "studied_credits",
    "active_days_before_cutoff",
    "click_trend_w3_w4",
    "n_submitted_before_cutoff",
)
AVERAGE_SCORE_FEATURE = ("avg_score_submitted",)
BINARY_FEATURES = ("imd_band_missing", "insufficient_data")
PREPROCESSING_INPUT_COLUMNS = (
    *NOMINAL_FEATURES,
    *ORDINAL_FEATURES,
    *IMD_FEATURE,
    *NUMERIC_FEATURES,
    *LOG_NUMERIC_FEATURES,
    *AVERAGE_SCORE_FEATURE,
    *BINARY_FEATURES,
)
TARGET_COLUMN = "label"
AUDIT_ONLY_COLUMNS = ("final_result",)
METADATA_COLUMNS = ("code_presentation", "module_presentation", "split")
FORBIDDEN_OUTPUT_COLUMNS = {
    TARGET_COLUMN,
    *AUDIT_ONLY_COLUMNS,
    "id_student",
    "code_presentation",
    "module_presentation",
    "split",
}

EDUCATION_ORDER = (
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
)


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    keys: pd.DataFrame
    metadata: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


@dataclass
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
    """Transform-only facade; fitting remains internal to the build stage."""

    _pipeline: ColumnTransformer
    manifest: dict[str, object]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["output_feature_names"])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        _validate_input_frame(X)
        output = np.asarray(self._pipeline.transform(X), dtype=np.float64)
        if output.ndim != 2 or output.shape[1] != len(self.feature_names):
            raise ValueError("transformed_feature_shape_mismatch")
        if not np.isfinite(output).all():
            raise ValueError("transformed_non_finite_values")
        return output


def _as_float(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _missing_flag(values: Any) -> np.ndarray:
    return pd.isna(values).astype(np.float64)


def _build_transformer() -> ColumnTransformer:
    nominal = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    ordinal = Pipeline([
        ("encode", OrdinalEncoder(
            categories=[list(EDUCATION_ORDER), list(AGE_BANDS)],
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64,
        )),
        ("scale", StandardScaler()),
    ])
    imd = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encode", OrdinalEncoder(
            categories=[["__MISSING__", *IMD_BANDS]],
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float64,
        )),
        ("scale", StandardScaler()),
    ])
    log_numeric = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    average_score = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("nominal", nominal, list(NOMINAL_FEATURES)),
            ("ordinal", ordinal, list(ORDINAL_FEATURES)),
            ("imd", imd, list(IMD_FEATURE)),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            ("log_numeric", log_numeric, list(LOG_NUMERIC_FEATURES)),
            ("avg_score", average_score, list(AVERAGE_SCORE_FEATURE)),
            ("avg_score_missing", FunctionTransformer(
                _missing_flag, feature_names_out="one-to-one"
            ), list(AVERAGE_SCORE_FEATURE)),
            ("binary", FunctionTransformer(
                _as_float, feature_names_out="one-to-one"
            ), list(BINARY_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def _fit_for_split(
    transformer: ColumnTransformer, X: pd.DataFrame, split_name: str
) -> np.ndarray:
    if split_name != "train":
        raise ValueError(f"preprocessor_fit_forbidden:{split_name}:required=train")
    _validate_input_frame(X)
    return np.asarray(transformer.fit_transform(X), dtype=np.float64)


def _validate_input_frame(X: pd.DataFrame) -> None:
    if list(X.columns) != list(PREPROCESSING_INPUT_COLUMNS):
        raise ValueError("preprocessing_input_schema_or_order_mismatch")
    if X.empty:
        raise ValueError("preprocessing_input_empty")


def _prepare_split(name: str, frame: pd.DataFrame) -> PreparedSplit:
    ordered = frame.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    keys = ordered[list(KEYS)].copy()
    if keys.duplicated().any():
        raise ValueError(f"preprocessing_duplicate_keys:{name}")
    X = ordered[list(PREPROCESSING_INPUT_COLUMNS)].copy()
    _validate_input_frame(X)
    return PreparedSplit(
        name=name,
        keys=keys,
        metadata=ordered[list(METADATA_COLUMNS) + list(AUDIT_ONLY_COLUMNS)].copy(),
        X=X,
        y=ordered[TARGET_COLUMN].copy(),
    )


def _prepare_splits(snapshot_path: Path) -> tuple[dict[str, PreparedSplit], dict[str, object]]:
    _, ledger = validate_snapshot_artifact(snapshot_path)
    loaded = load_snapshot_splits(snapshot_path)
    return {
        name: _prepare_split(name, loaded.frame(name)) for name in SPLITS
    }, ledger


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def _canonical_hash(value: object) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_hash(keys: pd.DataFrame, frame: pd.DataFrame) -> str:
    key_frame = keys.reset_index(drop=True).rename(
        columns={column: f"key__{column}" for column in keys.columns}
    )
    combined = pd.concat([key_frame, frame.reset_index(drop=True)], axis=1)
    schema = "\n".join(f"{column}:{combined[column].dtype}" for column in combined.columns)
    values = pd.util.hash_pandas_object(combined, index=False, categorize=True).to_numpy()
    digest = sha256(schema.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _series_hash(keys: pd.DataFrame, values: pd.Series) -> str:
    return _frame_hash(keys, values.rename(values.name or "value").to_frame())


def _matrix_hash(keys: pd.DataFrame, values: np.ndarray, names: list[str]) -> str:
    array = np.ascontiguousarray(values, dtype=np.float64)
    digest = sha256(json.dumps(names, separators=(",", ":")).encode())
    key_values = pd.util.hash_pandas_object(keys, index=False, categorize=True).to_numpy()
    digest.update(key_values.tobytes())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _packages() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _contract_payload() -> dict[str, object]:
    return {
        "schema_version": PREPROCESSOR_SCHEMA_VERSION,
        "input_columns": PREPROCESSING_INPUT_COLUMNS,
        "nominal_features": NOMINAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "imd_feature": IMD_FEATURE,
        "log_numeric_features": LOG_NUMERIC_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "average_score_feature": AVERAGE_SCORE_FEATURE,
        "binary_features": BINARY_FEATURES,
        "education_order": EDUCATION_ORDER,
        "age_order": AGE_BANDS,
        "imd_order": IMD_BANDS,
        "nominal_unknown": "ignore/all-zero",
        "imd_missing": "__MISSING__",
        "avg_score_imputation": "train_median+missing_indicator",
        "numeric_scaling": "StandardScaler(train)",
        "click_transform": "log1p+StandardScaler(train)",
        "output": "dense-float64",
        "packages": _packages(),
    }


def preprocessing_contract_hash() -> str:
    return _canonical_hash(_contract_payload())


def _learned_state(transformer: ColumnTransformer) -> dict[str, object]:
    nominal = transformer.named_transformers_["nominal"]
    ordinal = transformer.named_transformers_["ordinal"]
    imd = transformer.named_transformers_["imd"]
    numeric = transformer.named_transformers_["numeric"]
    log_numeric = transformer.named_transformers_["log_numeric"]
    avg_score = transformer.named_transformers_["avg_score"]
    return _jsonable({
        "nominal_categories": [list(values) for values in nominal.categories_],
        "ordinal_categories": [list(values) for values in ordinal["encode"].categories_],
        "ordinal_mean": ordinal["scale"].mean_,
        "ordinal_scale": ordinal["scale"].scale_,
        "imd_categories": [list(values) for values in imd["encode"].categories_],
        "imd_imputer_statistics": imd["impute"].statistics_,
        "imd_mean": imd["scale"].mean_,
        "imd_scale": imd["scale"].scale_,
        "numeric_mean": numeric.mean_,
        "numeric_scale": numeric.scale_,
        "log_numeric_mean": log_numeric["scale"].mean_,
        "log_numeric_scale": log_numeric["scale"].scale_,
        "avg_score_imputer_statistics": avg_score["impute"].statistics_,
        "avg_score_mean": avg_score["scale"].mean_,
        "avg_score_scale": avg_score["scale"].scale_,
        "output_feature_names": list(transformer.get_feature_names_out()),
    })


def _verify_source_run(data_run_dir: Path, snapshot_path: Path) -> None:
    reports_dir = data_run_dir / "reports"
    required = {
        "pipeline": reports_dir / "pipeline_report.json",
        "leakage": reports_dir / "leakage_audit.json",
        "quality": reports_dir / "quality_report.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"preprocessing_source_reports_missing:{','.join(missing)}")
    reports = {name: json.loads(path.read_text()) for name, path in required.items()}
    if reports["pipeline"].get("status") != "ok":
        raise ValueError("preprocessing_source_pipeline_not_ok")
    stages = reports["pipeline"].get("stages", [])
    if not stages or any(stage.get("status") != "ok" for stage in stages):
        raise ValueError("preprocessing_source_stage_not_ok")
    if reports["leakage"].get("status") != "ok" or reports["leakage"].get("violations"):
        raise ValueError("preprocessing_source_leakage_not_ok")
    quality = reports["quality"]
    if quality.get("status") != "ok" or quality.get("errors"):
        raise ValueError("preprocessing_source_quality_not_ok")
    gates = quality.get("gate_status", {})
    if gates.get("schema") != "passed" or gates.get("missingness") != "passed":
        raise ValueError("preprocessing_source_required_gate_not_passed")
    if any(value == "failed" for value in gates.values()):
        raise ValueError("preprocessing_source_gate_failed")

    fresh_leakage = audit_snapshot(snapshot_path, data_run_dir / "ingestion", CUTOFF_DAY)
    if fresh_leakage.status != "ok" or fresh_leakage.violations:
        raise ValueError("preprocessing_fresh_leakage_audit_failed")
    fresh_quality = validate_snapshot(snapshot_path)
    if fresh_quality.status != "ok":
        raise ValueError("preprocessing_fresh_quality_gate_failed")


def _verification_report(
    transformer: ColumnTransformer,
    version: str,
    prepared: dict[str, PreparedSplit],
) -> PreprocessingVerificationReport:
    names = [str(value) for value in transformer.get_feature_names_out()]
    report = PreprocessingVerificationReport("ok", version, len(names), names)
    if len(names) != len(set(names)):
        report.errors.append("output_feature_names_not_unique")
    for forbidden in FORBIDDEN_OUTPUT_COLUMNS:
        if any(name == forbidden or name.endswith(f"__{forbidden}") for name in names):
            report.errors.append(f"forbidden_output_feature:{forbidden}")
    reference_width = len(names)
    for split_name in SPLITS:
        split = prepared[split_name]
        transformed = np.asarray(transformer.transform(split.X), dtype=np.float64)
        if transformed.shape != (len(split.X), reference_width):
            report.errors.append(f"output_shape_mismatch:{split_name}")
        if not np.isfinite(transformed).all():
            report.errors.append(f"output_non_finite:{split_name}")
        nominal_width = sum(
            len(values) for values in transformer.named_transformers_["nominal"].categories_
        )
        if nominal_width and not np.isin(transformed[:, :nominal_width], [0.0, 1.0]).all():
            report.errors.append(f"one_hot_domain_invalid:{split_name}")
        report.split_results[split_name] = {
            "row_count": len(split.X),
            "key_hash": _frame_hash(split.keys, pd.DataFrame(index=split.keys.index)),
            "target_hash": _series_hash(split.keys, split.y),
            "transformed_semantic_hash": _matrix_hash(split.keys, transformed, names),
            "nan_count": int(np.isnan(transformed).sum()),
            "inf_count": int(np.isinf(transformed).sum()),
        }
    if report.errors:
        report.status = "failed"
    return report


def build_preprocessing_artifact(
    data_run_dir: Path | str,
    artifacts_dir: Path | str = Path("artifacts"),
    run_id: str | None = None,
) -> PreprocessingBuildResult:
    data_run_dir = Path(data_run_dir)
    generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = validate_run_id(generated if run_id is None else run_id)
    parent = Path(artifacts_dir) / "preprocessing-runs"
    target = parent / run_id
    if target.exists():
        raise FileExistsError(f"preprocessing_run_id_exists:{run_id}")
    snapshot_path = data_run_dir / "snapshot" / "snapshot.parquet"
    validate_snapshot_artifact(snapshot_path)
    _verify_source_run(data_run_dir, snapshot_path)
    prepared, ledger = _prepare_splits(snapshot_path)

    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        transformer = _build_transformer()
        _fit_for_split(transformer, prepared["train"].X, "train")
        names = [str(value) for value in transformer.get_feature_names_out()]
        learned_state = _learned_state(transformer)
        state_hash = _canonical_hash(learned_state)
        train_hash = _frame_hash(prepared["train"].keys, prepared["train"].X)
        contract_hash = preprocessing_contract_hash()
        version = (
            f"{PREPROCESSOR_SCHEMA_VERSION}:{contract_hash[:8]}:"
            f"{train_hash[:8]}:{state_hash[:8]}"
        )
        verification = _verification_report(transformer, version, prepared)
        if verification.status != "ok":
            raise ValueError(f"preprocessing_verification_failed:{','.join(verification.errors)}")

        artifact_path = staging / "preprocessor.joblib"
        verification_path = staging / "preprocessor.verification.json"
        manifest_path = staging / "preprocessor.manifest.json"
        marker_path = staging / "preprocessor.success.json"
        joblib.dump(transformer, artifact_path, compress=3)
        verification_path.write_text(
            json.dumps(_jsonable(asdict(verification)), indent=2, sort_keys=True)
        )
        manifest = {
            "schema_version": PREPROCESSOR_SCHEMA_VERSION,
            "preprocessor_version": version,
            "preprocessing_contract_hash": contract_hash,
            "snapshot_version": ledger["snapshot_version"],
            "snapshot_content_hash": ledger["snapshot_content_hash"],
            "snapshot_contract_hash": ledger["contract_hash"],
            "cutoff_day": CUTOFF_DAY,
            "split_rules": SPLIT_RULES,
            "input_columns": list(PREPROCESSING_INPUT_COLUMNS),
            "input_dtypes": {
                column: str(prepared["train"].X[column].dtype)
                for column in PREPROCESSING_INPUT_COLUMNS
            },
            "key_columns": list(KEYS),
            "target_column": TARGET_COLUMN,
            "audit_only_columns": list(AUDIT_ONLY_COLUMNS),
            "metadata_columns": list(METADATA_COLUMNS),
            "train_row_count": len(prepared["train"].X),
            "train_feature_semantic_hash": train_hash,
            "learned_state_hash": state_hash,
            "learned_state": learned_state,
            "output_feature_names": names,
            "output_feature_count": len(names),
            "packages": _packages(),
            "artifact_sha256": _file_hash(artifact_path),
            "verification_sha256": _file_hash(verification_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))
        marker = {
            "schema_version": PREPROCESSOR_SCHEMA_VERSION,
            "preprocessor_version": version,
            "manifest_sha256": _file_hash(manifest_path),
            "artifact_sha256": manifest["artifact_sha256"],
            "verification_sha256": manifest["verification_sha256"],
        }
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True))
        os.replace(staging, target)
        return PreprocessingBuildResult(
            "ok", run_id, target, version, len(names),
            target / manifest_path.name,
            target / verification_path.name,
            target / marker_path.name,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_preprocessing_artifact(
    artifact_dir: Path | str, snapshot_path: Path | str
) -> FittedPreprocessor:
    artifact_dir, snapshot_path = Path(artifact_dir), Path(snapshot_path)
    artifact_path = artifact_dir / "preprocessor.joblib"
    manifest_path = artifact_dir / "preprocessor.manifest.json"
    verification_path = artifact_dir / "preprocessor.verification.json"
    marker_path = artifact_dir / "preprocessor.success.json"
    for path in (artifact_path, manifest_path, verification_path, marker_path):
        if not path.is_file():
            raise FileNotFoundError(f"preprocessing_artifact_missing:{path.name}")
    marker = json.loads(marker_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if marker.get("schema_version") != PREPROCESSOR_SCHEMA_VERSION:
        raise ValueError("preprocessor_success_schema_mismatch")
    if marker.get("manifest_sha256") != _file_hash(manifest_path):
        raise ValueError("preprocessor_manifest_hash_mismatch")
    if manifest.get("schema_version") != PREPROCESSOR_SCHEMA_VERSION:
        raise ValueError("preprocessor_manifest_schema_mismatch")
    contract_hash = preprocessing_contract_hash()
    if manifest.get("preprocessing_contract_hash") != contract_hash:
        raise ValueError("preprocessor_contract_hash_mismatch")
    if manifest.get("packages") != _packages():
        raise ValueError("preprocessor_runtime_packages_mismatch")
    artifact_hash = _file_hash(artifact_path)
    verification_hash = _file_hash(verification_path)
    if artifact_hash != manifest.get("artifact_sha256") or artifact_hash != marker.get("artifact_sha256"):
        raise ValueError("preprocessor_artifact_hash_mismatch")
    if verification_hash != manifest.get("verification_sha256") or verification_hash != marker.get("verification_sha256"):
        raise ValueError("preprocessor_verification_hash_mismatch")

    _, ledger = validate_snapshot_artifact(snapshot_path)
    expected_snapshot = {
        "snapshot_version": ledger["snapshot_version"],
        "snapshot_content_hash": ledger["snapshot_content_hash"],
        "snapshot_contract_hash": ledger["contract_hash"],
    }
    for field, expected in expected_snapshot.items():
        if manifest.get(field) != expected:
            raise ValueError(f"preprocessor_{field}_mismatch")
    invariants = {
        "cutoff_day": CUTOFF_DAY,
        "split_rules": SPLIT_RULES,
        "input_columns": list(PREPROCESSING_INPUT_COLUMNS),
        "key_columns": list(KEYS),
        "target_column": TARGET_COLUMN,
        "audit_only_columns": list(AUDIT_ONLY_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
    }
    for field, expected in invariants.items():
        if manifest.get(field) != expected:
            raise ValueError(f"preprocessor_manifest_{field}_mismatch")
    prepared, _ = _prepare_splits(snapshot_path)
    train_hash = _frame_hash(prepared["train"].keys, prepared["train"].X)
    if manifest.get("train_row_count") != len(prepared["train"].X):
        raise ValueError("preprocessor_train_row_count_mismatch")
    if manifest.get("train_feature_semantic_hash") != train_hash:
        raise ValueError("preprocessor_train_feature_hash_mismatch")

    transformer = joblib.load(artifact_path)
    if not isinstance(transformer, ColumnTransformer):
        raise TypeError("preprocessor_artifact_type_invalid")
    state = _learned_state(transformer)
    if _canonical_hash(state) != manifest.get("learned_state_hash"):
        raise ValueError("preprocessor_learned_state_hash_mismatch")
    if state != manifest.get("learned_state"):
        raise ValueError("preprocessor_learned_state_manifest_mismatch")
    names = [str(value) for value in transformer.get_feature_names_out()]
    if names != manifest.get("output_feature_names"):
        raise ValueError("preprocessor_output_feature_names_mismatch")
    if len(names) != manifest.get("output_feature_count"):
        raise ValueError("preprocessor_output_feature_count_mismatch")
    expected_version = (
        f"{PREPROCESSOR_SCHEMA_VERSION}:{contract_hash[:8]}:"
        f"{train_hash[:8]}:{str(manifest['learned_state_hash'])[:8]}"
    )
    if manifest.get("preprocessor_version") != expected_version or marker.get(
        "preprocessor_version"
    ) != expected_version:
        raise ValueError("preprocessor_version_mismatch")
    verification = json.loads(verification_path.read_text())
    if verification.get("status") != "ok" or verification.get(
        "preprocessor_version"
    ) != expected_version:
        raise ValueError("preprocessor_verification_report_invalid")
    return FittedPreprocessor(transformer, manifest)


def verify_preprocessing_artifact(
    artifact_dir: Path | str, data_run_dir: Path | str
) -> PreprocessingVerificationReport:
    data_run_dir = Path(data_run_dir)
    snapshot_path = data_run_dir / "snapshot" / "snapshot.parquet"
    bundle = load_preprocessing_artifact(artifact_dir, snapshot_path)
    prepared, _ = _prepare_splits(snapshot_path)
    return _verification_report(
        bundle._pipeline, str(bundle.manifest["preprocessor_version"]), prepared
    )

