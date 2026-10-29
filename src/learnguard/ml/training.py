"""Phase 5 / M4 train-only Logistic Regression baseline.

The public training boundary accepts verified artifact directories rather than
arbitrary matrices or split names. Preprocessing is transform-only here.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from pandas.api.types import is_integer_dtype
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from learnguard.data.snapshot_contract import (
    CUTOFF_DAY,
    KEYS,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_SCHEMA_VERSION,
    SPLIT_RULES,
    validate_ledger_metadata,
)
from learnguard.ml.preprocessing import (
    AUDIT_ONLY_COLUMNS,
    FORBIDDEN_OUTPUT_COLUMNS,
    METADATA_COLUMNS,
    PREPROCESSING_INPUT_COLUMNS,
    PREPROCESSOR_SCHEMA_VERSION,
    TARGET_COLUMN,
    FittedPreprocessor,
    _canonical_hash,
    _file_hash,
    _frame_hash,
    _learned_state,
    _matrix_hash,
    _packages as preprocessing_packages,
    _series_hash,
    preprocessing_contract_hash,
)
from learnguard.settings import validate_run_id


MODEL_ARTIFACT_SCHEMA_VERSION = "logistic-model-v1"
MODEL_SEMANTIC_VERSION_PREFIX = "logreg-v1"
MODEL_HASH_ALGORITHM = "learnguard-model-hash-v1"
NATIVE_THREAD_LIMIT = 1
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0

LOGISTIC_BASELINE_PARAMS: dict[str, object] = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "class_weight": None,
    "max_iter": 2000,
    "tol": 1e-4,
    "fit_intercept": True,
    "random_state": 42,
    "n_jobs": None,
}


@dataclass(frozen=True)
class VerifiedTrainMatrix:
    keys: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    snapshot_ledger: dict[str, object]
    preprocessor_manifest: dict[str, object]
    preprocessing_file_hashes: dict[str, str]
    training_matrix_semantic_hash: str
    training_target_semantic_hash: str


@dataclass(frozen=True)
class LogisticTrainingResult:
    status: str
    run_id: str
    artifact_dir: Path
    model_semantic_version: str
    model_contract_hash: str
    learned_state_hash: str
    training_matrix_semantic_hash: str
    training_target_semantic_hash: str
    published_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ModelVerificationReport:
    status: str
    model_semantic_version: str
    model_contract_hash: str
    train_row_count: int
    output_feature_count: int
    learned_classes: tuple[int, ...]
    convergence_status: str
    training_matrix_semantic_hash: str
    training_target_semantic_hash: str
    train_probability_semantic_hash: str


@dataclass(frozen=True)
class LoadedLogisticBaseline:
    """Transform/predict facade; the fitted estimator is intentionally private."""

    _model: LogisticRegression
    _preprocessor: FittedPreprocessor
    _manifest: dict[str, object]
    verification: ModelVerificationReport

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._manifest["output_feature_names"])

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(self._manifest)

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        matrix = self._preprocessor.transform(X)
        probabilities = _validate_probabilities(self._model, matrix)
        positive_index = int(np.flatnonzero(self._model.classes_ == POSITIVE_CLASS)[0])
        return probabilities[:, positive_index].copy()


def _model_packages() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }


def _resolved_estimator_params() -> dict[str, object]:
    return LogisticRegression(**LOGISTIC_BASELINE_PARAMS).get_params(deep=False)


def _model_contract_payload() -> dict[str, object]:
    return {
        "artifact_schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "semantic_version_prefix": MODEL_SEMANTIC_VERSION_PREFIX,
        "model_type": "sklearn.linear_model.LogisticRegression",
        "estimator_parameters": _resolved_estimator_params(),
        "target_column": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "train_split_rule": {
            presentation: split
            for presentation, split in SPLIT_RULES.items()
            if split == "train"
        },
        "key_columns": list(KEYS),
        "input_dtype": "float64",
        "row_order": list(KEYS),
        "native_thread_limit": NATIVE_THREAD_LIMIT,
        "convergence_policy": {
            "convergence_warning": "fail",
            "n_iter_must_be_positive": True,
            "n_iter_must_be_less_than_max_iter": True,
            "auto_retry": False,
        },
        "hash_algorithm": MODEL_HASH_ALGORITHM,
        "packages": _model_packages(),
    }


def model_contract_hash() -> str:
    return _canonical_hash(_model_contract_payload())


def _array_hash(name: str, values: np.ndarray, dtype: str) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    digest = sha256(f"{MODEL_HASH_ALGORITHM}:{name}:{array.dtype}:{array.shape}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _prediction_hash(keys: pd.DataFrame, positive_probability: np.ndarray) -> str:
    probabilities = np.ascontiguousarray(positive_probability, dtype="<f8")
    key_hashes = pd.util.hash_pandas_object(
        keys.reset_index(drop=True), index=False, categorize=True
    ).to_numpy(dtype="<u8", copy=False)
    digest = sha256(
        f"{MODEL_HASH_ALGORITHM}:positive-class:{POSITIVE_CLASS}:{probabilities.shape}".encode()
    )
    digest.update(key_hashes.tobytes())
    digest.update(probabilities.tobytes())
    return digest.hexdigest()


def _model_state(model: LogisticRegression) -> dict[str, object]:
    state = {
        "coefficient_semantic_hash": _array_hash("coefficient", model.coef_, "<f8"),
        "intercept_semantic_hash": _array_hash("intercept", model.intercept_, "<f8"),
        "learned_classes": [int(value) for value in model.classes_.tolist()],
        "n_features_in": int(model.n_features_in_),
        "n_iter": [int(value) for value in model.n_iter_.tolist()],
    }
    return {**state, "learned_state_hash": _canonical_hash(state)}


def _read_json(path: Path, error_prefix: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{error_prefix}_invalid:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{error_prefix}_invalid:not_object")
    return value


def _verify_data_run_metadata(data_run_dir: Path) -> tuple[Path, dict[str, object]]:
    snapshot_path = data_run_dir / "snapshot" / "snapshot.parquet"
    required = (
        snapshot_path,
        data_run_dir / "snapshot" / "snapshot.ledger.json",
        data_run_dir / "snapshot" / "snapshot.success.json",
        data_run_dir / "reports" / "pipeline_report.json",
        data_run_dir / "reports" / "leakage_audit.json",
        data_run_dir / "reports" / "quality_report.json",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model_source_files_missing:{','.join(missing)}")

    pipeline = _read_json(required[3], "model_source_pipeline_report")
    leakage = _read_json(required[4], "model_source_leakage_report")
    quality = _read_json(required[5], "model_source_quality_report")
    stages = pipeline.get("stages")
    if pipeline.get("status") != "ok" or not isinstance(stages, list) or not stages or any(
        not isinstance(stage, dict) or stage.get("status") != "ok" for stage in stages
    ):
        raise ValueError("model_source_pipeline_not_ok")
    if leakage.get("status") != "ok" or leakage.get("violations"):
        raise ValueError("model_source_leakage_not_ok")
    if quality.get("status") != "ok" or quality.get("errors"):
        raise ValueError("model_source_quality_not_ok")
    gates = quality.get("gate_status")
    if not isinstance(gates, dict) or gates.get("schema") != "passed" or gates.get(
        "missingness"
    ) != "passed" or "failed" in gates.values():
        raise ValueError("model_source_quality_gate_not_ok")

    ledger = _read_json(required[1], "model_snapshot_ledger")
    metadata_errors = validate_ledger_metadata(ledger)
    if metadata_errors:
        raise ValueError(f"model_snapshot_ledger_invalid:{','.join(metadata_errors)}")
    if ledger.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("model_snapshot_schema_unsupported")
    if ledger.get("output_schema") != list(SNAPSHOT_COLUMNS):
        raise ValueError("model_snapshot_output_schema_mismatch")
    if not isinstance(ledger.get("snapshot_content_hash"), str) or len(
        str(ledger.get("snapshot_content_hash"))
    ) != 64:
        raise ValueError("model_snapshot_content_hash_invalid")
    marker = _read_json(required[2], "model_snapshot_success_marker")
    for field in ("snapshot_version", "snapshot_content_hash"):
        if marker.get(field) != ledger.get(field):
            raise ValueError(f"model_snapshot_success_{field}_mismatch")
    return snapshot_path, ledger


def _preprocessing_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "artifact": artifact_dir / "preprocessor.joblib",
        "manifest": artifact_dir / "preprocessor.manifest.json",
        "verification": artifact_dir / "preprocessor.verification.json",
        "success": artifact_dir / "preprocessor.success.json",
    }


def _load_preprocessor_for_training(
    artifact_dir: Path, ledger: dict[str, object]
) -> tuple[FittedPreprocessor, dict[str, object], dict[str, object], dict[str, str]]:
    paths = _preprocessing_paths(artifact_dir)
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model_preprocessing_files_missing:{','.join(missing)}")
    hashes = {name: _file_hash(path) for name, path in paths.items()}
    marker = _read_json(paths["success"], "model_preprocessor_success")
    manifest = _read_json(paths["manifest"], "model_preprocessor_manifest")
    verification = _read_json(paths["verification"], "model_preprocessor_verification")

    if marker.get("schema_version") != PREPROCESSOR_SCHEMA_VERSION:
        raise ValueError("model_preprocessor_success_schema_mismatch")
    if marker.get("manifest_sha256") != hashes["manifest"]:
        raise ValueError("model_preprocessor_manifest_hash_mismatch")
    if manifest.get("schema_version") != PREPROCESSOR_SCHEMA_VERSION:
        raise ValueError("model_preprocessor_manifest_schema_mismatch")
    if manifest.get("preprocessing_contract_hash") != preprocessing_contract_hash():
        raise ValueError("model_preprocessor_contract_hash_mismatch")
    if manifest.get("packages") != preprocessing_packages():
        raise ValueError("model_preprocessor_runtime_packages_mismatch")
    if manifest.get("artifact_sha256") != hashes["artifact"] or marker.get(
        "artifact_sha256"
    ) != hashes["artifact"]:
        raise ValueError("model_preprocessor_artifact_hash_mismatch")
    if manifest.get("verification_sha256") != hashes["verification"] or marker.get(
        "verification_sha256"
    ) != hashes["verification"]:
        raise ValueError("model_preprocessor_verification_hash_mismatch")

    provenance = {
        "snapshot_version": ledger["snapshot_version"],
        "snapshot_content_hash": ledger["snapshot_content_hash"],
        "snapshot_contract_hash": ledger["contract_hash"],
    }
    for field, expected in provenance.items():
        if manifest.get(field) != expected:
            raise ValueError(f"model_preprocessor_{field}_mismatch")
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
            raise ValueError(f"model_preprocessor_manifest_{field}_mismatch")

    if verification.get("status") != "ok" or verification.get(
        "preprocessor_version"
    ) != manifest.get("preprocessor_version"):
        raise ValueError("model_preprocessor_verification_invalid")
    if verification.get("output_feature_names") != manifest.get("output_feature_names"):
        raise ValueError("model_preprocessor_verification_feature_names_mismatch")
    if verification.get("output_feature_count") != manifest.get("output_feature_count"):
        raise ValueError("model_preprocessor_verification_feature_count_mismatch")

    transformer = joblib.load(paths["artifact"])
    if not isinstance(transformer, ColumnTransformer):
        raise TypeError("model_preprocessor_artifact_type_invalid")
    state = _learned_state(transformer)
    if _canonical_hash(state) != manifest.get("learned_state_hash") or state != manifest.get(
        "learned_state"
    ):
        raise ValueError("model_preprocessor_learned_state_mismatch")
    names = [str(value) for value in transformer.get_feature_names_out()]
    if names != manifest.get("output_feature_names") or len(names) != manifest.get(
        "output_feature_count"
    ):
        raise ValueError("model_preprocessor_output_features_mismatch")
    expected_version = (
        f"{PREPROCESSOR_SCHEMA_VERSION}:{str(manifest['preprocessing_contract_hash'])[:8]}:"
        f"{str(manifest['train_feature_semantic_hash'])[:8]}:"
        f"{str(manifest['learned_state_hash'])[:8]}"
    )
    if manifest.get("preprocessor_version") != expected_version or marker.get(
        "preprocessor_version"
    ) != expected_version:
        raise ValueError("model_preprocessor_version_mismatch")
    return FittedPreprocessor(transformer, manifest), manifest, verification, hashes


def _load_verified_train_matrix(
    data_run_dir: Path | str, preprocessing_artifact_dir: Path | str
) -> VerifiedTrainMatrix:
    snapshot_path, ledger = _verify_data_run_metadata(Path(data_run_dir))
    bundle, manifest, verification, hashes = _load_preprocessor_for_training(
        Path(preprocessing_artifact_dir), ledger
    )

    columns = list(
        dict.fromkeys([*KEYS, *PREPROCESSING_INPUT_COLUMNS, TARGET_COLUMN, "split"])
    )
    frame = pd.read_parquet(
        snapshot_path, columns=columns, filters=[("split", "==", "train")]
    )
    if frame.empty:
        raise ValueError("model_train_empty")
    if not frame["split"].eq("train").all():
        raise ValueError("model_non_train_rows_loaded")
    train_presentations = set(frame["code_presentation"].unique())
    expected_presentations = {
        presentation for presentation, split in SPLIT_RULES.items() if split == "train"
    }
    if not train_presentations <= expected_presentations:
        raise ValueError("model_train_split_rule_mismatch")
    frame = frame.sort_values(list(KEYS), kind="mergesort").reset_index(drop=True)
    keys = frame[list(KEYS)].copy()
    if keys.duplicated().any():
        raise ValueError("model_train_duplicate_keys")
    raw_X = frame[list(PREPROCESSING_INPUT_COLUMNS)].copy()
    target = frame[TARGET_COLUMN]
    if not is_integer_dtype(target.dtype):
        raise ValueError("model_target_dtype_invalid")
    y = target.to_numpy(dtype=np.int64, copy=True)
    unique = np.unique(y)
    if unique.size != 2 or not np.array_equal(unique, np.array([0, 1], dtype=np.int64)):
        if unique.size < 2:
            raise ValueError("model_train_single_class")
        raise ValueError("model_target_domain_invalid")

    raw_hash = _frame_hash(keys, raw_X)
    if manifest.get("train_row_count") != len(frame):
        raise ValueError("model_preprocessor_train_row_count_mismatch")
    if manifest.get("train_feature_semantic_hash") != raw_hash:
        raise ValueError("model_preprocessor_train_feature_hash_mismatch")
    target_hash = _series_hash(keys, target)
    split_results = verification.get("split_results")
    train_result = split_results.get("train") if isinstance(split_results, dict) else None
    if not isinstance(train_result, dict):
        raise ValueError("model_preprocessor_train_verification_missing")
    if train_result.get("row_count") != len(frame) or train_result.get(
        "target_hash"
    ) != target_hash:
        raise ValueError("model_preprocessor_train_verification_mismatch")

    X = np.ascontiguousarray(bundle.transform(raw_X), dtype=np.float64)
    if X.ndim != 2 or X.shape != (len(frame), len(bundle.feature_names)):
        raise ValueError("model_training_matrix_shape_mismatch")
    if not np.isfinite(X).all():
        raise ValueError("model_training_matrix_non_finite")
    matrix_hash = _matrix_hash(keys, X, list(bundle.feature_names))
    if train_result.get("transformed_semantic_hash") != matrix_hash:
        raise ValueError("model_training_matrix_hash_mismatch")
    return VerifiedTrainMatrix(
        keys,
        X,
        y,
        bundle.feature_names,
        ledger,
        manifest,
        hashes,
        matrix_hash,
        target_hash,
    )


def _validate_fitted_model(
    model: LogisticRegression, train: VerifiedTrainMatrix, caught: list[warnings.WarningMessage]
) -> dict[str, object]:
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ValueError("model_convergence_warning")
    if not isinstance(model, LogisticRegression):
        raise TypeError("model_type_invalid")
    if not np.array_equal(model.classes_, np.array([NEGATIVE_CLASS, POSITIVE_CLASS])):
        raise ValueError("model_learned_classes_invalid")
    if model.coef_.shape != (1, train.X.shape[1]):
        raise ValueError("model_coefficient_shape_invalid")
    if model.intercept_.shape != (1,):
        raise ValueError("model_intercept_shape_invalid")
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("model_learned_state_non_finite")
    if model.n_iter_.shape != (1,) or int(model.n_iter_[0]) <= 0:
        raise ValueError("model_iteration_count_invalid")
    if int(model.n_iter_[0]) >= int(LOGISTIC_BASELINE_PARAMS["max_iter"]):
        raise ValueError("model_convergence_not_reached")
    if int(model.n_features_in_) != train.X.shape[1]:
        raise ValueError("model_feature_count_mismatch")
    return _model_state(model)


def _validate_training_arrays(train: VerifiedTrainMatrix) -> None:
    if train.X.ndim != 2 or train.X.shape[0] == 0 or train.X.shape[1] == 0:
        raise ValueError("model_train_empty")
    if train.y.ndim != 1 or len(train.y) != len(train.X):
        raise ValueError("model_training_target_shape_mismatch")
    if train.X.dtype != np.float64:
        raise ValueError("model_training_matrix_dtype_invalid")
    if not np.isfinite(train.X).all():
        raise ValueError("model_training_matrix_non_finite")
    if not np.array_equal(np.unique(train.y), np.array([0, 1], dtype=np.int64)):
        if np.unique(train.y).size < 2:
            raise ValueError("model_train_single_class")
        raise ValueError("model_target_domain_invalid")
    if train.X.shape[1] != len(train.feature_names):
        raise ValueError("model_training_feature_count_mismatch")


def _validate_probabilities(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    if probabilities.shape != (len(X), 2):
        raise ValueError("model_probability_shape_invalid")
    if not np.isfinite(probabilities).all():
        raise ValueError("model_probability_non_finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("model_probability_out_of_range")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("model_probability_sum_invalid")
    return probabilities


def _assert_preprocessor_unchanged(
    artifact_dir: Path, expected_hashes: dict[str, str]
) -> None:
    actual = {
        name: _file_hash(path) for name, path in _preprocessing_paths(artifact_dir).items()
    }
    if actual != expected_hashes:
        raise ValueError("model_preprocessing_artifact_mutated")


def train_logistic_baseline(
    data_run_dir: Path | str,
    preprocessing_artifact_dir: Path | str,
    artifacts_dir: Path | str = Path("artifacts"),
    run_id: str | None = None,
) -> LogisticTrainingResult:
    generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = validate_run_id(generated if run_id is None else run_id)
    preprocessing_artifact_dir = Path(preprocessing_artifact_dir)
    parent = Path(artifacts_dir) / "model-runs"
    target = parent / run_id
    if target.exists():
        raise FileExistsError(f"model_run_id_exists:{run_id}")

    train = _load_verified_train_matrix(data_run_dir, preprocessing_artifact_dir)
    _validate_training_arrays(train)
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{run_id}.lock"
    staging = parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    lock_acquired = False
    try:
        with lock_path.open("x"):
            pass
        lock_acquired = True
        if target.exists():
            raise FileExistsError(f"model_run_id_exists:{run_id}")
        staging.mkdir()

        model = LogisticRegression(**LOGISTIC_BASELINE_PARAMS)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(
            limits=NATIVE_THREAD_LIMIT
        ):
            warnings.simplefilter("always")
            model.fit(train.X, train.y)
        fit_duration = time.perf_counter() - started
        state = _validate_fitted_model(model, train, caught)
        probabilities = _validate_probabilities(model, train.X)
        positive_index = int(np.flatnonzero(model.classes_ == POSITIVE_CLASS)[0])
        probability_hash = _prediction_hash(train.keys, probabilities[:, positive_index])

        forbidden = [
            name
            for name in train.feature_names
            if any(name == field or name.endswith(f"__{field}") for field in FORBIDDEN_OUTPUT_COLUMNS)
        ]
        if forbidden:
            raise ValueError(f"model_forbidden_output_features:{','.join(forbidden)}")

        contract_hash = model_contract_hash()
        semantic_version = (
            f"{MODEL_SEMANTIC_VERSION_PREFIX}:{contract_hash[:8]}:"
            f"{train.training_matrix_semantic_hash[:8]}:"
            f"{str(state['learned_state_hash'])[:8]}"
        )
        artifact_path = staging / "model.joblib"
        report_path = staging / "model.training-report.json"
        manifest_path = staging / "model.manifest.json"
        marker_path = staging / "model.success.json"
        joblib.dump(model, artifact_path, compress=3)

        reloaded = joblib.load(artifact_path)
        if not isinstance(reloaded, LogisticRegression):
            raise TypeError("model_serialized_type_invalid")
        reloaded_probabilities = _validate_probabilities(reloaded, train.X)
        if not np.array_equal(probabilities, reloaded_probabilities):
            raise ValueError("model_reload_probability_mismatch")

        class_counts = {
            str(value): int(np.count_nonzero(train.y == value))
            for value in (NEGATIVE_CLASS, POSITIVE_CLASS)
        }
        report = {
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "status": "ok",
            "train_row_count": len(train.y),
            "feature_count": train.X.shape[1],
            "class_counts": class_counts,
            "class_rates": {key: count / len(train.y) for key, count in class_counts.items()},
            "fit_duration_seconds": fit_duration,
            "convergence_status": "converged",
            "iterations": [int(value) for value in model.n_iter_.tolist()],
            "maximum_absolute_coefficient": float(np.max(np.abs(model.coef_))),
            "minimum_train_probability": float(probabilities.min()),
            "maximum_train_probability": float(probabilities.max()),
            "nan_count": int(np.isnan(train.X).sum() + np.isnan(probabilities).sum()),
            "inf_count": int(np.isinf(train.X).sum() + np.isinf(probabilities).sum()),
            "train_probability_semantic_hash": probability_hash,
            "warnings": [f"{item.category.__name__}: {item.message}" for item in caught],
            "errors": [],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))

        ledger = train.snapshot_ledger
        preprocessor = train.preprocessor_manifest
        manifest = {
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_semantic_version": semantic_version,
            "model_contract_hash": contract_hash,
            "model_type": "sklearn.linear_model.LogisticRegression",
            "estimator_parameters": model.get_params(deep=False),
            "target_column": TARGET_COLUMN,
            "positive_class": POSITIVE_CLASS,
            "negative_class": NEGATIVE_CLASS,
            "train_split_rule": {
                presentation: split
                for presentation, split in SPLIT_RULES.items()
                if split == "train"
            },
            "key_columns": list(KEYS),
            "train_row_count": len(train.y),
            "train_class_counts": class_counts,
            "train_class_rates": report["class_rates"],
            "snapshot_version": ledger["snapshot_version"],
            "snapshot_content_hash": ledger["snapshot_content_hash"],
            "snapshot_contract_hash": ledger["contract_hash"],
            "preprocessor_version": preprocessor["preprocessor_version"],
            "preprocessor_contract_hash": preprocessor["preprocessing_contract_hash"],
            "train_feature_semantic_hash": preprocessor["train_feature_semantic_hash"],
            "preprocessor_learned_state_hash": preprocessor["learned_state_hash"],
            "preprocessor_artifact_sha256": preprocessor["artifact_sha256"],
            "output_feature_names": list(train.feature_names),
            "output_feature_count": len(train.feature_names),
            "input_dtype": "float64",
            "training_matrix_semantic_hash": train.training_matrix_semantic_hash,
            "training_target_semantic_hash": train.training_target_semantic_hash,
            "coefficient_semantic_hash": state["coefficient_semantic_hash"],
            "intercept_semantic_hash": state["intercept_semantic_hash"],
            "learned_state_hash": state["learned_state_hash"],
            "learned_classes": state["learned_classes"],
            "n_features_in": state["n_features_in"],
            "n_iter": state["n_iter"],
            "convergence_status": "converged",
            "train_probability_semantic_hash": probability_hash,
            "packages": _model_packages(),
            "hash_algorithm": MODEL_HASH_ALGORITHM,
            "model_artifact_sha256": _file_hash(artifact_path),
            "training_report_sha256": _file_hash(report_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        marker = {
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_semantic_version": semantic_version,
            "manifest_sha256": _file_hash(manifest_path),
            "model_artifact_sha256": manifest["model_artifact_sha256"],
            "training_report_sha256": manifest["training_report_sha256"],
        }
        marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True))

        _assert_preprocessor_unchanged(preprocessing_artifact_dir, train.preprocessing_file_hashes)
        if target.exists():
            raise FileExistsError(f"model_run_id_exists:{run_id}")
        os.replace(staging, target)
        published = tuple(
            target / name
            for name in (
                "model.joblib",
                "model.manifest.json",
                "model.training-report.json",
                "model.success.json",
            )
        )
        return LogisticTrainingResult(
            "ok",
            run_id,
            target,
            semantic_version,
            contract_hash,
            str(state["learned_state_hash"]),
            train.training_matrix_semantic_hash,
            train.training_target_semantic_hash,
            published,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if lock_acquired:
            lock_path.unlink(missing_ok=True)


def load_logistic_model_artifact(
    model_artifact_dir: Path | str,
    data_run_dir: Path | str,
    preprocessing_artifact_dir: Path | str,
) -> LoadedLogisticBaseline:
    model_artifact_dir = Path(model_artifact_dir)
    paths = {
        "artifact": model_artifact_dir / "model.joblib",
        "manifest": model_artifact_dir / "model.manifest.json",
        "report": model_artifact_dir / "model.training-report.json",
        "success": model_artifact_dir / "model.success.json",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"model_artifact_files_missing:{','.join(missing)}")
    hashes = {name: _file_hash(path) for name, path in paths.items()}
    marker = _read_json(paths["success"], "model_success_marker")
    if marker.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("model_success_schema_mismatch")
    if marker.get("manifest_sha256") != hashes["manifest"]:
        raise ValueError("model_manifest_hash_mismatch")
    if marker.get("model_artifact_sha256") != hashes["artifact"]:
        raise ValueError("model_artifact_hash_mismatch")
    if marker.get("training_report_sha256") != hashes["report"]:
        raise ValueError("model_training_report_hash_mismatch")
    manifest = _read_json(paths["manifest"], "model_manifest")
    report = _read_json(paths["report"], "model_training_report")
    if manifest.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("model_manifest_schema_mismatch")
    if manifest.get("model_artifact_sha256") != hashes["artifact"]:
        raise ValueError("model_artifact_hash_mismatch")
    if manifest.get("training_report_sha256") != hashes["report"]:
        raise ValueError("model_training_report_hash_mismatch")
    expected_contract_hash = model_contract_hash()
    if manifest.get("model_contract_hash") != expected_contract_hash:
        raise ValueError("model_contract_hash_mismatch")
    if manifest.get("packages") != _model_packages():
        raise ValueError("model_dependency_mismatch")
    if manifest.get("model_type") != "sklearn.linear_model.LogisticRegression":
        raise ValueError("model_manifest_type_mismatch")
    if manifest.get("estimator_parameters") != _resolved_estimator_params():
        raise ValueError("model_estimator_parameters_mismatch")
    if report.get("status") != "ok" or report.get("convergence_status") != "converged":
        raise ValueError("model_training_report_invalid")

    train = _load_verified_train_matrix(data_run_dir, preprocessing_artifact_dir)
    provenance = {
        "snapshot_version": train.snapshot_ledger["snapshot_version"],
        "snapshot_content_hash": train.snapshot_ledger["snapshot_content_hash"],
        "snapshot_contract_hash": train.snapshot_ledger["contract_hash"],
        "preprocessor_version": train.preprocessor_manifest["preprocessor_version"],
        "preprocessor_contract_hash": train.preprocessor_manifest["preprocessing_contract_hash"],
        "train_feature_semantic_hash": train.preprocessor_manifest["train_feature_semantic_hash"],
        "preprocessor_learned_state_hash": train.preprocessor_manifest["learned_state_hash"],
        "preprocessor_artifact_sha256": train.preprocessor_manifest["artifact_sha256"],
    }
    for field, expected in provenance.items():
        if manifest.get(field) != expected:
            raise ValueError(f"model_{field}_mismatch")
    invariants = {
        "target_column": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "negative_class": NEGATIVE_CLASS,
        "train_split_rule": {
            presentation: split
            for presentation, split in SPLIT_RULES.items()
            if split == "train"
        },
        "key_columns": list(KEYS),
        "output_feature_names": list(train.feature_names),
        "output_feature_count": len(train.feature_names),
        "train_row_count": len(train.y),
        "training_matrix_semantic_hash": train.training_matrix_semantic_hash,
        "training_target_semantic_hash": train.training_target_semantic_hash,
        "input_dtype": "float64",
        "hash_algorithm": MODEL_HASH_ALGORITHM,
    }
    for field, expected in invariants.items():
        if manifest.get(field) != expected:
            raise ValueError(f"model_manifest_{field}_mismatch")

    class_counts = {
        str(value): int(np.count_nonzero(train.y == value))
        for value in (NEGATIVE_CLASS, POSITIVE_CLASS)
    }
    class_rates = {key: count / len(train.y) for key, count in class_counts.items()}
    if manifest.get("train_class_counts") != class_counts or manifest.get(
        "train_class_rates"
    ) != class_rates:
        raise ValueError("model_manifest_train_class_distribution_mismatch")

    model = joblib.load(paths["artifact"])
    if not isinstance(model, LogisticRegression):
        raise TypeError("model_artifact_type_invalid")
    state = _validate_fitted_model(model, train, [])
    for field in (
        "coefficient_semantic_hash",
        "intercept_semantic_hash",
        "learned_state_hash",
        "learned_classes",
        "n_features_in",
        "n_iter",
    ):
        if manifest.get(field) != state.get(field):
            raise ValueError(f"model_{field}_mismatch")
    expected_version = (
        f"{MODEL_SEMANTIC_VERSION_PREFIX}:{expected_contract_hash[:8]}:"
        f"{train.training_matrix_semantic_hash[:8]}:{str(state['learned_state_hash'])[:8]}"
    )
    if manifest.get("model_semantic_version") != expected_version or marker.get(
        "model_semantic_version"
    ) != expected_version:
        raise ValueError("model_semantic_version_mismatch")

    probabilities = _validate_probabilities(model, train.X)
    positive_index = int(np.flatnonzero(model.classes_ == POSITIVE_CLASS)[0])
    probability_hash = _prediction_hash(train.keys, probabilities[:, positive_index])
    if manifest.get("train_probability_semantic_hash") != probability_hash or report.get(
        "train_probability_semantic_hash"
    ) != probability_hash:
        raise ValueError("model_train_probability_hash_mismatch")
    if report.get("train_row_count") != len(train.y) or report.get(
        "feature_count"
    ) != len(train.feature_names):
        raise ValueError("model_training_report_shape_mismatch")
    if report.get("class_counts") != class_counts or report.get("class_rates") != class_rates:
        raise ValueError("model_training_report_class_distribution_mismatch")
    if report.get("iterations") != state["n_iter"]:
        raise ValueError("model_training_report_iterations_mismatch")
    if report.get("nan_count") != 0 or report.get("inf_count") != 0:
        raise ValueError("model_training_report_non_finite_count")
    if not np.isclose(
        float(report.get("minimum_train_probability", np.nan)),
        float(probabilities.min()),
        rtol=0.0,
        atol=0.0,
    ) or not np.isclose(
        float(report.get("maximum_train_probability", np.nan)),
        float(probabilities.max()),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("model_training_report_probability_range_mismatch")

    bundle, _, _, _ = _load_preprocessor_for_training(
        Path(preprocessing_artifact_dir), train.snapshot_ledger
    )
    verification = ModelVerificationReport(
        "ok",
        expected_version,
        expected_contract_hash,
        len(train.y),
        len(train.feature_names),
        tuple(int(value) for value in model.classes_),
        "converged",
        train.training_matrix_semantic_hash,
        train.training_target_semantic_hash,
        probability_hash,
    )
    return LoadedLogisticBaseline(model, bundle, manifest, verification)


def verify_logistic_model_artifact(
    model_artifact_dir: Path | str,
    data_run_dir: Path | str,
    preprocessing_artifact_dir: Path | str,
) -> ModelVerificationReport:
    return load_logistic_model_artifact(
        model_artifact_dir, data_run_dir, preprocessing_artifact_dir
    ).verification
