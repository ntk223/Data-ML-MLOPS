"""Fixed Logistic Regression baseline layered on the published data contract."""

from __future__ import annotations

import json
import os
import platform
import re
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from learnguard.data.features import KEYS, SPLIT_RULES
from learnguard.data.loading import TARGET_COLUMN, load_snapshot_splits
from learnguard.ml.preprocessing import (
    FORBIDDEN_OUTPUT_COLUMNS,
    PREPROCESSING_INPUT_COLUMNS,
    FittedPreprocessor,
    _file_hash,
    _matrix_hash,
    _prepare_split,
    _series_hash,
    load_preprocessing_artifact,
)

MODEL_ARTIFACT_SCHEMA_VERSION = "logistic-model-v1"
MODEL_SEMANTIC_VERSION_PREFIX = "logreg-v1"
POSITIVE_CLASS, NEGATIVE_CLASS = 1, 0
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

LOGISTIC_BASELINE_PARAMS: dict[str, object] = {
    "penalty": "l2", "C": 1.0, "solver": "lbfgs", "class_weight": None,
    "max_iter": 2000, "tol": 1e-4, "fit_intercept": True,
    "random_state": 42, "n_jobs": None,
}


@dataclass(frozen=True)
class VerifiedTrainMatrix:
    keys: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    snapshot_version: str
    preprocessor_manifest: dict[str, object]
    matrix_hash: str
    target_hash: str


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
    _model: LogisticRegression
    _preprocessor: FittedPreprocessor
    _manifest: dict[str, object]
    verification: ModelVerificationReport

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self._manifest["output_feature_names"])

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(self._manifest)

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        matrix = self._preprocessor.transform(X)
        return _probabilities(self._model, matrix)[:, 1].copy()


def _packages() -> dict[str, str]:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }


def _canonical_hash(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _resolved_estimator_params() -> dict[str, object]:
    return LogisticRegression(**LOGISTIC_BASELINE_PARAMS).get_params(deep=False)


def model_contract_hash() -> str:
    return _canonical_hash({
        "schema": MODEL_ARTIFACT_SCHEMA_VERSION,
        "model": "sklearn.linear_model.LogisticRegression",
        "params": _resolved_estimator_params(),
        "target": TARGET_COLUMN,
        "classes": [0, 1],
        "split_rules": SPLIT_RULES,
        "input_dtype": "float64",
        "native_threads": 1,
        "packages": _packages(),
    })


def _array_hash(name: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    digest = sha256(f"{name}:{array.shape}:{array.dtype}".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _prediction_hash(keys: pd.DataFrame, values: np.ndarray) -> str:
    digest = sha256(pd.util.hash_pandas_object(keys, index=False, categorize=True).to_numpy().tobytes())
    digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _state(model: LogisticRegression) -> dict[str, object]:
    value = {
        "coefficient_semantic_hash": _array_hash("coef", model.coef_),
        "intercept_semantic_hash": _array_hash("intercept", model.intercept_),
        "learned_classes": [int(x) for x in model.classes_],
        "n_features_in": int(model.n_features_in_),
        "n_iter": [int(x) for x in model.n_iter_],
    }
    return {**value, "learned_state_hash": _canonical_hash(value)}


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid_run_id")
    return run_id


def _verified_train(snapshot_path: Path | str, preprocessor_dir: Path | str) -> VerifiedTrainMatrix:
    snapshot_path = Path(snapshot_path)
    splits = load_snapshot_splits(snapshot_path)
    prepared = _prepare_split("train", splits.train)
    bundle = load_preprocessing_artifact(preprocessor_dir, snapshot_path)
    X = np.ascontiguousarray(bundle.transform(prepared.X), dtype=np.float64)
    y = prepared.y.to_numpy(dtype=np.int64)
    if X.size == 0 or X.dtype != np.float64 or not np.isfinite(X).all():
        raise ValueError("model_training_matrix_invalid")
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        raise ValueError("model_train_requires_classes_0_1")
    names = bundle.feature_names
    matrix_hash = _matrix_hash(prepared.keys, X, list(names))
    verification = json.loads((Path(preprocessor_dir) / "preprocessor.verification.json").read_text())
    if verification["split_results"]["train"]["transformed_semantic_hash"] != matrix_hash:
        raise ValueError("model_training_matrix_hash_mismatch")
    return VerifiedTrainMatrix(
        prepared.keys, X, y, names, splits.snapshot_version, bundle.manifest,
        matrix_hash, _series_hash(prepared.keys, prepared.y),
    )


def _validate_model(model: LogisticRegression, train: VerifiedTrainMatrix, caught: list[warnings.WarningMessage]) -> dict[str, object]:
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise ValueError("model_convergence_warning")
    if not np.array_equal(model.classes_, np.array([0, 1])):
        raise ValueError("model_classes_invalid")
    if model.coef_.shape != (1, train.X.shape[1]) or model.intercept_.shape != (1,):
        raise ValueError("model_learned_shape_invalid")
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("model_learned_state_non_finite")
    if int(model.n_iter_[0]) <= 0 or int(model.n_iter_[0]) >= int(LOGISTIC_BASELINE_PARAMS["max_iter"]):
        raise ValueError("model_convergence_not_reached")
    return _state(model)


def _probabilities(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict_proba(X), dtype=np.float64)
    if values.shape != (len(X), 2) or not np.isfinite(values).all():
        raise ValueError("model_probability_invalid")
    if ((values < 0) | (values > 1)).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("model_probability_domain_invalid")
    return values


def train_logistic_baseline(snapshot_path: Path | str, preprocessing_artifact_dir: Path | str, artifacts_dir: Path | str = Path("artifacts"), run_id: str = "logistic") -> LogisticTrainingResult:
    run_id = _validate_run_id(run_id)
    target = Path(artifacts_dir) / "model-runs" / run_id
    if target.exists():
        raise FileExistsError(f"model_run_id_exists:{run_id}")
    train = _verified_train(snapshot_path, preprocessing_artifact_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        model = LogisticRegression(**LOGISTIC_BASELINE_PARAMS)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
            warnings.simplefilter("always")
            model.fit(train.X, train.y)
        state = _validate_model(model, train, caught)
        probabilities = _probabilities(model, train.X)
        probability_hash = _prediction_hash(train.keys, probabilities[:, 1])
        contract_hash = model_contract_hash()
        version = f"{MODEL_SEMANTIC_VERSION_PREFIX}:{contract_hash[:8]}:{train.matrix_hash[:8]}:{str(state['learned_state_hash'])[:8]}"
        artifact = staging / "model.joblib"
        report_path = staging / "model.training-report.json"
        manifest_path = staging / "model.manifest.json"
        marker_path = staging / "model.success.json"
        joblib.dump(model, artifact, compress=3)
        reloaded = joblib.load(artifact)
        if not np.array_equal(probabilities, _probabilities(reloaded, train.X)):
            raise ValueError("model_reload_probability_mismatch")
        counts = {str(x): int(np.count_nonzero(train.y == x)) for x in (0, 1)}
        report = {
            "status": "ok", "train_row_count": len(train.y),
            "feature_count": train.X.shape[1], "class_counts": counts,
            "class_rates": {k: v / len(train.y) for k, v in counts.items()},
            "fit_duration_seconds": time.perf_counter() - started,
            "convergence_status": "converged", "iterations": state["n_iter"],
            "maximum_absolute_coefficient": float(np.abs(model.coef_).max()),
            "minimum_train_probability": float(probabilities.min()),
            "maximum_train_probability": float(probabilities.max()),
            "nan_count": 0, "inf_count": 0,
            "train_probability_semantic_hash": probability_hash,
            "warnings": [f"{x.category.__name__}: {x.message}" for x in caught], "errors": [],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        manifest = {
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_semantic_version": version, "model_contract_hash": contract_hash,
            "model_type": "sklearn.linear_model.LogisticRegression",
            "estimator_parameters": model.get_params(deep=False),
            "target_column": TARGET_COLUMN, "positive_class": 1, "negative_class": 0,
            "train_split_rule": {k: v for k, v in SPLIT_RULES.items() if v == "train"},
            "train_row_count": len(train.y), "train_class_counts": counts,
            "snapshot_version": train.snapshot_version,
            "preprocessor_version": train.preprocessor_manifest["preprocessor_version"],
            "preprocessor_contract_hash": train.preprocessor_manifest["preprocessing_contract_hash"],
            "preprocessor_learned_state_hash": train.preprocessor_manifest["learned_state_hash"],
            "output_feature_names": list(train.feature_names), "output_feature_count": len(train.feature_names),
            "training_matrix_semantic_hash": train.matrix_hash,
            "training_target_semantic_hash": train.target_hash,
            **state, "convergence_status": "converged",
            "train_probability_semantic_hash": probability_hash,
            "packages": _packages(), "model_artifact_sha256": _file_hash(artifact),
            "training_report_sha256": _file_hash(report_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        marker_path.write_text(json.dumps({
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_semantic_version": version,
            "manifest_sha256": _file_hash(manifest_path),
            "model_artifact_sha256": manifest["model_artifact_sha256"],
            "training_report_sha256": manifest["training_report_sha256"],
        }, indent=2, sort_keys=True))
        if target.exists():
            raise FileExistsError(f"model_run_id_exists:{run_id}")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    paths = tuple(target / name for name in ("model.joblib", "model.manifest.json", "model.training-report.json", "model.success.json"))
    return LogisticTrainingResult("ok", run_id, target, version, contract_hash, str(state["learned_state_hash"]), train.matrix_hash, train.target_hash, paths)


def load_logistic_model_artifact(model_artifact_dir: Path | str, snapshot_path: Path | str, preprocessing_artifact_dir: Path | str) -> LoadedLogisticBaseline:
    root = Path(model_artifact_dir)
    artifact, manifest_path = root / "model.joblib", root / "model.manifest.json"
    report_path, marker_path = root / "model.training-report.json", root / "model.success.json"
    if any(not p.is_file() for p in (artifact, manifest_path, report_path, marker_path)):
        raise FileNotFoundError("model_artifact_missing")
    marker = json.loads(marker_path.read_text())
    if marker.get("manifest_sha256") != _file_hash(manifest_path):
        raise ValueError("model_manifest_hash_mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_contract_hash") != model_contract_hash() or manifest.get("packages") != _packages():
        raise ValueError("model_contract_or_dependency_mismatch")
    if manifest.get("model_artifact_sha256") != _file_hash(artifact) or marker.get("model_artifact_sha256") != _file_hash(artifact):
        raise ValueError("model_artifact_hash_mismatch")
    if manifest.get("training_report_sha256") != _file_hash(report_path) or marker.get("training_report_sha256") != _file_hash(report_path):
        raise ValueError("model_report_hash_mismatch")
    train = _verified_train(snapshot_path, preprocessing_artifact_dir)
    expected = {
        "snapshot_version": train.snapshot_version,
        "preprocessor_version": train.preprocessor_manifest["preprocessor_version"],
        "output_feature_names": list(train.feature_names),
        "training_matrix_semantic_hash": train.matrix_hash,
        "training_target_semantic_hash": train.target_hash,
    }
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("model_provenance_mismatch")
    model = joblib.load(artifact)
    state = _validate_model(model, train, [])
    if manifest.get("learned_state_hash") != state["learned_state_hash"]:
        raise ValueError("model_state_mismatch")
    probabilities = _probabilities(model, train.X)
    probability_hash = _prediction_hash(train.keys, probabilities[:, 1])
    if manifest.get("train_probability_semantic_hash") != probability_hash:
        raise ValueError("model_probability_hash_mismatch")
    bundle = load_preprocessing_artifact(preprocessing_artifact_dir, snapshot_path)
    verification = ModelVerificationReport(
        "ok", str(manifest["model_semantic_version"]), str(manifest["model_contract_hash"]),
        len(train.y), len(train.feature_names), tuple(int(x) for x in model.classes_),
        "converged", train.matrix_hash, train.target_hash, probability_hash,
    )
    return LoadedLogisticBaseline(model, bundle, manifest, verification)


def verify_logistic_model_artifact(model_artifact_dir: Path | str, snapshot_path: Path | str, preprocessing_artifact_dir: Path | str) -> ModelVerificationReport:
    return load_logistic_model_artifact(model_artifact_dir, snapshot_path, preprocessing_artifact_dir).verification
