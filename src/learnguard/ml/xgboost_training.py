"""Deterministic, train-only XGBoost artifact for LearnGuard."""

from __future__ import annotations

import json
import os
import platform
import shutil
import time
import uuid
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
import xgboost
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from learnguard.data.features import SPLIT_RULES
from learnguard.data.loading import TARGET_COLUMN
from learnguard.ml.preprocessing import _file_hash, load_preprocessing_artifact
from learnguard.ml.json_utils import json_safe, strict_json_dumps
from learnguard.ml.training import (
    SHAP_BACKGROUND_FILENAME,
    VerifiedTrainMatrix,
    _prediction_hash,
    _validate_run_id,
    _verified_train,
)

LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION = "xgboost-model-v1"
XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION = "xgboost-model-v2"
XGBOOST_MODEL_SEMANTIC_VERSION_PREFIX = "xgb-v2"
XGBOOST_MODEL_TYPE = "xgboost.sklearn.XGBClassifier"
XGBOOST_MODEL_FILENAME = "model.ubj"
XGBOOST_SHAP_BACKGROUND_ROWS = 128

LEGACY_XGBOOST_BASELINE_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0,
}

# This is the single production estimator contract.  Benchmark selection and
# train-xgboost both import this mapping so they cannot silently diverge.
XGBOOST_PRODUCTION_CANDIDATE = "xgboost_regularized"
XGBOOST_PRODUCTION_PARAMS: dict[str, object] = {
    **LEGACY_XGBOOST_BASELINE_PARAMS,
    "n_estimators": 400,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.05,
    "reg_lambda": 5.0,
}

# Kept as a source-compatible alias for callers that imported the old name.
XGBOOST_BASELINE_PARAMS = LEGACY_XGBOOST_BASELINE_PARAMS


@dataclass(frozen=True)
class XGBoostTrainingResult:
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
class XGBoostVerificationReport:
    status: str
    model_semantic_version: str
    model_contract_hash: str
    train_row_count: int
    output_feature_count: int
    learned_classes: tuple[int, ...]
    training_matrix_semantic_hash: str
    training_target_semantic_hash: str
    train_probability_semantic_hash: str


@dataclass(frozen=True)
class LoadedXGBoostModel:
    _model: XGBClassifier
    _preprocessor: object
    _manifest: dict[str, object]
    verification: XGBoostVerificationReport

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._manifest["output_feature_names"])

    @property
    def manifest(self) -> Mapping[str, object]:
        return MappingProxyType(self._manifest)

    def predict_positive_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._preprocessor.transform(frame)
        return _probabilities(self._model, matrix)[:, 1].copy()


def _packages() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "xgboost": xgboost.__version__,
    }


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode()).hexdigest()


def _resolved_estimator_params(
    params: Mapping[str, object] = XGBOOST_PRODUCTION_PARAMS,
) -> dict[str, object]:
    return XGBClassifier(**params).get_params(deep=False)


def xgboost_model_contract_hash(
    *,
    schema: str = XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    params: Mapping[str, object] = XGBOOST_PRODUCTION_PARAMS,
) -> str:
    return _canonical_hash({
        "schema": schema,
        "model": XGBOOST_MODEL_TYPE,
        "params": _resolved_estimator_params(params),
        "target": TARGET_COLUMN,
        "classes": [0, 1],
        "split_rules": SPLIT_RULES,
        "input_dtype": "float64",
        "native_threads": 1,
        "payload_format": "ubjson",
        "explanation": "tree-shap-interventional-raw-log-odds-v1",
        "packages": _packages(),
    })


def _probabilities(model: XGBClassifier, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    if values.shape != (len(matrix), 2) or not np.isfinite(values).all():
        raise ValueError("xgboost_probability_invalid")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("xgboost_probability_domain_invalid")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-7, rtol=0):
        raise ValueError("xgboost_probability_sum_invalid")
    return values


def _state(model: XGBClassifier, train: VerifiedTrainMatrix) -> dict[str, object]:
    if not np.array_equal(np.asarray(model.classes_), np.array([0, 1])):
        raise ValueError("xgboost_classes_invalid")
    booster = model.get_booster()
    if booster.num_features() != train.X.shape[1]:
        raise ValueError("xgboost_learned_shape_invalid")
    raw = bytes(booster.save_raw(raw_format="ubj"))
    config = json.loads(booster.save_config())
    return {
        "learned_state_hash": sha256(raw).hexdigest(),
        "booster_config_semantic_hash": _canonical_hash(config),
        "learned_classes": [0, 1],
        "n_features_in": int(booster.num_features()),
        "boosted_rounds": int(booster.num_boosted_rounds()),
    }


def _background(matrix: np.ndarray) -> np.ndarray:
    count = min(XGBOOST_SHAP_BACKGROUND_ROWS, len(matrix))
    indices = np.linspace(0, len(matrix) - 1, num=count, dtype=np.int64)
    return np.ascontiguousarray(matrix[indices], dtype=np.float64)


def train_xgboost_model(
    snapshot_path: Path | str,
    preprocessing_artifact_dir: Path | str,
    artifacts_dir: Path | str = Path("artifacts"),
    run_id: str = "xgboost",
) -> XGBoostTrainingResult:
    """Fit the fixed CPU XGBoost contract using the train partition only."""
    run_id = _validate_run_id(run_id)
    target = Path(artifacts_dir) / "model-runs" / run_id
    if target.exists():
        raise FileExistsError(f"model_run_id_exists:{run_id}")
    train = _verified_train(snapshot_path, preprocessing_artifact_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        model = XGBClassifier(**XGBOOST_PRODUCTION_PARAMS)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(train.X, train.y)
        fit_seconds = time.perf_counter() - started
        state = _state(model, train)
        probabilities = _probabilities(model, train.X)
        probability_hash = _prediction_hash(train.keys, probabilities[:, 1])
        contract_hash = xgboost_model_contract_hash()
        version = (
            f"{XGBOOST_MODEL_SEMANTIC_VERSION_PREFIX}:{contract_hash[:8]}:"
            f"{train.matrix_hash[:8]}:{str(state['learned_state_hash'])[:8]}"
        )
        artifact = staging / XGBOOST_MODEL_FILENAME
        report_path = staging / "model.training-report.json"
        manifest_path = staging / "model.manifest.json"
        marker_path = staging / "model.success.json"
        background_path = staging / SHAP_BACKGROUND_FILENAME
        model.save_model(artifact)
        np.savez_compressed(background_path, values=_background(train.X))

        reloaded = XGBClassifier(**XGBOOST_PRODUCTION_PARAMS)
        reloaded.load_model(artifact)
        if not np.array_equal(probabilities, _probabilities(reloaded, train.X)):
            raise ValueError("xgboost_reload_probability_mismatch")
        counts = {str(value): int(np.count_nonzero(train.y == value)) for value in (0, 1)}
        report = {
            "status": "ok",
            "train_row_count": len(train.y),
            "feature_count": train.X.shape[1],
            "class_counts": counts,
            "class_rates": {key: value / len(train.y) for key, value in counts.items()},
            "negative_positive_ratio": counts["0"] / counts["1"],
            "class_weight_policy": "none_mild_imbalance",
            "fit_duration_seconds": fit_seconds,
            "boosted_rounds": state["boosted_rounds"],
            "minimum_train_probability": float(probabilities[:, 1].min()),
            "maximum_train_probability": float(probabilities[:, 1].max()),
            "nan_count": 0,
            "inf_count": 0,
            "train_probability_semantic_hash": probability_hash,
            "warnings": [],
            "errors": [],
        }
        report_path.write_text(strict_json_dumps(report, indent=2, sort_keys=True))
        manifest = {
            "schema_version": XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_semantic_version": version,
            "model_contract_hash": contract_hash,
            "model_type": XGBOOST_MODEL_TYPE,
            "model_payload_filename": XGBOOST_MODEL_FILENAME,
            "model_payload_format": "ubjson",
            "estimator_parameters": json_safe(model.get_params(deep=False)),
            "production_candidate": XGBOOST_PRODUCTION_CANDIDATE,
            "objective": XGBOOST_PRODUCTION_PARAMS["objective"],
            "eval_metric": XGBOOST_PRODUCTION_PARAMS["eval_metric"],
            "target_column": TARGET_COLUMN,
            "positive_class": 1,
            "negative_class": 0,
            "train_split_rule": {key: value for key, value in SPLIT_RULES.items() if value == "train"},
            "train_row_count": len(train.y),
            "train_class_counts": counts,
            "class_weight_policy": "none_mild_imbalance",
            "snapshot_version": train.snapshot_version,
            "preprocessor_version": train.preprocessor_manifest["preprocessor_version"],
            "preprocessor_contract_hash": train.preprocessor_manifest["preprocessing_contract_hash"],
            "preprocessor_learned_state_hash": train.preprocessor_manifest["learned_state_hash"],
            "output_feature_names": list(train.feature_names),
            "output_feature_count": len(train.feature_names),
            "training_matrix_semantic_hash": train.matrix_hash,
            "training_target_semantic_hash": train.target_hash,
            **state,
            "train_probability_semantic_hash": probability_hash,
            "packages": _packages(),
            "model_artifact_sha256": _file_hash(artifact),
            "training_report_sha256": _file_hash(report_path),
            "explanation": {
                "method": "tree-shap-interventional-v1",
                "output_space": "log_odds",
                "model_output": "raw",
                "background_split": "train",
                "background_selection": "sorted_evenly_spaced_v1",
                "background_row_count": min(XGBOOST_SHAP_BACKGROUND_ROWS, len(train.X)),
                "background_filename": SHAP_BACKGROUND_FILENAME,
                "background_sha256": _file_hash(background_path),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(strict_json_dumps(manifest, indent=2, sort_keys=True))
        marker_path.write_text(strict_json_dumps({
            "schema_version": XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
            "model_semantic_version": version,
            "manifest_sha256": _file_hash(manifest_path),
            "model_artifact_sha256": manifest["model_artifact_sha256"],
            "training_report_sha256": manifest["training_report_sha256"],
            "background_sha256": manifest["explanation"]["background_sha256"],
        }, indent=2, sort_keys=True))
        if target.exists():
            raise FileExistsError(f"model_run_id_exists:{run_id}")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    paths = tuple(target / name for name in (
        XGBOOST_MODEL_FILENAME,
        SHAP_BACKGROUND_FILENAME,
        "model.manifest.json",
        "model.training-report.json",
        "model.success.json",
    ))
    return XGBoostTrainingResult(
        "ok", run_id, target, version, contract_hash,
        str(state["learned_state_hash"]), train.matrix_hash, train.target_hash, paths,
    )


def load_xgboost_model_artifact(
    model_artifact_dir: Path | str,
    snapshot_path: Path | str,
    preprocessing_artifact_dir: Path | str,
) -> LoadedXGBoostModel:
    root = Path(model_artifact_dir)
    artifact = root / XGBOOST_MODEL_FILENAME
    manifest_path = root / "model.manifest.json"
    report_path = root / "model.training-report.json"
    marker_path = root / "model.success.json"
    background_path = root / SHAP_BACKGROUND_FILENAME
    if any(not path.is_file() for path in (
        artifact, manifest_path, report_path, marker_path, background_path,
    )):
        raise FileNotFoundError("xgboost_model_artifact_missing")
    marker = json.loads(marker_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if marker.get("manifest_sha256") != _file_hash(manifest_path):
        raise ValueError("xgboost_manifest_hash_mismatch")
    schema = manifest.get("schema_version")
    if schema not in {
        LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
        XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported_model_schema_version:{manifest.get('schema_version')}")
    if manifest.get("model_type") != XGBOOST_MODEL_TYPE:
        raise ValueError("xgboost_model_type_mismatch")
    params = (
        LEGACY_XGBOOST_BASELINE_PARAMS
        if schema == LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION
        else XGBOOST_PRODUCTION_PARAMS
    )
    expected_contract = xgboost_model_contract_hash(schema=str(schema), params=params)
    if manifest.get("model_contract_hash") != expected_contract:
        raise ValueError("xgboost_model_contract_mismatch")
    if manifest.get("packages") != _packages():
        raise ValueError("xgboost_dependency_mismatch")
    checks = (
        (manifest.get("model_artifact_sha256"), _file_hash(artifact), "xgboost_artifact_hash_mismatch"),
        (marker.get("model_artifact_sha256"), _file_hash(artifact), "xgboost_artifact_hash_mismatch"),
        (manifest.get("training_report_sha256"), _file_hash(report_path), "xgboost_report_hash_mismatch"),
        (marker.get("training_report_sha256"), _file_hash(report_path), "xgboost_report_hash_mismatch"),
        (manifest.get("explanation", {}).get("background_sha256"), _file_hash(background_path), "xgboost_background_hash_mismatch"),
        (marker.get("background_sha256"), _file_hash(background_path), "xgboost_background_hash_mismatch"),
    )
    for expected, actual, error in checks:
        if expected != actual:
            raise ValueError(error)
    train = _verified_train(snapshot_path, preprocessing_artifact_dir)
    expected_provenance = {
        "snapshot_version": train.snapshot_version,
        "preprocessor_version": train.preprocessor_manifest["preprocessor_version"],
        "output_feature_names": list(train.feature_names),
        "training_matrix_semantic_hash": train.matrix_hash,
        "training_target_semantic_hash": train.target_hash,
    }
    if any(manifest.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("xgboost_model_provenance_mismatch")
    model = XGBClassifier(**params)
    model.load_model(artifact)
    state = _state(model, train)
    if manifest.get("learned_state_hash") != state["learned_state_hash"]:
        raise ValueError("xgboost_model_state_mismatch")
    probabilities = _probabilities(model, train.X)
    probability_hash = _prediction_hash(train.keys, probabilities[:, 1])
    if manifest.get("train_probability_semantic_hash") != probability_hash:
        raise ValueError("xgboost_probability_hash_mismatch")
    preprocessor = load_preprocessing_artifact(preprocessing_artifact_dir, snapshot_path)
    verification = XGBoostVerificationReport(
        "ok", str(manifest["model_semantic_version"]),
        str(manifest["model_contract_hash"]), len(train.y), len(train.feature_names),
        tuple(int(value) for value in model.classes_), train.matrix_hash,
        train.target_hash, probability_hash,
    )
    return LoadedXGBoostModel(model, preprocessor, manifest, verification)


def verify_xgboost_model_artifact(
    model_artifact_dir: Path | str,
    snapshot_path: Path | str,
    preprocessing_artifact_dir: Path | str,
) -> XGBoostVerificationReport:
    return load_xgboost_model_artifact(
        model_artifact_dir, snapshot_path, preprocessing_artifact_dir,
    ).verification

