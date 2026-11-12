"""Concrete `ScoringModel` adapters wrapping algorithm-specific artifacts.

Each adapter translates one trained-algorithm artifact into the algorithm-agnostic `ScoringModel`
protocol from `model_interface.py`, so scoring/export/DB code never sees the
estimator class. Swapping the training algorithm means adding an adapter here
and registering it in `MODEL_TYPE_BUILDERS`; nothing downstream changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np
import pandas as pd

from learnguard.ml.contracts import PredictionSignal
from learnguard.ml.model_interface import ExplanationBatch, TOP_SIGNAL_COUNT
from learnguard.ml.training import (
    LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    SHAP_BACKGROUND_FILENAME,
    LoadedLogisticBaseline,
    load_logistic_model_artifact,
)
from learnguard.ml.xgboost_training import (
    LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    XGBOOST_MODEL_TYPE,
    LoadedXGBoostModel,
    load_xgboost_model_artifact,
    verify_xgboost_model_artifact,
)

if TYPE_CHECKING:
    from learnguard.ml.model_interface import ScoringModel

LOGISTIC_MODEL_TYPE = "sklearn.linear_model.LogisticRegression"


class LogisticScoringModel:
    """`ScoringModel` adapter for the logistic baseline in `training.py`.

    Explain keeps the exact linear-model semantics the batch scorer has always
    used: the contribution of feature i is `coef_[0, i] * x_transformed_i`;
    these are associations, not causal effects.
    """

    def __init__(self, baseline: LoadedLogisticBaseline) -> None:
        self._baseline = baseline

    @property
    def version(self) -> str:
        return str(self._baseline.manifest["model_semantic_version"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._baseline.feature_names

    @property
    def manifest(self) -> Mapping[str, object]:
        return self._baseline.manifest

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Return the transformed model-input matrix for raw snapshot rows."""
        return self._baseline._preprocessor.transform(X)

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability for each raw snapshot row of `X`."""
        return self._baseline.predict_positive_proba(X)

    def explain(self, feature_row: np.ndarray) -> list[PredictionSignal]:
        """Return the top coef-based signals for ONE transformed row.

        `feature_row` must be the post-preprocessing numeric row (what the
        model consumes, e.g. a row of `FittedPreprocessor.transform` output),
        NOT the raw snapshot frame. Returns at most `TOP_SIGNAL_COUNT` signals
        sorted by absolute contribution (descending, stable on ties).
        """
        values = np.asarray(feature_row, dtype=np.float64)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        coefficients = np.asarray(self._baseline._model.coef_, dtype=np.float64)
        if values.ndim != 1 or values.size != coefficients.shape[1]:
            raise ValueError("scoring_explain_row_invalid")
        names = [str(name) for name in self.feature_names]
        if len(names) != values.size:
            raise ValueError("scoring_explain_feature_names_mismatch")
        # Keep this single-row API backward compatible. New scoring uses the
        # vectorized SHAP API below, which applies the train-background shift.
        contributions = coefficients[0] * values
        order = np.argsort(-np.abs(contributions), kind="stable")[:TOP_SIGNAL_COUNT]
        return [
            PredictionSignal(
                feature_name=names[index],
                contribution=float(contributions[index]),
                direction="increases_risk" if contributions[index] > 0 else "decreases_risk",
            )
            for index in order
        ]

    def explain_many(self, feature_matrix: np.ndarray) -> ExplanationBatch:
        matrix = np.ascontiguousarray(feature_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("scoring_explain_matrix_invalid")
        schema = str(self._baseline.manifest["schema_version"])
        model = self._baseline._model
        if schema == MODEL_ARTIFACT_SCHEMA_VERSION:
            root = self._baseline.manifest.get("artifact_dir")
            if not isinstance(root, str):
                raise ValueError("model_artifact_dir_missing")
            with np.load(Path(root) / SHAP_BACKGROUND_FILENAME) as data:
                mean = np.asarray(data["mean"], dtype=np.float64)
                covariance = np.asarray(data["covariance"], dtype=np.float64)
            try:
                import shap
            except ModuleNotFoundError:
                # Exact interventional Linear SHAP identity. This is not the
                # legacy coef*x contribution: the train background is removed.
                values = (matrix - mean) * np.asarray(model.coef_[0], dtype=np.float64)
                expected = float(model.intercept_[0] + np.dot(mean, model.coef_[0]))
                method = "linear-shap-interventional-exact-v1"
            else:
                explainer = shap.LinearExplainer(
                    model, (mean, covariance), feature_perturbation="interventional",
                )
                values = np.asarray(explainer.shap_values(matrix), dtype=np.float64)
                expected = float(explainer.expected_value)
                method = "linear-shap-interventional-v1"
        else:
            values = matrix * np.asarray(model.coef_[0], dtype=np.float64)
            expected = float(model.intercept_[0])
            method = "legacy-linear-contribution"
        outputs = np.asarray(model.decision_function(matrix), dtype=np.float64)
        return ExplanationBatch(
            values=values,
            expected_value=expected,
            output_values=outputs,
            feature_names=self.feature_names,
            method=method,
            output_space="log_odds",
        )


class XGBoostScoringModel:
    """Scoring and train-background interventional Tree SHAP adapter."""

    def __init__(self, bundle: LoadedXGBoostModel) -> None:
        self._bundle = bundle
        self._explainer: object | None = None

    @property
    def version(self) -> str:
        return str(self._bundle.manifest["model_semantic_version"])

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._bundle.feature_names

    @property
    def manifest(self) -> Mapping[str, object]:
        return self._bundle.manifest

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._bundle._preprocessor.transform(X)

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._bundle.predict_positive_proba(X)

    def _tree_explainer(self) -> object:
        if self._explainer is None:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/learnguard-matplotlib")
            import shap

            root = self._bundle.manifest.get("artifact_dir")
            if not isinstance(root, str):
                raise ValueError("model_artifact_dir_missing")
            with np.load(Path(root) / SHAP_BACKGROUND_FILENAME) as data:
                background = np.asarray(data["values"], dtype=np.float64)
            self._explainer = shap.TreeExplainer(
                self._bundle._model,
                background,
                feature_perturbation="interventional",
                model_output="raw",
                feature_names=list(self.feature_names),
            )
        return self._explainer

    def explain_many(self, feature_matrix: np.ndarray) -> ExplanationBatch:
        matrix = np.ascontiguousarray(feature_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("scoring_explain_matrix_invalid")
        explainer = self._tree_explainer()
        explanation = explainer(matrix, check_additivity=False)  # type: ignore[operator]
        values = np.asarray(explanation.values, dtype=np.float64)
        base_values = np.asarray(explanation.base_values, dtype=np.float64)
        if values.shape != matrix.shape or not np.isfinite(values).all():
            raise ValueError("xgboost_shap_values_invalid")
        if base_values.ndim == 0:
            expected = float(base_values)
        elif base_values.shape == (len(matrix),) and np.allclose(
            base_values, base_values[0], atol=1e-12, rtol=0,
        ):
            expected = float(base_values[0])
        else:
            raise ValueError("xgboost_shap_expected_value_invalid")
        outputs = np.asarray(
            self._bundle._model.predict(matrix, output_margin=True),
            dtype=np.float64,
        )
        if outputs.shape != (len(matrix),) or not np.isfinite(outputs).all():
            raise ValueError("xgboost_margin_invalid")
        return ExplanationBatch(
            values=values,
            expected_value=expected,
            output_values=outputs,
            feature_names=self.feature_names,
            method="tree-shap-interventional-v1",
            output_space="log_odds",
        )

    def explain(self, feature_row: np.ndarray) -> list[PredictionSignal]:
        values = np.asarray(feature_row, dtype=np.float64)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1 or values.size != len(self.feature_names):
            raise ValueError("scoring_explain_row_invalid")
        contributions = self.explain_many(values.reshape(1, -1)).values[0]
        order = np.argsort(-np.abs(contributions), kind="stable")[:TOP_SIGNAL_COUNT]
        return [
            PredictionSignal(
                feature_name=self.feature_names[index],
                contribution=float(contributions[index]),
                direction=("increases_risk" if contributions[index] > 0 else
                           "decreases_risk" if contributions[index] < 0 else "neutral"),
            )
            for index in order
        ]


def load_logistic_scoring_model(manifest: dict[str, object], model_artifact_dir: Path | str,
                                snapshot_path: Path | str,
                                preprocessing_artifact_dir: Path | str) -> LogisticScoringModel:
    """Route entry for logistic artifacts: validate schema, load, and wrap."""
    if manifest.get("schema_version") not in {
        LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION, MODEL_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported_model_schema_version:{manifest.get('schema_version')}")
    baseline = load_logistic_model_artifact(
        model_artifact_dir, snapshot_path, preprocessing_artifact_dir,
    )
    baseline._manifest["artifact_dir"] = str(Path(model_artifact_dir).resolve())
    return LogisticScoringModel(baseline)


def load_xgboost_scoring_model(
    manifest: dict[str, object], model_artifact_dir: Path | str,
    snapshot_path: Path | str, preprocessing_artifact_dir: Path | str,
) -> XGBoostScoringModel:
    if manifest.get("schema_version") not in {
        LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
        XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported_model_schema_version:{manifest.get('schema_version')}")
    bundle = load_xgboost_model_artifact(
        model_artifact_dir, snapshot_path, preprocessing_artifact_dir,
    )
    bundle._manifest["artifact_dir"] = str(Path(model_artifact_dir).resolve())
    return XGBoostScoringModel(bundle)


MODEL_TYPE_BUILDERS: dict[str, Callable[[dict[str, object], Path | str, Path | str, Path | str], ScoringModel]] = {
    LOGISTIC_MODEL_TYPE: load_logistic_scoring_model,
    XGBOOST_MODEL_TYPE: load_xgboost_scoring_model,
}

MODEL_TYPE_VERIFIERS: dict[str, Callable[[Path | str, Path | str, Path | str], object]] = {
    LOGISTIC_MODEL_TYPE: lambda model, snapshot, preprocessor: load_logistic_model_artifact(
        model, snapshot, preprocessor,
    ).verification,
    XGBOOST_MODEL_TYPE: verify_xgboost_model_artifact,
}

SUPPORTED_MODEL_SCHEMAS = {
    LEGACY_MODEL_ARTIFACT_SCHEMA_VERSION,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    LEGACY_XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
    XGBOOST_MODEL_ARTIFACT_SCHEMA_VERSION,
}
