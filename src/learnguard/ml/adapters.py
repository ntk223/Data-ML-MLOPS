"""Concrete `ScoringModel` adapters wrapping algorithm-specific artifacts.

Each adapter translates one trained-algorithm artifact — today the logistic
baseline from `training.py` — into the algorithm-agnostic `ScoringModel`
protocol from `model_interface.py`, so scoring/export/DB code never sees the
estimator class. Swapping the training algorithm means adding an adapter here
and registering it in `MODEL_TYPE_BUILDERS`; nothing downstream changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np
import pandas as pd

from learnguard.ml.contracts import PredictionSignal
from learnguard.ml.model_interface import TOP_SIGNAL_COUNT
from learnguard.ml.training import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    LoadedLogisticBaseline,
    load_logistic_model_artifact,
)

if TYPE_CHECKING:
    from pathlib import Path

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


def load_logistic_scoring_model(manifest: dict[str, object], model_artifact_dir: Path | str,
                                snapshot_path: Path | str,
                                preprocessing_artifact_dir: Path | str) -> LogisticScoringModel:
    """Route entry for logistic artifacts: validate schema, load, and wrap."""
    if manifest.get("schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported_model_schema_version:{manifest.get('schema_version')}")
    return LogisticScoringModel(
        load_logistic_model_artifact(model_artifact_dir, snapshot_path, preprocessing_artifact_dir)
    )


MODEL_TYPE_BUILDERS: dict[str, Callable[[dict[str, object], Path | str, Path | str, Path | str], ScoringModel]] = {
    LOGISTIC_MODEL_TYPE: load_logistic_scoring_model,
}
