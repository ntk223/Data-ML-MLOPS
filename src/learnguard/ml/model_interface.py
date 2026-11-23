"""Algorithm-agnostic scoring model interface for M5 batch scoring.

Scoring, export, and database code consume trained models ONLY through the
`ScoringModel` protocol and the `load_scoring_model` factory, so swapping the
training algorithm (e.g. logistic → XGBoost) never touches anything downstream
of the model artifact. Adding an algorithm means adding an adapter in
`adapters.py` and registering it there — this module stays untouched.

The `ScoringModel` contract:

- `version` is the model's semantic version as published in its manifest.
- `feature_names` is the ordered tuple of names of the TRANSFORMED features
  the model consumes (post-preprocessor).
- `transform(X)` maps raw snapshot rows (a pandas DataFrame) to the
  transformed model-input matrix, so one transform can feed both
  probabilities and explain.
- `predict_positive_proba(X)` maps raw snapshot rows to positive-class
  probabilities in [0, 1].
- `explain(feature_row)` takes ONE transformed feature vector (float64,
  `len(feature_row) == len(feature_names)`) and returns at most
  `TOP_SIGNAL_COUNT` `PredictionSignal`s sorted by |contribution| descending
  (stable on ties). Direction is "increases_risk" when the contribution is
  > 0, "decreases_risk" when < 0, and "neutral" when exactly 0. Contributions
  are associations, not causal effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from learnguard.ml.contracts import PredictionSignal

TOP_SIGNAL_COUNT = 5
MODEL_MANIFEST_FILENAME = "model.manifest.json"


@dataclass(frozen=True)
class ExplanationBatch:
    """Vectorized additive explanations in the model's native output space."""

    values: np.ndarray
    expected_value: float
    output_values: np.ndarray
    feature_names: tuple[str, ...]
    method: str
    output_space: str


@runtime_checkable
class ScoringModel(Protocol):
    """Structural interface every scoring-capable model artifact satisfies."""

    version: str
    feature_names: tuple[str, ...]

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Map raw snapshot rows to the transformed model-input matrix."""
        ...

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Positive-class probability for each raw snapshot row of `X`."""
        ...

    def explain(self, feature_row: np.ndarray) -> list[PredictionSignal]:
        """Top signals for ONE transformed row (see module docstring)."""
        ...

    def explain_many(self, feature_matrix: np.ndarray) -> ExplanationBatch:
        """Explain a transformed matrix without rebuilding an explainer per row."""
        ...


def load_scoring_model(model_artifact_dir: Path | str, snapshot_path: Path | str,
                       preprocessing_artifact_dir: Path | str) -> ScoringModel:
    """Load a published model artifact as an algorithm-agnostic `ScoringModel`.

    Reads only `model.manifest.json` — never the serialized estimator — to
    inspect `schema_version`/`model_type` and route to the registered adapter;
    each adapter then fully re-validates the artifact (hashes, provenance,
    learned state) against its own contract. Raises `FileNotFoundError` when
    the manifest is missing and `ValueError("unsupported_model_type:<value>")`
    for unregistered model types.
    """
    manifest_path = Path(model_artifact_dir) / MODEL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model_manifest_missing:{manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model_type = str(manifest.get("model_type"))
    # Imported lazily so this interface module never imports a concrete
    # adapter at load time (adapters.py imports this module's protocol).
    from learnguard.ml.adapters import MODEL_TYPE_BUILDERS
    builder = MODEL_TYPE_BUILDERS.get(model_type)
    if builder is None:
        raise ValueError(f"unsupported_model_type:{model_type}")
    return builder(manifest, model_artifact_dir, snapshot_path, preprocessing_artifact_dir)


def verify_model_artifact(model_artifact_dir: Path | str, snapshot_path: Path | str,
                          preprocessing_artifact_dir: Path | str) -> object:
    """Dispatch integrity/provenance verification using manifest model_type."""
    manifest_path = Path(model_artifact_dir) / MODEL_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model_manifest_missing:{manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model_type = str(manifest.get("model_type"))
    from learnguard.ml.adapters import MODEL_TYPE_VERIFIERS
    verifier = MODEL_TYPE_VERIFIERS.get(model_type)
    if verifier is None:
        raise ValueError(f"unsupported_model_type:{model_type}")
    return verifier(model_artifact_dir, snapshot_path, preprocessing_artifact_dir)
