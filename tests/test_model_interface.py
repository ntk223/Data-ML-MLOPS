"""M5 ScoringModel interface tests: factory routing, adapter conformance, decoupling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learnguard.ml import scoring
from learnguard.ml.adapters import LogisticScoringModel
from learnguard.ml.contracts import load_prediction_batch_manifest
from learnguard.ml.model_interface import ScoringModel, load_scoring_model
from learnguard.ml.preprocessing import _prepare_split
from learnguard.ml.scoring import BatchScorer
from learnguard.ml.training import load_logistic_model_artifact, train_logistic_baseline
from learnguard.data.loading import load_snapshot_splits
from tests.test_preprocessing import preprocessing_run


@pytest.fixture(scope="module")
def model_run(preprocessing_run):
    root, snapshot, preprocessor = preprocessing_run
    result = train_logistic_baseline(
        snapshot, preprocessor.artifact_dir, root / "artifacts", "model"
    )
    return root, snapshot, preprocessor, result


class _ConstantScoringModel:
    """Stub `ScoringModel` that is NOT logistic: constant positive probability."""

    def __init__(self, version: str, feature_names: tuple[str, ...], probability: float) -> None:
        self.version = version
        self.feature_names = feature_names
        self._probability = probability

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(X), len(self.feature_names)), dtype=np.float64)

    def predict_positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._probability, dtype=np.float64)

    def explain(self, feature_row: np.ndarray) -> list:
        return []


def test_factory_returns_conforming_logistic_model(model_run):
    _, snapshot, preprocessor, model = model_run
    scoring_model = load_scoring_model(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    assert isinstance(scoring_model, LogisticScoringModel)
    assert isinstance(scoring_model, ScoringModel)  # runtime_checkable protocol
    manifest = json.loads((model.artifact_dir / "model.manifest.json").read_text())
    assert scoring_model.version == manifest["model_semantic_version"]
    assert scoring_model.feature_names == tuple(manifest["output_feature_names"])


def test_factory_rejects_unregistered_model_type(tmp_path):
    (tmp_path / "model.manifest.json").write_text(json.dumps({
        "schema_version": "logistic-model-v1",
        "model_type": "example.UnknownClassifier",
    }))
    with pytest.raises(ValueError, match="unsupported_model_type:example.UnknownClassifier"):
        load_scoring_model(tmp_path, "snapshot.parquet", "preprocessor")


def test_factory_missing_manifest_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scoring_model(tmp_path, "snapshot.parquet", "preprocessor")


def test_adapter_explain_matches_legacy_coef_formula(model_run):
    _, snapshot, preprocessor, model = model_run
    loaded = load_logistic_model_artifact(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    scoring_model = load_scoring_model(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    splits_frame = _prepare_split("train", load_snapshot_splits(snapshot).train)
    matrix = loaded._preprocessor.transform(splits_frame.X)
    for row_index in (0, 1, 2):
        signals = scoring_model.explain(matrix[row_index])
        contributions = loaded._model.coef_[0] * matrix[row_index]
        expected_order = np.argsort(-np.abs(contributions), kind="stable")[:5]
        assert [signal.feature_name for signal in signals] == [
            loaded.feature_names[i] for i in expected_order
        ]
        for signal, index in zip(signals, expected_order):
            assert signal.contribution == pytest.approx(float(contributions[index]))
            expected_direction = "increases_risk" if contributions[index] > 0 else "decreases_risk"
            assert signal.direction == expected_direction


def test_batch_scorer_accepts_non_logistic_scoring_model(model_run, monkeypatch, tmp_path):
    """THE decoupling proof: a stub model drives the full scoring pipeline."""
    _, snapshot, preprocessor, _ = model_run
    stub = _ConstantScoringModel("stub-v1", ("f0", "f1"), 0.3)
    monkeypatch.setattr(scoring, "load_scoring_model", lambda *args, **kwargs: stub)
    batch_dir = tmp_path / "prediction-batches" / "stub"
    report = BatchScorer().score(
        snapshot, model_run[3].artifact_dir, preprocessor.artifact_dir,
        batch_dir / "predictions.parquet", 0.5, "stub",
    )
    assert report.status == "ok"
    assert report.model_version == "stub-v1"
    predictions = pd.read_parquet(report.output_path)
    assert (predictions["risk_score"] == 0.3).all()
    # Constant 0.3 gives uncertainty 0.4 < HIGH_UNCERTAINTY_THRESHOLD, so rows
    # are either scored or flagged insufficient-data by the snapshot itself.
    assert (predictions["status"] != "manual_review").all()
    scored = predictions[predictions["status"] == "scored"]
    assert (scored["predicted_label"] == 0).all()
    manifest = load_prediction_batch_manifest(batch_dir / "manifest.json")
    assert manifest.model_version == "stub-v1"
