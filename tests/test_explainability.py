import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learnguard.data.loading import load_snapshot_splits
from learnguard.ml.explainability import build_shap_artifact
from learnguard.ml.model_interface import load_scoring_model
from learnguard.ml.preprocessing import _prepare_split
from tests.test_model_interface import model_run
from tests.test_preprocessing import preprocessing_run


def test_linear_shap_additivity_and_background(model_run):
    _, snapshot, preprocessor, model = model_run
    scoring_model = load_scoring_model(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    prepared = _prepare_split("validation", load_snapshot_splits(snapshot).validation)
    matrix = scoring_model.transform(prepared.X)
    explanation = scoring_model.explain_many(matrix)
    assert explanation.method.startswith("linear-shap")
    assert explanation.output_space == "log_odds"
    assert explanation.values.shape == matrix.shape
    assert np.isfinite(explanation.values).all()
    np.testing.assert_allclose(
        explanation.expected_value + explanation.values.sum(axis=1),
        explanation.output_values, atol=1e-8, rtol=0,
    )
    manifest = json.loads((model.artifact_dir / "model.manifest.json").read_text())
    assert manifest["explanation"]["background_split"] == "train"


def test_explainability_artifact_is_non_identifying(model_run, tmp_path):
    _, snapshot, preprocessor, model = model_run
    report = build_shap_artifact(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        tmp_path, "shap-test", splits=("validation",),
    )
    assert report.status == "ok"
    assert report.maximum_additivity_error <= 1e-8
    frame = pd.read_parquet(report.artifact_dir / "shap_values.parquet")
    assert "id_student" not in frame.columns
    assert len(frame) == len(load_snapshot_splits(snapshot).validation)
    assert (report.artifact_dir / "plots/summary_beeswarm.png").is_file()


def test_requested_shap_split_never_loads_holdout(model_run, tmp_path, monkeypatch):
    _, snapshot, preprocessor, model = model_run
    original = pd.read_parquet
    snapshot_filters: list[object] = []

    def spy(path, *args, **kwargs):
        if Path(path) == Path(snapshot):
            snapshot_filters.append(kwargs.get("filters"))
            assert kwargs.get("filters") in (
                [("split", "in", ["train"])],
                [("split", "in", ["validation"])],
            )
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    report = build_shap_artifact(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        tmp_path, "validation-shap", splits=("validation",),
    )
    assert report.row_count > 0
    assert [("split", "in", ["validation"])] in snapshot_filters
    assert all(filters is not None and "holdout" not in filters[0][2]
               for filters in snapshot_filters)
