"""Phase 5 train-only Logistic Regression contract and artifact tests."""

import json
import shutil
import warnings
from dataclasses import replace
from hashlib import sha256

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from learnguard.cli import build_parser, main
from learnguard.ml.preprocessing import build_preprocessing_artifact
from learnguard.ml.training import (
    LOGISTIC_BASELINE_PARAMS,
    MODEL_ARTIFACT_SCHEMA_VERSION,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    VerifiedTrainMatrix,
    _load_verified_train_matrix,
    _model_state,
    _resolved_estimator_params,
    _validate_fitted_model,
    _validate_training_arrays,
    load_logistic_model_artifact,
    model_contract_hash,
    train_logistic_baseline,
    verify_logistic_model_artifact,
)
from learnguard.pipeline import run_data_pipeline
from tests.conftest import write_csvs
from tests.test_loading import _tables_all_presentations


@pytest.fixture(scope="module")
def logistic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("logistic")
    source = write_csvs(root / "source", _tables_all_presentations())
    report = run_data_pipeline(source, root / "artifacts", run_id="data")
    assert report.status == "ok"
    data_run = root / "artifacts/data-runs/data"
    preprocessor = build_preprocessing_artifact(
        data_run, root / "artifacts", run_id="preprocessor"
    )
    model = train_logistic_baseline(
        data_run, preprocessor.artifact_dir, root / "artifacts", run_id="model"
    )
    return root, data_run, preprocessor, model


@pytest.fixture(scope="module")
def changed_train_feature_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("logistic-train-feature")
    tables = _tables_all_presentations()
    mask = (
        tables["studentVle"]["code_presentation"].eq("2013B")
        & tables["studentVle"]["date"].le(27)
    )
    index = tables["studentVle"].index[mask][0]
    tables["studentVle"].loc[index, "sum_click"] += 7
    source = write_csvs(root / "source", tables)
    report = run_data_pipeline(source, root / "artifacts", run_id="data")
    assert report.status == "ok"
    data_run = root / "artifacts/data-runs/data"
    preprocessor = build_preprocessing_artifact(
        data_run, root / "artifacts", run_id="preprocessor"
    )
    model = train_logistic_baseline(
        data_run, preprocessor.artifact_dir, root / "artifacts", run_id="model"
    )
    return root, data_run, preprocessor, model


def _manifest(model_dir):
    return json.loads((model_dir / "model.manifest.json").read_text())


def _rehash_manifest(model_dir):
    manifest_path = model_dir / "model.manifest.json"
    marker_path = model_dir / "model.success.json"
    marker = json.loads(marker_path.read_text())
    marker["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True))


def test_fixed_logistic_v1_contract():
    assert LOGISTIC_BASELINE_PARAMS == {
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
    resolved = _resolved_estimator_params()
    assert resolved["class_weight"] is None
    assert resolved["solver"] == "lbfgs"
    assert len(model_contract_hash()) == 64
    assert (NEGATIVE_CLASS, POSITIVE_CLASS) == (0, 1)


def test_verified_boundary_reads_only_train(logistic_run, monkeypatch):
    _, data_run, preprocessor, _ = logistic_run
    original = pd.read_parquet
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr("learnguard.ml.training.pd.read_parquet", tracked)
    train = _load_verified_train_matrix(data_run, preprocessor.artifact_dir)
    assert train.X.dtype == np.float64
    assert set(train.y) == {0, 1}
    assert len(train.feature_names) == train.X.shape[1]
    assert calls == [{
        "columns": list(dict.fromkeys([
            "code_module", "code_presentation", "id_student",
            "gender", "region", "disability", "highest_education", "age_band",
            "imd_band", "num_of_prev_attempts", "studied_credits",
            "active_days_before_cutoff", "click_trend_w3_w4",
            "n_submitted_before_cutoff", "n_clicks_before_cutoff",
            "avg_score_submitted", "imd_band_missing", "insufficient_data",
            "label", "split",
        ])),
        "filters": [("split", "==", "train")],
    }]


def test_artifact_contract_and_round_trip(logistic_run):
    _, data_run, preprocessor, result = logistic_run
    assert result.status == "ok"
    assert {path.name for path in result.published_paths} == {
        "model.joblib", "model.manifest.json", "model.training-report.json",
        "model.success.json",
    }
    manifest = _manifest(result.artifact_dir)
    assert manifest["schema_version"] == MODEL_ARTIFACT_SCHEMA_VERSION
    assert manifest["target_column"] == "label"
    assert manifest["learned_classes"] == [0, 1]
    assert manifest["input_dtype"] == "float64"
    assert not any(
        name.endswith(("__label", "__id_student", "__final_result", "__split"))
        for name in manifest["output_feature_names"]
    )
    assert not ({"threshold", "validation_metrics", "holdout_metrics"} & set(manifest))
    verified = verify_logistic_model_artifact(
        result.artifact_dir, data_run, preprocessor.artifact_dir
    )
    assert verified.status == "ok"
    assert verified.model_semantic_version == result.model_semantic_version


def test_loaded_model_is_predict_only_facade(logistic_run):
    _, data_run, preprocessor, result = logistic_run
    loaded = load_logistic_model_artifact(
        result.artifact_dir, data_run, preprocessor.artifact_dir
    )
    assert not hasattr(loaded, "fit")
    assert len(loaded.feature_names) == loaded.verification.output_feature_count


def test_identical_runs_have_identical_semantic_model(logistic_run):
    root, data_run, preprocessor, first = logistic_run
    second = train_logistic_baseline(
        data_run, preprocessor.artifact_dir, root / "artifacts", run_id="model-2"
    )
    left, right = _manifest(first.artifact_dir), _manifest(second.artifact_dir)
    fields = (
        "model_semantic_version", "model_contract_hash",
        "training_matrix_semantic_hash", "training_target_semantic_hash",
        "coefficient_semantic_hash", "intercept_semantic_hash",
        "learned_state_hash", "train_probability_semantic_hash",
        "output_feature_names", "learned_classes",
    )
    assert all(left[field] == right[field] for field in fields)


@pytest.mark.parametrize("filename", [
    "model.joblib", "model.manifest.json", "model.training-report.json",
])
def test_byte_tampering_is_rejected(logistic_run, tmp_path, filename):
    _, data_run, preprocessor, result = logistic_run
    copied = tmp_path / "tampered"
    shutil.copytree(result.artifact_dir, copied)
    with (copied / filename).open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_logistic_model_artifact(copied, data_run, preprocessor.artifact_dir)


def test_coordinated_manifest_marker_tampering_is_rejected(logistic_run, tmp_path):
    _, data_run, preprocessor, result = logistic_run
    copied = tmp_path / "tampered"
    shutil.copytree(result.artifact_dir, copied)
    path = copied / "model.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["train_row_count"] += 1
    path.write_text(json.dumps(manifest, sort_keys=True))
    _rehash_manifest(copied)
    with pytest.raises(ValueError, match="train_row_count_mismatch"):
        load_logistic_model_artifact(copied, data_run, preprocessor.artifact_dir)


def test_dependency_mismatch_is_rejected(logistic_run, tmp_path):
    _, data_run, preprocessor, result = logistic_run
    copied = tmp_path / "dependency"
    shutil.copytree(result.artifact_dir, copied)
    path = copied / "model.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["packages"]["scikit_learn"] = "0.0"
    path.write_text(json.dumps(manifest, sort_keys=True))
    _rehash_manifest(copied)
    with pytest.raises(ValueError, match="dependency_mismatch"):
        load_logistic_model_artifact(copied, data_run, preprocessor.artifact_dir)


def test_feature_order_mismatch_is_rejected(logistic_run, tmp_path):
    _, data_run, preprocessor, result = logistic_run
    copied = tmp_path / "features"
    shutil.copytree(result.artifact_dir, copied)
    path = copied / "model.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_feature_names"] = list(reversed(manifest["output_feature_names"]))
    path.write_text(json.dumps(manifest, sort_keys=True))
    _rehash_manifest(copied)
    with pytest.raises(ValueError, match="output_feature_names_mismatch"):
        load_logistic_model_artifact(copied, data_run, preprocessor.artifact_dir)


def test_coefficient_state_mismatch_is_rejected(logistic_run, tmp_path):
    _, data_run, preprocessor, result = logistic_run
    copied = tmp_path / "state"
    shutil.copytree(result.artifact_dir, copied)
    model_path = copied / "model.joblib"
    model = joblib.load(model_path)
    model.coef_[0, 0] += 1.0
    joblib.dump(model, model_path, compress=3)
    artifact_hash = sha256(model_path.read_bytes()).hexdigest()
    manifest_path = copied / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model_artifact_sha256"] = artifact_hash
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    marker_path = copied / "model.success.json"
    marker = json.loads(marker_path.read_text())
    marker["model_artifact_sha256"] = artifact_hash
    marker["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker, sort_keys=True))
    with pytest.raises(ValueError, match="coefficient_semantic_hash_mismatch"):
        load_logistic_model_artifact(copied, data_run, preprocessor.artifact_dir)


@pytest.mark.parametrize("split,column", [
    ("validation", "n_clicks_before_cutoff"),
    ("validation", "label"),
    ("holdout", "n_clicks_before_cutoff"),
    ("holdout", "label"),
])
def test_non_train_mutation_does_not_change_model(
    logistic_run, tmp_path, split, column
):
    root, data_run, preprocessor, original = logistic_run
    copied = tmp_path / f"data-{split}-{column}"
    shutil.copytree(data_run, copied)
    path = copied / "snapshot/snapshot.parquet"
    frame = pd.read_parquet(path)
    index = frame.index[frame["split"].eq(split)][0]
    frame.loc[index, column] = (
        1 - int(frame.loc[index, column]) if column == "label"
        else float(frame.loc[index, column]) + 12345.0
    )
    frame.to_parquet(path, index=False)
    changed = train_logistic_baseline(
        copied, preprocessor.artifact_dir, tmp_path / "models", run_id="changed"
    )
    left, right = _manifest(original.artifact_dir), _manifest(changed.artifact_dir)
    for field in (
        "model_semantic_version", "coefficient_semantic_hash",
        "intercept_semantic_hash", "learned_state_hash",
        "train_probability_semantic_hash",
    ):
        assert left[field] == right[field]


def test_train_row_permutation_is_normalized(logistic_run, tmp_path):
    _, data_run, preprocessor, original = logistic_run
    copied = tmp_path / "permuted"
    shutil.copytree(data_run, copied)
    path = copied / "snapshot/snapshot.parquet"
    frame = pd.read_parquet(path).sample(frac=1, random_state=42)
    frame.to_parquet(path, index=False)
    changed = train_logistic_baseline(
        copied, preprocessor.artifact_dir, tmp_path / "models", run_id="permuted"
    )
    assert changed.model_semantic_version == original.model_semantic_version
    assert changed.learned_state_hash == original.learned_state_hash


def test_train_feature_mutation_changes_matrix_and_model(
    logistic_run, changed_train_feature_run
):
    _, _, _, original = logistic_run
    _, _, _, changed = changed_train_feature_run
    left, right = _manifest(original.artifact_dir), _manifest(changed.artifact_dir)
    assert left["training_matrix_semantic_hash"] != right["training_matrix_semantic_hash"]
    assert left["learned_state_hash"] != right["learned_state_hash"]
    assert left["model_semantic_version"] != right["model_semantic_version"]


def test_train_label_mutation_changes_target_and_model(logistic_run, tmp_path):
    _, _, _, original = logistic_run
    tables = _tables_all_presentations()
    mask = (
        tables["studentInfo"]["code_presentation"].eq("2013B")
        & tables["studentInfo"]["id_student"].eq(101)
    )
    tables["studentInfo"].loc[mask, "final_result"] = "Fail"
    source = write_csvs(tmp_path / "source", tables)
    report = run_data_pipeline(source, tmp_path / "artifacts", run_id="data")
    assert report.status == "ok"
    data_run = tmp_path / "artifacts/data-runs/data"
    preprocessor = build_preprocessing_artifact(
        data_run, tmp_path / "artifacts", run_id="preprocessor"
    )
    changed = train_logistic_baseline(
        data_run, preprocessor.artifact_dir, tmp_path / "artifacts", run_id="model"
    )
    left, right = _manifest(original.artifact_dir), _manifest(changed.artifact_dir)
    assert left["training_target_semantic_hash"] != right["training_target_semantic_hash"]
    assert left["learned_state_hash"] != right["learned_state_hash"]
    assert left["model_semantic_version"] != right["model_semantic_version"]


def test_snapshot_and_preprocessor_mismatch_is_rejected(
    logistic_run, changed_train_feature_run
):
    _, _, _, original = logistic_run
    _, changed_data, changed_preprocessor, _ = changed_train_feature_run
    with pytest.raises(ValueError, match="model_(snapshot|preprocessor|train_feature)"):
        load_logistic_model_artifact(
            original.artifact_dir, changed_data, changed_preprocessor.artifact_dir
        )


@pytest.mark.parametrize("kind", ["empty", "single", "nan", "inf", "dtype"])
def test_invalid_training_arrays_are_rejected(logistic_run, kind):
    _, data_run, preprocessor, _ = logistic_run
    train = _load_verified_train_matrix(data_run, preprocessor.artifact_dir)
    if kind == "empty":
        changed = replace(train, X=train.X[:0], y=train.y[:0])
        expected = "train_empty"
    elif kind == "single":
        changed = replace(train, y=np.zeros_like(train.y))
        expected = "single_class"
    elif kind == "nan":
        values = train.X.copy(); values[0, 0] = np.nan
        changed, expected = replace(train, X=values), "non_finite"
    elif kind == "inf":
        values = train.X.copy(); values[0, 0] = np.inf
        changed, expected = replace(train, X=values), "non_finite"
    else:
        changed, expected = replace(train, X=train.X.astype(np.float32)), "dtype_invalid"
    with pytest.raises(ValueError, match=expected):
        _validate_training_arrays(changed)


def test_wrong_coefficient_shape_is_rejected(logistic_run):
    _, data_run, preprocessor, _ = logistic_run
    train = _load_verified_train_matrix(data_run, preprocessor.artifact_dir)
    model = LogisticRegression(**LOGISTIC_BASELINE_PARAMS).fit(train.X, train.y)
    model.coef_ = model.coef_[:, :-1]
    with pytest.raises(ValueError, match="coefficient_shape_invalid"):
        _validate_fitted_model(model, train, [])


def test_convergence_failure_does_not_publish(logistic_run, tmp_path, monkeypatch):
    _, data_run, preprocessor, _ = logistic_run
    original_fit = LogisticRegression.fit

    def forced_failure(self, X, y, *args, **kwargs):
        result = original_fit(self, X, y, *args, **kwargs)
        self.n_iter_[:] = self.max_iter
        return result

    monkeypatch.setattr("learnguard.ml.training.LogisticRegression.fit", forced_failure)
    with pytest.raises(ValueError, match="convergence_not_reached"):
        train_logistic_baseline(
            data_run, preprocessor.artifact_dir, tmp_path, run_id="failed"
        )
    assert not (tmp_path / "model-runs/failed").exists()
    assert not list((tmp_path / "model-runs").glob(".failed.staging-*"))


def test_convergence_warning_does_not_publish(logistic_run, tmp_path, monkeypatch):
    _, data_run, preprocessor, _ = logistic_run
    original_fit = LogisticRegression.fit

    def warned_fit(self, X, y, *args, **kwargs):
        result = original_fit(self, X, y, *args, **kwargs)
        warnings.warn("injected", ConvergenceWarning)
        return result

    monkeypatch.setattr("learnguard.ml.training.LogisticRegression.fit", warned_fit)
    with pytest.raises(ValueError, match="convergence_warning"):
        train_logistic_baseline(
            data_run, preprocessor.artifact_dir, tmp_path, run_id="warned"
        )
    assert not (tmp_path / "model-runs/warned").exists()


def test_serialization_failure_does_not_publish(logistic_run, tmp_path, monkeypatch):
    _, data_run, preprocessor, _ = logistic_run

    def fail_dump(*args, **kwargs):
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr("learnguard.ml.training.joblib.dump", fail_dump)
    with pytest.raises(RuntimeError, match="injected serialization failure"):
        train_logistic_baseline(
            data_run, preprocessor.artifact_dir, tmp_path, run_id="failed"
        )
    assert not (tmp_path / "model-runs/failed").exists()
    assert not list((tmp_path / "model-runs").glob(".failed.staging-*"))


@pytest.mark.parametrize(
    "run_id", ["", ".", "..", "../escape", "nested/run", r"..\escape", "/tmp/x"]
)
def test_unsafe_model_run_ids_are_rejected_before_path_use(tmp_path, run_id):
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="invalid_run_id"):
        train_logistic_baseline(tmp_path, tmp_path, artifacts, run_id=run_id)
    assert not artifacts.exists()


def test_existing_model_run_is_not_overwritten(logistic_run):
    root, data_run, preprocessor, _ = logistic_run
    with pytest.raises(FileExistsError, match="run_id_exists"):
        train_logistic_baseline(
            data_run, preprocessor.artifact_dir, root / "artifacts", run_id="model"
        )


def test_cli_surface_has_no_model_override_flags():
    parser = build_parser()
    trained = parser.parse_args([
        "ml", "train-logistic", "--data-run-dir", "data",
        "--preprocessing-artifact-dir", "pre", "--run-id", "m1",
    ])
    verified = parser.parse_args([
        "ml", "verify-model", "--model-artifact-dir", "model",
        "--data-run-dir", "data", "--preprocessing-artifact-dir", "pre",
    ])
    assert trained.command == "train-logistic"
    assert verified.command == "verify-model"
    assert not any(
        hasattr(trained, name)
        for name in ("split", "threshold", "class_weight", "C", "solver")
    )


def test_cli_verify_model(logistic_run):
    _, data_run, preprocessor, result = logistic_run
    code = main([
        "ml", "verify-model", "--model-artifact-dir", str(result.artifact_dir),
        "--data-run-dir", str(data_run),
        "--preprocessing-artifact-dir", str(preprocessor.artifact_dir),
    ])
    assert code == 0


def test_cli_train_failure_returns_nonzero(tmp_path):
    code = main([
        "ml", "train-logistic", "--data-run-dir", str(tmp_path / "missing"),
        "--preprocessing-artifact-dir", str(tmp_path / "missing-pre"),
        "--artifacts-dir", str(tmp_path / "artifacts"), "--run-id", "failed",
    ])
    assert code == 2
