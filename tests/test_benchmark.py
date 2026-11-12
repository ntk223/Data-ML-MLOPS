import json

import numpy as np
import pandas as pd
import pytest

from learnguard.cli import build_parser
from learnguard.ml import benchmark as benchmark_module
from learnguard.ml.benchmark import (
    _candidate_models,
    _selection_key,
    run_benchmark,
)
from tests.test_preprocessing import preprocessing_run


def test_benchmark_locks_before_one_shot_holdout(
    preprocessing_run, tmp_path, monkeypatch,
):
    _, snapshot, preprocessor = preprocessing_run
    requested: list[tuple[str, ...]] = []
    original = benchmark_module.load_snapshot_partitions
    original_read_parquet = pd.read_parquet

    def assert_locked_before_holdout_access():
        staging_dirs = list((tmp_path / "benchmark-runs").glob(
            ".bench-test.staging-*"
        ))
        assert len(staging_dirs) == 1
        locked_path = staging_dirs[0] / "locked_config.json"
        assert locked_path.is_file(), "holdout_loaded_before_config_lock"
        locked_at_access = json.loads(locked_path.read_text())
        assert locked_at_access["holdout_accessed"] is False
        assert locked_at_access["holdout_used_for_selection"] is False

    def lock_guarded_loader(path, names):
        requested.append(tuple(names))
        if tuple(names) == ("holdout",):
            assert_locked_before_holdout_access()
        return original(path, names)

    def parquet_lock_guard(*args, **kwargs):
        filters = kwargs.get("filters")
        reads_all_splits = filters is None
        reads_holdout = filters is not None and any(
            condition[0] == "split" and "holdout" in condition[2]
            for condition in filters
        )
        if reads_all_splits or reads_holdout:
            assert_locked_before_holdout_access()
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(
        benchmark_module, "load_snapshot_partitions", lock_guarded_loader,
    )
    monkeypatch.setattr("learnguard.data.loading.pd.read_parquet", parquet_lock_guard)
    report = run_benchmark(snapshot, preprocessor.artifact_dir, tmp_path, "bench-test")
    assert requested == [("train", "validation"), ("holdout",)]
    assert report.status == "ok"
    locked = json.loads(report.locked_config_path.read_text())
    holdout = json.loads(report.holdout_report_path.read_text())
    final_report = json.loads(
        (report.artifact_dir / "benchmark_report.json").read_text()
    )
    assert locked["holdout_used_for_selection"] is False
    assert locked["holdout_accessed"] is False
    assert holdout["prediction_calls"] == 1
    assert holdout["locked_config_sha256"]
    assert final_report["holdout_accessed"] is True
    validation = pd.read_csv(report.artifact_dir / "validation_metrics.csv")
    assert {"dummy_prior", "logistic_baseline", "extra_trees_a", "hist_gradient_boosting_a"} <= set(validation["candidate"])
    assert (report.artifact_dir / "success.json").is_file()


def test_selection_never_trades_lower_pr_auc_for_recall_within_one_millionth():
    higher_pr_auc = {
        "pr_auc": 0.7000001, "recall_at_top_20_pct": 0.2,
        "ece": 0.2, "inference_seconds": 2.0,
    }
    higher_recall = {
        "pr_auc": 0.7000000, "recall_at_top_20_pct": 0.9,
        "ece": 0.01, "inference_seconds": 0.1,
    }
    assert _selection_key(higher_pr_auc) > _selection_key(higher_recall)


def test_random_forest_candidate_contracts_are_fixed():
    candidates = dict(_candidate_models())
    assert len(candidates) == 10
    expected = {
        "random_forest_a": {
            "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
            "max_features": "sqrt", "class_weight": None,
        },
        "random_forest_b": {
            "n_estimators": 500, "max_depth": None, "min_samples_leaf": 5,
            "max_features": "sqrt", "class_weight": "balanced_subsample",
        },
        "random_forest_c": {
            "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 10,
            "max_features": 0.7, "class_weight": "balanced_subsample",
        },
    }
    for name, contract in expected.items():
        params = candidates[name].get_params(deep=False)
        assert {key: params[key] for key in contract} == contract
        assert params["bootstrap"] is True
        assert params["random_state"] == 42
        assert params["n_jobs"] == 1


def test_validation_only_never_uses_full_loader_or_publishes_holdout(
    preprocessing_run, tmp_path, monkeypatch,
):
    _, snapshot, preprocessor = preprocessing_run
    requested: list[tuple[str, ...]] = []
    parquet_filters: list[object] = []
    original = benchmark_module.load_snapshot_partitions
    original_read_parquet = pd.read_parquet

    def selected_loader(path, names):
        requested.append(tuple(names))
        return original(path, names)

    def parquet_spy(*args, **kwargs):
        parquet_filters.append(kwargs.get("filters"))
        return original_read_parquet(*args, **kwargs)

    monkeypatch.setattr(benchmark_module, "load_snapshot_partitions", selected_loader)
    monkeypatch.setattr("learnguard.data.loading.pd.read_parquet", parquet_spy)
    report = run_benchmark(
        snapshot, preprocessor.artifact_dir, tmp_path, "validation-only",
        evaluation_scope="validation_only",
    )
    assert requested == [("train", "validation")]
    assert parquet_filters == [
        [("split", "in", ["train", "validation"])],
        [("split", "in", ["train"])],
    ]
    assert report.winner is None
    assert report.validation_leader
    assert report.locked_config_path is None
    assert report.holdout_report_path is None
    assert report.evaluation_scope == "validation_only"
    payload = json.loads((report.artifact_dir / "benchmark_report.json").read_text())
    assert payload["holdout_accessed"] is False
    assert payload["production_release_eligible"] is False
    assert payload["holdout_prediction_calls"] == 0
    assert payload["candidate_count"] == 10
    names = [str(path.relative_to(report.artifact_dir))
             for path in report.artifact_dir.rglob("*") if path.is_file()]
    assert not any("holdout" in name or "locked_config" in name for name in names)
    predictions = pd.read_parquet(report.artifact_dir / "validation_predictions.parquet")
    assert "id_student" not in predictions
    assert len([c for c in predictions if c.startswith("risk_score__")]) == 10


def test_cli_parser_accepts_validation_only():
    args = build_parser().parse_args([
        "ml", "benchmark", "--snapshot", "snapshot.parquet",
        "--preprocessing-artifact-dir", "pre", "--run-id", "rf",
        "--validation-only",
    ])
    assert args.validation_only is True


@pytest.mark.parametrize("run_id", ["", "..", "../x", "nested/x", "/tmp/x"])
def test_benchmark_rejects_unsafe_run_id(tmp_path, run_id):
    with pytest.raises(ValueError, match="invalid_run_id"):
        run_benchmark("missing", "missing", tmp_path, run_id,
                      evaluation_scope="validation_only")


def test_random_forest_seed_is_deterministic():
    model = dict(_candidate_models())["random_forest_a"]
    twin = dict(_candidate_models())["random_forest_a"]
    X = np.arange(400, dtype=np.float64).reshape(100, 4)
    y = np.tile([0, 1], 50)
    model.fit(X, y)
    twin.fit(X, y)
    np.testing.assert_array_equal(model.predict_proba(X), twin.predict_proba(X))
