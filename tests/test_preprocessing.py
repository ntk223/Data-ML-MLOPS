"""Phase 4 train-only preprocessing contract and artifact tests."""

import json
import shutil
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from learnguard.cli import build_parser, main
from learnguard.ml.preprocessing import (
    AUDIT_ONLY_COLUMNS,
    FORBIDDEN_OUTPUT_COLUMNS,
    PREPROCESSING_INPUT_COLUMNS,
    _build_transformer,
    _fit_for_split,
    _frame_hash,
    _learned_state,
    _prepare_split,
    _prepare_splits,
    _validate_input_frame,
    build_preprocessing_artifact,
    load_preprocessing_artifact,
    preprocessing_contract_hash,
    verify_preprocessing_artifact,
)
from learnguard.pipeline import run_data_pipeline
from tests.conftest import write_csvs
from tests.test_loading import _tables_all_presentations


@pytest.fixture(scope="module")
def preprocessing_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("preprocessing")
    source = write_csvs(root / "source", _tables_all_presentations())
    data_report = run_data_pipeline(source, root / "artifacts", run_id="data")
    assert data_report.status == "ok"
    data_run = root / "artifacts/data-runs/data"
    result = build_preprocessing_artifact(
        data_run, root / "artifacts", run_id="preprocessing"
    )
    return root, data_run, result


def test_phase4_feature_contract_replaces_presentation_with_module():
    assert "code_module" in PREPROCESSING_INPUT_COLUMNS
    assert "module_presentation" not in PREPROCESSING_INPUT_COLUMNS
    assert "label" not in PREPROCESSING_INPUT_COLUMNS
    assert not set(AUDIT_ONLY_COLUMNS) & set(PREPROCESSING_INPUT_COLUMNS)
    assert len(PREPROCESSING_INPUT_COLUMNS) == len(set(PREPROCESSING_INPUT_COLUMNS)) == 16


def test_fit_guard_rejects_non_train_split(preprocessing_run):
    _, data_run, _ = preprocessing_run
    prepared, _ = _prepare_splits(data_run / "snapshot/snapshot.parquet")
    with pytest.raises(ValueError, match="preprocessor_fit_forbidden"):
        _fit_for_split(_build_transformer(), prepared["validation"].X, "validation")


def test_missing_extra_or_reordered_input_is_rejected(preprocessing_run):
    _, data_run, _ = preprocessing_run
    prepared, _ = _prepare_splits(data_run / "snapshot/snapshot.parquet")
    X = prepared["train"].X
    with pytest.raises(ValueError, match="schema_or_order_mismatch"):
        _validate_input_frame(X.drop(columns=[X.columns[0]]))
    with pytest.raises(ValueError, match="schema_or_order_mismatch"):
        _validate_input_frame(X[list(reversed(X.columns))])
    with pytest.raises(ValueError, match="schema_or_order_mismatch"):
        _validate_input_frame(X.assign(extra=1))


def test_shared_transformer_has_stable_finite_output(preprocessing_run):
    _, data_run, result = preprocessing_run
    snapshot = data_run / "snapshot/snapshot.parquet"
    bundle = load_preprocessing_artifact(result.artifact_dir, snapshot)
    prepared, _ = _prepare_splits(snapshot)
    widths = set()
    for split in prepared.values():
        values = bundle.transform(split.X)
        widths.add(values.shape[1])
        assert values.shape[0] == len(split.X)
        assert np.isfinite(values).all()
    assert widths == {result.output_feature_count}
    assert all(
        not any(name == field or name.endswith(f"__{field}") for name in bundle.feature_names)
        for field in FORBIDDEN_OUTPUT_COLUMNS
    )


def test_unknown_validation_category_does_not_change_feature_space(preprocessing_run):
    _, data_run, _ = preprocessing_run
    prepared, _ = _prepare_splits(data_run / "snapshot/snapshot.parquet")
    transformer = _build_transformer()
    _fit_for_split(transformer, prepared["train"].X, "train")
    before_names = transformer.get_feature_names_out().tolist()
    validation = prepared["validation"].X.copy()
    validation.loc[validation.index[0], "region"] = "Future Region"
    output = transformer.transform(validation)
    assert output.shape[1] == len(before_names)
    assert transformer.get_feature_names_out().tolist() == before_names
    assert "Future Region" not in _learned_state(transformer)["nominal_categories"][2]


def test_validation_and_holdout_mutation_do_not_change_learned_state(preprocessing_run):
    _, data_run, _ = preprocessing_run
    prepared, _ = _prepare_splits(data_run / "snapshot/snapshot.parquet")
    transformer = _build_transformer()
    _fit_for_split(transformer, prepared["train"].X, "train")
    before = _learned_state(transformer)
    prepared["validation"].X.loc[0, "n_clicks_before_cutoff"] = 10**12
    prepared["holdout"].X.loc[0, "region"] = "Holdout Only"
    assert _learned_state(transformer) == before


def test_train_mutation_changes_train_semantic_hash(preprocessing_run):
    _, data_run, _ = preprocessing_run
    prepared, _ = _prepare_splits(data_run / "snapshot/snapshot.parquet")
    train = prepared["train"]
    before = _frame_hash(train.keys, train.X)
    changed = train.X.copy()
    changed.loc[0, "n_clicks_before_cutoff"] += 1
    assert _frame_hash(train.keys, changed) != before


def test_row_order_permutation_is_normalized(preprocessing_run):
    _, data_run, _ = preprocessing_run
    snapshot = pd.read_parquet(data_run / "snapshot/snapshot.parquet")
    train = snapshot[snapshot["split"] == "train"]
    first = _prepare_split("train", train)
    shuffled = train.sample(frac=1, random_state=42)
    second = _prepare_split("train", shuffled)
    assert _frame_hash(first.keys, first.X) == _frame_hash(second.keys, second.X)


def test_avg_score_missing_indicator_preserves_zero_and_missing(preprocessing_run):
    _, data_run, result = preprocessing_run
    bundle = load_preprocessing_artifact(
        result.artifact_dir, data_run / "snapshot/snapshot.parquet"
    )
    prepared, _ = _prepare_splits(data_run / "snapshot/snapshot.parquet")
    sample = prepared["train"].X.iloc[[0, 0]].copy()
    sample.loc[sample.index[0], "avg_score_submitted"] = np.nan
    sample.index = [0, 1]
    sample.loc[1, "avg_score_submitted"] = 0.0
    values = bundle.transform(sample)
    indicator = bundle.feature_names.index("avg_score_missing__avg_score_submitted")
    assert values[:, indicator].tolist() == [1.0, 0.0]


def test_artifact_round_trip_and_verification(preprocessing_run):
    _, data_run, result = preprocessing_run
    report = verify_preprocessing_artifact(result.artifact_dir, data_run)
    manifest = json.loads(result.manifest_path.read_text())
    assert report.status == "ok"
    assert report.preprocessor_version == result.preprocessor_version
    assert manifest["preprocessing_contract_hash"] == preprocessing_contract_hash()
    assert manifest["train_row_count"] > 0
    assert set(report.split_results) == {"train", "validation", "holdout"}


def test_manifest_tampering_is_rejected(preprocessing_run, tmp_path):
    _, data_run, result = preprocessing_run
    copied = tmp_path / "tampered"
    shutil.copytree(result.artifact_dir, copied)
    manifest_path = copied / "preprocessor.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train_row_count"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest_hash_mismatch"):
        load_preprocessing_artifact(copied, data_run / "snapshot/snapshot.parquet")


def test_coordinated_manifest_marker_tampering_is_rejected(preprocessing_run, tmp_path):
    _, data_run, result = preprocessing_run
    copied = tmp_path / "tampered"
    shutil.copytree(result.artifact_dir, copied)
    manifest_path = copied / "preprocessor.manifest.json"
    marker_path = copied / "preprocessor.success.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train_row_count"] += 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    marker = json.loads(marker_path.read_text())
    marker["manifest_sha256"] = sha256(manifest_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="train_row_count_mismatch"):
        load_preprocessing_artifact(copied, data_run / "snapshot/snapshot.parquet")


def test_artifact_byte_tampering_is_rejected(preprocessing_run, tmp_path):
    _, data_run, result = preprocessing_run
    copied = tmp_path / "tampered"
    shutil.copytree(result.artifact_dir, copied)
    with (copied / "preprocessor.joblib").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact_hash_mismatch"):
        load_preprocessing_artifact(copied, data_run / "snapshot/snapshot.parquet")


def test_existing_preprocessing_run_is_not_overwritten(preprocessing_run):
    root, data_run, _ = preprocessing_run
    with pytest.raises(FileExistsError, match="run_id_exists"):
        build_preprocessing_artifact(
            data_run, root / "artifacts", run_id="preprocessing"
        )


@pytest.mark.parametrize(
    "run_id", ["", ".", "..", "../escape", "nested/run", r"..\escape", "/tmp/escape"]
)
def test_preprocessing_rejects_unsafe_run_ids_before_path_use(tmp_path, run_id):
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="invalid_run_id"):
        build_preprocessing_artifact(
            tmp_path / "missing-data-run", artifacts, run_id=run_id
        )
    assert not artifacts.exists()


def test_missing_source_reports_fail_before_fit(tmp_path, preprocessing_run):
    root, data_run, _ = preprocessing_run
    copied = tmp_path / "data-run"
    shutil.copytree(data_run, copied)
    (copied / "reports/leakage_audit.json").unlink()
    with pytest.raises(FileNotFoundError, match="source_reports_missing"):
        build_preprocessing_artifact(copied, root / "other-artifacts", run_id="failed")


def test_failed_serialization_does_not_publish_artifact(
    preprocessing_run, tmp_path, monkeypatch
):
    _, data_run, _ = preprocessing_run

    def fail_dump(*args, **kwargs):
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr("learnguard.ml.preprocessing.joblib.dump", fail_dump)
    with pytest.raises(RuntimeError, match="injected serialization failure"):
        build_preprocessing_artifact(data_run, tmp_path, run_id="failed")
    assert not (tmp_path / "preprocessing-runs/failed").exists()
    assert not list((tmp_path / "preprocessing-runs").glob(".failed.staging-*"))


def test_cli_exposes_preprocessing_commands():
    parser = build_parser()
    parsed = parser.parse_args([
        "ml", "preprocess", "--data-run-dir", "data-run", "--run-id", "p1"
    ])
    verified = parser.parse_args([
        "ml", "verify-preprocessor", "--artifact-dir", "artifact",
        "--data-run-dir", "data-run",
    ])
    assert (parsed.area, parsed.command) == ("ml", "preprocess")
    assert verified.command == "verify-preprocessor"


def test_cli_verify_preprocessor(preprocessing_run):
    _, data_run, result = preprocessing_run
    code = main([
        "ml", "verify-preprocessor",
        "--artifact-dir", str(result.artifact_dir),
        "--data-run-dir", str(data_run),
    ])
    assert code == 0


def test_cli_preprocess_returns_nonzero_for_invalid_data_run(tmp_path):
    code = main([
        "ml", "preprocess",
        "--data-run-dir", str(tmp_path / "missing"),
        "--artifacts-dir", str(tmp_path / "artifacts"),
        "--run-id", "failed",
    ])
    assert code == 2

