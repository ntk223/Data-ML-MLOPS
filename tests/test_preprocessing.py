"""ML preprocessing tests against the unchanged data-owner snapshot contract."""

import json
import shutil

import pytest

from learnguard.data.features import build_feature_snapshot
from learnguard.ml.preprocessing import (
    PREPROCESSING_INPUT_COLUMNS,
    build_preprocessing_artifact,
    load_preprocessing_artifact,
    verify_preprocessing_artifact,
)
from tests.test_features import write_canonical
from tests.test_loading import _tables_all_presentations


@pytest.fixture(scope="module")
def preprocessing_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml-preprocessing")
    processed = write_canonical(root, _tables_all_presentations())
    snapshot = build_feature_snapshot(processed, root / "snapshot/snapshot.parquet")
    result = build_preprocessing_artifact(snapshot.snapshot_path, root / "artifacts", "pre")
    return root, snapshot.snapshot_path, result


def test_uses_published_model_feature_contract():
    assert "module_presentation" in PREPROCESSING_INPUT_COLUMNS
    assert "label" not in PREPROCESSING_INPUT_COLUMNS
    assert "id_student" not in PREPROCESSING_INPUT_COLUMNS


def test_train_only_artifact_round_trip(preprocessing_run):
    _, snapshot, result = preprocessing_run
    report = verify_preprocessing_artifact(result.artifact_dir, snapshot)
    assert report.status == "ok"
    assert set(report.split_results) == {"train", "validation", "holdout"}
    assert all(item["nan_count"] == item["inf_count"] == 0 for item in report.split_results.values())
    bundle = load_preprocessing_artifact(result.artifact_dir, snapshot)
    assert len(bundle.feature_names) == result.output_feature_count


def test_artifact_tampering_is_rejected(preprocessing_run, tmp_path):
    _, snapshot, result = preprocessing_run
    copied = tmp_path / "tampered"
    shutil.copytree(result.artifact_dir, copied)
    with (copied / "preprocessor.joblib").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact_hash_mismatch"):
        load_preprocessing_artifact(copied, snapshot)


def test_manifest_records_data_owner_snapshot_version(preprocessing_run):
    _, _, result = preprocessing_run
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["snapshot_version"].startswith("snapshot-v1:cutoff27:")
    assert manifest["target_column"] == "label"


@pytest.mark.parametrize("run_id", ["", "..", "../x", "nested/x", "/tmp/x"])
def test_unsafe_run_id_rejected(preprocessing_run, tmp_path, run_id):
    _, snapshot, _ = preprocessing_run
    with pytest.raises(ValueError, match="invalid_run_id"):
        build_preprocessing_artifact(snapshot, tmp_path, run_id)
