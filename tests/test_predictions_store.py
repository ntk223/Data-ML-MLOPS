"""M6 prediction store tests: split selection + shared PostgreSQL upsert/read.

DB tests run against the REAL local PostgreSQL and skip when it is
unavailable. They never touch `ml_predictions`: every test uses a dedicated
`ml_predictions_test__<uuid>` table that is dropped afterwards.
"""

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learnguard.cli import main
from learnguard.data.loading import SnapshotPartitions, load_snapshot_splits
from learnguard.ml import scoring as scoring_module
from learnguard.ml.predictions_store import (
    DEFAULT_TABLE,
    PredictionStore,
    PredictionStoreConfig,
)
from learnguard.ml.scoring import PREDICTION_COLUMNS, score_snapshot
from learnguard.ml.adapters import LogisticScoringModel
from learnguard.ml.training import load_logistic_model_artifact
from tests.test_preprocessing import preprocessing_run  # noqa: F401  (fixture chain)
from tests.test_scoring import model_run

PG_DSN = "postgresql://learnguard:learnguard@localhost:5432/learnguard"


def _test_table() -> str:
    return f"ml_predictions_test__{uuid.uuid4().hex}"


@pytest.fixture()
def store():
    """Real-PostgreSQL store on a dedicated throwaway table; skips without PG."""
    candidate = PredictionStore(
        PredictionStoreConfig(dsn=PG_DSN, table=_test_table())
    )
    try:
        candidate.ensure_schema()
    except Exception:
        candidate.close()
        pytest.skip("postgres unavailable")
    yield candidate
    candidate.drop_table()
    candidate.close()


def _sample_frame(rows: int = 3) -> pd.DataFrame:
    """Hand-built frame with exactly the prediction-batch.v1 parquet columns."""
    records: list[dict[str, object]] = []
    for index in range(rows):
        records.append({
            "student_key": f"{index:016x}",
            "code_module": "AAA",
            "code_presentation": "2014J",
            "split": "holdout",
            "risk_score": 0.1 * (index + 1),
            "predicted_label": index % 2,
            "threshold": 0.5,
            "uncertainty": 0.2,
            "status": "insufficient_data" if index == 0 else "scored",
            "review_reason": "insufficient_data" if index == 0 else None,
            "model_version": "logreg-v1:test",
            "snapshot_version": "snapshot-v1:test",
            "cutoff_day": 27,
            "top_signals_json": "[]" if index == 0 else json.dumps(
                [{"contribution": 0.5, "direction": "increases_risk",
                  "feature_name": "n_clicks_before_cutoff"}]
            ),
        })
    return pd.DataFrame(records, columns=list(PREDICTION_COLUMNS))


# ---- split selection (no database) ------------------------------------------


def test_score_holdout_only_yields_holdout_rows(model_run):
    root, snapshot, preprocessor, model = model_run
    batch_dir = root / "artifacts" / "prediction-batches" / "holdout-only"
    report = score_snapshot(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        batch_dir / "predictions.parquet", 0.5, "holdout-only",
        splits=("holdout",),
    )
    splits = load_snapshot_splits(snapshot)
    predictions = pd.read_parquet(report.output_path)
    assert len(predictions) == len(splits.holdout)
    assert set(predictions["split"]) == {"holdout"}
    assert report.input_rows == len(predictions)


def test_score_multiple_splits_concatenates(model_run):
    root, snapshot, preprocessor, model = model_run
    batch_dir = root / "artifacts" / "prediction-batches" / "two-splits"
    report = score_snapshot(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        batch_dir / "predictions.parquet", 0.5, "two-splits",
        splits=("train", "holdout"),
    )
    predictions = pd.read_parquet(report.output_path)
    assert set(predictions["split"]) == {"train", "holdout"}


def test_unknown_split_name_rejected(model_run):
    root, snapshot, preprocessor, model = model_run
    batch_dir = root / "artifacts" / "prediction-batches" / "bad-split"
    with pytest.raises(ValueError, match="unknown_split_name"):
        score_snapshot(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            batch_dir / "predictions.parquet", 0.5, "bad-split",
            splits=("holdout", "demo"),
        )


def test_duplicate_split_rejected(model_run):
    root, snapshot, preprocessor, model = model_run
    batch_dir = root / "artifacts" / "prediction-batches" / "dup-split"
    with pytest.raises(ValueError, match="duplicate_split_name"):
        score_snapshot(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            batch_dir / "predictions.parquet", 0.5, "dup-split",
            splits=("holdout", "holdout"),
        )


def test_empty_split_rejected(model_run, tmp_path, monkeypatch):
    """A requested split with zero rows fails loudly before any publish.

    `load_snapshot_splits` normally rejects split-less snapshots even earlier
    (`split_empty`), so this stubs a splits object with an empty validation
    frame to exercise the scorer's own defensive `empty_split` guard.
    """
    root, snapshot, preprocessor, model = model_run
    splits = load_snapshot_splits(snapshot)
    loaded = load_logistic_model_artifact(model.artifact_dir, snapshot, preprocessor.artifact_dir)
    empty_validation = SnapshotPartitions(
        snapshot_version=splits.snapshot_version,
        partitions={"validation": splits.train.iloc[0:0].copy()},
    )
    monkeypatch.setattr(
        scoring_module, "load_snapshot_partitions",
        lambda _path, _names: empty_validation,
    )
    monkeypatch.setattr(
        scoring_module, "load_scoring_model",
        lambda *_args, **_kwargs: LogisticScoringModel(loaded),
    )
    batch_dir = tmp_path / "artifacts" / "prediction-batches" / "empty-validation"
    with pytest.raises(ValueError, match="empty_split:validation"):
        score_snapshot(
            snapshot, model.artifact_dir, preprocessor.artifact_dir,
            batch_dir / "predictions.parquet", 0.5, "empty-validation",
            splits=("validation",),
        )
    assert not batch_dir.exists()


# ---- store configuration -----------------------------------------------------


def test_config_from_environment_defaults_to_unconfigured():
    config = PredictionStoreConfig.from_environment({})
    assert config.dsn == ""
    assert config.table == DEFAULT_TABLE
    assert not config.is_configured()


def test_config_from_environment_reads_dsn(monkeypatch):
    monkeypatch.setenv("LEARNGUARD_PG_DSN", PG_DSN)
    config = PredictionStoreConfig.from_environment()
    assert config.dsn == PG_DSN
    assert config.is_configured()


def test_missing_dsn_fails_at_engine_creation():
    store = PredictionStore(PredictionStoreConfig(dsn=""))
    with pytest.raises(ValueError, match="prediction_store_dsn_missing"):
        store.ensure_schema()


def test_unsupported_dsn_scheme_rejected():
    store = PredictionStore(PredictionStoreConfig(dsn="mysql://x/y"))
    with pytest.raises(ValueError, match="prediction_store_dsn_unsupported"):
        store.ensure_schema()


def test_table_name_must_be_safe_identifier():
    with pytest.raises(ValueError, match="prediction_store_table_name_invalid"):
        PredictionStore(PredictionStoreConfig(dsn=PG_DSN, table="ml; DROP TABLE x"))


# ---- store behaviour against real PostgreSQL ---------------------------------


def test_ensure_schema_is_idempotent(store):
    store.ensure_schema()  # second call must not raise
    empty = store.fetch_predictions()
    assert empty.empty
    assert list(empty.columns) == [
        *PREDICTION_COLUMNS, "run_id", "scored_at",
    ]


def test_upsert_overwrites_same_student(store):
    frame = _sample_frame()
    assert store.write_predictions(frame, "run-1") == len(frame)
    assert store.write_predictions(frame, "run-2") == len(frame)
    stored = store.fetch_predictions()
    assert len(stored) == len(frame)  # row count unchanged
    assert (stored["run_id"] == "run-2").all()  # newest run wins
    assert store.count_run("run-1") == 0
    assert store.count_run("run-2") == len(frame)


def test_write_rolls_back_on_bad_row(store):
    frame = _sample_frame()
    store.write_predictions(frame, "run-1")
    bad = frame.copy()
    bad.loc[bad.index[-1], "student_key"] = "x" * 20  # varchar(16) overflow
    with pytest.raises(ValueError, match="prediction_store_write_failed"):
        store.write_predictions(bad, "run-2")
    assert store.count_run("run-2") == 0  # nothing from the failed write
    assert store.count_run("run-1") == len(frame)  # earlier rows intact


def test_features_json_round_trip(store):
    """features_json phải lên DB nguyên vẹn (JSON text) qua upsert."""
    frame = _sample_frame()
    frame["features_json"] = json.dumps(
        {"avg_score_submitted": 45.5, "n_clicks_before_cutoff": 120.0,
         "studied_credits": 60.0}, separators=(",", ":"), sort_keys=True,
    )
    store.write_predictions(frame, "run-features")
    stored = store.fetch_predictions()
    assert len(stored) == len(frame)
    for value in stored["features_json"]:
        features = json.loads(value)
        assert features["avg_score_submitted"] == 45.5
        assert features["n_clicks_before_cutoff"] == 120.0


def test_write_rejects_wrong_columns(store):
    frame = _sample_frame().drop(columns=["uncertainty"])
    with pytest.raises(ValueError, match="prediction_store_columns_invalid"):
        store.write_predictions(frame, "run-1")
    assert store.fetch_predictions().empty


def test_fetch_filters_by_status(store):
    store.write_predictions(_sample_frame(), "run-1")
    scored = store.fetch_predictions(status="scored")
    insufficient = store.fetch_predictions(status="insufficient_data")
    assert len(scored) == 2 and len(insufficient) == 1
    assert (scored["status"] == "scored").all()
    assert (insufficient["review_reason"] == "insufficient_data").all()
    assert list(scored["student_key"]) == sorted(scored["student_key"])


def test_empty_signals_stored_as_null(store):
    store.write_predictions(_sample_frame(), "run-1")
    stored = store.fetch_predictions().set_index("student_key")
    assert pd.isna(stored.loc["0000000000000000", "top_signals_json"])
    assert not pd.isna(stored.loc["0000000000000001", "top_signals_json"])


def test_round_trip_score_holdout_write_fetch(store, model_run):
    root, snapshot, preprocessor, model = model_run
    batch_dir = root / "artifacts" / "prediction-batches" / "store-round-trip"
    report = score_snapshot(
        snapshot, model.artifact_dir, preprocessor.artifact_dir,
        batch_dir / "predictions.parquet", 0.5, "store-round-trip",
        splits=("holdout",),
    )
    frame = pd.read_parquet(report.output_path)
    assert "id_student" not in frame.columns
    assert store.write_predictions(frame, "store-round-trip") == len(frame)
    stored = store.fetch_predictions()
    assert len(stored) == len(frame)
    assert "id_student" not in stored.columns
    assert list(stored["student_key"]) == list(
        frame.sort_values(["code_module", "code_presentation", "student_key"])[
            "student_key"
        ]
    )
    assert store.count_run("store-round-trip") == len(frame)


# ---- CLI end-to-end with the database -----------------------------------------


def _postgres_available() -> bool:
    """True when the real local PostgreSQL accepts connections."""
    probe = PredictionStore(PredictionStoreConfig(dsn=PG_DSN, table=_test_table()))
    try:
        probe.ensure_schema()
        probe.drop_table()
        return True
    except Exception:
        return False
    finally:
        probe.close()


def test_cli_score_writes_predictions_to_postgres(model_run, capsys):
    if not _postgres_available():
        pytest.skip("postgres unavailable")  # cùng guard với các test DB khác
    root, snapshot, preprocessor, model = model_run
    table = _test_table()
    run_id = f"store-cli-{uuid.uuid4().hex[:8]}"
    store = PredictionStore(PredictionStoreConfig(dsn=PG_DSN, table=table))
    try:
        code = main([
            "ml", "score", "--snapshot", str(snapshot),
            "--model-artifact-dir", str(model.artifact_dir),
            "--preprocessing-artifact-dir", str(preprocessor.artifact_dir),
            "--artifacts-dir", str(root / "artifacts"),
            "--run-id", run_id,
            "--split", "holdout", "--pg-dsn", PG_DSN, "--pg-table", table,
        ])
        assert code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["database_write"] == "ok"
        assert report["database_rows"] == report["input_rows"]
        assert report["database_rows"] > 0
        assert Path(report["output_path"]).is_file()
        assert Path(report["batch_dir"], "success.json").is_file()
        assert store.count_run(run_id) == report["database_rows"]
    finally:
        store.drop_table()
        store.close()


def test_text_coerces_nan_to_null_no_db():
    """Hồi quy: NaN (float/np) phải thành NULL trong DB, không phải chuỗi 'nan'."""
    from learnguard.ml.predictions_store import _text

    assert _text(None) is None
    assert _text(pd.NA) is None
    assert _text(float("nan")) is None
    assert _text(np.nan) is None
    assert _text("high_uncertainty") == "high_uncertainty"
    assert _text("") == ""


def test_cli_score_without_dsn_skips_database_write(model_run, capsys, monkeypatch):
    root, snapshot, preprocessor, model = model_run
    monkeypatch.delenv("LEARNGUARD_PG_DSN", raising=False)
    code = main([
        "ml", "score", "--snapshot", str(snapshot),
        "--model-artifact-dir", str(model.artifact_dir),
        "--preprocessing-artifact-dir", str(preprocessor.artifact_dir),
        "--artifacts-dir", str(root / "artifacts"),
        "--run-id", "store-cli-file-only", "--split", "holdout",
    ])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["database_write"] == "skipped_no_dsn"
    assert report["database_rows"] is None
