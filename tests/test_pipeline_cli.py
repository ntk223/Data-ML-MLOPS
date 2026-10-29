"""Orchestration and CLI behavior for the M1–M3 data boundary."""

import json

import pytest

from learnguard.cli import build_parser, main
from learnguard.pipeline import PIPELINE_STAGES, run_data_pipeline
from learnguard.settings import Settings
from tests.conftest import write_csvs
from tests.test_loading import _tables_all_presentations


def test_data_pipeline_fails_fast_at_ingestion(tmp_path):
    artifacts = tmp_path / "artifacts"
    report = run_data_pipeline(
        tmp_path / "missing.zip", artifacts, run_id="failed-run"
    )
    assert report.status == "failed"
    assert [s.stage for s in report.stages] == ["ingestion"]
    assert report.report_path.is_file()
    ingestion_dir = artifacts / "data-runs/failed-run/ingestion"
    ingestion_report = ingestion_dir / "ingestion_report.json"
    assert report.stages[0].output_paths == [ingestion_report]
    assert ingestion_report.is_file()
    assert not (ingestion_dir / "ingestion.success.json").exists()
    assert not (ingestion_dir / "canonical").exists()
    serialized = json.loads(report.report_path.read_text())
    assert serialized["stages"][0]["output_paths"] == [str(ingestion_report)]
    assert PIPELINE_STAGES == (
        "ingestion", "feature_snapshot", "leakage_audit", "quality_gate"
    )


def test_pipeline_refuses_to_overwrite_existing_run(tmp_path):
    run_dir = tmp_path / "artifacts/data-runs/existing"
    run_dir.mkdir(parents=True)
    (run_dir / "marker").write_text("keep")
    try:
        run_data_pipeline(tmp_path / "missing.zip", tmp_path / "artifacts", run_id="existing")
    except FileExistsError as exc:
        assert "run_id already exists" in str(exc)
    else:
        raise AssertionError("existing run must not be overwritten")


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", "nested/run", r"..\escape", "/tmp/escape", "has space"],
)
def test_data_pipeline_rejects_unsafe_run_ids_before_path_creation(tmp_path, run_id):
    artifacts = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="invalid_run_id"):
        run_data_pipeline(tmp_path / "missing.zip", artifacts, run_id=run_id)
    assert not artifacts.exists()


def test_full_fixture_pipeline_is_reproducible_with_baseline(tmp_path):
    source = write_csvs(tmp_path / "source", _tables_all_presentations())
    first = run_data_pipeline(source, tmp_path / "artifacts", run_id="first")
    assert first.status == "ok", first.stages
    baseline = tmp_path / "artifacts/data-runs/first/snapshot/snapshot.parquet"
    second = run_data_pipeline(
        source, tmp_path / "artifacts", baseline_snapshot=baseline, run_id="second"
    )
    assert second.status == "ok", second.stages
    quality = second.stages[-1].summary
    assert quality["volume_ratio"] == 1.0
    assert quality["gate_status"]["volume"] == "passed"
    assert quality["gate_status"]["psi"] == "not_applicable"  # fixture < 200 rows


def test_cli_exposes_total_and_stage_commands():
    parser = build_parser()
    args = parser.parse_args(["data", "run", "--run-id", "x"])
    assert args.command == "run" and args.cutoff_day == 27
    cases = (
        ["data", "ingest", "--source", "x", "--output-dir", "y"],
        ["data", "snapshot", "--processed-dir", "x", "--output", "y"],
        ["data", "audit", "--snapshot", "x", "--processed-dir", "y"],
        ["data", "validate", "--snapshot", "x"],
    )
    assert [parser.parse_args(case).command for case in cases] == [
        "ingest", "snapshot", "audit", "validate"
    ]


def test_cli_returns_nonzero_on_failed_pipeline(tmp_path):
    code = main([
        "data", "run", "--source", str(tmp_path / "missing.zip"),
        "--artifacts-dir", str(tmp_path / "artifacts"), "--run-id", "cli-fail",
    ])
    assert code == 1


def test_cli_rejects_unsafe_run_id(tmp_path):
    code = main([
        "data", "run", "--source", str(tmp_path / "missing.zip"),
        "--artifacts-dir", str(tmp_path / "artifacts"), "--run-id", "../escape",
    ])
    assert code == 2
    assert not (tmp_path / "escape").exists()


def test_cli_rejects_cutoff_other_than_27():
    with pytest.raises(SystemExit) as exc:
        main(["data", "run", "--cutoff-day", "28"])
    assert exc.value.code == 2


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("LEARNGUARD_SOURCE", str(tmp_path / "source.zip"))
    monkeypatch.setenv("LEARNGUARD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("LEARNGUARD_CUTOFF_DAY", "27")
    settings = Settings.from_environment()
    assert settings.data_dir == tmp_path / "source.zip"
    assert settings.artifacts_dir == tmp_path / "artifacts"
    assert settings.cutoff_day == 27
