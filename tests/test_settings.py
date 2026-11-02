"""Settings.from_environment tests: all-optional env contract, no secrets."""

import pytest

from learnguard.settings import CUTOFF_DAY, Settings


def test_empty_environment_yields_defaults():
    settings = Settings.from_environment({})
    assert settings.cutoff_day == CUTOFF_DAY
    assert settings.data_dir.name == "oulad"
    assert settings.artifacts_dir == Settings().artifacts_dir
    assert settings.mlflow_tracking_uri == ""
    assert settings.pg_dsn == ""


def test_reads_pg_dsn_and_artifacts_dir():
    settings = Settings.from_environment({
        "LEARNGUARD_PG_DSN": "postgresql://learnguard:learnguard@localhost:5432/learnguard",
        "LEARNGUARD_ARTIFACTS_DIR": "/srv/artifacts",
        "LEARNGUARD_DATA_DIR": "data/oulad-v2",
        "LEARNGUARD_MLFLOW_TRACKING_URI": "http://localhost:5000",
    })
    assert settings.pg_dsn.startswith("postgresql://")
    assert settings.artifacts_dir.name == "artifacts"
    assert settings.data_dir.name == "oulad-v2"
    assert settings.mlflow_tracking_uri.startswith("http://")


def test_reads_os_environment_by_default(monkeypatch):
    monkeypatch.setenv("LEARNGUARD_PG_DSN", "postgresql://env-only")
    settings = Settings.from_environment()
    assert settings.pg_dsn == "postgresql://env-only"


def test_cutoff_day_override_and_invalid_value():
    assert Settings.from_environment({"LEARNGUARD_CUTOFF_DAY": "14"}).cutoff_day == 14
    with pytest.raises(ValueError, match="settings_cutoff_day_invalid"):
        Settings.from_environment({"LEARNGUARD_CUTOFF_DAY": "week4"})
