"""Architecture routing for the XGBoost runtime distribution."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def _xgboost_requirements() -> tuple[list[Requirement], list[Requirement]]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pyproject = [
        Requirement(value) for value in project["project"]["dependencies"]
        if Requirement(value).name.startswith("xgboost")
    ]
    locked = [
        Requirement(line) for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#") and Requirement(line).name.startswith("xgboost")
    ]
    return pyproject, locked


def _selected(requirements: list[Requirement], system: str, machine: str) -> list[str]:
    environment = {
        "platform_system": system,
        "platform_machine": machine,
        "python_version": "3.11",
        "python_full_version": "3.11.15",
        "implementation_name": "cpython",
        "implementation_version": "3.11.15",
        "os_name": "posix" if system != "Windows" else "nt",
        "platform_python_implementation": "CPython",
        "sys_platform": {"Linux": "linux", "Darwin": "darwin", "Windows": "win32"}[system],
        "extra": "",
    }
    return [req.name for req in requirements if req.marker is None or req.marker.evaluate(environment)]


def test_xgboost_distribution_markers_are_complete_and_exclusive():
    pyproject, locked = _xgboost_requirements()
    cases = {
        ("Linux", "x86_64"): ["xgboost-cpu"],
        ("Linux", "aarch64"): ["xgboost"],
        ("Linux", "arm64"): ["xgboost"],
        ("Darwin", "arm64"): ["xgboost"],
        ("Darwin", "x86_64"): ["xgboost"],
        ("Windows", "AMD64"): ["xgboost"],
    }
    for environment, expected in cases.items():
        assert _selected(pyproject, *environment) == expected
        assert _selected(locked, *environment) == expected
