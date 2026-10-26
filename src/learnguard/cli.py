"""CLI contracts for module-level and end-to-end commands."""

from pathlib import Path
from typing import Sequence


def build_parser():
    """TODO: define pipeline, ingest, snapshot, train, and score commands."""
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """TODO: parse arguments, dispatch to core modules, and return exit code."""
    raise NotImplementedError


def run_pipeline_command(data_dir: Path, artifacts_dir: Path,
                         cutoff_day: int) -> int:
    """TODO: invoke pipeline and report its final state."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
