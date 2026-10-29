"""Command-line interface for the OULAD data and preprocessing pipelines."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from learnguard.data.features import build_feature_snapshot
from learnguard.data.ingestion import ingest_oulad
from learnguard.data.leakage import audit_snapshot
from learnguard.data.quality import validate_snapshot
from learnguard.pipeline import run_data_pipeline
from learnguard.settings import CUTOFF_DAY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learnguard")
    root = parser.add_subparsers(dest="area", required=True)
    data = root.add_parser("data", help="OULAD data pipeline commands")
    commands = data.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run ingestion through quality gate")
    run.add_argument("--source", type=Path, default=Path("data/archive.zip"))
    run.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    run.add_argument("--cutoff-day", type=int, choices=[CUTOFF_DAY], default=CUTOFF_DAY)
    run.add_argument("--baseline-snapshot", type=Path)
    run.add_argument("--run-id")

    ingest = commands.add_parser("ingest", help="validate CSV/ZIP and write canonical Parquet")
    ingest.add_argument("--source", type=Path, required=True)
    ingest.add_argument("--output-dir", type=Path, required=True)

    snapshot = commands.add_parser("snapshot", help="build the day-27 point-in-time snapshot")
    snapshot.add_argument("--processed-dir", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--cutoff-day", type=int, choices=[CUTOFF_DAY], default=CUTOFF_DAY)

    audit = commands.add_parser("audit", help="run leakage and provenance audit")
    audit.add_argument("--snapshot", type=Path, required=True)
    audit.add_argument("--processed-dir", type=Path, required=True)
    audit.add_argument("--cutoff-day", type=int, choices=[CUTOFF_DAY], default=CUTOFF_DAY)

    validate = commands.add_parser("validate", help="run snapshot quality gates")
    validate.add_argument("--snapshot", type=Path, required=True)
    validate.add_argument("--baseline-snapshot", type=Path)

    ml = root.add_parser("ml", help="ML preparation commands")
    ml_commands = ml.add_subparsers(dest="command", required=True)
    preprocess = ml_commands.add_parser(
        "preprocess", help="fit and publish the train-only preprocessor"
    )
    preprocess.add_argument("--data-run-dir", type=Path, required=True)
    preprocess.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    preprocess.add_argument("--run-id")
    verify = ml_commands.add_parser(
        "verify-preprocessor", help="verify and replay a preprocessing artifact"
    )
    verify.add_argument("--artifact-dir", type=Path, required=True)
    verify.add_argument("--data-run-dir", type=Path, required=True)
    train_logistic = ml_commands.add_parser(
        "train-logistic", help="fit and publish the fixed train-only Logistic baseline"
    )
    train_logistic.add_argument("--data-run-dir", type=Path, required=True)
    train_logistic.add_argument(
        "--preprocessing-artifact-dir", type=Path, required=True
    )
    train_logistic.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    train_logistic.add_argument("--run-id")
    verify_model = ml_commands.add_parser(
        "verify-model", help="verify a Logistic model artifact and its provenance"
    )
    verify_model.add_argument("--model-artifact-dir", type=Path, required=True)
    verify_model.add_argument("--data-run-dir", type=Path, required=True)
    verify_model.add_argument(
        "--preprocessing-artifact-dir", type=Path, required=True
    )
    return parser


def _print(value: object) -> None:
    print(json.dumps(asdict(value), indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.area == "ml":
            from learnguard.ml.preprocessing import (
                build_preprocessing_artifact,
                verify_preprocessing_artifact,
            )
            from learnguard.ml.training import (
                train_logistic_baseline,
                verify_logistic_model_artifact,
            )

            if args.command == "preprocess":
                result = build_preprocessing_artifact(
                    args.data_run_dir, args.artifacts_dir, args.run_id
                )
            elif args.command == "verify-preprocessor":
                result = verify_preprocessing_artifact(
                    args.artifact_dir, args.data_run_dir
                )
            elif args.command == "train-logistic":
                result = train_logistic_baseline(
                    args.data_run_dir,
                    args.preprocessing_artifact_dir,
                    args.artifacts_dir,
                    args.run_id,
                )
            else:
                result = verify_logistic_model_artifact(
                    args.model_artifact_dir,
                    args.data_run_dir,
                    args.preprocessing_artifact_dir,
                )
            _print(result)
            return 0 if result.status == "ok" else 1
        if args.command == "run":
            report = run_data_pipeline(
                args.source, args.artifacts_dir, args.cutoff_day,
                args.baseline_snapshot, args.run_id,
            )
            _print(report)
            return 0 if report.status == "ok" else 1
        if args.command == "ingest":
            report = ingest_oulad(args.source, args.output_dir)
            _print(report)
            return 0 if report.status == "ok" else 1
        if args.command == "snapshot":
            _print(build_feature_snapshot(args.processed_dir, args.output, args.cutoff_day))
            return 0
        if args.command == "audit":
            report = audit_snapshot(args.snapshot, args.processed_dir, args.cutoff_day)
            _print(report)
            return 0 if report.status == "ok" else 1
        report = validate_snapshot(args.snapshot, args.baseline_snapshot)
        _print(report)
        return 0 if report.status == "ok" else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def run_pipeline_command(data_dir: Path, artifacts_dir: Path,
                         cutoff_day: int) -> int:
    report = run_data_pipeline(data_dir, artifacts_dir, cutoff_day)
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
