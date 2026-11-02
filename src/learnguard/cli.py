"""ML-owner CLI layered on the data pipeline contract from PIPELINE_GUIDE."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from learnguard.ml.scoring import DEFAULT_SCORE_SPLITS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learnguard")
    root = parser.add_subparsers(dest="area", required=True)
    ml = root.add_parser("ml", help="ML artifact commands")
    commands = ml.add_subparsers(dest="command", required=True)

    preprocess = commands.add_parser("preprocess", help="fit train-only preprocessing")
    preprocess.add_argument("--snapshot", type=Path, required=True)
    preprocess.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    preprocess.add_argument("--run-id", required=True)

    verify_pre = commands.add_parser("verify-preprocessor")
    verify_pre.add_argument("--artifact-dir", type=Path, required=True)
    verify_pre.add_argument("--snapshot", type=Path, required=True)

    train = commands.add_parser("train-logistic", help="fit fixed Logistic baseline")
    train.add_argument("--snapshot", type=Path, required=True)
    train.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    train.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--run-id", required=True)

    verify = commands.add_parser("verify-model")
    verify.add_argument("--model-artifact-dir", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--preprocessing-artifact-dir", type=Path, required=True)

    score = commands.add_parser("score", help="score a snapshot and publish prediction-batch.v1")
    score.add_argument("--snapshot", type=Path, required=True)
    score.add_argument("--model-artifact-dir", type=Path, required=True)
    score.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    score.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    score.add_argument("--run-id", required=True)
    score.add_argument("--threshold", type=float, default=0.5)
    score.add_argument(
        "--split", action="append", default=None,
        help="split to score (repeatable or comma-separated; default: all three)",
    )
    score.add_argument(
        "--pg-dsn", default=None,
        help="PostgreSQL DSN for the shared prediction store "
             "(falls back to LEARNGUARD_PG_DSN; file-only when absent). "
             "Production mode per docs/M5_PREDICTION_BATCH.md writes here",
    )
    score.add_argument("--pg-table", default="ml_predictions",
                       help="prediction store table name (default: ml_predictions)")
    return parser


def _requested_splits(values: Sequence[str] | None) -> tuple[str, ...]:
    """Parse repeatable/comma-separated `--split` values; empty = all three."""
    if not values:
        return DEFAULT_SCORE_SPLITS
    names = tuple(
        name.strip() for value in values for name in value.split(",") if name.strip()
    )
    return names or DEFAULT_SCORE_SPLITS


def _write_predictions_to_store(report: object, args: argparse.Namespace) -> None:
    """After the file publish, upsert the same rows into PostgreSQL.

    The published files are the audit trail and stay untouched on DB failure:
    an exception here propagates to the caller, which exits 2 while the batch
    directory remains. With no DSN (flag or `LEARNGUARD_PG_DSN`) the write is
    skipped and the report says so. The DSN itself is never echoed.
    """
    import os

    from learnguard.ml.predictions_store import (
        PredictionStore,
        PredictionStoreConfig,
    )

    dsn = args.pg_dsn or os.environ.get("LEARNGUARD_PG_DSN", "")
    if not dsn:
        return
    import pandas as pd

    from learnguard.ml.scoring import ScoringReport

    if not isinstance(report, ScoringReport):
        raise ValueError("prediction_store_report_invalid")
    frame = pd.read_parquet(report.output_path)
    store = PredictionStore(PredictionStoreConfig(dsn=dsn, table=args.pg_table))
    with store:
        store.ensure_schema()
        written = store.write_predictions(frame, args.run_id)
        stored = store.count_run(args.run_id)
    if stored != written:
        raise ValueError(
            f"prediction_store_rowcount_mismatch:written={written}:stored={stored}"
        )
    report.database_write = "ok"
    report.database_rows = stored


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preprocess":
            from learnguard.ml.preprocessing import build_preprocessing_artifact
            result = build_preprocessing_artifact(
                args.snapshot, args.artifacts_dir, args.run_id
            )
        elif args.command == "verify-preprocessor":
            from learnguard.ml.preprocessing import verify_preprocessing_artifact
            result = verify_preprocessing_artifact(args.artifact_dir, args.snapshot)
        elif args.command == "train-logistic":
            from learnguard.ml.training import train_logistic_baseline
            result = train_logistic_baseline(
                args.snapshot,
                args.preprocessing_artifact_dir,
                args.artifacts_dir,
                args.run_id,
            )
        elif args.command == "score":
            from learnguard.ml.scoring import score_snapshot
            result = score_snapshot(
                args.snapshot,
                args.model_artifact_dir,
                args.preprocessing_artifact_dir,
                args.artifacts_dir / "prediction-batches" / args.run_id / "predictions.parquet",
                args.threshold,
                args.run_id,
                splits=_requested_splits(args.split),
            )
            _write_predictions_to_store(result, args)
        else:
            from learnguard.ml.training import verify_logistic_model_artifact
            result = verify_logistic_model_artifact(
                args.model_artifact_dir,
                args.snapshot,
                args.preprocessing_artifact_dir,
            )
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0 if result.status == "ok" else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def run_pipeline_command(data_dir: Path, artifacts_dir: Path, cutoff_day: int) -> int:
    """Data orchestration remains owned by the existing pipeline guide."""
    raise NotImplementedError("use PIPELINE_GUIDE data commands")


if __name__ == "__main__":
    raise SystemExit(main())
