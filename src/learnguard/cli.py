"""ML-owner CLI layered on the data pipeline contract from PIPELINE_GUIDE."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


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
    return parser


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
