"""ML-owner CLI layered on the data pipeline contract from PIPELINE_GUIDE."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from learnguard.ml.scoring import DEFAULT_SCORE_SPLITS

_DEFAULT_SNAPSHOT = Path("artifacts/feature_snapshot_m2/snapshot.parquet")


def _default_eval_run_id(prefix: str) -> str:
    """Default run-id cho evaluate: eval-{data|ml|all}-YYYYMMDD-HHMMSS."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}"


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

    train_xgb = commands.add_parser(
        "train-xgboost", help="fit fixed train-only CPU XGBoost production contract",
    )
    train_xgb.add_argument("--snapshot", type=Path, required=True)
    train_xgb.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    train_xgb.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    train_xgb.add_argument("--run-id", required=True)

    verify = commands.add_parser("verify-model")
    verify.add_argument("--model-artifact-dir", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--preprocessing-artifact-dir", type=Path, required=True)

    score = commands.add_parser("score", help="score a snapshot and publish prediction-batch.v1")
    score.add_argument("--snapshot", type=Path, required=True)
    score.add_argument("--model-artifact-dir", type=Path)
    score.add_argument("--preprocessing-artifact-dir", type=Path)
    score.add_argument("--release-artifact-dir", type=Path)
    score.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    score.add_argument("--run-id", required=True)
    score.add_argument("--threshold", type=float, default=None)
    score.add_argument(
        "--split", action="append", default=None,
        help=(
            "split to score (repeatable or comma-separated; raw XGBoost "
            "defaults to validation, legacy Logistic defaults to all three, "
            "release mode requires an explicit split)"
        ),
    )
    score.add_argument(
        "--pg-dsn", default=None,
        help="PostgreSQL DSN for the shared prediction store "
             "(falls back to LEARNGUARD_PG_DSN; file-only when absent). "
             "Production mode per docs/M5_PREDICTION_BATCH.md writes here",
    )
    score.add_argument("--pg-table", default="ml_predictions",
                       help="prediction store table name (default: ml_predictions)")
    score.add_argument(
        "--max-students", type=int, default=None,
        help=(
            "deterministic per-student subsample: keep only N students before "
            "publish (requires exactly one --split; forbidden in release mode)"
        ),
    )
    score.add_argument(
        "--seed", type=int, default=42,
        help="seed for --max-students subsampling (default: 42)",
    )

    explain = commands.add_parser("explain-shap", help="publish model-native SHAP artifacts")
    explain.add_argument("--snapshot", type=Path, required=True)
    explain.add_argument("--model-artifact-dir", type=Path, required=True)
    explain.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    explain.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    explain.add_argument("--run-id", required=True)
    explain.add_argument("--split", action="append", default=None)

    benchmark = commands.add_parser("benchmark", help="run validation selection and one-shot holdout evaluation")
    benchmark.add_argument("--snapshot", type=Path, required=True)
    benchmark.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    benchmark.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    benchmark.add_argument("--run-id", required=True)
    benchmark.add_argument(
        "--validation-only", action="store_true",
        help="fit/select on train+validation without loading or evaluating holdout",
    )

    comparison = commands.add_parser(
        "publish-model-comparison",
        help="render a verified validation-only Logistic/RF/XGBoost comparison",
    )
    comparison.add_argument("--benchmark-artifact-dir", type=Path, required=True)
    comparison.add_argument("--xgboost-model-artifact-dir", type=Path, required=True)
    comparison.add_argument("--logistic-model-artifact-dir", type=Path, required=True)
    comparison.add_argument("--snapshot", type=Path, required=True)
    comparison.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    comparison.add_argument(
        "--output", type=Path,
        default=Path("docs/MODEL_BENCHMARK_COMPARISON.md"),
    )

    release_eval = commands.add_parser(
        "evaluate-validation-release",
        help="build validation-only threshold and confusion-matrix artifacts",
    )
    release_eval.add_argument("--snapshot", type=Path, required=True)
    release_eval.add_argument("--benchmark-artifact-dir", type=Path, required=True)
    release_eval.add_argument("--model-artifact-dir", type=Path, required=True)
    release_eval.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    release_eval.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    release_eval.add_argument("--run-id", required=True)
    release_eval.add_argument(
        "--candidate-name", default=None,
        help="benchmark candidate matching the model artifact (default: dispatch by model type)",
    )

    create_release = commands.add_parser(
        "create-release", help="publish an immutable model release or validation candidate",
    )
    create_release.add_argument("--snapshot", type=Path, required=True)
    create_release.add_argument("--model-artifact-dir", type=Path, required=True)
    create_release.add_argument("--preprocessing-artifact-dir", type=Path, required=True)
    create_release.add_argument("--evaluation-artifact-dir", type=Path, required=True)
    create_release.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    create_release.add_argument("--run-id", required=True)
    create_release.add_argument(
        "--release-status", choices=("production", "validation_candidate"),
        default=None, help="default: status proven by the evaluation artifact",
    )
    create_release.add_argument(
        "--promotion-decision", type=Path, default=None,
        help="promotion-decision.v2 governance artifact required for XGBoost production",
    )
    create_release.add_argument(
        "--ml-gate-artifact-dir", type=Path, default=None,
        help="published passing ML-gate directory required for XGBoost production",
    )
    create_release.add_argument(
        "--rollback-release-artifact-dir", type=Path, default=None,
        help="immutable Logistic production release required as XGBoost rollback baseline",
    )

    verify_release = commands.add_parser(
        "verify-release", help="verify a model-release.v1/v2/v3/v4 artifact",
    )
    verify_release.add_argument("--release-artifact-dir", type=Path, required=True)
    verify_release.add_argument("--snapshot", type=Path, required=True)

    # ---- evaluate: evaluation gates cấp 1 (scorecard eval-report.v1) --------
    evaluate = root.add_parser(
        "evaluate", help="evaluation gates publishing eval-report.v1 scorecards",
    )
    eval_commands = evaluate.add_subparsers(dest="eval_command", required=True)

    eval_data = eval_commands.add_parser("data", help="Gate 1 — data quality scorecard")
    eval_data.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    eval_data.add_argument("--baseline-snapshot", type=Path, default=None)
    eval_data.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    eval_data.add_argument("--run-id", default=None)

    eval_ml = eval_commands.add_parser("ml", help="Gate 2 — ML model scorecard")
    eval_ml.add_argument("--model-artifact-dir", type=Path, default=None)
    eval_ml.add_argument("--preprocessing-artifact-dir", type=Path, default=None)
    eval_ml.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    eval_ml.add_argument("--min-roc-auc", type=float, default=0.70)
    eval_ml.add_argument("--baseline-model-dir", type=Path, default=None)
    eval_ml.add_argument(
        "--baseline-preprocessing-artifact-dir", type=Path, default=None,
        help="default: dùng preprocessing hiện tại",
    )
    eval_ml.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    eval_ml.add_argument("--run-id", default=None)

    eval_llm = eval_commands.add_parser(
        "llm", help="Gate 3 — LLM advisor-brief quality scorecard (read-only DB)",
    )
    eval_llm.add_argument(
        "--pg-dsn", default=None,
        help="PostgreSQL DSN của store chia sẻ "
             "(falls back to LEARNGUARD_PG_DSN; mọi check skipped khi trống)",
    )
    eval_llm.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    eval_llm.add_argument("--run-id", default=None)
    eval_llm.add_argument(
        "--limit", type=int, default=5000,
        help="số brief/calls tối đa quét (default: 5000)",
    )
    eval_llm.add_argument(
        "--judge-sample", type=int, default=10,
        help="số brief MỚI NHẤT judge chấm (default: 10; 0 = tắt judge_quality)",
    )
    eval_llm.add_argument(
        "--llm-api-key", default=None,
        help="API key Gemini cho judge_quality (falls back to env "
             "LLM_API_KEY rồi GEMINI_API_KEY; không key → check skipped). "
             "Key không bao giờ được ghi vào scorecard.",
    )
    eval_llm.add_argument(
        "--judge-model", default=None,
        help="model Gemini làm judge (default: gemini-3.1-flash-lite)",
    )

    eval_all = eval_commands.add_parser("all", help="Gate 1+2+3 — data + ml + llm cùng run-id")
    eval_all.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    eval_all.add_argument("--baseline-snapshot", type=Path, default=None)
    eval_all.add_argument("--model-artifact-dir", type=Path, default=None)
    eval_all.add_argument("--preprocessing-artifact-dir", type=Path, default=None)
    eval_all.add_argument("--min-roc-auc", type=float, default=0.70)
    eval_all.add_argument("--baseline-model-dir", type=Path, default=None)
    eval_all.add_argument(
        "--baseline-preprocessing-artifact-dir", type=Path, default=None,
        help="default: dùng preprocessing hiện tại",
    )
    eval_all.add_argument(
        "--pg-dsn", default=None,
        help="PostgreSQL DSN cho LLM gate (falls back to LEARNGUARD_PG_DSN)",
    )
    eval_all.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    eval_all.add_argument("--run-id", default=None)
    eval_all.add_argument("--limit", type=int, default=5000)
    eval_all.add_argument(
        "--judge-sample", type=int, default=10,
        help="pass-through LLM gate: số brief judge chấm (default: 10)",
    )
    eval_all.add_argument(
        "--llm-api-key", default=None,
        help="pass-through LLM gate: API key Gemini cho judge_quality "
             "(falls back to env LLM_API_KEY/GEMINI_API_KEY)",
    )
    eval_all.add_argument(
        "--judge-model", default=None,
        help="pass-through LLM gate: model Gemini làm judge",
    )
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
    if not report.database_write_allowed:
        raise ValueError("prediction_store_candidate_release_forbidden")
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
        if args.area == "evaluate":
            return _run_evaluate(args)
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
        elif args.command == "train-xgboost":
            from learnguard.ml.xgboost_training import train_xgboost_model
            result = train_xgboost_model(
                args.snapshot,
                args.preprocessing_artifact_dir,
                args.artifacts_dir,
                args.run_id,
            )
        elif args.command == "score":
            output_path = (
                args.artifacts_dir / "prediction-batches" / args.run_id
                / "predictions.parquet"
            )
            if args.release_artifact_dir is not None:
                if (
                    args.model_artifact_dir is not None
                    or args.preprocessing_artifact_dir is not None
                    or args.threshold is not None
                ):
                    raise ValueError("release_score_legacy_arguments_forbidden")
                if args.max_students is not None:
                    raise ValueError("max_students_release_mode_forbidden")
                if not args.split:
                    raise ValueError("release_score_explicit_split_required")
                from learnguard.ml.scoring import score_release_snapshot
                result = score_release_snapshot(
                    args.snapshot, args.release_artifact_dir, output_path,
                    args.run_id, splits=_requested_splits(args.split),
                )
            else:
                if (
                    args.model_artifact_dir is None
                    or args.preprocessing_artifact_dir is None
                ):
                    raise ValueError("legacy_score_artifact_arguments_required")
                from learnguard.ml.scoring import score_snapshot
                result = score_snapshot(
                    args.snapshot,
                    args.model_artifact_dir,
                    args.preprocessing_artifact_dir,
                    output_path,
                    0.5 if args.threshold is None else args.threshold,
                    args.run_id,
                    splits=(
                        None if args.split is None
                        else _requested_splits(args.split)
                    ),
                    max_students=args.max_students,
                    seed=args.seed,
                )
            _write_predictions_to_store(result, args)
        elif args.command == "explain-shap":
            from learnguard.ml.explainability import build_shap_artifact
            result = build_shap_artifact(
                args.snapshot, args.model_artifact_dir,
                args.preprocessing_artifact_dir, args.artifacts_dir,
                args.run_id, splits=(
                    None if args.split is None else _requested_splits(args.split)
                ),
            )
        elif args.command == "benchmark":
            from learnguard.ml.benchmark import run_benchmark
            result = run_benchmark(
                args.snapshot, args.preprocessing_artifact_dir,
                args.artifacts_dir, args.run_id,
                evaluation_scope=(
                    "validation_only" if args.validation_only else "full"
                ),
            )
        elif args.command == "publish-model-comparison":
            from learnguard.ml.model_comparison import publish_model_comparison
            result = publish_model_comparison(
                args.benchmark_artifact_dir,
                args.xgboost_model_artifact_dir,
                args.logistic_model_artifact_dir,
                args.snapshot,
                args.preprocessing_artifact_dir,
                args.output,
            )
        elif args.command == "evaluate-validation-release":
            from learnguard.ml.release import build_validation_evaluation_artifact
            result = build_validation_evaluation_artifact(
                args.snapshot, args.benchmark_artifact_dir,
                args.model_artifact_dir, args.preprocessing_artifact_dir,
                args.artifacts_dir, args.run_id, args.candidate_name,
            )
        elif args.command == "create-release":
            from learnguard.ml.release import build_model_release
            result = build_model_release(
                args.snapshot, args.model_artifact_dir,
                args.preprocessing_artifact_dir, args.evaluation_artifact_dir,
                args.artifacts_dir, args.run_id, args.release_status,
                args.promotion_decision, args.ml_gate_artifact_dir,
                args.rollback_release_artifact_dir,
            )
        elif args.command == "verify-release":
            from learnguard.ml.release import load_and_verify_model_release
            release = load_and_verify_model_release(
                args.release_artifact_dir, args.snapshot,
            )
            result = {
                "status": "ok", "release_id": release.release_id,
                "model_version": release.model_version,
                "snapshot_version": release.snapshot_version,
                "production_threshold": release.production_threshold,
                "release_status": release.release_status,
                "database_write_allowed": release.database_write_allowed,
            }
        else:
            from learnguard.ml.model_interface import verify_model_artifact
            result = verify_model_artifact(
                args.model_artifact_dir,
                args.snapshot,
                args.preprocessing_artifact_dir,
            )
        payload = result if isinstance(result, dict) else asdict(result)
        print(json.dumps(payload, indent=2, default=str))
        status = payload.get("status")
        return 0 if status == "ok" else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _require_ml_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    """`evaluate ml`/`all` cần model + preprocessing artifact — fail-loudly nếu thiếu."""
    if args.model_artifact_dir is None:
        raise ValueError("evaluate_ml_model_artifact_required")
    if args.preprocessing_artifact_dir is None:
        raise ValueError("evaluate_ml_preprocessing_artifact_required")
    return args.model_artifact_dir, args.preprocessing_artifact_dir


def _run_evaluate(args: argparse.Namespace) -> int:
    """Gate 1/2/3 — chạy evaluation gates, publish scorecard, exit 0/1 theo FAIL."""
    from learnguard.evaluation.data_gate import run_data_gate
    from learnguard.evaluation.llm_gate import resolve_pg_dsn, run_llm_gate
    from learnguard.evaluation.ml_gate import run_ml_gate
    from learnguard.evaluation.scorecard import (
        evaluate_exit_code,
        format_summary,
        publish_evaluation_run,
    )

    command = args.eval_command
    run_id = args.run_id or _default_eval_run_id(f"eval-{command}")
    if command == "data":
        scorecards = [run_data_gate(args.snapshot, args.baseline_snapshot, run_id)]
    elif command == "ml":
        model_dir, preprocessing_dir = _require_ml_inputs(args)
        scorecards = [run_ml_gate(
            model_dir, preprocessing_dir, args.snapshot, run_id,
            min_roc_auc=args.min_roc_auc,
            baseline_model_dir=args.baseline_model_dir,
            baseline_preprocessing_artifact_dir=(
                args.baseline_preprocessing_artifact_dir
            ),
        )]
    elif command == "llm":
        scorecards = [run_llm_gate(
            resolve_pg_dsn(args.pg_dsn), run_id, limit=args.limit,
            judge_sample=args.judge_sample,
            llm_api_key=args.llm_api_key,
            judge_model=args.judge_model,
        )]
    else:  # eval_command == "all"
        model_dir, preprocessing_dir = _require_ml_inputs(args)
        scorecards = [
            run_data_gate(args.snapshot, args.baseline_snapshot, run_id),
            run_ml_gate(
                model_dir, preprocessing_dir, args.snapshot, run_id,
                min_roc_auc=args.min_roc_auc,
                baseline_model_dir=args.baseline_model_dir,
                baseline_preprocessing_artifact_dir=(
                    args.baseline_preprocessing_artifact_dir
                ),
            ),
            # Gate 3 — LLM gate đọc chung PostgreSQL, publish thêm eval_llm.json
            run_llm_gate(
                resolve_pg_dsn(args.pg_dsn), run_id, limit=args.limit,
                judge_sample=args.judge_sample,
                llm_api_key=args.llm_api_key,
                judge_model=args.judge_model,
            ),
        ]
    run_dir = publish_evaluation_run(args.artifacts_dir, run_id, scorecards)
    for card in scorecards:
        print(format_summary(card))
    total: dict[str, int] = {}
    for card in scorecards:
        for key, value in card.summary().items():
            total[key] = total.get(key, 0) + value
    print(
        f"combined: total={total.get('total', 0)} pass={total.get('pass', 0)} "
        f"warn={total.get('warn', 0)} fail={total.get('fail', 0)} "
        f"skipped={total.get('skipped', 0)}"
    )
    print(f"evaluation_run_dir: {run_dir}")
    return evaluate_exit_code(scorecards)


def run_pipeline_command(data_dir: Path, artifacts_dir: Path, cutoff_day: int) -> int:
    """Data orchestration remains owned by the existing pipeline guide."""
    raise NotImplementedError("use PIPELINE_GUIDE data commands")


if __name__ == "__main__":
    raise SystemExit(main())
