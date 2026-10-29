"""Fail-fast orchestration for the OULAD M1–M3 data pipeline."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from learnguard.data.features import build_feature_snapshot
from learnguard.data.ingestion import ingest_oulad
from learnguard.data.leakage import audit_snapshot
from learnguard.data.quality import validate_snapshot
from learnguard.data.snapshot_contract import validate_cutoff
from learnguard.settings import CUTOFF_DAY, validate_run_id


PIPELINE_STAGES = ("ingestion", "feature_snapshot", "leakage_audit", "quality_gate")


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    artifacts_dir: Path
    cutoff_day: int = CUTOFF_DAY
    baseline_snapshot: Path | None = None
    run_id: str | None = None


@dataclass
class StageResult:
    stage: str
    status: str
    output_paths: list[Path] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineReport:
    run_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    stages: list[StageResult] = field(default_factory=list)
    report_path: Path | None = None


class LearnGuardPipeline:
    """Run ingestion, snapshot, leakage audit and quality gate in order."""

    def __init__(self, config: PipelineConfig) -> None:
        validate_cutoff(config.cutoff_day)
        self.config = config
        generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_id = validate_run_id(
            generated if config.run_id is None else config.run_id
        )
        self.run_dir = Path(config.artifacts_dir) / "data-runs" / self.run_id
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(f"run_id already exists:{self.run_id}")
        self.ingestion_dir = self.run_dir / "ingestion"
        self.snapshot_path = self.run_dir / "snapshot" / "snapshot.parquet"
        self.reports_dir = self.run_dir / "reports"

    def run(self) -> PipelineReport:
        report = PipelineReport(self.run_id, "running", datetime.now(timezone.utc))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        for stage in PIPELINE_STAGES:
            try:
                result = self.run_stage(stage)
            except Exception as exc:
                result = StageResult(stage, "failed", error=f"{type(exc).__name__}: {exc}")
            report.stages.append(result)
            if result.status != "ok":
                report.status = "failed"
                break
        else:
            report.status = "ok"
        report.finished_at = datetime.now(timezone.utc)
        report.report_path = self.reports_dir / "pipeline_report.json"
        report.report_path.write_text(json.dumps(asdict(report), indent=2, default=str))
        return report

    def run_stage(self, stage_name: str) -> StageResult:
        if stage_name not in PIPELINE_STAGES:
            raise ValueError(f"unknown_stage:{stage_name}")
        if stage_name == "ingestion":
            result = ingest_oulad(self.config.data_dir, self.ingestion_dir)
            published_outputs = [
                path
                for path in (
                    self.ingestion_dir / "ingestion_report.json",
                    self.ingestion_dir / "ingestion.success.json",
                    self.ingestion_dir / "canonical",
                )
                if path.exists()
            ]
            return StageResult(
                stage_name, "ok" if result.status == "ok" else "failed",
                published_outputs,
                {"row_counts": result.row_counts,
                 "quarantined_row_counts": result.quarantined_row_counts,
                 "deduped_row_counts": result.deduped_row_counts,
                 "canonical_content_hashes": result.canonical_content_hashes,
                 "warnings": result.warnings},
                ";".join(result.errors) if result.status != "ok" else None,
            )
        if stage_name == "feature_snapshot":
            result = build_feature_snapshot(
                self.ingestion_dir, self.snapshot_path, self.config.cutoff_day
            )
            return StageResult(
                stage_name, "ok", [result.snapshot_path, result.ledger_path,
                                    result.snapshot_path.parent / "snapshot.success.json"],
                {"row_count": result.row_count, "snapshot_version": result.snapshot_version,
                 "insufficient_data_count": result.insufficient_data_count,
                 "cohort_exclusions": result.cohort_exclusions},
            )
        if stage_name == "leakage_audit":
            audit = audit_snapshot(self.snapshot_path, self.ingestion_dir, self.config.cutoff_day)
            path = self.reports_dir / "leakage_audit.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(audit), indent=2, default=str))
            return StageResult(
                stage_name, "ok" if audit.status == "ok" else "failed", [path],
                {"checked_features": audit.checked_features,
                 "source_checks": audit.source_checks,
                 "violations": len(audit.violations)},
                None if audit.status == "ok" else "leakage audit violated",
            )
        quality = validate_snapshot(self.snapshot_path, self.config.baseline_snapshot)
        path = self.reports_dir / "quality_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(quality), indent=2, default=str))
        return StageResult(
            stage_name, "ok" if quality.status == "ok" else "failed", [path],
            {"row_count": quality.row_count, "volume_ratio": quality.volume_ratio,
             "gate_status": quality.gate_status, "psi_by_feature": quality.psi_by_feature},
            None if quality.status == "ok" else ";".join(quality.errors),
        )


def run_data_pipeline(source: Path, artifacts_dir: Path, cutoff_day: int = CUTOFF_DAY,
                      baseline_snapshot: Path | None = None,
                      run_id: str | None = None) -> PipelineReport:
    """Public M1–M3 pipeline entry point."""
    config = PipelineConfig(Path(source), Path(artifacts_dir), cutoff_day,
                            Path(baseline_snapshot) if baseline_snapshot else None, run_id)
    return LearnGuardPipeline(config).run()


def run_pipeline(data_dir: Path, artifacts_dir: Path,
                 cutoff_day: int = CUTOFF_DAY) -> PipelineReport:
    """Backward-compatible alias for the data-only pipeline."""
    return run_data_pipeline(data_dir, artifacts_dir, cutoff_day)
