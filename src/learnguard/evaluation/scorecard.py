"""Scorecard schema `eval-report.v1` — hợp đồng dùng chung bởi mọi evaluation gate.

Một gate trả về một `Scorecard`: danh sách `GateCheck` (pass/warn/fail/skipped)
+ summary đếm theo trạng thái. Publish ATOMIC vào
`artifacts/evaluations/<run-id>/` (scorecard JSON + manifest.json +
success.json chứa manifest_sha256) — copy pattern staging + `os.replace` của
scoring.py: run-id trùng → `FileExistsError`, không bao giờ ghi đè.

Exit code do CLI quyết định: 0 khi không check nào FAIL, 1 khi ≥1 FAIL
(WARN không fail).
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from learnguard.ml.preprocessing import _file_hash, _validate_run_id

EVAL_SCHEMA_VERSION = "eval-report.v1"
CHECK_STATUSES: tuple[str, ...] = ("pass", "warn", "fail", "skipped")
SUBJECTS: tuple[str, ...] = ("data", "ml", "llm")
SCORECARD_FILENAME: dict[str, str] = {
    "data": "eval_data.json", "ml": "eval_ml.json", "llm": "eval_llm.json",
}
EVALUATIONS_DIRNAME = "evaluations"
MANIFEST_FILENAME = "manifest.json"
SUCCESS_FILENAME = "success.json"


@dataclass(frozen=True)
class GateCheck:
    """Một check trong scorecard — id cố định theo hợp đồng gate."""

    id: str
    name: str
    status: str
    observed: dict[str, object] = field(default_factory=dict)
    threshold: str | None = None
    details: str = ""

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError(
                f"evaluation_check_status_invalid:{self.status}:"
                f"allowed={','.join(CHECK_STATUSES)}"
            )


@dataclass
class Scorecard:
    """Bảng điểm của một gate cho một subject (`data` | `ml` | `llm`)."""

    run_id: str
    subject: str
    checks: list[GateCheck] = field(default_factory=list)
    inputs: dict[str, str | None] = field(default_factory=dict)
    schema_version: str = EVAL_SCHEMA_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.subject not in SUBJECTS:
            raise ValueError(
                f"evaluation_subject_invalid:{self.subject}:"
                f"allowed={','.join(SUBJECTS)}"
            )

    def add(self, check: GateCheck) -> GateCheck:
        self.checks.append(check)
        return check

    def summary(self) -> dict[str, int]:
        counts = {status: 0 for status in CHECK_STATUSES}
        for check in self.checks:
            counts[check.status] += 1
        return {"total": len(self.checks), **counts}

    def has_failure(self) -> bool:
        return any(check.status == "fail" for check in self.checks)

    def worst_status(self) -> str:
        if self.has_failure():
            return "fail"
        if any(check.status == "warn" for check in self.checks):
            return "warn"
        return "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "subject": self.subject,
            "created_at": self.created_at,
            "inputs": dict(self.inputs),
            "checks": [
                {
                    "id": check.id,
                    "name": check.name,
                    "status": check.status,
                    "threshold": check.threshold,
                    "observed": _jsonable(check.observed),
                    "details": check.details,
                }
                for check in self.checks
            ],
            "summary": self.summary(),
        }


def publish_evaluation_run(
    artifacts_dir: Path | str, run_id: str, scorecards: list[Scorecard],
) -> Path:
    """Publish ATOMIC một hoặc nhiều scorecard vào `evaluations/<run-id>/`.

    Cùng run-id = cùng một thư mục chạy đánh giá; `evaluate all` publish cả
    `eval_data.json` lẫn `eval_ml.json` vào đó. Run-id đã tồn tại →
    `FileExistsError` (không ghi đè artifact đánh giá).
    """
    run_id = _validate_run_id(run_id)
    if not scorecards:
        raise ValueError("evaluation_scorecards_empty")
    subjects = [card.subject for card in scorecards]
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"evaluation_duplicate_subject:{','.join(subjects)}")
    root = Path(artifacts_dir) / EVALUATIONS_DIRNAME
    target = root / run_id
    if target.exists():
        raise FileExistsError(f"evaluation_run_id_exists:{run_id}")
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".{run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        files: dict[str, str] = {}
        for card in scorecards:
            path = staging / SCORECARD_FILENAME[card.subject]
            path.write_text(
                json.dumps(card.to_dict(), indent=2, ensure_ascii=False) + "\n"
            )
            files[path.name] = _file_hash(path)
        manifest = {
            "schema_version": EVAL_SCHEMA_VERSION,
            "run_id": run_id,
            "subjects": subjects,
            "scorecards": files,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        worst = max((card.worst_status() for card in scorecards), key=lambda s: ("pass", "warn", "fail").index(s))
        (staging / SUCCESS_FILENAME).write_text(json.dumps({
            "schema_version": EVAL_SCHEMA_VERSION,
            "run_id": run_id,
            "status": worst,
            "manifest_sha256": _file_hash(manifest_path),
        }, indent=2, sort_keys=True) + "\n")
        if target.exists():
            raise FileExistsError(f"evaluation_run_id_exists:{run_id}")
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def format_summary(card: Scorecard) -> str:
    """Bảng tóm tắt stdout: một dòng mỗi check `[STATUS] id — details`."""
    lines = [f"== scorecard {card.subject} (run_id={card.run_id}) =="]
    for check in card.checks:
        lines.append(f"[{check.status.upper()}] {check.id} — {check.details}")
    summary = card.summary()
    lines.append(
        "summary: total={total} pass={pass} warn={warn} fail={fail} "
        "skipped={skipped}".format(**summary)
    )
    return "\n".join(lines)


def evaluate_exit_code(scorecards: list[Scorecard]) -> int:
    """0 khi không check nào FAIL; 1 khi ≥1 FAIL. WARN không fail."""
    return 1 if any(card.has_failure() for card in scorecards) else 0


def _jsonable(value: object) -> object:
    """Chuẩn hoá observed về kiểu JSON an toàn (numpy → python, float16/32 → float)."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value
    # numpy scalar / numpy bool: quy về python gốc qua item() khi được.
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)
