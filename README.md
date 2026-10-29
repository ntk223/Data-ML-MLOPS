# LearnGuard ML/Data

Batch repository for LearnGuard. The OULAD data boundary (M1–M3) is implemented:

- M1 validates seven OULAD tables from a ZIP or CSV directory, quarantines invalid rows, deduplicates exact events and writes canonical Parquet.
- M2 creates a leakage-safe point-in-time snapshot at day 27, including total clicks,
  active days, click trend and eligible assessment features.
- M3 recomputes the full snapshot, injects future events, validates provenance and enforces quality gates.
- Phase 4 fits a deterministic preprocessing pipeline on the temporal train split only,
  then verifies validation and holdout with the same fitted state.

Phase 5 fits a fixed Logistic Regression baseline from the approved preprocessing
artifact. Evaluation, threshold selection, scoring and prediction-batch export remain
separate work. See [`docs/M4_LOGISTIC_BASELINE.md`](docs/M4_LOGISTIC_BASELINE.md) for
the model-only runbook and [`docs/IMPLEMENTATION_TODO.md`](docs/IMPLEMENTATION_TODO.md)
for later phases.

## Quickstart

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest==9.1.1
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest tests -q

.venv/bin/learnguard data run \
  --source data/archive.zip \
  --artifacts-dir artifacts \
  --cutoff-day 27
```

The command reads the tracked ZIP directly; manual extraction is not required. Outputs are
isolated under `artifacts/data-runs/<run-id>/` and the pipeline stops on the first failed
stage or gate.

For a drift/reproducibility run, pass an approved earlier snapshot:

```bash
.venv/bin/learnguard data run \
  --source data/archive.zip \
  --artifacts-dir artifacts \
  --baseline-snapshot artifacts/data-runs/<first-run>/snapshot/snapshot.parquet
```

See [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md) for stage commands, artifact
contracts and verification results.

After a successful data run, build and verify the train-only preprocessor:

```bash
.venv/bin/learnguard ml preprocess \
  --data-run-dir artifacts/data-runs/<data-run-id> \
  --artifacts-dir artifacts \
  --run-id <preprocessing-run-id>

.venv/bin/learnguard ml verify-preprocessor \
  --artifact-dir artifacts/preprocessing-runs/<preprocessing-run-id> \
  --data-run-dir artifacts/data-runs/<data-run-id>
```

The fitted artifact is bound to its verified `snapshot-v3`. Treat Joblib files as
trusted local artifacts only: checksums detect accidental or partial tampering, but do
not make an untrusted pickle safe to load.

Train and verify the fixed Logistic Regression baseline as a separate ML stage:

```bash
.venv/bin/learnguard ml train-logistic \
  --data-run-dir artifacts/data-runs/<data-run-id> \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<preprocessing-run-id> \
  --artifacts-dir artifacts \
  --run-id <model-run-id>

.venv/bin/learnguard ml verify-model \
  --model-artifact-dir artifacts/model-runs/<model-run-id> \
  --data-run-dir artifacts/data-runs/<data-run-id> \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<preprocessing-run-id>
```

The command fits only transformed train rows. It does not load validation or holdout,
refit preprocessing, tune parameters/thresholds, or calculate predictive metrics.
Model Joblib files have the same trusted-local-source restriction as preprocessing.

## Development

- Python: 3.11+; verified with 3.11.
- Runtime dependencies are pinned in `requirements.txt`.
- Pull requests run fixture-based tests; the manual/scheduled
  `full-oulad-verification` workflow executes the complete 464 MB dataset twice.
- Raw extracted data, virtual environments and generated artifacts are ignored by Git.
