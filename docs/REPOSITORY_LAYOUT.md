# ML/Data repository boundary

This Git repository owns M1–M5 batch processing and is designed to deploy independently from the backend and frontend repositories.

## Modules

- M1 ingestion and schema: `src/learnguard/data/ingestion.py`, `schema.py`
- M2 point-in-time features (cutoff day 27): `src/learnguard/data/features.py`
- M3 leakage and data quality gates: `src/learnguard/data/leakage.py`, `quality.py`
- M4 model training and evaluation: `src/learnguard/ml/training.py`, `evaluation.py`
- M5 scoring and batch contract: `src/learnguard/ml/scoring.py`, `contracts.py`
- Pipeline orchestration: `src/learnguard/pipeline.py`, `cli.py`

The code defines module/class/function structure and TODOs; it does not implement the end-to-end business pipeline yet.

## Deployment seam to backend

The intended M5 → backend contract is `prediction-batch.v1`, a JSON manifest and prediction artifact. Expected provenance fields include `run_id`, `cutoff_day`, `snapshot_version`, `model_version`, creation time, row count, artifact path, and SHA-256. The backend validates and imports this format without importing this repository as a Python package.

Exact schema, artifact transport/storage, authentication, retries, and retention are TODO. See the workspace-level [repository boundaries](../../project/docs/REPOSITORY_LAYOUT.md) for the cross-component view.
