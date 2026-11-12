# LearnGuard ML/Data

Independent repository for the implemented LearnGuard M1–M5 batch pipeline, including leakage-safe data processing, Logistic/XGBoost model artifacts, evaluation gates, releases, scoring, and PostgreSQL export.

## Scope

- M1: OULAD ingestion and schema contracts.
- M2: point-in-time feature building with the day-27 cutoff.
- M3: leakage audits and data quality checks.
- M4: model training and evaluation interfaces.
- M5: scoring and prediction batch export contract.
- Pipeline and CLI entry points to compose these stages later.

Backend and frontend are separate Git repositories. This repository must publish prediction batches through the versioned contract described in [`docs/REPOSITORY_LAYOUT.md`](docs/REPOSITORY_LAYOUT.md), without importing backend code.

## Structure

```text
src/learnguard/
├── data/       # ingestion, schema, features, leakage, quality
├── ml/         # training, evaluation, scoring, contracts
├── pipeline.py
├── settings.py
└── cli.py
data/oulad/     # local data location; raw data is not committed
```

## Development

```bash
# Tạo môi trường
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt

# Hoặc dùng pip
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

# Cài package ở chế độ editable (cho CLI/scripts)
.venv/bin/pip install -e .

# Chạy bộ test M1
.venv/bin/python -m pytest tests/ -q

# Chạy ingestion trên OULAD thật
.venv/bin/python -c "from learnguard.data.ingestion import ingest_oulad; ingest_oulad('data/oulad/archive', 'artifacts/ingestion')"
```

Dependencies runtime nằm trong `requirements.txt` (pandas, pyarrow); công cụ notebook/EDA nằm trong `requirements-dev.txt`. Notebook EDA: [`notebooks/01_oulad_eda.ipynb`](notebooks/01_oulad_eda.ipynb).

Python package metadata is in [`pyproject.toml`](pyproject.toml). Setup and run commands will be added as the TODO stages are implemented. No Docker Compose or production pipeline is included at this scaffold stage.

## Logistic Regression baseline

Run the unchanged data pipeline exactly as documented in
[`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md), then use the resulting
`snapshot.parquet` with the model-only commands documented in
[`docs/M4_LOGISTIC_BASELINE.md`](docs/M4_LOGISTIC_BASELINE.md).

Model v2 bổ sung train-background Linear SHAP và bounded validation-first benchmark.
Xem hướng dẫn M4 cho hai lệnh `learnguard ml explain-shap` và
`learnguard ml benchmark`.

Logistic releases remain verified immutable rollback artifacts. Production releases are published separately from immutable training
artifacts. `learnguard ml evaluate-validation-release` derives the validation
Top-20% alert-budget threshold and confusion matrices; `create-release` packages
that policy with verified model/preprocessor payloads. Release-mode scoring reads
the locked threshold and never silently falls back to `0.5`.

Runbook kiểm chứng đầy đủ: [`docs/MODEL_EVALUATION_RELEASE_GUIDE.md`](docs/MODEL_EVALUATION_RELEASE_GUIDE.md).

## XGBoost production primary

The CPU-only `xgboost_regularized` contract is train-only and produces a native
UBJSON `xgboost-model-v2` artifact with train-background interventional Tree
SHAP. A `model-release.v4` may become the operational primary only after the
validation gate and a bound `promotion-decision.v2` governance approval pass.
That decision permits PostgreSQL writes but is not a scientific superiority
claim; the consumed 2014J holdout remains forbidden.

Dependency routing uses `xgboost-cpu` only on Linux `x86_64`; Linux ARM and
other platforms install the regular `xgboost` distribution.

```bash
.venv/bin/learnguard ml train-xgboost \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<pre-run> \
  --artifacts-dir artifacts --run-id <xgb-run>
```

See [`docs/M4_XGBOOST.md`](docs/M4_XGBOOST.md) for the complete workflow and
[`docs/MODEL_BENCHMARK_COMPARISON.md`](docs/MODEL_BENCHMARK_COMPARISON.md) for
the reproducible Logistic/Random Forest B/XGBoost comparison.
