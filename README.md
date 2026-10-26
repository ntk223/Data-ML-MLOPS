# LearnGuard ML/Data

Independent repository for the LearnGuard M1–M5 batch pipeline. The repository provides the initial package structure, typed contracts, core classes/functions, and TODOs; the pipeline logic is intentionally not implemented yet.

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
