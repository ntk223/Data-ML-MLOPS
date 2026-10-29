# OULAD Data, Preprocessing and Logistic Baseline Guide

Phạm vi tài liệu là M1–M3 tại cutoff day 27, Phase 4 train-only preprocessing và
Phase 5/M4 Logistic baseline. Model training vẫn là ML stage riêng, không được gộp
vào data pipeline M1–M3.

## 1. Luồng xử lý và cài đặt

```text
archive.zip hoặc thư mục 7 CSV
  → ingestion + quarantine + canonical Parquet
  → snapshot-v3 point-in-time
  → full recompute + future-event leakage probes
  → schema/missingness/volume/PSI gates
```

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest==9.1.1
.venv/bin/pip install -e . --no-deps
.venv/bin/pytest tests -q
```

`data/archive.zip` đã chứa trực tiếp bảy CSV. Pipeline đọc ZIP tại chỗ, hash nội dung
từng CSV và không cần giải nén. Thư mục CSV vẫn được hỗ trợ để phát triển cục bộ.

## 2. Chạy toàn bộ M1–M3

```bash
.venv/bin/learnguard data run \
  --source data/archive.zip \
  --artifacts-dir artifacts \
  --cutoff-day 27
```

Mỗi run nằm riêng dưới `artifacts/data-runs/<run-id>/`:

```text
ingestion/
  canonical/*.parquet
  quarantine/*.csv
  ingestion_report.json
  ingestion.success.json
snapshot/
  snapshot.parquet
  snapshot.ledger.json
  snapshot.success.json
reports/
  leakage_audit.json
  quality_report.json
  pipeline_report.json
```

Pipeline fail-fast theo thứ tự `ingestion`, `feature_snapshot`, `leakage_audit`,
`quality_gate`. Exit code 0 chỉ khi cả bốn stage đạt.

`run-id` phải là một tên đơn gồm 1–128 ký tự ASCII: chữ, số, `.`, `_`, `-`, và
ký tự đầu phải là chữ hoặc số. Absolute path, `..`, slash, backslash và khoảng trắng
bị từ chối trước khi pipeline tạo thư mục. Quy tắc này áp dụng cho cả data run và
preprocessing run.

Để đo volume/PSI và kiểm tra tái lập, chạy lần hai với snapshot đã duyệt làm baseline:

```bash
.venv/bin/learnguard data run \
  --source data/archive.zip \
  --artifacts-dir artifacts \
  --baseline-snapshot artifacts/data-runs/<run-1>/snapshot/snapshot.parquet
```

Ở run đầu không có baseline, volume và PSI được ghi rõ là `not_applicable`; chúng không
được báo pass giả. Run sau mới đánh giá hai gate này.

## 3. Chạy từng stage

```bash
.venv/bin/learnguard data ingest \
  --source data/archive.zip --output-dir artifacts/manual/ingestion

.venv/bin/learnguard data snapshot \
  --processed-dir artifacts/manual/ingestion \
  --output artifacts/manual/snapshot/snapshot.parquet

.venv/bin/learnguard data audit \
  --snapshot artifacts/manual/snapshot/snapshot.parquet \
  --processed-dir artifacts/manual/ingestion

.venv/bin/learnguard data validate \
  --snapshot artifacts/manual/snapshot/snapshot.parquet \
  --baseline-snapshot artifacts/data-runs/<approved>/snapshot/snapshot.parquet
```

Python entry point tương đương là `run_data_pipeline(source, artifacts_dir,
cutoff_day=27, baseline_snapshot=None, run_id=None)`.

## 4. Hợp đồng snapshot-v3

- Khóa: `code_module`, `code_presentation`, `id_student`.
- Target model: `label`; `label=1` cho Fail/Withdrawn, ngược lại bằng 0.
  `final_result` chỉ là audit field và không đi vào model matrix.
- Split: 2013B+2013J là train, 2014B là validation, 2014J là holdout.
- Feature tĩnh: gender, region, education, IMD, age, previous attempts, credits,
  disability và module/presentation.
- Feature hành vi chỉ dùng dữ liệu `date <= 27`; Exam không có ngày bị loại.
- `active_days_before_cutoff` đếm số ngày VLE khác nhau tới ngày 27, gồm cả ngày âm
  vì hoạt động xem tài liệu trước khai giảng là tín hiệu hợp lệ.
- `imd_band` và `avg_score_submitted` là hai cột duy nhất được phép null.
- `insufficient_data` là cờ, không phải lý do âm thầm loại sinh viên.

Ingestion report chứa semantic content hash cho từng canonical table. Ledger chứa source
hashes, full contract hash, cutoff, row count và snapshot semantic content hash.
`load_snapshot_splits()` từ chối ledger legacy, snapshot bị sửa, split sai hoặc
contract không khớp. Version hiện đã kiểm chứng:
`snapshot-v3:cutoff27:f917f578:f3ee9894`.

## 5. Leakage và quality gates

Leakage audit kiểm catalog, ledger, tái tạo toàn bộ snapshot (gồm cohort, static fields,
flags, label, split) và 16 future-event probes. Bất kỳ violation nào đều chặn pipeline.

Quality gate kiểm exact schema, dtype/domain, PK, label/result consistency, split,
derived flags, missingness, volume và PSI. Baseline phải là snapshot đã duyệt; không dùng
snapshot hiện tại làm baseline cho chính nó trong production.

## 6. Kết quả nghiệm thu OULAD thật (2026-10-29)

Hai run độc lập từ ZIP đã hoàn thành:

| Chỉ số | Kết quả |
|---|---:|
| Canonical studentVle | 9,868,110 |
| Exact duplicates loại minh bạch | 787,170 |
| Quarantine | 0 |
| Snapshot rows | 27,535 |
| Insufficient-data rows | 796 |
| Leakage probes / violations | 16 / 0 |
| Run-2 volume ratio | 1.0 |
| Run-2 PSI | 0.0 cho 14 feature |
| Semantic content hash | `d941abb1412566109da3c78be3846b61adf49ecbbbb7e9201f4afc10ea71c0d5` |

Hai run có cùng snapshot version, row count và semantic hash; timestamp và run ID được
phép khác.

## 7. Phase 4 — train-only preprocessing

Phase 4 chỉ nhận một data-run M1–M3 đã thành công. Khi build, stage xác minh lại
snapshot/ledger, report, leakage audit và quality gate trước khi fit.

```bash
.venv/bin/learnguard ml preprocess \
  --data-run-dir artifacts/data-runs/<data-run-id> \
  --artifacts-dir artifacts \
  --run-id <preprocessing-run-id>

.venv/bin/learnguard ml verify-preprocessor \
  --artifact-dir artifacts/preprocessing-runs/<preprocessing-run-id> \
  --data-run-dir artifacts/data-runs/<data-run-id>
```

Preprocessor được fit duy nhất trên train. Validation và holdout chỉ gọi transform.
`code_module` là categorical feature; `module_presentation`, entity keys,
`final_result`, `label` và `split` không đi vào ma trận model. Pipeline dùng one-hot
cho nominal, business ordering cho education/age/IMD, train median cho score thiếu,
`log1p` cho clicks và StandardScaler fit trên train.

Artifact được publish atomically dưới `artifacts/preprocessing-runs/<run-id>/`:

```text
preprocessor.joblib
preprocessor.manifest.json
preprocessor.verification.json
preprocessor.success.json
```

Manifest ràng buộc snapshot-v3, preprocessing contract, train semantic hash, learned
state, output feature order, dependency versions và checksums. Không materialize các
ma trận transformed. Joblib chỉ được load từ nguồn cục bộ đáng tin cậy.

Nghiệm thu OULAD thật hai lần cho cùng kết quả:

| Chỉ số | Kết quả |
|---|---:|
| Preprocessor version | `preprocessor-v1:af8c78d7:43a459f7:84a02baf` |
| Train / validation / holdout | 11,811 / 6,510 / 9,214 |
| Output features | 36 |
| NaN / Inf sau transform | 0 / 0 |
| Train median cho `avg_score_submitted` | 76.0 |

Hai run có cùng contract hash, train hash, learned-state hash, feature order và
transformed semantic hash cho cả ba split.

## 8. Phase 5 / M4 — Logistic Regression baseline

Phase 4 preprocessing đã hoàn tất trước khi bắt đầu bước này. Training chỉ dùng fitted
preprocessor đã xác minh và chỉ đọc temporal train; không refit preprocessing, không đọc
validation/holdout, không tuning và không tính metric.

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

Artifact được publish atomically dưới `artifacts/model-runs/<run-id>/`: `model.joblib`,
`model.manifest.json`, `model.training-report.json`, `model.success.json`. Baseline v1 là
L2 Logistic Regression, `C=1.0`, solver `lbfgs`, không class weighting, `max_iter=2000`,
`tol=1e-4`, seed 42. CLI không cho override contract này.

Training report chỉ chứa convergence và structural sanity checks. Precision/Recall,
AUC, threshold, calibration, fairness, model comparison và holdout evaluation thuộc
phase đánh giá sau. Joblib chỉ được load từ nguồn cục bộ đáng tin cậy; checksum không
làm pickle từ nguồn không tin cậy trở nên an toàn.

## 9. CI

- `.github/workflows/test.yml`: fixture tests trên pull request và push vào main.
- `.github/workflows/full-oulad.yml`: chạy manual hoặc hằng tuần, thực thi full dataset
  hai lần, so version/hash, volume ratio, PSI và hai preprocessing semantic runs, sau đó
  lưu report/manifest làm CI artifact.
