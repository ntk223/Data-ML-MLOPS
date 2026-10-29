# Phase 5 / M4 — Logistic Regression baseline

Tài liệu này chỉ mô tả model-training stage. Data pipeline M1–M3 và Phase 4
preprocessing tiếp tục tuân theo `PIPELINE_GUIDE.md` và không bị model stage thay đổi.

## Chạy baseline

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

Training chỉ dùng fitted preprocessor đã xác minh và temporal train. Stage không refit
preprocessing, không đọc validation/holdout, không tuning và không tính metric.

## Model contract

Baseline v1 là L2 Logistic Regression với `C=1.0`, solver `lbfgs`,
`class_weight=None`, `max_iter=2000`, `tol=1e-4`, `fit_intercept=True` và seed 42.
CLI không cho override contract này.

Artifact được publish atomically dưới `artifacts/model-runs/<run-id>/`:

```text
model.joblib
model.manifest.json
model.training-report.json
model.success.json
```

Training report chỉ chứa convergence và structural sanity checks. Precision/Recall,
AUC, threshold, calibration, fairness, model comparison và holdout evaluation thuộc
phase đánh giá sau.

Joblib chỉ được load từ nguồn cục bộ đáng tin cậy. Checksum phát hiện artifact bị sửa,
nhưng không làm pickle từ nguồn không tin cậy trở nên an toàn.

## Full OULAD verification

- Train rows: 11.811.
- Input features: 36 dense `float64` features từ preprocessing-v1.
- Learned classes: `[0, 1]`.
- Convergence: 27 iterations, không warning.
- Model semantic version: `logreg-v1:39911ebf:04c126ab:2f7d4d10`.
- Hai identical runs có cùng matrix, target, coefficient, intercept, learned-state và
  train-probability semantic hashes.
