# Phase 5 / M4 — Logistic Regression baseline

Tài liệu này chỉ mô tả model-training stage. Data pipeline M1–M3 và Phase 4
preprocessing tiếp tục tuân theo `PIPELINE_GUIDE.md` và không bị model stage thay đổi.

## Chạy baseline

```bash
.venv/bin/learnguard ml preprocess \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --artifacts-dir artifacts \
  --run-id <preprocessing-run-id>

.venv/bin/learnguard ml train-logistic \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<preprocessing-run-id> \
  --artifacts-dir artifacts \
  --run-id <model-run-id>

.venv/bin/learnguard ml verify-model \
  --model-artifact-dir artifacts/model-runs/<model-run-id> \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<preprocessing-run-id>
```

Training chỉ dùng fitted preprocessor đã xác minh và temporal train. Stage không refit
preprocessing, không dùng validation/holdout để fit, không tuning và không tính metric.
Việc load snapshot luôn đi qua `load_snapshot_splits()` theo đúng `PIPELINE_GUIDE.md`;
model trainer chỉ lấy thuộc tính `train` từ kết quả loader.

## Model contract

Baseline v1 là L2 Logistic Regression với `C=1.0`, solver `lbfgs`,
`class_weight=None`, `max_iter=2000`, `tol=1e-4`, `fit_intercept=True` và seed 42.
CLI không cho override contract này.

Artifact được publish atomically dưới `artifacts/model-runs/<run-id>/`:

```text
model.joblib
shap_background.npz
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
- Data contract: `snapshot-v1:cutoff27:35de8103:39e10cfa`, giữ nguyên 21 cột
  snapshot và 15 cột đầu vào được công bố trong `PIPELINE_GUIDE.md`.
- Model matrix: 38 dense `float64` features từ preprocessing-v1 sau encoding.
- Learned classes: `[0, 1]`.
- Convergence: 39 iterations, không warning.
- Preprocessor semantic version: `preprocessor-v1:b8c61096:4cfed679:cfb51a0d`.
- Model semantic version: `logreg-v1:bab3d0df:48843ee8:1ccd9891`.
- Hai identical runs có cùng matrix, target, coefficient, intercept, learned-state và
  train-probability semantic hashes.

## Linear SHAP và benchmark

Model artifact `logistic-model-v2` lưu `shap_background.npz` gồm mean/covariance
của transformed train matrix. Prediction batch mới dùng interventional Linear SHAP
trên thang log-odds (`coef * (x - train_mean)`) cho `top_signals_json`; model v1
vẫn load được với contribution legacy. Giải thích là association, không phải quan hệ
nhân quả.

```bash
learnguard ml explain-shap \
  --snapshot artifacts/manual-runs/<data-run>/snapshot/snapshot.parquet \
  --model-artifact-dir artifacts/model-runs/<model-v2-run> \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<pre-run> \
  --artifacts-dir artifacts --run-id <shap-run>

learnguard ml benchmark \
  --snapshot artifacts/manual-runs/<data-run>/snapshot/snapshot.parquet \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<pre-run> \
  --artifacts-dir artifacts --run-id <benchmark-run>

# Sau khi holdout đã được tiêu thụ: chỉ benchmark train/validation
learnguard ml benchmark \
  --snapshot artifacts/manual-runs/<data-run>/snapshot/snapshot.parquet \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/<pre-run> \
  --artifacts-dir artifacts --run-id <validation-run> --validation-only
```

Benchmark fit candidate chỉ trên train, chọn model/threshold trên validation,
ghi `locked_config.json`, rồi gọi holdout prediction đúng một lần. Candidate winner
chỉ phục vụ báo cáo lịch sử; Logistic hiện được giữ làm fallback/rollback.

Full OULAD run `benchmark-20261105-shap` chọn HistGradientBoosting B trên validation
(PR-AUC 0.7211, ROC-AUC 0.7283, threshold 0.30135). Final holdout một lần đạt
PR-AUC 0.5832, ROC-AUC 0.6956 và Recall@top-20% 0.3007. SHAP run
`shap-20261105-full` xuất 27.535 dòng × 38 feature với maximum additivity error
`1.78e-15`.

Random Forest A/B/C được đăng ký dưới dạng bounded candidate. Vì holdout 2014J đã
được tiêu thụ, mọi thử nghiệm Random Forest mới phải dùng `--validation-only`:
loader chỉ đọc train+validation, report ghi `production_release_eligible=false`,
và không tạo locked config/holdout artifact. Kết quả validation không đủ để release
production; final evaluation cần temporal holdout mới hoặc quyết định split mới.

## Validation evaluation và production release

Logistic là verified fallback/rollback và các production release cũ vẫn bất biến.
XGBoost là operational primary theo governance policy trong `M4_XGBOOST.md`.
Threshold của mỗi release là scalar threshold
suy ra từ Top-20% validation score của đúng binary production; max-F1 chỉ là
comparator. Nếu score bằng nhau tại biên, policy chọn scalar gần ngân sách nhất và
ưu tiên không vượt ngân sách khi sai lệch bằng nhau. Release fail nếu sai lệch vượt
1 điểm phần trăm (ngoại trừ lượng tử hóa không tránh được ở fixture rất nhỏ).

```bash
learnguard ml evaluate-validation-release \
  --snapshot <snapshot> \
  --benchmark-artifact-dir <validation-only-benchmark> \
  --model-artifact-dir <logistic-v2-model> \
  --preprocessing-artifact-dir <preprocessor> \
  --artifacts-dir artifacts --run-id <evaluation-run>

learnguard ml create-release \
  --snapshot <snapshot> \
  --model-artifact-dir <logistic-v2-model> \
  --preprocessing-artifact-dir <preprocessor> \
  --evaluation-artifact-dir artifacts/evaluation-runs/<evaluation-run> \
  --artifacts-dir artifacts --run-id <release-id>

learnguard ml verify-release \
  --release-artifact-dir artifacts/model-releases/<release-id> \
  --snapshot <snapshot>
```

Evaluation chỉ chấp nhận benchmark `validation_only` có checksum hợp lệ và binary
Logistic trùng release model. Release tự chứa model/preprocessor bằng relative
paths, không sửa artifact nguồn. Với snapshot hiện tại, release ghi
`holdout_access_policy=forbidden_prior_consumed`; Random Forest B vẫn chỉ là
benchmark challenger và không có production adapter/release trong phạm vi này.
