# Model Evaluation & Production Release Guide

**Mục tiêu:** kiểm chứng Model Evaluation, threshold Top-20%, XGBoost operational
production release và Logistic rollback mà không truy cập lại holdout `2014J`.

Release mới dùng `model-release.v4`; readers vẫn hỗ trợ Logistic release v1–v3
và validation candidate cũ. XGBoost production bắt buộc có `promotion-decision.v2` gắn
đúng model/benchmark/evaluation/ML-gate hashes. ML gate phải không có check fail,
artifact integrity phải pass và regression phải dùng đúng model/preprocessor
bên trong immutable Logistic rollback release được chỉ định. Release v4 đóng
gói rollback release và bind type/version/manifest/payload hashes; xem
`M4_XGBOOST.md`.

**Phạm vi an toàn:** các lệnh trong guide chỉ dùng train để verify model và dùng
validation artifact đã có để tính metric/threshold. Không chạy full benchmark,
không score holdout. Chỉ release production đã verify mới được ghi PostgreSQL.

---

## 1. Artifact nguồn

```text
Snapshot       artifacts/manual-runs/20261105-004543/snapshot/snapshot.parquet
Preprocessor   artifacts/preprocessing-runs/pre-20261105-004543
Logistic v2    artifacts/model-runs/model-shap-v2-20261105
XGBoost v2     artifacts/model-runs/xgboost-production-v2-20261109
Benchmark      artifacts/benchmark-runs/<validation-only-run>
XGB release    artifacts/model-releases/<xgb-production-release>
Rollback       artifacts/model-releases/logistic-production-20261105
```

XGBoost là operational primary; Logistic là rollback và Random Forest B chỉ là
benchmark challenger. Đây là governance promotion trên validation, không phải
kết luận superiority từ holdout mới.

---

## 2. Chuẩn bị môi trường

```bash
cd /home/hoang/Data-ML-MLOPS

.venv/bin/python --version
.venv/bin/learnguard --help
```

Nếu chưa có môi trường:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .
```

---

## 3. Chạy kiểm thử

Kiểm thử tập trung cho evaluation, release, SHAP leakage guard và scoring:

```bash
.venv/bin/python -m pytest \
  tests/test_evaluation.py \
  tests/test_release.py \
  tests/test_explainability.py \
  tests/test_scoring.py \
  tests/test_training.py \
  tests/test_benchmark.py -q
```

Chạy toàn bộ repository:

```bash
.venv/bin/python -m pytest tests/ -q
```

Kết quả tham chiếu hiện tại:

```text
Chạy suite hiện tại và ghi lại số pass/skip thực tế; không hard-code kết quả cũ.
```

Các test PostgreSQL được skip khi máy không có DB local là hành vi bình thường.

---

## 4. Verify production release hiện có

```bash
.venv/bin/learnguard ml verify-release \
  --release-artifact-dir artifacts/model-releases/logistic-production-20261105 \
  --snapshot artifacts/manual-runs/20261105-004543/snapshot/snapshot.parquet
```

Kỳ vọng:

```json
{
  "status": "ok",
  "release_id": "logistic-production-20261105",
  "model_version": "logreg-v2:cea6cdea:48843ee8:1ccd9891",
  "production_threshold": 0.8103480496322563
}
```

Verifier phải fail nếu model/preprocessor bị sửa, threshold không khớp
`threshold_policy.json`, hoặc `holdout_access_policy` không còn là
`forbidden_prior_consumed`.

---

## 5. Tạo evaluation và release mới

Mỗi lần chạy phải dùng run-id mới. Lệnh dưới đây tạo ID từ thời gian hiện tại:

```bash
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
EVAL_ID="logistic-release-evaluation-${RUN_TAG}"
RELEASE_ID="logistic-production-${RUN_TAG}"
```

### 5.1 Tạo validation evaluation

```bash
.venv/bin/learnguard ml evaluate-validation-release \
  --snapshot artifacts/manual-runs/20261105-004543/snapshot/snapshot.parquet \
  --benchmark-artifact-dir artifacts/benchmark-runs/random-forest-validation-20261105-retry3 \
  --model-artifact-dir artifacts/model-runs/model-shap-v2-20261105 \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/pre-20261105-004543 \
  --artifacts-dir artifacts \
  --run-id "${EVAL_ID}"
```

Lệnh chỉ chấp nhận benchmark `validation_only` có checksum hợp lệ và Logistic
candidate có binary hash trùng production model.

### 5.2 Đóng gói release

```bash
.venv/bin/learnguard ml create-release \
  --snapshot artifacts/manual-runs/20261105-004543/snapshot/snapshot.parquet \
  --model-artifact-dir artifacts/model-runs/model-shap-v2-20261105 \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/pre-20261105-004543 \
  --evaluation-artifact-dir "artifacts/evaluation-runs/${EVAL_ID}" \
  --artifacts-dir artifacts \
  --run-id "${RELEASE_ID}"
```

### 5.3 Verify release mới

```bash
.venv/bin/learnguard ml verify-release \
  --release-artifact-dir "artifacts/model-releases/${RELEASE_ID}" \
  --snapshot artifacts/manual-runs/20261105-004543/snapshot/snapshot.parquet
```

Không chạy lại cùng `EVAL_ID` hoặc `RELEASE_ID`; pipeline cố ý từ chối ghi đè.

---

## 6. Kiểm tra threshold và metric

```bash
jq '{
  policy,
  population,
  validation_sample_count,
  budget_count,
  production_threshold,
  realized_alert_count,
  realized_alert_rate,
  tie_policy,
  production_threshold_metrics,
  max_f1_comparator,
  holdout_used
}' "artifacts/evaluation-runs/${EVAL_ID}/threshold_policy.json"
```

Kết quả tham chiếu:

```text
production threshold : 0.8103480496322563
alert count          : 1,302 / 6,510
alert rate           : 20%
precision            : 0.7573
recall               : 0.3127
F1                   : 0.4426
confusion matrix     : [[3041, 316], [2167, 986]]
```

Max-F1 threshold `0.181018...` chỉ là comparator, không phải production policy.

---

## 7. Kiểm tra confusion matrix

```bash
find "artifacts/evaluation-runs/${EVAL_ID}/confusion_matrices" \
  -type f -name '*.csv' | sort

find "artifacts/evaluation-runs/${EVAL_ID}/plots" \
  -type f -name 'confusion_matrix*.png' | sort
```

Kỳ vọng có 21 CSV và 21 PNG:

```bash
find "artifacts/evaluation-runs/${EVAL_ID}/confusion_matrices" \
  -type f -name '*.csv' | wc -l

find "artifacts/evaluation-runs/${EVAL_ID}/plots" \
  -type f -name 'confusion_matrix*.png' | wc -l
```

Xem matrix production Logistic:

```bash
column -s, -t < \
  "artifacts/evaluation-runs/${EVAL_ID}/confusion_matrices/logistic_baseline__production-top20.csv"
```

Quy ước:

```text
[[TN, FP],
 [FN, TP]]
```

---

## 8. Kiểm tra checksum

Evaluation manifest:

```bash
(
  cd "artifacts/evaluation-runs/${EVAL_ID}"
  jq -r '.files | to_entries[] | "\(.value)  \(.key)"' manifest.json \
    | sha256sum -c -
)
```

Release được kiểm tra bằng CLI vì verifier còn đối chiếu threshold policy,
evaluation metadata, model/preprocessor provenance và holdout policy:

```bash
.venv/bin/learnguard ml verify-release \
  --release-artifact-dir "artifacts/model-releases/${RELEASE_ID}" \
  --snapshot artifacts/manual-runs/20261105-004543/snapshot/snapshot.parquet
```

---

## 9. Optional: score validation bằng release

Chỉ dùng để smoke-test file output; không thêm `--pg-dsn`.

```bash
SCORE_ID="release-validation-check-${RUN_TAG}"

.venv/bin/learnguard ml score \
  --snapshot artifacts/manual-runs/20261105-004543/snapshot/snapshot.parquet \
  --release-artifact-dir "artifacts/model-releases/${RELEASE_ID}" \
  --artifacts-dir artifacts \
  --run-id "${SCORE_ID}" \
  --split validation
```

Kiểm tra threshold và nhãn:

```bash
.venv/bin/python -c "
import pandas as pd
p = 'artifacts/prediction-batches/${SCORE_ID}/predictions.parquet'
d = pd.read_parquet(p)
t = d['threshold'].unique()
valid = d['predicted_label'].notna()
print('rows:', len(d))
print('threshold:', t.tolist())
print('labels_match:', bool(
    (d.loc[valid, 'predicted_label'].astype(int)
     == (d.loc[valid, 'risk_score'] >= d.loc[valid, 'threshold']).astype(int)).all()
))
print('contains_id_student:', 'id_student' in d.columns)
"
```

Kiểm tra release audit sidecar:

```bash
jq '.' "artifacts/prediction-batches/${SCORE_ID}/release.provenance.json"
```

---

## 10. Các lệnh bị cấm với snapshot hiện tại

Không chạy:

```text
learnguard ml benchmark ...                 # full mode
learnguard ml score ... --split holdout
learnguard ml score ... --pg-dsn ...        # trong lúc kiểm chứng local
```

Release hiện tại chủ động fail với `scoring_release_holdout_forbidden` nếu yêu
cầu `--split holdout`. Muốn đánh giá candidate mới cần temporal presentation mới
hoặc quyết định split mới được trưởng nhóm phê duyệt.

---

## 11. Checklist hoàn thành

- [ ] Targeted tests pass.
- [ ] Full suite pass.
- [ ] Evaluation manifest checksum hợp lệ.
- [ ] Có 21 confusion matrix CSV và 21 PNG.
- [ ] Alert rate validation đúng 20%.
- [ ] `verify-release` trả `status=ok`.
- [ ] Threshold manifest khớp threshold policy và validation evaluation.
- [ ] `holdout_reused=false`.
- [ ] `holdout_access_policy=forbidden_prior_consumed`.
- [ ] Random Forest B vẫn là `validation_release_candidate`.
- [ ] Không ghi PostgreSQL và không truy cập holdout.
