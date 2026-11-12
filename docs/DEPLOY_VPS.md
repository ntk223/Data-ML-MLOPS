# DEPLOY_VPS — Triển khai batch runner LearnGuard lên VPS

**Dành cho:** người triển khai vận hành trên VPS.
**Mô hình deploy:** **batch runner, KHÔNG phải service** — không mở port, không chạy liên tục. Cron định kỳ chạy chuỗi M1→M5, prediction nằm sẵn trong DB để backend đọc.
**Phân định (chốt 2026-11-02, `PIPELINE_GUIDE.md` mục 10):** PostgreSQL do **backend team** vận hành — VPS này KHÔNG cài DB, chỉ nhận DSN.

---

## 1. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| OS | Ubuntu 22.04+ (Linux bất kỳ có cron) |
| Python | >= 3.11 |
| Disk | >= 5GB (dữ liệu ~500MB + artifacts + venv) |
| RAM | >= 2GB |
| Mạng | Ra được tới PostgreSQL của backend (port 5432) |
| Cron | Có sẵn (`which cron`) |

---

## 2. Cài đặt (chạy 1 lần)

```bash
# 2.1. Clone repo
sudo mkdir -p /opt/learnguard && sudo chown "$USER" /opt/learnguard
cd /opt/learnguard
git clone <URL_REPO> ml-data && cd ml-data

# 2.2. Môi trường Python
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/pip install -r requirements-dev.txt   # để chạy pytest kiểm tra; có thể bỏ sau

# 2.3. Dữ liệu — copy zip OULAD (hoặc LMS extract) lên VPS rồi giải nén
unzip data/archive.zip -d data/oulad/archive
# Kiểm tra đủ 7 CSV:
ls data/oulad/archive/
# kỳ vọng: assessments courses studentAssessment studentInfo studentRegistration studentVle vle

# 2.4. Kiểm tra cài đặt
.venv/bin/python -m pytest tests/ -q
# Kỳ vọng: toàn pass; các test PostgreSQL tự SKIP (chưa có DSN — bình thường)
```

---

## 3. Cấu hình bí mật — file `.env` (KHÔNG bao giờ commit)

```bash
cat > .env <<'EOF'
# DSN do BACKEND cấp — user này có quyền GHI bảng ml_predictions
LEARNGUARD_PG_DSN=postgresql://ml_writer:MAT_KHAU@HOST_BACKEND:5432/learnguard
EOF
chmod 600 .env
```

Xác nhận an toàn: `git check-ignore .env` phải in ra `.env` (đã gitignore).

> **An ninh:** DSN là secret — nhận qua kênh riêng từ backend, không gửi qua chat/ticket công khai. Khi backend cấp DSN mới (rotate) → chỉ cần sửa `.env`, không cần đổi code.
>
> **Quyết định đã chấp nhận (dự án học tập):** dùng **chung credential root/user duy nhất** cho cả ml-data (ghi) và backend (đọc) — KHÔNG tách user. Chấp nhận được vì: (1) ml-data viết qua đúng 1 câu UPSERT cố định có bảng validate + rows parameterized, không có đường SQL tự do; (2) backend chỉ SELECT qua API. Guardrail sở hữu ghi chỉ tồn tại ở **tầng code**, không ở tầng DB — nếu leak DSN thì toàn bộ DB bị lộ. Khi chuyển thành product: tách 2 user ngay.

---

## 4. Chạy thử thủ công — trước khi bật cron

```bash
cd /opt/learnguard/ml-data

# 4.1. Chuỗi M1→M3 (quickstart mục 4 của PIPELINE_GUIDE.md)
#   → kỳ vọng in "PIPELINE OK — snapshot-v1:..."

# 4.2. M4: train XGBoost contract (chạy 1 lần mỗi khi ra model mới)
.venv/bin/learnguard ml preprocess \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --artifacts-dir artifacts --run-id prod-pre-v1
.venv/bin/learnguard ml train-xgboost \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --preprocessing-artifact-dir artifacts/preprocessing-runs/prod-pre-v1 \
  --artifacts-dir artifacts --run-id prod-xgb-v2

# Chạy validation benchmark, ML gate, evaluation và governance promotion theo
# docs/M4_XGBOOST.md; cấu hình immutable release ID sau khi verify.

# 4.3. M5: score thử (chế độ file-only, chưa cần DSN)
.venv/bin/learnguard ml score \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --release-artifact-dir "artifacts/model-releases/${LEARNGUARD_RELEASE_ID}" \
  --artifacts-dir artifacts --run-id prod-score-test --split validation
# Kỳ vọng: artifacts/prediction-batches/prod-score-test/ có predictions.parquet + manifest.json + success.json
```

---

## 5. Cronjob — chạy định kỳ

### 5.1. Tạo script bọc (bao cả nạp .env + kiểm tra lỗi)

Tạo `scripts/cron_pipeline.sh` (nếu repo chưa có):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Nạp secret
set -a; [ -f .env ] && source .env; set +a

PY=.venv/bin/python
RUN_ID="cron-$(date +%Y%m%d-%H%M)"

# --- Chuỗi M1→M3: ingest → snapshot → audit + gates (fail = dừng, KHÔNG score) ---
"$PY" - <<'PYEOF'
from pathlib import Path
from learnguard.data.ingestion import ingest_oulad
from learnguard.data.features import build_feature_snapshot
from learnguard.data.leakage import LeakageAuditor
from learnguard.data.quality import validate_snapshot

ing = ingest_oulad(Path('data/oulad/archive'), Path('artifacts/ingestion_m1_verify'))
snap = build_feature_snapshot(Path('artifacts/ingestion_m1_verify'),
                              Path('artifacts/feature_snapshot_m2/snapshot.parquet'))
audit = LeakageAuditor().audit_snapshot(snap.snapshot_path, Path('artifacts/ingestion_m1_verify'), 27)
gate = validate_snapshot(snap.snapshot_path)
assert ing.status == 'ok' and audit.status == 'ok' and gate.status == 'ok', \
    f'GATES FAILED: {ing.status}/{audit.status}/{gate.status}'
print('PIPELINE OK —', snap.snapshot_version)
PYEOF

# --- M5: verify rồi score immutable release đang dùng ---
"$PY" -m learnguard.cli ml verify-release \
  --release-artifact-dir "artifacts/model-releases/${LEARNGUARD_RELEASE_ID}" \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet
"$PY" -m learnguard.cli ml score \
  --snapshot artifacts/feature_snapshot_m2/snapshot.parquet \
  --release-artifact-dir "artifacts/model-releases/${LEARNGUARD_RELEASE_ID}" \
  --artifacts-dir artifacts --run-id "$RUN_ID" \
  --split validation \
  --pg-dsn "$LEARNGUARD_PG_DSN" --pg-table ml_predictions

echo "CRON OK — $RUN_ID"
```

Bổ sung release ID bất biến vào `.env`:

```bash
LEARNGUARD_RELEASE_ID=xgboost-production-v4
```

`chmod +x scripts/cron_pipeline.sh`

### 5.2. Đăng ký crontab

```bash
mkdir -p logs
crontab -e
# Thêm (ví dụ: 06:30 thứ Hai hàng tuần — giờ SERVER, kiểm tra date trước):
30 6 * * 1 cd /opt/learnguard/ml-data && ./scripts/cron_pipeline.sh >> logs/cron.log 2>&1
```

Kiểm tra cron đã đăng ký: `crontab -l`. Chạy thử ngay không cần chờ cron: `./scripts/cron_pipeline.sh`.

---

## 6. Bảo trì & sự cố

| Tình huống | Xử lý |
|---|---|
| Cron log có `GATES FAILED` | Đúng thiết kế — pipeline tự chặn, KHÔNG score. Xem lỗi trong log + `artifacts/ingestion_m1_verify/quarantine/` |
| `unknown_presentation_for_split` | Kỳ học mới chưa đăng ký rule → báo data engineer thêm vào `SPLIT_RULES` |
| `database_write` báo lỗi kết nối | DSN sai/hết hạn hoặc PG backend chặt tường lửa — liên hệ backend |
| `artifacts/` phình to | Prediction batch mỗi run ~vài MB; xoá batch cũ > 90 ngày (giữ manifest/success.json của run gần nhất) |
| Cập nhật code | `git pull` → chạy lại `pytest` → chạy thử `cron_pipeline.sh` thủ công → mới để cron tiếp tục |
| Model mới release | Chạy lại bước 4.2 với run-id mới → cập nhật `MODEL_RUN_ID` trong `.env` |

---

## 7. Checklist go-live

- [ ] Repo clone + venv + `pytest` xanh (test PG skip là bình thường)
- [ ] 7 CSV đúng chỗ, quickstart M1→M3 in `PIPELINE OK`
- [ ] Model release đầu tiên đã train (`artifacts/model-runs/<run-id>/model.success.json` tồn tại)
- [ ] `.env` có đủ 3 biến (`LEARNGUARD_PG_DSN`, `MODEL_RUN_ID`, `PREPROCESS_RUN_ID`), chmod 600, gitignore xác nhận
- [ ] Backend xác nhận: server PG chạy, tường lửa cho phép VPS, credential đã cấp (dự án học tập: dùng chung credential — xem ghi nhận ở mục 3)
- [ ] Score thủ công 1 lần: `database_write: ok` trong JSON output
- [ ] Crontab đăng ký + chạy thử `./scripts/cron_pipeline.sh` thành công
- [ ] Backend xác nhận đọc được dự đoán mới nhất qua bảng `ml_predictions`

**Tài liệu tham chiếu trên VPS:** `PIPELINE_GUIDE.md` (runbook + luồng vận hành) · `M5_PREDICTION_BATCH.md` (hợp đồng bảng/manifest) · `M4_LOGISTIC_BASELINE.md` (model).
