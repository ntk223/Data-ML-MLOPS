# M5 — Prediction batch artifact (giao cho backend)

**Bên xuất & GHI DB:** repo `ml-data` (M5 scoring).
**Bên tiêu thụ:** repo `backend` — chỉ ĐỌC bảng `ml_predictions` (xem §3).
**Chế độ tích hợp đã chốt (2026-10-30, bản 2):** ml-data ghi thẳng vào PostgreSQL dùng chung. Backend **không implement import service** — là consumer chỉ đọc. File `prediction-batch.v1` vẫn xuất song song làm audit trail + xác minh chéo.

## 1. Artifact mỗi lần chạy `learnguard ml score`

```text
artifacts/prediction-batches/<run-id>/
├── predictions.parquet   # 15 cột, payload
├── manifest.json         # prediction-batch.v1 — 9 trường
└── success.json          # {schema_version, run_id, manifest_sha256, predictions_sha256}
```

Backend xác minh theo đúng trình tự 4 bước ở §4.3 của DB_CONTRACT.md (schema_version → sha256 parquet → sha256 manifest qua success.json → row_count).

## 2. Công thức `student_key` (nguồn chân lý — backend tham chiếu khi join)

```text
student_key = sha256( "learnguard:prediction-batch.v1:{code_module}:{code_presentation}:{id_student}" ).hexdigest()[:16]
```

- Băm chuỗi UTF-8, lấy **16 ký tự hex đầu** (lowercase).
- `id_student` là số nguyên OULAD, ghép **không có khoảng trắng** trong mẫu định dạng trên.
- Key là pseudonymous: pipeline **không xuất** `id_student` hay mapping. Backend sinh key từ danh sách SV của mình bằng đúng công thức này để join — không cần bảng mapping.
- Bất kỳ thay đổi nào của chuỗi tiền tố/độ dài key = **breaking change**, phải nâng `schema_version` và thông báo backend.

**Test vector kiểm tra chéo (bắt buộc pass trước khi implement):**

| code_module | code_presentation | id_student | student_key |
|---|---|---|---|
| AAA | 2013B | 1 | `4c337e13fc7b6871` |
| AAA | 2013J | 11391 | `7bd60aced1ccab8c` |

## 3. Chế độ vận hành

| Chế độ | Lệnh | Ghi chú |
|---|---|---|
| **Production — ghi DB** | `learnguard ml score ... --pg-dsn $LEARNGUARD_PG_DSN` | Sau khi publish file, upsert vào bảng `ml_predictions` (all-or-nothing; lỗi DB → lệnh fail, file vẫn còn làm audit trail) |
| **Production — chỉ file** | `learnguard ml score ...` (không DSN) | Dùng khi backend chưa sẵn sàng DB hoặc cho môi trường offline |
| **Chọn tập chấm** | `--split train\|validation\|holdout` (kết hợp được; mặc định cả 3) | Holdout chỉ dùng khi đánh giá cuối kỳ — xem §5.3 `PIPELINE_GUIDE.md` |

## 4. Hợp đồng đọc — bảng `ml_predictions` (backend implement API trên bảng này)

- **Bảng:** `ml_predictions` — **sở hữu ghi: ml-data**; backend chỉ SELECT.
- **PK:** `(code_module, code_presentation, student_key)` — mỗi SV 1 dòng/1 lớp học phần, **luôn là dự đoán mới nhất** (run mới upsert đè run cũ). Không cần JOIN "batch mới nhất".
- **Cột:** `student_key CHAR(16)`, `code_module`, `code_presentation`, `split`, `risk_score` (float8), `predicted_label` (smallint, NULL khi không gán nhãn), `threshold`, `uncertainty`, `status`, `review_reason`, `model_version`, `snapshot_version`, `cutoff_day`, `top_signals_json` (text), `features_json` (text), `run_id`, `scored_at` (timestamptz).
- **`features_json` (mới):** JSON object `{"tên_cột": giá trị_gốc | null}` với đúng 6 cột thống kê **hành vi/học thuật** tại cutoff: `n_clicks_before_cutoff`, `click_trend_w3_w4`, `n_submitted_before_cutoff`, `avg_score_submitted`, `num_of_prev_attempts`, `studied_credits`. Giá trị là GỐC (chưa qua preprocessing) — dùng hiển thị thống kê cho cố vấn. **Cố ý không chứa demographics** (gender/region/highest_education/imd_band/age_band/disability) — ranh giới PII; demographics chỉ tồn tại trong snapshot. Bảng `ml_predictions` có sẵn migration tại chỗ (`ALTER TABLE … ADD COLUMN IF NOT EXISTS`) qua `ensure_schema()`.
- **`status`:** `scored` / `manual_review` / `insufficient_data` / `no_prediction` — backend PHẢI xử lý đủ 4.
- **Index sẵn có:** `(status)`, `(model_version)`.
- **Versioning:** đổi schema = breaking change — ml-data thông báo trước, backend đồng bộ.
- **Credential:** ml-data dùng DSN từ `LEARNGUARD_PG_DSN` (env), user DB cần quyền `INSERT/UPDATE/SELECT` trên `ml_predictions`; backend chỉ cần `SELECT`.

**Ví dụ query chuẩn cho advisor dashboard:**

```sql
-- Top nguy nhất
SELECT student_key, code_module, code_presentation, risk_score, top_signals_json
FROM ml_predictions
WHERE status = 'scored' AND predicted_label = 1
ORDER BY risk_score DESC LIMIT 50;

-- Hàng đợi review tay
SELECT student_key, risk_score, uncertainty, review_reason
FROM ml_predictions WHERE status = 'manual_review'
ORDER BY uncertainty DESC;
```

> **Điều backend phải cập nhật trong DB_CONTRACT.md:** §2 (bỏ PredictionBatchImportService, chuyển sang đọc `ml_predictions`), §6.4 (thay bảng `predictions` bằng contract bảng `ml_predictions` ở trên), Gap 1/2 (bỏ — không còn import). `student_key` §9 giữ nguyên — công thức ở §2 dưới đây vẫn cần cho backend join về SV thật.
