# M5 — Prediction batch artifact (giao cho backend)

**Bên xuất & GHI DB:** repo `ml-data` (M5 scoring).
**Bên tiêu thụ:** repo `backend` — chỉ ĐỌC bảng `ml_predictions` (xem §3).
**Chế độ tích hợp đã chốt (2026-10-30, bản 2):** ml-data ghi thẳng vào PostgreSQL dùng chung. Backend **không implement import service** — là consumer chỉ đọc. File `prediction-batch.v1` vẫn xuất song song làm audit trail + xác minh chéo.

## 1. Artifact mỗi lần chạy `learnguard ml score`

```text
artifacts/prediction-batches/<run-id>/
├── predictions.parquet   # 14 cột, payload
├── manifest.json         # prediction-batch.v1 — 9 trường
├── success.json          # checksum manifest/payload
└── release.provenance.json # chỉ có ở release mode; producer-only audit sidecar
```

Backend xác minh theo đúng trình tự 4 bước ở §4.3 của DB_CONTRACT.md (schema_version → sha256 parquet → sha256 manifest qua success.json → row_count).
Sidecar không thuộc `prediction-batch.v1`, không thêm cột Parquet/PostgreSQL và
backend cũ có thể bỏ qua. `success.json` ghi thêm checksum sidecar khi có release.

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

Release mode:

```bash
learnguard ml score \
  --snapshot <snapshot> \
  --release-artifact-dir artifacts/model-releases/<release-id> \
  --artifacts-dir artifacts --run-id <score-run> \
  --split <explicit-split>
```

Khi có `--release-artifact-dir`, model/preprocessor/threshold CLI bị cấm; scorer
verify release rồi dùng threshold đã khóa. Không có fallback `0.5`. Release hiện
tại cấm `holdout` vì 2014J đã bị tiêu thụ. Legacy mode vẫn giữ model + preprocessor
và mặc định `0.5` để tương thích caller cũ.

## 4. Hợp đồng đọc — bảng `ml_predictions` (backend implement API trên bảng này)

- **Bảng:** `ml_predictions` — **sở hữu ghi: ml-data**; backend chỉ SELECT.
- **PK:** `(code_module, code_presentation, student_key)` — mỗi SV 1 dòng/1 lớp học phần, **luôn là dự đoán mới nhất** (run mới upsert đè run cũ). Không cần JOIN "batch mới nhất".
- **Cột:** `student_key CHAR(16)`, `code_module`, `code_presentation`, `split`, `risk_score` (float8), `predicted_label` (smallint, NULL khi không gán nhãn), `threshold`, `uncertainty`, `status`, `review_reason`, `model_version`, `snapshot_version`, `cutoff_day`, `top_signals_json` (text), `run_id`, `scored_at` (timestamptz).
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
