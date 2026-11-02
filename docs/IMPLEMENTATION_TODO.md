# ML/Data implementation TODO

Đây là checklist cho repository `ml-data/`. Các module hiện có chỉ là scaffold, chưa phải mô tả chức năng đã hoàn tất.

## M1–M3 · Data pipeline

- [x] Chốt schema bảy bảng OULAD và kiểu dữ liệu chuẩn. *(schema.py: dtype + enum + range + null policy, truy vết EDA 2026-10-26)*
- [x] Implement ingestion, integrity checks, linting, quarantine và source hash. *(ingestion.py + 23 test pytest; verify OULAD thật: 0 quarantine, dedupe 787,170 exact-duplicate studentVle)*
- [x] Hoàn thiện catalog với source, aggregation, as-of rule và feature kind. *(5 features; rule `n_clicks_before_cutoff` đã đồng bộ `date <= cutoff`)*
- [x] Implement cohort và snapshot tới cutoff ngày 27, kèm as-of ledger/version. *(PointInTimeFeatureBuilder: cohort loại rút-lui-trước-cutoff; 5 features đúng as-of rule; ledger ghi source hash + version; 13 test — verify thật: 27,535 dòng, 0 leakage khi inject future-event)*
- [x] Implement leakage audit cho VLE và assessments. *(LeakageAuditor 3 lớp: catalog rules + recompute đối chiếu + 4 injection probes×4 SV; verify thật: 0 violation)*
- [x] Implement quality gate cho schema, missingness, volume và PSI. *(DataQualityGate 4 gates; PSI yêu cầu ≥200 mẫu/bên — chặn false alarm do nhiễu; verify thật: ok)*
- [x] Thêm kiểm thử cutoff, future-event injection, catalog và quality gates. *(53/53 test; test bắt được leaky-builder cấy chủ đích qua injection probe)*

## M4–M5 · ML pipeline

- [x] Implement baseline Logistic Regression train-only dưới dạng ML stage riêng;
  không sửa data pipeline contract.
- [ ] Implement model ứng viên.
- [ ] Dùng temporal split: train 2013B+2013J → validation 2014B → holdout 2014J. *(hợp đồng từ data engineer — SPLIT_RULES; ưu tiên train nhiều, holdout 2014J là kỳ duy nhất chưa tiêu hao)*
- [ ] Báo cáo Recall@top-k, AUC, ECE, fairness slices và false negatives.
- [ ] Tuning threshold chỉ trên validation; ghi model card/registry metadata.
- [x] Implement scoring, SHAP, uncertainty, insufficient-data và manual-review. *(BatchScorer: explain theo logistic contribution (thay SHAP), uncertainty `1−|2p−1|`, insufficient_data vẫn có risk_score nhưng không gán nhãn, manual_review khi uncertainty ≥ 0.8; verify thật: 27,535 dòng → 22,096 scored / 4,643 manual_review / 796 insufficient_data)*
- [x] Implement `prediction-batch.v1` export, manifest/checksum và stable pseudonymous key. *(PredictionBatchExporter + load_prediction_batch_manifest: manifest sha256 checksum, atomic publish (staging + os.replace), student_key = sha256 16 hex không lộ id_student; CLI `ml score`)*
- [x] Implement shared PostgreSQL prediction store cho serving. *(PRODUCTION MODE sau khi chốt 2026-10-30 bản 2: ml-data ghi thẳng bảng `ml_predictions` — PK (code_module, code_presentation, student_key), upsert transactional all-or-nothing, idempotent DDL + index; backend là consumer CHỈ ĐỌC, không implement import service. predictions_store.py + `ml score --pg-dsn`/`LEARNGUARD_PG_DSN`; file prediction-batch.v1 vẫn xuất song song làm audit trail; scoring có tham số `splits` — mặc định cả 3 split)*

## Orchestration/Release

- [ ] Implement stage orchestration, report và failure state.
- [ ] Chỉ xuất batch sau khi quality/leakage/evaluation gate đạt yêu cầu.
- [ ] Phối hợp backend repo để chốt schema và artifact transport. *(Đã chốt 2026-10-30 bản 2: direct DB write — backend đọc `ml_predictions`, bỏ PredictionBatchImportService; hợp đồng đọc + công thức student_key + test vector ở docs/M5_PREDICTION_BATCH.md; backend cần cập nhật DB_CONTRACT.md §2/§6.4 tương ứng)*
- [ ] Tạo image/job deployment độc lập sau khi batch contract ổn định.
- [ ] Ghi runbook và xác nhận metric bằng OULAD thật; không giả định tiêu chí đạt trước khi đo.
