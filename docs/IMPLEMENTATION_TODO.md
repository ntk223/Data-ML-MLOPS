# ML/Data implementation TODO

Đây là checklist cho repository `ml-data/`. Các module hiện có chỉ là scaffold, chưa phải mô tả chức năng đã hoàn tất.

## M1–M3 · Data pipeline

- [x] Chốt schema bảy bảng OULAD và kiểu dữ liệu chuẩn. *(schema.py: dtype + enum + range + null policy, truy vết EDA 2026-10-26)*
- [x] Implement ingestion, integrity checks, linting, quarantine và source hash. *(ingestion.py + 23 test pytest; verify OULAD thật: 0 quarantine, dedupe 787,170 exact-duplicate studentVle)*
- [x] Hoàn thiện catalog với source, aggregation, as-of rule và feature kind. *(5 features; rule `n_clicks_before_cutoff` đã đồng bộ `date <= cutoff`)*
- [ ] Implement cohort và snapshot tới cutoff ngày 27, kèm as-of ledger/version.
- [ ] Implement leakage audit cho VLE và assessments.
- [ ] Implement quality gate cho schema, missingness, volume và PSI.
- [ ] Thêm kiểm thử cutoff, future-event injection, catalog và quality gates.

## M4–M5 · ML pipeline

- [ ] Implement baseline Logistic Regression và model ứng viên.
- [ ] Dùng temporal split 2013 train/validation → 2014 holdout.
- [ ] Báo cáo Recall@top-k, AUC, ECE, fairness slices và false negatives.
- [ ] Tuning threshold chỉ trên validation; ghi model card/registry metadata.
- [ ] Implement scoring, SHAP, uncertainty, insufficient-data và manual-review.
- [ ] Implement `prediction-batch.v1` export, manifest/checksum và stable pseudonymous key.

## Orchestration/Release

- [ ] Implement stage orchestration, report và failure state.
- [ ] Chỉ xuất batch sau khi quality/leakage/evaluation gate đạt yêu cầu.
- [ ] Phối hợp backend repo để chốt schema và artifact transport.
- [ ] Tạo image/job deployment độc lập sau khi batch contract ổn định.
- [ ] Ghi runbook và xác nhận metric bằng OULAD thật; không giả định tiêu chí đạt trước khi đo.
