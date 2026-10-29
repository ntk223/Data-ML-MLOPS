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
- [ ] Implement scoring, SHAP, uncertainty, insufficient-data và manual-review.
- [ ] Implement `prediction-batch.v1` export, manifest/checksum và stable pseudonymous key.

## Orchestration/Release

- [ ] Implement stage orchestration, report và failure state.
- [ ] Chỉ xuất batch sau khi quality/leakage/evaluation gate đạt yêu cầu.
- [ ] Phối hợp backend repo để chốt schema và artifact transport.
- [ ] Tạo image/job deployment độc lập sau khi batch contract ổn định.
- [ ] Ghi runbook và xác nhận metric bằng OULAD thật; không giả định tiêu chí đạt trước khi đo.
