# ML/Data implementation TODO

Checklist hiện trạng cho repository `ml-data/`.

## M1–M3 · Data pipeline

- [x] Chốt schema bảy bảng OULAD và kiểu dữ liệu chuẩn. *(schema.py: dtype + enum + range + null policy, truy vết EDA 2026-10-26)*
- [x] Implement ingestion, integrity checks, linting, quarantine và source hash. *(ingestion.py + 23 test pytest; verify OULAD thật: 0 quarantine, dedupe 787,170 exact-duplicate studentVle)*
- [x] Hoàn thiện catalog với source, aggregation, as-of rule và feature kind. *(6 features; thêm `active_days_before_cutoff`, tính cả ngày âm)*
- [x] Implement cohort và snapshot tới cutoff ngày 27, kèm as-of ledger/version. *(snapshot-v3, 22 cột, semantic source/contract/content hashes; verify thật: 27,535 dòng)*
- [x] Implement leakage audit cho VLE và assessments. *(LeakageAuditor 3 lớp: catalog rules + recompute đối chiếu + 4 injection probes×4 SV; verify thật: 0 violation)*
- [x] Implement quality gate cho schema, missingness, volume và PSI. *(DataQualityGate 4 gates; PSI yêu cầu ≥200 mẫu/bên — chặn false alarm do nhiễu; verify thật: ok)*
- [x] Thêm kiểm thử cutoff, future-event injection, catalog, quality gates và safe run IDs. *(97 test M1–M3; full suite sau Phase 4: 123 pass)*
- [x] Hoàn thiện ZIP ingestion atomic, composite FK, snapshot-v3 semantic hash và loader verification.
- [x] Hoàn thiện data-only orchestration + CLI fail-fast cho M1–M3.
- [x] Thêm PR CI và workflow full OULAD hai run để kiểm tra reproducibility/PSI.

## M4–M5 · ML pipeline

- [x] Implement train-only preprocessing contract và artifact verification. *(fit chỉ trên train 2013B+2013J; `code_module` thay `module_presentation`; dense 36 features trên OULAD thật)*
- [x] Thêm imputation/encoding/scaling, unknown-category handling và preprocessing leakage probes.
- [x] Thêm manifest, semantic hashes, atomic publication và CLI `ml preprocess`/`ml verify-preprocessor`.
- [x] Implement Phase 5 / M4 baseline Logistic Regression train-only, fixed contract,
  semantic artifacts, trusted loader, atomic publication và CLI verification.
- [ ] Implement model ứng viên trong phase riêng sau baseline.
- [x] Baseline tuân thủ temporal split: fit train 2013B+2013J; training không đọc
  validation 2014B hoặc holdout 2014J.
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
