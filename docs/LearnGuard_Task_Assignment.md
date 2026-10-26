# KẾ HOẠCH GIAO VIỆC NHÓM 12 — LEARNGUARD

**Học phần:** INT3417 1 · **Thời gian:** 10 tuần · **Sản phẩm:** LearnGuard — Student Success Early-Warning Assistant
**Tài liệu tham chiếu:** `LearnGuard_Spec_Report.md` (mã REQ-xx / SPEC-xx / mục 2.x quy về báo cáo spec)

## 1. Phân vai theo workstream

| Thành viên | Vai trò | Phạm vi chính |
|---|---|---|
| Nguyễn Văn Thanh Tùng (23021716) | **ML & Evaluation Lead** | Mô hình dự đoán, calibration, SHAP, fairness audit, toàn bộ evaluation |
| Nguyễn Trung Kiên (23021590) | **Data Lead** | ETL, feature/snapshot, data quality gate, versioning, CI |
| Nguyễn Văn Huy Hoàng (23021561) | **LLM Lead** | Prompt, advisor brief, grounding, fallback template, human eval brief |
| Trần Việt Hưng (23021586) | **System/DevOps Lead** | Backend/API, dashboard, deployment, logging/monitoring, runbook |

> Vai trò có thể hoán đổi theo thế mạnh thực tế — thống nhất lại trong buổi nhóm tuần 1. Mỗi mảng vẫn luôn có đúng 1 owner chịu trách nhiệm cuối.

**Backup chéo (co-owner):** Tùng ↔ Kiên (data/model), Hoàng ↔ Hưng (LLM/system) — tránh single point of failure khi ai đó bận thi/đợt khác.

## 2. Nguyên tắc làm việc

- Mọi mã nguồn qua **Git**: nhánh feature → pull request → 1 người review chéo → merge. Không commit trực tiếp lên `main`.
- **Definition of Done (mỗi task):** code merged + test pass + có tài liệu/notebook giải thích + demo được cho nhóm.
- Họp nhóm **1 lần/tuần** (30 phút): demo kết quả, chốt block, phân công tuần sau.
- Mọi release model/system tuân theo **SPEC-06** (audit + test pass + model card cập nhật).
- Log quyết định kỹ thuật vào `docs/decisions.md` (design decision log) — phục vụ bảo vệ.

## 3. Timeline 10 tuần — milestone tổng

| Tuần | Milestone | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | Setup + spec chốt | Repo khởi tạo, môi trường chạy được, OULAD đã tải, EDA khái quát |
| 2 | **M1 — Dữ liệu sạch** | ETL xong 7 bảng, danh sách feature as-of được chốt |
| 3–4 | Model pipeline | Baseline + GBM train xong trên temporal holdout (2013 → 2014) |
| 5 | **M2 — Model v1** | REQ-01 (Recall@20% ≥ 0.6, AUC ≥ 0.75) + REQ-02 (audit 0 vi phạm) + REQ-05 (FNR gap ≤ 0.05) đạt trên holdout |
| 6–7 | **M3 — End-to-end demo** | Pipeline → inference → brief → dashboard chạy được 1 vòng hoàn chỉnh |
| 8 | Hệ thống cứng hoá | Quality gate + fallback + test tự động + container + monitoring hoạt động |
| 9 | **M4 — Evaluation đầy đủ** | Toàn bộ metric 2.1–2.5 của báo cáo spec được đo và ghi kết quả thật |
| 10 | **M5 — Bàn giao** | Demo + báo cáo cuối + slide + video backup |

## 4. Chi tiết công việc theo từng người

### 4.1 Nguyễn Văn Thanh Tùng — ML & Evaluation Lead

| Tuần | Công việc | Deliverable | Liên kết |
|---|---|---|---|
| 1 | Đọc tài liệu OULAD; EDA khái quát: phân bố `final_result`, missing, khác biệt giữa các course/presentation | Notebook EDA v1 | Mục 1.3 |
| 2 | Cùng Kiên chốt danh sách feature as-of (mỗi feature ghi rõ nguồn timestamp) | Feature definition doc | REQ-02 |
| 3 | Baseline: logistic regression + rule-based; dựng khung evaluation (temporal split 2013→2014) | Bảng benchmark baseline | REQ-01 |
| 4 | Train GBM (XGBoost/LightGBM) + calibration (isotonic/Platt); chạy snapshot tuần 4/8/12 | Model v0 + kết quả holdout | REQ-01 |
| 5 | SHAP per-student (top-3) + fairness audit FNR gap theo 4 nhóm; nếu vi phạm → mitigation; model card v1 | Model v1 + model card | REQ-03, REQ-05 |
| 6 | Cung cấp format context (feature + SHAP) cho LLM; thống nhất vùng uncertain | Context schema | SPEC-04, SPEC-08 |
| 7 | Thiết lập ngưỡng uncertain/OOD; re-train trên dữ liệu qua quality gate | Cấu hình inference | SPEC-08 |
| 8 | Viết test tự động model: temporal audit test, unit test feature, pin seed/dependency | Test suite pass trong CI | SPEC-06 |
| 9 | Chạy evaluation cuối: Recall@top-20%, AUC, ECE, lead time, fairness — ghi kết quả thật vào báo cáo | Evaluation report | Mục 2.1, 2.4 |
| 10 | Polish model card + slide phần ML + Q&A bảo vệ (sẵn sàng trả lời "vì sao top-k", "vì sao GBM+SHAP") | Slide + Q&A sheet | T-01, T-03 |

### 4.2 Nguyễn Trung Kiên — Data Lead

| Tuần | Công việc | Deliverable | Liên kết |
|---|---|---|---|
| 1 | Tải OULAD; mô tả schema 7 file CSV; init repo + cấu trúc project + quy ước Git | Repo + data dictionary | — |
| 2 | ETL đầy đủ 7 bảng → processed data; notebook EDA dùng chung | Processed dataset | Mục 1.4.1 |
| 3 | Feature store: hàm build snapshot as-of tuần 4/8/12 (as-of join, không rò tương lai); unit test đầu tiên | Snapshot builder | REQ-02, SPEC-01 |
| 4 | Tối ưu snapshot (chạy ≤ vài phút/cấu hình), chuẩn hoá output cho model | Snapshot pipeline ổn định | SPEC-01 |
| 5 | Data quality gate: schema/null/volume/PSI check + fail-safe chặn batch lỗi | Quality gate module | SPEC-07 |
| 6 | Ghép chu trình tuần: ingest → snapshot → handoff inference → trả kết quả; đóng gói CLI | Pipeline chạy được theo chu kỳ | SPEC-01 |
| 7 | Data versioning (hash/DVC phiên bản snapshot); kiểm chứng tái lập 2 lần chạy giống hệt nhau | Reproducibility check pass | SPEC-06 |
| 8 | Test tự động pipeline (unit + integration); CI chạy trước mỗi merge (GitHub Actions) | CI xanh trên main | SPEC-06 |
| 9 | Re-run pipeline sạch cho evaluation cuối; hỗ trợ debug tích hợp toàn hệ thống | Dữ liệu evaluation cuối | M4 |
| 10 | Tài liệu pipeline (sơ đồ + hướng dẫn chạy từ 0) + slide phần data | Tài liệu + slide | — |

### 4.3 Nguyễn Văn Huy Hoàng — LLM Lead

| Tuần | Công việc | Deliverable | Liên kết |
|---|---|---|---|
| 1 | So sánh LLM API vs local (chi phí, latency, tiếng Anh/Việt); ước lượng chi phí theo ASM-04 | Chốt lựa chọn LLM | ASM-04 |
| 2 | Thiết kế schema brief (mức rủi ro + 3 lý do + 2–3 gợi ý, ≤ 300 từ) + rubric groundedness | Brief spec + rubric | REQ-04 |
| 3 | Prototype brief với dữ liệu mẫu; prompt v1 + nén context (feature chính + SHAP top-5) | Brief mẫu đầu tiên | SPEC-04 |
| 4 | Xây thư viện template gợi ý hành động (khuyến nghị chỉ lấy từ template đã duyệt) | Template library | SPEC-04 |
| 5 | Đo groundedness/faithfulness trên batch brief (cùng Tùng với SHAP); chỉnh prompt | Kết quả đo + prompt v2 | Mục 2.5 |
| 6 | Gắn API sinh brief vào backend; đo p95 latency ≤ 15s, chi phí/brief | LLM endpoint | REQ-04 |
| 7 | Fallback template tĩnh khi LLM lỗi/timeout; log mọi sự kiện fallback/escalation | Fallback hoạt động | SPEC-08 |
| 8 | Test tự động LLM component: mock API, validate schema output, smoke test | Test suite pass | SPEC-06 |
| 9 | Human eval: ≥ 30 brief × 3–5 đánh giá viên, tính κ ≥ 0.4; sửa theo phản hồi | Kết quả human eval | Mục 2.3 |
| 10 | Tài liệu LLM component + slide phần brief + Q&A (grounding, vì sao template-only) | Tài liệu + slide | — |

### 4.4 Trần Việt Hưng — System/DevOps Lead

| Tuần | Công việc | Deliverable | Liên kết |
|---|---|---|---|
| 1 | Chọn stack (FastAPI + Streamlit/React); skeleton repo + Docker skeleton | Chốt stack + khung dự án | Mục 1.4.6 |
| 2 | Thiết kế API contract (danh sách risk, chi tiết SV, brief, feedback) + DB đơn giản | API spec + schema DB | 1.4.6 |
| 3 | Dashboard v1: bảng top-k, filter, trang chi tiết sinh viên | UI chạy được với mock | 1.4.6 |
| 4 | Tích hợp API với model output (mock trước, đợi format thật) | API + model connect | REQ-01 |
| 5 | Trang brief + form feedback đúng/sai + trạng thái can thiệp; disclaimer + hướng dẫn giao tiếp | Dashboard v2 | SPEC-05, REQ-04 |
| 6 | Hoàn thiện backend; kết nối LLM endpoint; log có cấu trúc mọi request/dự đoán | Backend hoàn chỉnh | SPEC-09 |
| 7 | Hàng đợi human review cho uncertain/OOD; hiển thị trạng thái "cần rà soát thủ công"; retry/fallback UI | Xử lý uncertain trên UI | SPEC-08 |
| 8 | Deployment: docker-compose chạy trên 1 máy; monitoring (latency/lỗi/PSI/số brief); runbook 1 trang | Hệ thống deploy được | SPEC-09 |
| 9 | Tổ chức usability study (nhiệm vụ mẫu, thu SUS ≥ 70); sửa UX theo feedback | Kết quả usability study | Mục 2.3 |
| 10 | Demo script + video backup; slide phần hệ thống; dọn repo + tag release cuối | Demo + release | M5 |

## 5. Ma trận trách nhiệm theo REQ/SPEC

| Hạng mục | Owner | Support |
|---|---|---|
| REQ-01, REQ-02 (dự đoán + anti-leakage) | Tùng | Kiên |
| REQ-03 (explainability) | Tùng | Hoàng |
| REQ-04 (advisor brief) | Hoàng | Hưng |
| REQ-05 (fairness) | Tùng | Kiên |
| SPEC-01, SPEC-02, SPEC-07 (pipeline, thiếu dữ liệu, quality gate) | Kiên | Tùng |
| SPEC-03, SPEC-06 (audit, versioning, test) | Kiên | Tùng, cả nhóm review |
| SPEC-04 (ràng buộc LLM) | Hoàng | Tùng |
| SPEC-05 (disclaimer/UI trách nhiệm) | Hưng | Hoàng |
| SPEC-08 (uncertainty/fallback) | Hoàng | Hưng |
| SPEC-09 (deployment/vận hành) | Hưng | Kiên |

## 6. Checklist theo dõi hàng tuần (chủ nhóm/Tùng cập nhật)

- [ ] Tuần này mỗi người hoàn thành Deliverable của mình chưa?
- [ ] PR được review chéo và merge chưa?
- [ ] CI/test còn xanh không?
- [ ] Có block nào cần hỗ trợ chéo không?
- [ ] `docs/decisions.md` có quyết định mới nào cần ghi lại không?
- [ ] Milestone tuần hiện tại đạt tiêu chí nghiệm thu chưa?

## 7. Ghi chú

- Phân công hiện theo giả định chia đều thế mạnh (ai mạnh mảng nào thì nhận mảng đó — điều chỉnh trong tuần 1).
- Nếu giữa kỳ có thành viên bận: backup chéo (mục 1) nhận tạm, không để task chết.
- Ưu tiên luôn cho **M2 (model v1 đạt REQ-01/02/05)** — đây là nền của mọi phần sau; nếu trễ, cắt giảm tính năng dashboard trước, không cắt evaluation.

### 7.1 Ưu tiên theo tinh thần môn MLOPS

Đây là môn **MLOPS** — điểm nằm ở *hệ thống quanh model*, không phải ở model. Thứ tự ưu tiên khi phân bổ thời gian:

1. **Pipeline có kiểm soát + reproducibility** (as-of snapshot, audit chống temporal leakage, versioning data/code/model, test tự động + CI) — trọng tâm các tuần 2, 5, 7, 8.
2. **Hệ thống end-to-end chạy được sớm** (M3, tuần 7): pipeline → model → LLM brief → dashboard demo trọn vòng. Rủi ro chết người nhất của bài tập lớn là dồn tích hợp xuống cuối kỳ — luôn có demo chạy được, kèm video backup.
3. **4 từ khóa đề bài gọi tên** (temporal leakage, explainability, fairness, dùng prediction có trách nhiệm) — chi phí thấp (mỗi mục ~1–2 ngày), giá trị chấm cao nhất; đã cụ thể hoá ở REQ-02/03/05 + SPEC-02/05/08.
4. **Model đủ tốt là dừng:** đạt REQ-01 (Recall@top-20% ≥ 0.6, AUC ≥ 0.75) thì ngừng tuning — đừng đốt quá 1 tuần cho +0.02 AUC.
5. **UI vừa đủ, đúng scope:** chức năng đầy đủ + SUS pass; 1 máy, batch tuần — không real-time/k8s.

**Cơ cấu thời gian tham khảo (10 tuần × 4 người):** ~30% data + model · ~40% pipeline/CI/deploy/monitoring · ~20% evaluation + audit + responsible AI · ~10% tài liệu/demo.
