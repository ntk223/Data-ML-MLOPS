# BASELINE TIÊU CHÍ ĐÁNH GIÁ — NHÓM 12, LEARNGUARD

**Học phần:** INT3417 1 · **Chủ đề:** Chủ đề 4 — LearnGuard: Student Success Early-Warning Assistant
**Mục đích tài liệu:** thống nhất với giảng viên bộ tiêu chí nghiệm thu sản phẩm **trước khi triển khai chi tiết**, để cả hai bên cùng có một mốc đối chiếu khách quan khi đánh giá cuối kỳ.
**Tài liệu gốc:** `LearnGuard_Spec_Report.md` (REQ-xx / SPEC-xx / mục 2.x) · `LearnGuard_Task_Assignment.md` (milestone)

---

## 1. Nguyên tắc đánh giá

1. **Evidence-based:** mỗi tiêu chí phải kèm cách đo và bằng chứng cụ thể (script tái lập, log CI, video, phiếu khảo sát) — không chấm theo tự nhận.
2. **Hai tầng:** tiêu chí *offline* (đo được trên OULAD + giả lập quy trình cố vấn) là điều kiện nghiệm thu; tiêu chí *online* (adoption, precision thực địa) chỉ áp dụng khi có pilot thật — ghi ở mục 6, không tính vào nghiệm thu giai đoạn này.
3. **Hai mức ngưỡng:** mỗi tiêu chí có mức **Baseline** (phải đạt để được tính "đạt") và mức **Mục tiêu** (phấn đấu, trình bày như điểm cộng).
4. **Tiêu chí gate:** một số tiêu chí là ràng buộc tuyệt đối (đạt/không đạt — thường là compliance về trách nhiệm), không có mức trung gian.

## 2. Nhóm A — Model quality (ML non-generative, tự huấn luyện)

| ID | Tiêu chí | Baseline (đạt) | Mục tiêu | Cách đo | Bằng chứng |
|---|---|---|---|---|---|
| A1 | Recall@top-20% tại snapshot tuần 8 | ≥ 0.60 | ≥ 0.70 | Script đánh giá trên temporal holdout (train 2013, test 2014) | Evaluation report + script tái lập |
| A2 | AUC-ROC tại snapshot tuần 8 | ≥ 0.75 | ≥ 0.80 | Như A1 | Evaluation report |
| A3 | Hiệu chỉnh xác suất (ECE) | ≤ 0.10 | ≤ 0.05 | Như A1 | Evaluation report |
| A4 | Warning lead time | ≥ 60% SV at-risk nằm trong top-20% tại ≥ 1 checkpoint trước `date_unregistration`; lead time TB ≥ 4 tuần | ≥ 70%; TB ≥ 6 tuần | So mốc gắn cờ vs `date_unregistration` trên holdout | Evaluation report |
| A5 | **[GATE] Toàn vẹn thời gian (anti-leakage)** | **0/100%** feature vi phạm as-of constraint | — | Audit script quét timestamp mọi feature, chạy tự động trong CI | Log audit CI |
| A6 | Fairness: max chênh lệch FNR giữa các nhóm (gender, age_band, disability, imd_band) | ≤ 0.05 | ≤ 0.03 | Fairness audit trên holdout | Báo cáo fairness + model card |

## 3. Nhóm B — System quality

| ID | Tiêu chí | Baseline (đạt) | Mục tiêu | Cách đo | Bằng chứng |
|---|---|---|---|---|---|
| B1 | Pipeline end-to-end (ingest → snapshot → inference → brief) | ≤ 1 giờ / 1 cohort | ≤ 20 phút | Đo thật trên máy demo, cohort cố định | Log pipeline |
| B2 | p95 latency sinh brief | ≤ 15 giây | ≤ 8 giây | Đo ≥ 30 brief thực tế | Log latency |
| B3 | Data quality gate chặn batch lỗi | 6/6 case lỗi chuẩn (schema sai, null 70%, volume ×10, PSI lệch…) bị chặn, 0 cảnh báo sinh từ dữ liệu lỗi | — | Bộ test nhiễm lỗi chuẩn hoá, chạy trong CI | Test report |
| B4 | Uptime dashboard trong demo/pilot | ≥ 99% | 100% | Monitoring log | Monitoring screenshot/log |
| B5 | Xử lý uncertain/OOD | 100% case uncertain → trạng thái "cần rà soát thủ công", 0 case tự động gắn cờ | — | Test với batch uncertain nhân tạo | Test report + demo |
| B6 | Fallback khi LLM API lỗi | Brief template từ SHAP/feature sinh đúng khi API chết; hệ thống không dừng | — | Test ngắt API chủ động | Video/test report |

## 4. Nhóm C — LLM quality (advisor brief)

| ID | Tiêu chí | Baseline (đạt) | Mục tiêu | Cách đo | Bằng chứng |
|---|---|---|---|---|---|
| C1 | Groundedness (khẳng định có bằng chứng trong context) | ≥ 90% | ≥ 95% | Checklist máy + spot-check ≥ 30 brief ngẫu nhiên | Kết quả human eval |
| C2 | Faithfulness (top-3 lý do khớp top-3 SHAP) | ≥ 80% | ≥ 90% | So khớp máy giữa brief và SHAP context | Bảng so khớp |
| C3 | Chi phí & giới hạn | ≤ 1 lần gọi API/brief; brief ≤ 300 từ | — | Đo trên batch thực tế | Log API |
| C4 | Validate đầu ra | 100% brief qua post-validation (schema + lý do map về SHAP); fail → regenerate → template | — | Bật/tắt validator trong test để so sánh | Test report |

## 5. Nhóm D — User quality (giả lập quy trình cố vấn)

| ID | Tiêu chí | Baseline (đạt) | Mục tiêu | Cách đo | Bằng chứng |
|---|---|---|---|---|---|
| D1 | Usability (SUS) | ≥ 70 | ≥ 75 | Panel 3–5 đánh giá viên, nhiệm vụ mẫu chuẩn | Phiếu SUS gốc |
| D2 | Hoàn thành nhiệm vụ mẫu | 100%, ≤ 10 phút/nhiệm vụ | ≤ 5 phút | Quan sát có ghi thời gian | Bảng ghi chép |
| D3 | Human eval brief | ≥ 70% brief đạt rubric; Cohen's κ ≥ 0.4 | ≥ 80%; κ ≥ 0.6 | ≥ 30 brief × 3–5 đánh giá viên | Phiếu chấm + bảng κ |

## 6. Nhóm E — MLOPS process (pipeline, reproducibility, vận hành)

| ID | Tiêu chí | Baseline (đạt) | Mục tiêu | Cách đo | Bằng chứng |
|---|---|---|---|---|---|
| E1 | **[GATE] Reproducibility** | Chạy lại pipeline 2 lần trên cùng data version → hash output giống 100% | — | Chạy trên máy sạch | Log 2 lần chạy + hash |
| E2 | Versioning | Code (Git) + data (hash/version) + model (registry + model card) đều được version hoá | — | Kiểm tra repo | Repo + model card |
| E3 | **[GATE] CI & automated testing** | CI xanh là điều kiện merge; bộ test phủ: unit feature, temporal audit, quality gate, API smoke | — | Xem lịch sử CI | GitHub Actions log |
| E4 | Release gated + rollback | Model chưa qua audit không thể vào serving; demo được rollback về version cũ | — | Demo quy trình | Video + runbook |
| E5 | Deployment | docker-compose chạy trên 1 máy từ trạng thái sạch | ≤ 2 lệnh khởi động | Cài đặt theo tài liệu | Video cài đặt |
| E6 | Logging/monitoring/runbook | Log có cấu trúc cho request/dự đoán/lỗi; monitoring: latency, lỗi, PSI drift; runbook ≥ 3 sự cố chính | — | Kiểm tra + demo | Runbook + monitoring |

## 7. Nhóm F — Responsible AI (4 từ khoá đề bài gọi tên)

| ID | Tiêu chí | Baseline (đạt) | Cách đo | Bằng chứng |
|---|---|---|---|---|
| F1 | **[GATE] Human-in-the-loop:** 0% can thiệp tự động tới sinh viên; mọi hành động qua phê duyệt cố vấn | Kiến trúc + code review + demo | Kiến trúc hệ thống |
| F2 | **[GATE] Trách nhiệm giao diện:** disclaimer bắt buộc + hướng dẫn giao tiếp không gắn nhãn | Kiểm tra UI | Screenshot + demo |
| F3 | Explainability: mỗi cảnh báo top-k có ≥ 3 yếu tố SHAP bằng ngôn ngữ thường | Kiểm tra ≥ 30 brief | Human eval kết quả |
| F4 | Privacy: dữ liệu pseudonymized; 0 PII trong log; dự đoán không hiển thị cho sinh viên | Kiểm tra log + code | Audit log |
| F5 | Fairness theo dõi liên tục: báo cáo FNR gap kèm mỗi release model | Model card | Model card |

## 8. Đối chiếu với 8 yêu cầu bắt buộc của đề

| Yêu cầu đề | Tiêu chí tương ứng |
|---|---|
| 1. ML non-generative tự huấn luyện trên dữ liệu công khai | A1–A3 (OULAD — nguồn công khai) |
| 2. Thành phần LLM (API/self-hosted) | C1–C4 |
| 3. Backend/API + giao diện | B1, B4, D1–D2 |
| 4. Data pipeline + kiểm tra chất lượng dữ liệu | B1, B3, E1–E2 |
| 5. Đánh giá riêng model quality và system quality | Nhóm A (model) tách bạch Nhóm B (system) |
| 6. Xử lý uncertainty/error (fallback, human review, escalation) | B5, B6, C4 |
| 7. Tái lập, versioning, kiểm thử tự động | E1–E4 |
| 8. Deployment, logging/monitoring, xử lý sự cố | E5–E6 |

## 9. Lịch đo tiêu chí

| Mốc | Nội dung đo |
|---|---|
| Tuần 5 (M2) | Nhóm A đo lần đầu trên temporal holdout; A5, A6 là điều kiện tiếp tục |
| Tuần 7 (M3) | Nhóm B đo trên demo end-to-end |
| Tuần 8 | Nhóm E (CI/test/deploy); C4 |
| Tuần 9 (M4) | Đo đầy đủ: A (chốt), B, C1–C3, D, F — ghi kết quả thật |
| Tuần 10 (M5) | Bảng tổng hợp kết quả vs baseline nộp cùng báo cáo; demo trực tiếp |

## 10. Online criteria — chỉ áp dụng khi có pilot thật (không tính nghiệm thu)

- Adoption: ≥ 80% cố vấn đăng nhập ≥ 1 lần/tuần.
- Precision@k thực địa theo feedback cố vấn ≥ 50% sau tháng đầu.
- Thời gian phản hồi cố vấn với cảnh báo ưu tiên ≤ 7 ngày.

> Ba mục này được ghi nhận như *mục tiêu sau triển khai*; giai đoạn học phần đánh giá bằng giả lập (Nhóm D).

## 11. Quy ước khi không đạt

- Tiêu chí **gate** (A5, E1, E3, F1, F2) không được thoả hiệp — nếu vi phạm, sản phẩm chưa đủ điều kiện trình demo chính thức.
- Tiêu chí mức: nếu trễ tiến độ, ưu tiên cắt tính năng dashboard trước; **không cắt** evaluation và audit.
- Mọi điều chỉnh ngưỡng (nếu có) phải cập nhật lại tài liệu này và được giảng viên xác nhận trước khi áp dụng — tài liệu này là mốc thống nhất duy nhất.

## 12. Xác nhận thống nhất

| Bên | Họ tên | Ý kiến | Chữ ký | Ngày |
|---|---|---|---|---|
| Nhóm 12 (trưởng nhóm) | Nguyễn Văn Thanh Tùng | | | |
| Giảng viên hướng dẫn | | | | |
