# BÁO CÁO BÀI TẬP LỚN — HỌC PHẦN INT3417 1

## Project Definition & Requirements Specification

| Thông tin | Chi tiết |
|---|---|
| **Lớp** | INT3417 1 |
| **Nhóm** | 12 |
| **Chủ đề** | Chủ đề 4 — LearnGuard: Student Success Early-Warning Assistant |

### Thành viên nhóm

| STT | Mã sinh viên | Họ và tên |
|---|---|---|
| 1 | 23021716 | Nguyễn Văn Thanh Tùng |
| 2 | 23021590 | Nguyễn Trung Kiên |
| 3 | 23021561 | Nguyễn Văn Huy Hoàng |
| 4 | 23021586 | Trần Việt Hưng |

---

## Bảng viết tắt

| Viết tắt | Thuật ngữ đầy đủ | Nghĩa tiếng Việt / Ghi chú |
|---|---|---|
| **AI** | Artificial Intelligence | Trí tuệ nhân tạo |
| **API** | Application Programming Interface | Giao diện lập trình ứng dụng |
| **ASM** | Assumption | Giả định (mã ASM-xx, mục 3.2) |
| **AUC** | Area Under the ROC Curve | Diện tích dưới đường cong ROC — thước đo khả năng phân biệt của mô hình |
| **CPU** | Central Processing Unit | Bộ xử lý trung tâm |
| **ECE** | Expected Calibration Error | Sai số hiệu chỉnh xác suất kỳ vọng |
| **ETL** | Extract – Transform – Load | Trích xuất – biến đổi – nạp dữ liệu |
| **FM** | Failure Mode | Kịch bản rủi ro/chế độ hỏng (mã FM-xx, mục 5) |
| **FN / FNR** | False Negative / False Negative Rate | Trường hợp bỏ sót / tỷ lệ bỏ sót (âm tính giả) |
| **GPU** | Graphics Processing Unit | Bộ xử lý đồ hoạ |
| **IMD** | Index of Multiple Deprivation | Chỉ số deprivation đa chiều (thước đo khó khăn kinh tế – xã hội ở UK; trường `imd_band` trong OULAD) |
| **LLM** | Large Language Model | Mô hình ngôn ngữ lớn |
| **LMS** | Learning Management System | Hệ thống quản lý học tập |
| **ML** | Machine Learning | Học máy |
| **OULAD** | Open University Learning Analytics Dataset | Bộ dữ liệu phân tích học tập của The Open University (UK) |
| **OOD** | Out-Of-Distribution | Dữ liệu/ví dụ nằm ngoài phân phối dữ liệu huấn luyện |
| **PII** | Personally Identifiable Information | Thông tin nhận dạng cá nhân |
| **PSI** | Population Stability Index | Chỉ số ổn định phân phối (so sánh train/test) |
| **REQ** | Requirement | Yêu cầu (mã REQ-xx, mục 3.1) |
| **SHAP** | SHapley Additive exPlanations | Phương pháp giải thích dự đoán dựa trên giá trị Shapley |
| **SPEC** | Specification | Đặc tả hành vi hệ thống (mã SPEC-xx, mục 3.3) |
| **SUS** | System Usability Scale | Thang đo khả năng sử dụng hệ thống (thang điểm 0–100) |
| **TA** | Teaching Assistant | Trợ giảng |
| **TMA** | Tutor Marked Assessment | Bài đánh giá do giảng viên/gia sư chấm (trong OULAD) |
| **T-xx** | Trade-off | Đánh đổi thiết kế (mã T-01…, mục 4.3) |
| **VLE** | Virtual Learning Environment | Môi trường học tập ảo — nền tảng học trực tuyến của The Open University |
| **top-k / @k** | — | k sinh viên/ví dụ có điểm rủi ro cao nhất được đưa vào danh sách ưu tiên |
| **κ** | Cohen's Kappa | Hệ số đồng thuận giữa các đánh giá viên |

---

# 1. Problem & Scope

## 1.1 Định nghĩa sản phẩm

> **"Dành cho cố vấn học tập và giảng viên phụ trách lớp học phần, LearnGuard sử dụng mô hình ML (không sinh tạo) để dự đoán rủi ro từ dữ liệu hành vi học tập, kết hợp LLM để soạn brief cố vấn, giúp phát hiện sớm và ưu tiên can thiệp cho sinh viên có nguy cơ trượt hoặc bỏ học trước khi quá muộn."**

## 1.2 Primary user & stakeholders

- **Primary user:** Cố vấn học tập (academic advisor) — người nhận cảnh báo, đọc brief, quyết định và thực hiện can thiệp.
- **Secondary stakeholders:**
  - Giảng viên phụ trách lớp học phần (cần bức tranh risk của lớp để điều chỉnh giảng dạy).
  - Sinh viên — người bị tác động trực tiếp bởi quyết định can thiệp (không tương tác trực tiếp với hệ thống trong phiên bản này).
  - Phòng Đào tạo / Công tác sinh viên — quan tâm tới tổng thể tỷ lệ fail/withdraw ở cấp trường.

## 1.3 Vấn đề cần giải quyết & giá trị mong muốn

Trong OULAD (Open University Learning Analytics Dataset), **≈ 45%** hồ sơ học phần kết thúc bằng **Fail hoặc Withdrawn** — tức là gần một nửa sinh viên thất bại ở một học phần. Hiện trạng của cố vấn/giảng viên:

- Phát hiện sinh viên gặp khó khăn **quá muộn** (thường chỉ khi điểm thi cuối kỳ/báo cáo bỏ học đã có).
- Quy trình phụ thuộc **quan sát thủ công**: 1 cố vấn phụ trách hàng trăm sinh viên, không thể theo dõi đều chi tiết hành vi học tập của từng người.
- Thông tin hành vi (điểm assessment giữa kỳ, mức độ tương tác trên LMS) **tồn tại nhưng không được tổng hợp** thành cảnh báo có thể hành động.

**Giá trị mong muốn:** chuyển từ phát hiện muộn → phát hiện sớm (giữa kỳ); từ bao phủ thủ công → xếp hạng ưu tiên tự động theo top-k phù hợp năng lực can thiệp; từ "con số điểm" → "lý do + gợi ý hành động" bằng ngôn ngữ tự nhiên giúp cố vấn hành động nhanh hơn.

## 1.4 In scope — hệ thống sẽ làm gì

1. Xây dựng pipeline dữ liệu trên **OULAD**: `studentInfo`, `studentVle`, `vle`, `assessments`, `studentAssessment`, `courses`, `studentRegistration`.
2. Sinh **snapshot đặc trưng theo mốc tuần** (week 4 / 8 / 12), chỉ dùng dữ liệu có thời điểm ≤ mốc cắt (chống temporal leakage).
3. **Mô hình ML phân loại at-risk** (nhãn = Fail hoặc Withdrawn) — gradient boosting + xác suất hiệu chỉnh (calibration).
4. **Explainability từng sinh viên** (SHAP): top yếu tố đóng góp, dịch sang ngôn ngữ dễ hiểu.
5. **LLM sinh advisor brief**: mức rủi ro + 3 lý do chính + 2–3 gợi ý hành động hỗ trợ, có grounding vào dữ liệu.
6. **Backend/API + web dashboard:** API phục vụ inference và sinh brief; dashboard cho cố vấn: danh sách sinh viên ưu tiên (top-k), xem brief, đánh giá đúng/sai + trạng thái can thiệp.
7. **Bộ đánh giá**: temporal split, Recall@top-k, calibration, fairness audit theo nhóm nhân khẩu, kiểm định groundedness của LLM.
8. **Data quality gate:** kiểm tra tự động chất lượng dữ liệu (schema, tỷ lệ thiếu, khối lượng, độ ổn định phân phối) trước mỗi lần inference.
9. **Vận hành sản phẩm hoàn chỉnh:** container hoá, deployment, logging/monitoring cơ bản, runbook xử lý sự cố.
10. **Reproducibility & kiểm thử tự động:** versioning dữ liệu/mã/mô hình, bộ test tự động chạy trước mỗi release.

## 1.5 Out of scope — hệ thống chủ động không làm

1. **Không tự động** gửi thông báo/can thiệp tới sinh viên — mọi hành động đều do cố vấn quyết định và thực hiện (human-in-the-loop).
2. Không chẩn đoán tâm lý/sức khỏe, không đưa kết luận về **năng lực hay hoàn cảnh cá nhân** của sinh viên; không thay thế tư vấn chuyên môn.
3. Không xử lý dữ liệu real-time/streaming — chu kỳ cập nhật **batch theo tuần**.
4. Không dùng LLM cho bước **dự đoán** rủi ro (LLM chỉ tổng hợp/giải thích/gợi ý); không fine-tune LLM trong phạm vi bài tập.
5. Không xây dựng quy trình thu thập dữ liệu LMS thực tế trong giai đoạn này — demo trên OULAD (dữ liệu công khai, đã ẩn danh).

## 1.6 Vì sao cần AI/ML? Có giải pháp đơn giản hơn không?

- **Rules tĩnh** (VD: "sinh viên giảm > 30% lượt truy cập VLE trong 2 tuần" hoặc "trượt TMA đầu tiên"): dễ triển khai, minh bạch, nhưng (i) ngưỡng phải thiết kế thủ công cho từng học phần, (ii) bỏ sót pattern đa chiều (tương tác giữa VLE + assessment + nhân khẩu học), (iii) không có cơ chế xếp hạng ưu tiên khi nguồn lực can thiệp giới hạn.
- **Xử lý thủ công:** 1 cố vấn / hàng trăm sinh viên → không khả thi về thời gian; chất lượng phụ thuộc kinh nghiệm cá nhân, không nhất quán.
- **ML (chọn):** học từ dữ liệu lịch sử để tổng hợp hàng chục tín hiệu thành một risk score + ranking top-k — đúng dạng bài toán **ưu tiên hoá** nguồn lực can thiệp. Yêu cầu kèm theo: explanation (SHAP) + fairness audit để dùng có trách nhiệm.
- **LLM (chọn):** đóng vai trò *lớp chuyển đổi* từ feature vector + SHAP → ngôn ngữ tự nhiên có tính hành động, giảm chi phí đọc hiểu của cố vấn. LLM **không** tham gia dự đoán.

# 2. Success Criteria

## 2.1 Primary outcome

Trên temporal holdout của OULAD, hệ thống phải **gắn cờ ≥ 60% sinh viên at-risk (Fail/Withdrawn) trong top-20% danh sách ưu tiên**, với **warning lead time trung bình ≥ 4 tuần** — cảnh báo xuất hiện ít nhất 4 tuần trước khi sinh viên bỏ học (`date_unregistration`) hoặc trước assessment quyết định, đủ sớm để can thiệp còn kịp thời gian phát huy tác dụng.

> Metric này đo được hoàn toàn offline: OULAD có sẵn nhãn kết quả và mốc thời gian bỏ học, không phụ thuộc người dùng thật.

## 2.2 Guardrails (điều không được phép xấu đi)

1. **Fairness:** chênh lệch False Negative Rate (FNR) giữa các nhóm được bảo vệ (giới tính, age band, disability, imd band) **≤ 0.05** trên tập kiểm định — không được bỏ sót có hệ thống một nhóm sinh viên nào.
2. **Trách nhiệm & an toàn:** 100% can thiệp bắt buộc qua phê duyệt cố vấn (không tự động hoá); brief không được chứa khẳng định ngoài dữ liệu (groundedness ≥ 90%); dự đoán không được hiển thị cho sinh viên.

## 2.3 System & User quality metrics

Đánh giá **riêng biệt** với model quality (2.4–2.5): model quality đo độ chính xác dự đoán; mục này đo chất lượng hệ thống và trải nghiệm người dùng.

**System quality:**

- Pipeline end-to-end ≤ 1 giờ/1 cohort (SPEC-01); p95 latency sinh brief ≤ 15 giây (mục 2.5).
- Data quality gate chặn 100% batch dữ liệu lỗi trước khi inference (SPEC-07).
- Dashboard khả dụng ≥ 99% trong giai đoạn demo/pilot (SPEC-09).

**User quality (giả lập quy trình cố vấn):**

- **Usability study:** panel 3–5 đánh giá viên đóng vai cố vấn (bạn học/TA/giảng viên) thực hiện nhiệm vụ mẫu *"chọn 3 sinh viên ưu tiên cao nhất → đọc brief → đề xuất can thiệp"*: **SUS ≥ 70** (trung bình ngành ~68), tỷ lệ hoàn thành nhiệm vụ **100%**, thời gian **≤ 10 phút/nhiệm vụ**.
- **Human eval brief:** 3–5 đánh giá viên chấm ≥ 30 brief ngẫu nhiên theo rubric (bằng chứng đúng + hữu ích + hành động khả thi): **≥ 70% brief đạt**, inter-rater agreement mức moderate (Cohen's κ ≥ 0.4).

## 2.4 ML model metrics

- **Recall@top-20%** ≥ 0.60 và **AUC ≥ 0.75** tại snapshot tuần 8 trên temporal holdout (train: kỳ 2013, test: kỳ 2014).
- **Calibration:** Expected Calibration Error (ECE) ≤ 0.10 — vì risk score được dùng để so sánh/ưu tiên giữa sinh viên.

## 2.5 LLM metrics

- **Groundedness** ≥ 90%: tỷ lệ khẳng định trong brief có bằng chứng trực tiếp trong context được cấp (kiểm bằng checklist + spot-check thủ công).
- **Faithfulness:** top-3 lý do nêu trong brief khớp top-3 SHAP factors ≥ 80%.
- **Latency:** p95 ≤ 15 giây/brief; độ dài ≤ 300 từ.

## 2.6 Tiêu chí hai tầng & lưu ý

Tiêu chí được tách theo **khả năng đo lường của từng giai đoạn**:

- **Offline criteria — nghiệm thu giai đoạn này:** toàn bộ metric ở 2.1–2.5, đo được trên dataset tĩnh OULAD + giả lập quy trình cố vấn, không cần người dùng thật.
- **Online criteria — future work (khi có pilot thật):** adoption (≥ 80% cố vấn đăng nhập ≥ 1 lần/tuần), precision@k thực địa theo feedback cố vấn (≥ 50% sau tháng đầu), thời gian phản hồi cố vấn với cảnh báo ưu tiên (≤ 7 ngày). Đây là điều kiện đánh giá *sau triển khai*, không phải điều kiện nghiệm thu hiện tại.

Không dùng duy nhất accuracy/F1 làm tiêu chí thành công: bài toán mất cân bằng lớp, bản chất là **ưu tiên hoá top-k** (chứ không phân loại toàn tập), và thành công thực sự phụ thuộc fairness, lead time và chất lượng brief.

# 3. Requirements — Assumptions — Specifications

## 3.1 Requirements (REQ)

> REQ-ID: Requirement + Measure/Acceptance criterion

- **REQ-01 — Chất lượng dự đoán:** Với snapshot tại tuần 8 trên temporal holdout của OULAD (train 2013, test 2014B/2014J), mô hình phải đạt **Recall@top-20% ≥ 0.60** và **AUC ≥ 0.75** với nhãn at-risk = {Fail, Withdrawn}.
- **REQ-02 — Toàn vẹn thời gian (anti-leakage):** Tại thời điểm dự đoán, pipeline chỉ được sử dụng dữ liệu có timestamp ≤ cuối tuần 8. Audit script tự động kiểm tra as-of constraint trên 100% feature — **0 vi phạm** được chấp nhận.
- **REQ-03 — Giải thích được:** Với mỗi sinh viên trong top-k, hệ thống hiển thị **≥ 3 yếu tố đóng góp chính** (SHAP) bằng ngôn ngữ thường; ≥ 80% đánh giá viên trong usability study (mục 2.3) trả lời "hiểu được vì sao sinh viên này bị gắn cờ".
- **REQ-04 — Advisor brief:** Với mỗi sinh viên top-k, LLM sinh brief **≤ 300 từ** gồm: mức rủi ro + 3 lý do chính + 2–3 gợi ý hành động; **groundedness ≥ 90%**; p95 latency ≤ 15 giây; chi phí ≤ 1 lần gọi API/brief.
- **REQ-05 — Fairness:** Chênh lệch FNR tối đa giữa các nhóm {gender, age_band, disability, imd_band} **≤ 0.05** trên tập kiểm định; nếu vi phạm → áp dụng biện pháp giảm thiểu (class weights, threshold theo nhóm) và ghi nhận trong model card.

## 3.2 Assumptions (ASM)

- **ASM-01:** OULAD đại diện đủ tốt cho dữ liệu LMS trong phạm vi demo/pilot (schema tương thích: nhân khẩu học, log VLE theo ngày, điểm assessment theo mốc); khi áp dụng thực tế chỉ cần viết adapter ETL, không đổi thiết kế hệ thống.
- **ASM-02:** Dữ liệu đầu vào đã/được **pseudonymize** trước khi vào hệ thống; hệ thống không lưu trữ PII thật của sinh viên.
- **ASM-03:** Cố vấn (hoặc đánh giá viên giả lập vai cố vấn trong giai đoạn đánh giá) có thiết bị truy cập web và dành đủ thời gian xem danh sách cảnh báo và phản hồi.
- **ASM-04:** Dịch vụ LLM (API hoặc model local nhỏ) khả dụng trong giờ làm việc với chi phí pilot ≤ 50 USD/tháng.

## 3.3 Specifications (SPEC)

- **SPEC-01 — Chu trình tuần:** Hàng tuần thực hiện: ingest → build snapshot (as-of cut date) → batch inference → rank top-k → sinh brief → cập nhật dashboard. Tổng thời gian pipeline ≤ 1 giờ cho toàn bộ cohort.
- **SPEC-02 — Xử lý dữ liệu thiếu:** Sinh viên có < 2 tuần dữ liệu VLE hoặc thiếu ≥ 60% feature quan trọng sẽ hiển thị trạng thái **"chưa đủ dữ liệu để đánh giá"** thay vì đưa ra dự đoán mạo hiểm.
- **SPEC-03 — Auditability:** Mỗi dự đoán lưu kèm model version, feature vector, SHAP top-k, mốc thời gian cắt — phục vụ tái lập và kiểm toán.
- **SPEC-04 — Ràng buộc LLM:** LLM chỉ nhận context từ dữ liệu đã tính toán (features, SHAP, thống kê lịch sử); prompt cấm sinh khẳng định ngoài context; các khuyến nghị chung chỉ được lấy từ thư viện template đã duyệt.
- **SPEC-05 — Giao diện có trách nhiệm:** Dashboard hiển thị disclaimer cố định: *"Đây là dự đoán hỗ trợ ra quyết định, không phải kết luận về năng lực hay hoàn cảnh của sinh viên"*, kèm hướng dẫn giao tiếp với sinh viên (không gắn nhãn, không quy kết).
- **SPEC-06 — Quản trị vòng đời & reproducibility:** Mọi mã nguồn, định nghĩa feature, model card, evaluation report được version control; pin seed + dependency + phiên bản dữ liệu để tái lập 100% kết quả; mỗi lần release model phải kèm fairness audit + temporal audit đạt 100% và bộ kiểm thử tự động (unit test feature engineering, temporal audit test, API smoke test) pass toàn bộ; design decision log ghi lại các quyết định kỹ thuật để bảo vệ/trình bày; cơ chế log feedback của cố vấn (đúng/sai, đã can thiệp) phục vụ tính precision@k trên holdout hiện tại và precision thực địa khi có pilot.
- **SPEC-07 — Data quality gate:** Trước mỗi lần inference, pipeline kiểm tra tự động: (a) schema khớp định nghĩa feature; (b) tỷ lệ thiếu/null trong ngưỡng cho phép; (c) khối lượng dữ liệu hợp lý so với lịch sử; (d) độ ổn định phân phối (PSI so với baseline < 0.2). Batch lỗi → bị chặn, không sinh cảnh báo từ dữ liệu lỗi (fail-safe) và báo lỗi cho nhóm.
- **SPEC-08 — Xử lý uncertainty & fallback:** (a) Xác suất dự đoán nằm trong vùng uncertain hoặc dữ liệu bất thường (OOD) → đưa vào hàng đợi human review với trạng thái "cần rà soát thủ công", không tự động gắn cờ; (b) LLM API lỗi/timeout → fallback sinh brief theo template tĩnh từ SHAP + feature values (không qua LLM), hệ thống vẫn hoạt động; (c) Mọi sự kiện fallback/escalation được log và tổng hợp trong báo cáo tuần.
- **SPEC-09 — Deployment & vận hành:** Toàn bộ hệ thống (data pipeline + API + dashboard) đóng gói bằng container, deploy được trên một máy duy nhất; log có cấu trúc cho mọi request/dự đoán/lỗi; monitoring theo dõi latency, tỷ lệ lỗi, PSI drift, số lượng cảnh báo/brief theo tuần; runbook xử lý các sự cố chính (LLM API chết, ingest dữ liệu lỗi, dashboard không truy cập được) kèm quy trình khôi phục.

# 4. Quality Attributes, Constraints & Trade-offs

## 4.1 Quality attributes (xếp theo thứ tự ưu tiên)

| Ưu tiên | Attribute | Lý do |
|---|---|---|
| 1 | **Interpretability & Trustworthiness** | Quyết định ảnh hưởng tới con người; cố vấn phải hiểu *vì sao* gắn cờ trước khi hành động. |
| 2 | **Fairness** | False negative bất đối xứng = bỏ sót chính những nhóm yếu thế cần hỗ trợ nhất — rủi ro đạo đức lớn nhất. |
| 3 | **Predictive accuracy** (dưới dạng Recall@k + calibration) | Đủ tốt để hữu ích; không tối đa hoá bằng mọi giá. |
| 4 | **Privacy & Security** | Dữ liệu nhạy cảm về giáo dục; pseudonymization + phân quyền truy cập. |
| 5 | **Latency / Cost** | Chạy batch theo tuần nên yêu cầu thấp; tối ưu chủ yếu ở chi phí LLM. |

## 4.2 Constraints

- **Dataset:** OULAD — công khai, cố định (32,593 sinh viên / 22 học phần / 4 kỳ 2013–2014); **không** bổ sung được nhãn mới.
- **Thời gian:** 10 tuần (theo yêu cầu học phần), 4 thành viên làm kiêm nhiệm.
- **Compute:** Colab/CPU + tối đa 1 GPU nhỏ; không train model lớn.
- **API budget:** ≤ ~50 USD/tháng cho LLM (pilot).
- **Kiến trúc bắt buộc:** non-generative ML cho dự đoán + LLM cho tổng hợp/gợi ý.
- **Ngôn ngữ:** dữ liệu gốc tiếng Anh → brief mặc định tiếng Anh, có thể song ngữ Anh–Việt nhờ LLM.

## 4.3 Trade-offs

- **T-01 — Accuracy ↔ Interpretability:** Chọn gradient boosting (XGBoost/LightGBM) + SHAP post-hoc thay vì logistic regression. Được recall cao hơn; mất tính minh bạch "bẩm sinh". Bù bằng SHAP + brief ngôn ngữ tự nhiên + model card. Nếu SHAP không ổn định giữa các lần chạy → chủ động hạ cấp về mô hình đơn giản hơn.
- **T-02 — LLM quality ↔ Inference cost:** Model lớn cho brief mượt nhưng đắt/chậm; model nhỏ rẻ nhưng dễ lệch khỏi context. Quyết định pilot: model tầm trung + 1 lần gọi/brief + nén context (chỉ feature chính + SHAP top-5) + đo groundedness; chỉ nâng cấp model nếu groundedness < 90%.
- **T-03 — Recall ↔ Advisor workload (precision):** Hạ threshold làm recall tăng nhưng gây alert fatigue → cố vấn tẩy chay hệ thống. Chọn **top-k cố định theo năng lực can thiệp** (k ≈ 20% cohort) thay vì threshold xác suất; k được hiệu chỉnh theo precision@k trên holdout và kết quả usability study.
- **T-04 — Temporal strictness ↔ Feature richness:** Ràng buộc as-of (REQ-02) làm feature "nghèo" hơn so với dùng dữ liệu cả kỳ — đây là ràng buộc không thoả hiệp. Giao dịch bằng cách thêm nhiều snapshot giữa kỳ (tuần 4/8/12) thay vì một snapshot giàu feature nhưng vi phạm thời gian.

# 5. Initial Risks & Controls

## 5.1 Failure modes

### FM-01 — Model/AI behavior failure: Temporal leakage tạo hiệu năng ảo

| Chuỗi rủi ro | Mô tả |
|---|---|
| **Cause** | Feature engineering dùng thông tin sau mốc cắt (VD: tổng lượt truy cập VLE cả kỳ, điểm exam cuối kỳ rò rỉ vào feature); tập train/test không tách theo thời gian. |
| **Failure** | AUC trên tập dev rất cao (> 0.9) nhưng sụp đổ khi chạy trên kỳ tương lai; cảnh báo sai lệch có hệ thống. |
| **Consequence** | Quyết định dựa trên số liệu ảo; bỏ sót sinh viên thật sự gặp nguy hiểm. |
| **Affected** | Cố vấn (quyết định sai), sinh viên (bị bỏ sót), nhóm (mất uy tín sản phẩm). |
| **Control** | As-of join bắt buộc; audit script tự động quét timestamp; split theo kỳ học (2013 → 2014); so sánh dev vs test, chênh > 0.1 AUC → điều tra leakage. |

### FM-02 — Data/environment/assumption failure: Missing data mang tính hệ thống

| Chuỗi rủi ro | Mô tả |
|---|---|
| **Cause** | Sinh viên ít tương tác VLE (do thiếu thiết bị, hoàn cảnh khó khăn) → ít dữ liệu → missingness tương quan với outcome. |
| **Failure** | Mô hình học "ít dữ liệu = an toàn" hoặc "ít dữ liệu = rủi ro" tuỳ cách impute; dự đoán bất ổn định với nhóm ít dữ liệu. |
| **Consequence** | Bỏ sót (FN) chính những sinh viên yếu thế cần hỗ trợ nhất; guardrail fairness bị phá vỡ. |
| **Affected** | Sinh viên yếu thế (disability, imd_band thấp, age_band lớn), cố vấn. |
| **Control** | Phân tích missingness trước khi train; missing-indicator + impute có kiểm soát; đo FNR/coverage theo từng nhóm; áp SPEC-02 hiển thị "chưa đủ dữ liệu" thay vì đoán. |

### FM-03 — Workflow/interface/human failure: Alert fatigue & stigma

| Chuỗi rủi ro | Mô tả |
|---|---|
| **Cause** | Top-k quá lớn so với năng lực cố vấn; brief dài/khó hiểu; cố vấn hiểu risk score là "kết luận"; giao tiếp thiếu khéo léo với sinh viên. |
| **Failure** | Cố vấn bỏ qua cảnh báo (ngừng dùng hệ thống) hoặc giao tiếp sai cách gây kỳ thị với sinh viên. |
| **Consequence** | Hệ thống bị bỏ không; sinh viên tổn thương (tự ti, bị gắn nhãn); hiệu quả can thiệp ≈ 0. |
| **Affected** | Sinh viên (trực tiếp — tổn thương tâm lý), cố vấn (mất thời gian), nhóm (adoption = 0). |
| **Control** | Giới hạn top-k theo năng lực xử lý; brief ≤ 300 từ, viết theo hành động; disclaimer + hướng dẫn giao tiếp bắt buộc (SPEC-05); đánh giá theo vòng: nếu precision@top-k trên holdout < 50% hoặc SUS < 70 → tạm dừng để điều chỉnh. |

## 5.2 Top risk — FM-01 Temporal leakage (phân tích đầy đủ)

Được chọn vì rủi ro này **âm thầm làm sai lệch mọi metric và mọi quyết định downstream**, khó phát hiện nếu không chủ động kiểm tra — toàn bộ giá trị của hệ thống (và của báo cáo đánh giá) phụ thuộc vào việc số liệu là thật.

| Giai đoạn | Kiểm soát |
|---|---|
| **Prevention** | Quy ước thiết kế feature **"as-of only"** ngay từ đầu; tài liệu hoá nguồn timestamp của từng feature; code review bắt buộc cho mọi thay đổi feature engineering; tách vai trò người viết feature và người viết audit; temporal split theo kỳ học (train 2013 → test 2014) là mặc định, không được thay bằng random split. |
| **Detection** | (a) Audit script tự động: 100% feature phải chứng minh được dữ liệu nguồn có timestamp ≤ cut date (REQ-02); (b) Sanity check: nếu AUC dev > 0.9 trong khi các nghiên cứu công bố trên OULAD thường ở mức ~0.75–0.85 → cờ đỏ; (c) Theo dõi PSI (population stability index) giữa train và test. |
| **Response** | Đóng băng release ngay lập tức; chạy audit để định danh feature vi phạm; loại bỏ/patch feature; báo cáo kết quả và nguyên nhân cho nhóm trong vòng 24 giờ. |
| **Recovery** | Rebuild feature pipeline sạch → re-train → re-evaluate (temporal + fairness) → cập nhật model card → chỉ mở lại dashboard khi temporal audit đạt 100%. |
| **Owner** | ML lead của nhóm (thành viên phụ trách mô hình) chịu trách nhiệm chính; toàn nhóm review và ký duyệt trước mỗi lần release model. |

---

## Tài liệu tham chiếu

1. Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics Dataset*. Scientific Data, 4, 170171.
2. OULAD chính thức: https://analyse.kmi2.open.ac.uk/open_dataset
