# TÀI LIỆU THIẾT KẾ HỆ THỐNG — Student Success Early-Warning Assistant

| | |
|---|---|
| **Đề bài** | CHỦ ĐỀ 4 — Student Success Early-Warning Assistant (`docs/CHỦ ĐỀ 4.docx`) |
| **Trạng thái** | Draft v1.0 — phục vụ thống nhất trong nhóm |
| **Phạm vi thiết kế** | Prototype end-to-end: dữ liệu OULAD → ML dự báo → LLM tạo Advisor Brief → Dashboard cho cố vấn |
| **Tài liệu liên quan** | `docs/CHỦ ĐỀ 4.docx` (spec), tóm tắt MLIP ch. 7, 9, 16, 24 (cơ sở pattern) |

---

## 1. Phạm vi và nguyên tắc thiết kế

### 1.1. Tóm tắt bài toán

Hệ thống hỗ trợ giảng viên/cố vấn học tập **phát hiện sớm** (cuối tuần 4) các sinh viên có nguy cơ không hoàn thành thành công học phần, dựa trên:

1. **Non-generative ML** — dự báo nguy cơ (risk score) từ dữ liệu học tập ≤ ngày 27 (OULAD).
2. **LLM** — tổng hợp tín hiệu thành **Advisor Brief** có căn cứ dữ liệu, kèm gợi ý hỗ trợ.
3. **Human oversight** — 100% quyết định hỗ trợ thuộc về cố vấn; hệ thống không tự hành động với sinh viên.

### 1.2. Nguyên tắc thiết kế (rút ra từ spec)

| # | Nguyên tắc | Căn cứ trong spec |
|---|---|---|
| P1 | **Temporal correctness là mối quan tâm kiến trúc hạng nhất** — tồn tại một điểm kiểm soát duy nhất cho "được phép biết gì tại ngày 27" | FR-02, Constraint "không dùng dữ liệu tương lai", Risk *Temporal Leakage* |
| P2 | **AI đề xuất, con người quyết định** — không có đường đi nào từ dự đoán đến hành động với sinh viên mà không qua cố vấn | Scope out, FR-07, Human oversight 100% |
| P3 | **Mọi đầu ra đều truy vết được** — mỗi nhận định của LLM ánh xạ về dữ liệu nguồn; mỗi dự đoán gắn model version | FR-03, FR-06, FR-09 |
| P4 | **Suy giảm rõ ràng, không suy diễn thay** — khi ML/LLM/dữ liệu lỗi, hiển thị trạng thái lỗi thay vì đoán | FR-08, Risk *ML/LLM/API Failure* |
| P5 | **Giảm thiểu dữ liệu** — chỉ gửi thông tin cần thiết cho LLM, kiểm soát quyền truy cập | Privacy & fairness, Risk *Privacy/Security* |

---

## 2. Kiến trúc tổng quan

### 2.1. Bức cảnh hệ thống (System Context)

```mermaid
flowchart LR
    ADV["Cố vấn học tập / Giảng viên<br/>(primary user)"]
    ADM["Quản trị hệ thống<br/>(vai trò phụ)"]
    SYS["Early-Warning Assistant"]
    OULAD[("OULAD<br/>7 bảng CSV, dữ liệu tĩnh")]
    LLM["LLM API<br/>(bên ngoài)"]

    ADV -->|"xem risk, review brief, ra quyết định"| SYS
    SYS -->|"danh sách ưu tiên, Advisor Brief"| ADV
    ADM -->|"quản trị tài khoản, xem giám sát"| SYS
    OULAD -->|"nạp theo batch"| SYS
    SYS <-->|"gọi API, đo cost / latency"| LLM

    style SYS fill:#e8f0fe,stroke:#1a73e8
```

**Ranh giới**: không kết nối LMS thật; không có kênh gửi thông báo tới sinh viên; mọi đầu ra dừng ở màn hình cố vấn (out of scope của đề bài).

### 2.2. Kiến trúc phân tầng

```mermaid
flowchart TB
    subgraph L5["TẦNG 1 · TRUY CẬP"]
        UI["M9 · Web Dashboard<br/>danh sách ưu tiên · hồ sơ SV · review workflow<br/>hiển thị uncertainty · trạng thái lỗi rõ ràng"]
    end

    subgraph L4["TẦNG 2 · DỊCH VỤ ỨNG DỤNG"]
        API["M6 · Backend API<br/>điều phối · RBAC · ghi nhận quyết định"]
        BRIEF["M8 · LLM module trong M6<br/>Advisor Brief grounded · citation bắt buộc<br/>đo cost/latency · fallback"]
    end

    subgraph L3["TẦNG 3 · AI / ML"]
        SCORE["M5 · Scoring Job (batch tuần 4)<br/>score + nhãn + top tín hiệu + uncertainty"]
        TRAIN["M4 · Training & Evaluation<br/>baseline + model chính · threshold tuning"]
        LEAK["M3 · Leakage Test Suite"]
        REG[("Model Registry<br/>version · config · feature list")]
    end

    subgraph L2["TẦNG 4 · DỮ LIỆU"]
        FEAT["M2 · Feature Pipeline<br/>Cutoff Enforcement ngày 0–27"]
        ING["M1 · Ingestion & Data Validation"]
        DB[("M7 · PostgreSQL<br/>predictions · briefs · decisions<br/>feedback · logs")]
        STORE[("File Store<br/>OULAD CSV · model artifacts")]
    end

    subgraph XC["LỚP NGANG · đè lên mọi tầng"]
        AUTH["Access Control"]
        AUDIT["Audit Log & Truy vết"]
        MON["M10 · Monitoring"]
        FALL["Failure States & Fallback"]
    end

    ADV["Cố vấn"] --> UI
    UI --> API
    API --> BRIEF
    API <--> DB
    BRIEF --> DB
    SCORE --> DB
    TRAIN --> REG
    REG --> SCORE
    FEAT --> TRAIN
    FEAT --> SCORE
    ING --> FEAT
    STORE --> ING
    API --> AUDIT
    BRIEF --> AUDIT
    SCORE --> AUDIT
    AUDIT --> MON
    DB --> MON
```

### 2.3. Các pattern áp dụng

| # | Pattern | Cơ sở / dẫn chiếu | Vấn đề trong bài toán | Điểm áp dụng |
|---|---|---|---|---|
| PT-1 | **Human-in-the-Loop Decision** | MLIP ch. 7.3a — thiết kế tương tác Human-AI | Human oversight 100%; không tự quyết can thiệp | Toàn bộ luồng C: mọi brief/đề xuất phải qua cố vấn (FR-07) |
| PT-2 | **Two-Stage Prediction** | MLIP ch. 9 — "model rẻ sàng lọc toàn bộ, phần đáng ngờ mới xử lý kỹ" | Alert overload: cố vấn không thể xem hết mọi hồ sơ | ML batch sàng lọc cohort → cố vấn chỉ review phần được gắn cờ, ưu tiên theo score |
| PT-3 | **Guardrails** | MLIP ch. 7.3c — hàng rào bảo vệ | LLM hallucination / nhận định không có căn cứ | M8: output schema bắt buộc + validation citation, fail → regenerate → fallback |
| PT-4 | **Design for Failure (explicit failure states + fallback)** | MLIP ch. 7.3 | FR-08; Risk *ML/LLM/API Failure* | Lớp ngang Failure States; M8 fallback; M9 hiển thị trạng thái lỗi |
| PT-5 | **Point-in-Time Feature Pipeline (cutoff enforcement)** | Thực hành MLOps — point-in-time join; ASM-02 | Temporal leakage — kết quả offline giả tạo | M2: feature catalog khai báo as-of; M3: leakage test như CI gate |
| PT-6 | **Data Schema Validation + Data Linting** | MLIP ch. 16.4 | Dữ liệu OULAD nhiễu, thiếu, lệch schema | M1: schema check, integrity check, data linting, quarantine |
| PT-7 | **Model Registry + Versioning (không ghi đè)** | MLIP ch. 24 — chống-pattern ghi đè model production | FR-03 (model version); truy vết dự đoán về model/config | Registry trong M4; mọi prediction gắn `model_version` |
| PT-8 | **Prediction Logging** | MLIP ch. 24.6c — "chìa khóa truy vết" | FR-09; debug dự đoán sai; đánh giá Product/User value | M5 + M7: ghi input snapshot hash, model version, tín hiệu, thời điểm |
| PT-9 | **Drift Monitoring + Slice Evaluation** | MLIP ch. 16.5 | Risk *Domain Shift / Bias* — chất lượng không đồng đều giữa học phần/kỳ | M10: slice metrics theo module/presentation; cảnh báo lệch phân bố |
| PT-10 | **Containment & Isolation (data minimization)** | MLIP ch. 7.3f; Privacy & fairness | Thông tin cá nhân lộ ra ngoài thẩm quyền | M8 chỉ nhận context tối thiểu ≤ cutoff; module có interface cô lập, mock được; log bảo vệ |

### 2.4. Lý do lựa chọn kiến trúc và phương án bị loại

**Kiến trúc chọn: Modular Monolith (API + DB + UI, gồm LLM module) + Batch ML Pipelines.**

| Quyết định | Phương án loại bỏ | Lý do |
|---|---|---|
| **PA-1: Scoring batch mô phỏng cuối tuần 4** | Online serving qua API | OULAD là dữ liệu lịch sử tĩnh — không có dòng sự kiện thật chảy vào. Batch cho kết quả tất định, tái lập được (MLIP ch. 24), đủ cho demo + đánh giá. Online chỉ cần thiết khi có LMS thật (out of scope). |
| **PA-2: LLM module có ranh giới nội bộ trong Backend API** | Triển khai LLM thành service/container riêng | Prototype chỉ có một backend và một máy demo; gọi module nội bộ giảm container và hop mạng. Module vẫn tách interface để mock khi kiểm thử, xử lý timeout/fallback, giảm thiểu context và đo cost/latency độc lập. Chỉ cân nhắc tách service khi có nhu cầu triển khai hoặc mở rộng độc lập. |
| **PA-3: Modular monolith** | Microservices đầy đủ | Đội quy mô nhóm sinh viên, 1 deployment target (demo). Microservices tăng chi phí vận hành không mang lại giá trị ở quy mô này; ranh giới module nội bộ (M1–M10) vẫn giữ được khả năng tách sau. |
| **PA-4: PostgreSQL** | SQLite | FR-09 cần RBAC + audit log thực dụng; Postgres cho concurrency giữa API — Scoring Job, và JSONB lưu tín hiệu/citation. |
| **PA-5: React dashboard** | Streamlit | FR-07 cần workflow trạng thái (edit/accept/reject) + FR-09 cần phân quyền theo vai trò — Streamlit yếu ở cả hai. React + FastAPI chi phí chấp nhận được. |
| **PA-6: Grounding trực tiếp (không RAG) giai đoạn đầu** | RAG với guidelines | Đề bài cho phép bỏ RAG nếu grounding khác đạt yêu cầu. Guidelines là bộ tài liệu nhỏ, tĩnh → nhét đủ vào context. Chỉ nâng cấp RAG nếu rubric LLM cho thấy thất bại (mục 3.8). |

### 2.5. Quan điểm triển khai (Deployment)

```mermaid
flowchart LR
    subgraph HOST["Máy demo — docker-compose"]
        FE["container: web-frontend<br/>React build"]
        BE["container: api-backend<br/>FastAPI"]
        PG[("container: postgres")]
        MLF["container: mlflow<br/>model registry + tracking"]
        JOB["container: batch-jobs<br/>train / score (chạy theo lệnh)"]
    end
    EXT["LLM API bên ngoài"]

    USER["Trình duyệt cố vấn"] --> FE --> BE
    BE --> PG
    BE --> EXT
    BE --> MLF
    JOB --> PG
    JOB --> MLF
```

Toàn bộ hệ thống chạy trên một máy demo qua `docker-compose`; batch jobs kích bằng lệnh (mô phỏng mốc "cuối tuần 4"). Chi tiết từng container xem mục 3.

---

## 3. Thiết kế chi tiết từng thành phần

### 3.1. M1 — Ingestion & Data Validation

**Trách nhiệm**: nạp 7 bảng OULAD vào file store chuẩn hoá (Parquet), kiểm tra chất lượng trước khi bất kỳ thành phần nào khác được phép dùng dữ liệu.

```mermaid
flowchart TB
    SRC[("OULAD CSV<br/>studentInfo · courses · studentRegistration<br/>assessments · studentAssessment · vle · studentVle")]
    LOAD["Bước 1 · Load thô + kiểm schema<br/>(cột, kiểu dữ liệu, giá trị hợp lệ)"]
    INTEG["Bước 2 · Referential integrity<br/>studentVle → vle → courses<br/>studentAssessment → assessments<br/>studentInfo → courses"]
    LINT["Bước 3 · Data linting<br/>(data smells: giá trị đáng ngờ,<br/>mã hóa bất thường, outlier khối lượng click)"]
    QUAR["Khu cách ly<br/>(bản ghi lỗi + lý do)"]
    REPORT["Báo cáo chất lượng dữ liệu<br/>(tỷ lệ thiếu, phân bố, số dòng bị loại)"]
    CLEAN[("Parquet đã làm sạch<br/>+ metadata phiên bản nạp")]

    SRC --> LOAD --> INTEG --> LINT
    LINT -->|"bất thường"| QUAR
    LINT -->|"đạt"| CLEAN
    LINT --> REPORT
    QUAR --> REPORT
```

**Quy tắc chính**:
- Schema của từng bảng khai báo tĩnh; sai schema → chặn toàn bộ lần nạp (fail fast).
- Bản ghi vi phạm integrity → **cách ly + ghi lý do**, không âm thầm bỏ.
- Báo cáo chất lượng được M10 lưu lại làm baseline so sánh drift các lần nạp sau.

### 3.2. M2 — Feature Pipeline & Cutoff Enforcement

**Trách nhiệm**: biến dữ liệu đã làm sạch thành feature snapshot tại mốc dự báo (ngày 0–27), với **bằng chứng thời gian** cho từng feature. Đây là **điểm kiểm soát leakage duy nhất** của hệ thống (P1).

**Cơ chế cốt lõi — Feature Catalog**: mỗi feature khai báo (bảng nguồn, cách tổng hợp, điều kiện thời gian hợp lệ). Pipeline **tự động áp cutoff** cho mọi feature dựa trên khai báo; feature không chứng minh được as-of time → **bị loại khỏi benchmark** (ASM-02).

```mermaid
flowchart TB
    CLEAN[("Parquet sạch (từ M1)")]
    CAT["Feature Catalog<br/>mỗi feature khai báo:<br/>bảng nguồn · công thức · as-of rule"]
    COHORT["Xác định cohort dự báo<br/>sinh viên còn đăng ký tại ngày 27<br/>(loại Withdrawn trước/ngày 28)"]
    CUT["CUTOFF ENFORCEMENT<br/>chỉ nhận sự kiện có date ≤ 27<br/>mọi bảng đều lọc theo cùng mốc"]
    AGG["Tổng hợp theo student–module–presentation<br/>tần suất · xu hướng · nộp bài · đặc trưng học phần"]
    LED[("As-of Ledger<br/>feature → bằng chứng thời gian")]
    SNAP[("Feature Snapshot (Parquet)<br/>1 dòng = 1 SV–học phần–kỳ")]

    CLEAN --> CAT
    CAT --> CUT
    COHORT --> CUT
    CUT --> AGG --> SNAP
    CAT --> LED
    CUT --> LED
```

**Ví dụ khai báo trong catalog**:

| Feature | Bảng nguồn | As-of rule | Tổng hợp |
|---|---|---|---|
| `n_clicks_before_cutoff` | studentVle | `date <= 27` (gồm date âm — xem trước trước khai giảng) | tổng số click |
| `click_trend_w3_w4` | studentVle | `date BETWEEN 14 AND 27` | so sánh tuần 3–4 vs tuần 1–2 |
| `n_submitted_before_cutoff` | studentAssessment ⋈ assessments | `date_submitted ≤ 27` | số bài nộp |
| `avg_score_submitted` | studentAssessment ⋈ assessments | `date_submitted ≤ 27` | điểm TB các bài đã nộp |
| `module_presentation` | courses | tĩnh | mã học phần/kỳ |
| ~~`final_result`~~ | studentInfo | — | **bị loại — là target** |

**Output schema của snapshot**: `id_student, code_module, code_presentation, [features...], feature_list_hash, snapshot_version`.

### 3.3. M3 — Leakage Test Suite

**Trách nhiệm**: chứng minh cơ chế cutoff đúng — chạy như **CI gate**, không pipeline nào đạt trạng thái "sẵn sàng" nếu fail.

```mermaid
flowchart TB
    RUN["Chạy bộ test trên pipeline M2"]
    T1["T1 · Inject sự kiện tương lai<br/>(date = 60) vào studentVle giả lập<br/>→ snapshot KHÔNG được đổi"]
    T2["T2 · Cố ý thêm final_result<br/>vào danh sách feature<br/>→ pipeline phải TỪ CHỐI build"]
    T3["T3 · Thêm assessment có date sau ngày 27<br/>→ không feature nào phát sinh từ nó"]
    T4["T4 · Audit catalog<br/>100% feature có as-of rule<br/>được kiểm chứng"]
    GATE{"Tất cả PASS?"}
    OK["Pipeline đạt trạng thái sẵn sàng<br/>cho M4 / M5"]
    BLOCK["CHẶN — báo cáo vi phạm<br/>+ bổ sung kiểm soát"]

    RUN --> T1 & T2 & T3 & T4
    T1 & T2 & T3 & T4 --> GATE
    GATE -->|"yes"| OK
    GATE -->|"no"| BLOCK
```

Đối ứng tiêu chí "Temporal correctness: 100% trường hợp cố ý đưa dữ liệu tương lai hoặc target vào features phải bị phát hiện/chặn".

### 3.4. M4 — Training & Evaluation

**Trách nhiệm**: huấn luyện baseline + model chính theo giao thức chung, chọn threshold trên validation, đăng ký model vào Registry.

```mermaid
flowchart TB
    SNAP[("Feature Snapshot (từ M2)")]
    LABEL["Gắn nhãn từ final_result<br/>Fail/Withdrawn sau cutoff = 1 (at-risk)<br/>Pass/Distinction = 0"]
    SPLIT["Split theo presentation (theo thời gian)<br/>kỳ cũ → train · kỳ giữa → validation · kỳ mới → test"]
    BASE["Baseline: Logistic Regression / Decision Tree"]
    MAIN["Model chính: Gradient Boosting<br/>(XGBoost / LightGBM)"]
    TH["Threshold tuning trên validation<br/>theo ngân sách review của cố vấn<br/>(đánh giá số SV bị gắn cờ)"]
    EVAL["Đánh giá trên TEST set<br/>Precision · Recall · F1 · PR-AUC<br/>confusion matrix lớp at-risk<br/>Precision@K / Recall@K"]
    SLICE["Slice evaluation<br/>theo học phần / kỳ học"]
    FN["Phân tích false negatives"]
    REG[("Model Registry<br/>model artifact · version · config<br/>feature_list_hash · metrics")]

    SNAP --> LABEL --> SPLIT
    SPLIT --> BASE & MAIN
    BASE & MAIN --> TH --> EVAL --> SLICE --> FN
    EVAL --> REG
    TH --> REG
```

**Quy ước bắt buộc**:
- Split **theo thời gian** (presentation) chứ không trộn ngẫu nhiên — đối ứng yêu cầu "validation theo thời gian" trong bảng rủi ro.
- Cùng một split, cùng test set cho baseline và model chính — đối ứng "sử dụng chung train/validation/test protocol".
- Chỉ model **đã qua M3** mới được đăng ký.
- Threshold là **một phần của artifact** đăng ký kèm model (đổi threshold = đổi version).

### 3.5. M5 — Scoring Job (batch cuối tuần 4)

```mermaid
sequenceDiagram
    autonumber
    participant OP as Người vận hành / Scheduler
    participant SJ as Scoring Job
    participant REG as Model Registry
    participant FS as Feature Snapshot (M2)
    participant DB as PostgreSQL (M7)

    OP->>SJ: Kích chạy batch "cuối tuần 4"
    SJ->>REG: Lấy model version hiện hành (không ghi đè)
    REG-->>SJ: model artifact + config + feature_list_hash
    SJ->>FS: Đọc snapshot khớp feature_list_hash
    FS-->>SJ: cohort (SV còn đăng ký ngày 27)
    SJ->>SJ: Predict → score · nhãn · uncertainty<br/>· top tín hiệu (SHAP/hệ số)
    SJ->>DB: Ghi predictions<br/>(score, label, signals, uncertainty, model_version, snapshot_version)
    SJ->>DB: Ghi prediction log (PT-8: input hash, thời điểm, version)
    SJ-->>OP: Báo cáo: số SV gắn cờ, phân bố score, cảnh báo dữ liệu thiếu
```

**Ghi chú thiết kế**:
- Job **không bao giờ sửa** prediction đã ghi — chạy lại = version mới (tất định, tái lập).
- Sinh viên thiếu dữ liệu → status `insufficient_data`, không score (FR-08).
- Top tín hiệu: với LR dùng hệ số × giá trị chuẩn hoá; với GBM dùng SHAP values. Là **tín hiệu tương quan, không phải nhân quả** — ghi chú này được gắn kèm trong dữ liệu hiển thị (FR-05).

### 3.6. M6 — Backend API

**Trách nhiệm**: cổng duy nhất cho Dashboard; thực thi RBAC; ghi nhận quyết định cố vấn; điều phối M8.

```mermaid
flowchart LR
    REQ["Request"] --> AUTHMW["Auth middleware<br/>(JWT session, vai trò)"]
    AUTHMW --> AUDITMW["Audit middleware<br/>(ghi actor · action · payload)"]
    AUDITMW --> VALID["Request validation<br/>(Pydantic)"]
    VALID --> H["Handlers<br/>students · predictions · briefs<br/>decisions · admin"]
    H --> DB[("PostgreSQL")]
    H --> LLM["M8 · LLM module<br/>(in-process)"]
```

**Danh sách endpoint chính**:

| Method | Path | Vai trò | Chức năng | FR |
|---|---|---|---|---|
| POST | `/auth/login` | — | Đăng nhập, cấp session | FR-09 |
| GET | `/students?risk=&sort=priority` | advisor | Danh sách SV kèm risk/priority | FR-04 |
| GET | `/students/{id}` | advisor | Hồ sơ: tín hiệu, uncertainty, trạng thái dữ liệu | FR-04, FR-05 |
| POST | `/briefs/generate` | advisor | Yêu cầu tạo brief cho 1 hồ sơ | FR-06 |
| GET | `/briefs/{id}` | advisor | Đọc brief + citation | FR-06 |
| PUT | `/briefs/{id}` | advisor | Sửa nội dung brief (tạo version mới) | FR-07 |
| POST | `/briefs/{id}/decision` | advisor | accept / reject + phản hồi | FR-07 |
| GET | `/admin/monitoring/*` | admin | Metrics dữ liệu/model/LLM/hệ thống | FR-09 |

**Nguyên tắc**: Lời gọi tới LLM chỉ đi từ M8 bên trong M6; Dashboard không gọi LLM API trực tiếp. M8 dùng client có timeout, retry giới hạn và fallback. Không có endpoint nào thực hiện hành động với sinh viên — kiến trúc physically enforce P2.

### 3.7. M7 — Database

```mermaid
erDiagram
    USERS ||--o{ DECISIONS : "ra quyết định"
    USERS ||--o{ AUDIT_LOGS : "thực hiện"
    MODEL_VERSIONS ||--o{ PREDICTIONS : "sinh ra"
    PREDICTIONS ||--o{ BRIEFS : "có"
    BRIEFS ||--o{ DECISIONS : "được quyết định"
    BRIEFS ||--o{ LLM_CALLS : "sinh ra bởi"

    USERS {
        int user_id PK
        string full_name
        string role "advisor | admin"
        string password_hash
    }
    MODEL_VERSIONS {
        string model_version PK
        string algorithm
        jsonb config
        string feature_list_hash
        jsonb metrics_test
        float threshold
        datetime trained_at
    }
    PREDICTIONS {
        int prediction_id PK
        int id_student
        string code_module
        string code_presentation
        float risk_score
        int predicted_label
        float uncertainty
        jsonb top_signals
        string status "scored | insufficient_data | no_prediction"
        string model_version FK
        string snapshot_version
        datetime scored_at
    }
    BRIEFS {
        int brief_id PK
        int prediction_id FK
        int version
        jsonb content "sections + text"
        jsonb citations "nhận định → dữ liệu nguồn"
        string status
        string llm_model
        int latency_ms
        float cost_usd
    }
    DECISIONS {
        int decision_id PK
        int brief_id FK
        int user_id FK
        string action "accept | reject | edit"
        string feedback
        datetime decided_at
    }
    LLM_CALLS {
        int call_id PK
        int brief_id FK
        string model
        int prompt_tokens
        int completion_tokens
        float cost_usd
        int latency_ms
        string status "ok | validation_failed | api_error"
    }
    AUDIT_LOGS {
        int log_id PK
        int user_id FK
        string action
        string entity
        jsonb payload
        datetime at
    }
```

**Ghi chú**:
- `BRIEFS` giữ **mọi version** (không ghi đè — cùng chống-pattern như model, MLIP ch. 24): sửa brief → version mới, version cũ còn nguyên phục vụ truy vết.
- `PREDICTIONS.status` hiện thực hoá P4: `insufficient_data` và `no_prediction` là trạng thái hợp lệ, không phải lỗi âm thầm.
- `DECISIONS` là sự kiện first-class (P2): nguồn dữ liệu cho Human-oversight 100% và bài so sánh Product/User value.

### 3.8. M8 — LLM module trong Backend API

**Trách nhiệm**: tạo Advisor Brief **grounded** — mỗi nhận định thực tế phải dẫn được về dữ liệu nguồn; tuân thủ toàn bộ ràng buộc mục 2.2 của đề bài.

```mermaid
sequenceDiagram
    autonumber
    participant API as Backend API (M6)
    participant LS as LLM module (M8, cùng process với M6)
    participant DB as PostgreSQL (M7)
    participant EXT as LLM API bên ngoài

    API->>DB: Đọc dữ liệu hồ sơ (≤ cutoff) + risk + top tín hiệu
    API->>LS: Gọi module tạo brief (context đã giảm thiểu dữ liệu)
    LS->>DB: Đọc advisor guidelines (bộ hướng dẫn chung)
    LS->>LS: Build prompt:<br/>dữ liệu + guidelines + schema output bắt buộc
    LS->>EXT: Gọi LLM (ghi đếm token/latency)
    EXT-->>LS: Bản nháp brief (JSON)
    LS->>LS: Validation: đúng schema?<br/>mọi nhận định thực tế có citation khớp dữ liệu nguồn?<br/>không ngôn ngữ nhân quả / suy diễn cá nhân?
    alt Validation FAIL (lần 1)
        LS->>EXT: Gọi lại với phản hồi lỗi cụ thể
    else Validation FAIL (lần 2)
        LS-->>API: brief = UNAVAILABLE + lý do (fallback, không tự bịa)
    else Validation PASS
        LS->>DB: Lưu brief + citations + cost + latency
        LS-->>API: Brief hợp lệ
    end
```

**Ràng buộc đề bài → cơ chế hiện thực**:

| Ràng buộc (spec 2.2) | Cơ chế trong M8 |
|---|---|
| Không suy diễn cá nhân ngoài dữ liệu | Prompt cấm + validator kiểm tra nhận định có citation |
| Không biến feature importance thành nhân quả | Prompt dùng khung "tín hiệu liên quan" + validator từ chối ngôn ngữ nhân quả |
| Không nhận định động cơ/năng lực/tâm lý | Prompt cấm + checklist rubric khi đánh giá |
| Không tự quyết can thiệp / gửi SV | Brief chỉ chứa "gợi ý để cân nhắc" — schema không có trường hành động |
| Thiếu dữ liệu → nêu giới hạn | Context builder gắn cờ `insufficient_data` vào prompt → brief có mục "Giới hạn dữ liệu" |

**Fallback khi LLM không khả dụng** (FR-08): M8 trong Backend API đánh dấu brief "UNAVAILABLE" kèm lý do; Dashboard hiện trạng thái này để cố vấn viết brief thủ công từ tín hiệu đã hiển thị — **không bao giờ** hiển thị brief cũ như thể nó mới, không tự sinh thay thế.

**Về RAG**: mặc định grounding trực tiếp (guidelines nhỏ, tĩnh). Kích hoạt RAG chỉ khi rubric đánh giá cho thấy LLM bỏ sót nguyên tắc trong guidelines.

### 3.9. M9 — Dashboard

**Trách nhiệm**: toàn bộ giao diện cố vấn. Thiết kế theo nguyên tắc: dự đoán là **đầu vào để xem xét**, không phải kết luận; uncertainty luôn hiển thị; mọi trạng thái lỗi thấy được.

**Các màn hình**:

| Màn hình | Nội dung chính | FR |
|---|---|---|
| S1 · Đăng nhập | Xác thực theo vai trò | FR-09 |
| S2 · Danh sách ưu tiên | SV gắn cờ, sắp theo priority (score × độ không chắc chắn), cờ trạng thái dữ liệu, bộ lọc theo học phần | FR-04 |
| S3 · Hồ sơ sinh viên | Tín hiệu học tập (kèm nhãn "tương quan, không nhân quả"), khoảng cách tới threshold, dữ liệu còn thiếu | FR-05 |
| S4 · Review brief | Brief + citation bấm được → đối chiếu dữ liệu nguồn; sửa/chấp nhận/từ chối + phản hồi | FR-06, FR-07 |
| S5 · Trạng thái hệ thống | Cờ khi ML/LLM/dữ liệu lỗi, hướng dẫn fallback | FR-08 |

**Vòng đời brief (state machine)**:

```mermaid
stateDiagram-v2
    state "Chưa tạo brief" as NG
    state "Đang tạo (LLM)" as GEN
    state "Đang kiểm tra citation" as VAL
    state "Đã tạo — chờ xem xét" as GEN_OK
    state "Đã sửa bởi cố vấn" as EDITED
    state "Không khả dụng (LLM lỗi)" as FAIL
    state "Bản thảo thủ công" as MANUAL
    state "ĐÃ CHẤP NHẬN" as ACC
    state "ĐÃ TỪ CHỐI" as REJ

    [*] --> NG
    NG --> GEN: cố vấn yêu cầu
    GEN --> VAL: LLM trả bản nháp
    VAL --> GEN_OK: pass validation
    VAL --> GEN: fail — sinh lại (lần 1)
    VAL --> FAIL: fail — sinh lại (lần 2)
    GEN_OK --> EDITED: cố vấn chỉnh sửa
    GEN_OK --> ACC: chấp nhận
    GEN_OK --> REJ: từ chối + phản hồi
    EDITED --> ACC: chấp nhận
    EDITED --> REJ: từ chối + phản hồi
    FAIL --> MANUAL: cố vấn viết từ tín hiệu
    MANUAL --> ACC
    MANUAL --> REJ
    ACC --> [*]
    REJ --> [*]
```

Mọi chuyển trạng thái ghi vào `DECISIONS` + `AUDIT_LOGS` (P3). Không có trạng thái nào tự chuyển sang `ACC` mà không qua thao tác của cố vấn.

### 3.9-bis. Hiển thị uncertainty (FR-05) — quy ước chung

| Dữ liệu | Cách hiển thị | Điều cấm |
|---|---|---|
| Risk score | Xác suất + nhãn "điểm gợi ý ưu tiên" | Gọi là "kết luận" |
| Vị trí threshold | Thanh trực quan score vs threshold | Ẩn threshold |
| Độ không chắc chắn | Khoảng cách tới threshold + trạng thái calibration | Chỉ hiện nhãn 0/1 |
| Tín hiệu | Top-N đóng góp, ghi chú tương quan | Diễn giải nhân quả |
| Thiếu dữ liệu | Nhãn lý do rõ ràng | Im lặng, score mặc định |

### 3.10. M10 — Monitoring & Audit

```mermaid
flowchart LR
    M1L["Log M1<br/>chất lượng nạp"] --> AGG["Tổng hợp metrics"]
    M4L["Log M4/M5<br/>model + slice metrics"] --> AGG
    M8L["Log M8<br/>LLM cost · latency · tỷ lệ validation fail"] --> AGG
    DBL[("DB · truy vết<br/>prediction log · decisions")] --> AGG
    AGG --> DQ["Dashboard chất lượng dữ liệu<br/>(drift giữa các lần nạp)"]
    AGG --> DM["Dashboard model<br/>(slice theo học phần/kỳ)"]
    AGG --> DL["Dashboard LLM<br/>(cost, p50/p95, tỷ lệ fallback)"]
    AGG --> AL["Cảnh báo<br/>(lệch phân bố, API chết, chi phí vượt ngân sách)"]
```

| Chiều đo (bảng chất lượng đề bài) | Metric | Nguồn |
|---|---|---|
| ML quality | P/R/F1/PR-AUC, confusion matrix | M4, ghi vào Registry |
| Early detection | Recall tại tuần 4 + phân tích FN | M4 |
| Alert workload | Precision@K, tỷ lệ SV gắn cờ | M4, M7 |
| LLM quality | Rubric có người kiểm tra (đánh giá ngoài hệ thống, M8 cấp dữ liệu) | M8 |
| Temporal correctness | Tỷ lệ pass M3 = 100% | M3 |
| System performance | p50/p95 latency brief, thời gian batch, chi phí LLM | M7 (`LLM_CALLS`) |
| Privacy | Số truy cập ngoài thẩm quyền = 0; dữ liệu gửi LLM ≤ cutoff | M6 audit |
| Reliability | Tỷ lệ fallback, trạng thái lỗi | M10 |

---

## 4. Ma trận truy vết Yêu cầu → Thiết kế

| FR | Component | Cơ chế |
|---|---|---|
| FR-01 | M1, M2 | Ingestion + feature pipeline tạo features tại mốc xác định |
| FR-02 | M2, M3 | Cutoff enforcement + leakage test CI gate |
| FR-03 | M4, M5 | Registry + scoring ghi score/nhãn/model version |
| FR-04 | M6, M9 | API + Dashboard danh sách ưu tiên |
| FR-05 | M5, M9 | Uncertainty trong prediction record + quy ước hiển thị (3.9-bis) |
| FR-06 | M8 | Brief grounded, citation đối chiếu dữ liệu nguồn |
| FR-07 | M6, M7, M9 | Workflow brief + DECISIONS/feedback |
| FR-08 | M8, M9 | Fallback + trạng thái lỗi hiển thị rõ |
| FR-09 | M6, M7, M10 | RBAC + audit log + prediction logging + monitoring |

| Rủi ro (spec mục 6) | Kiểm soát bắt buộc | Điểm hiện thực |
|---|---|---|
| Temporal/Data Leakage | Feature cutoff, validation theo thời gian, leakage test | M2, M3, split M4 |
| False Negative | Đánh giá recall, phân tích missed cases | M4 (FN analysis) |
| False Positive / Alert Overload | Threshold tuning, Precision@K, review workload | M4 (TH), M9 (S2) |
| Domain Shift / Bias | Slice evaluation, monitoring | M4 (SLICE), M10 |
| LLM Hallucination | Grounding, kiểm tra nguồn, human review | M8 (validation), M9 (S4), PT-1 |
| Privacy / Security | Data minimization, access control, bảo vệ log | M6 (RBAC), M8 (context tối thiểu), M7 |
| Human/Workflow Failure | UI hỗ trợ kiểm chứng, lưu quyết định | M9 (S3, S4), M7 (DECISIONS) |
| ML/LLM/API Failure | Error handling, explicit failure states, fallback | Lớp ngang + M8 + M9 (S5) |

| Giả định (ASM) | Xác minh ở đâu |
|---|---|
| ASM-01 | Slice evaluation M4/M10 — nêu giới hạn tổng quát hoá trong báo cáo |
| ASM-02 | Feature Catalog + T4 audit (M3) — feature không chứng minh as-of bị loại |
| ASM-03 | Hiển thị proxy + nhãn "tín hiệu tương quan" ở M9 |
| ASM-04 | Kịch bản demo S2→S4; đo thời gian review phục vụ Product/User value |
| ASM-05 | Fallback M8/M5 + monitoring M10 — kiểm thử tình huống tắt LLM/API |

---

## 5. Hướng mở rộng (ngoài phạm vi hiện tại)

- **RAG** cho guidelines nếu rubric LLM thất bại ở grounding trực tiếp.
- **Online serving** khi có LMS thật (thay batch bằng stream + re-scoring định kỳ).
- **Continuous learning**: retrain theo lịch khi thêm kỳ mới —Registry sẵn sàng cho việc này (chỉ thêm job, không đổi kiến trúc).
- Mở rộng slice evaluation fairness (nhóm demographics) nếu yêu cầu đánh giá fairness chi tiết hơn.

---

## Phụ lục A. Quy tắc đặt tên & versioning

| Đối tượng | Định dạng | Ví dụ |
|---|---|---|
| Feature snapshot | `snapshot_v{n}_{feature_list_hash[:8]}` | `snapshot_v3_1a2b3c4d` |
| Model version | `model_{alg}_{date}_{hash[:8]}` | `model_gbm_20261018_9f8e7d6c` |
| Brief | `brief_{prediction_id}_v{n}` | `brief_1042_v2` |
| Prompt template | `prompt_{purpose}_v{n}` (nằm trong git) | `prompt_brief_v5` |

## Phụ lục B. Quyết định mở cần nhóm chốt trước khi cài đặt

1. Model chính: XGBoost hay LightGBM (chi tiết M4).
2. Ngân sách review giả định cho threshold tuning: gắn cờ tối đa ~top K% cohort? (chi tiết M4/TH).
3. LLM provider cụ thể theo budget nhóm (chi tiết M8).
4. Số lượng kịch bản demo cho bài đánh giá Product/User value (chi tiết ASM-04).
