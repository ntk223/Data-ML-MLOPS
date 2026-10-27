# PIPELINE_GUIDE — Hướng dẫn vận hành pipeline dữ liệu LearnGuard

**Dành cho:** ML engineer / bất kỳ ai cần chạy pipeline `ml-data` hoặc tiêu thụ feature snapshot.
**Phạm vi:** M1 (ingestion) → M2 (feature snapshot) → M3 (leakage audit + quality gates). Phần M4/M5 do ML owner phát triển trên đầuContract được mô tả ở mục 5.
**Người chịu trách nhiệm pipeline:** data engineer (người viết tài liệu này). Mọi thắc mắc về split/feature/quarantine → hỏi data engineer, **không tự sửa**.

---

## 1. Bức tranh tổng thể

```
CSV OULAD (7 file, giữ nguyên)
   │
   ▼  M1 — ingest_oulad()          validate schema → quarantine dòng lỗi → dedupe
   │                                → Parquet canonical + SHA-256 + ingestion_report.json
   ▼  M2 — build_feature_snapshot() cohort point-in-time (cutoff ngày 27)
   │                                → snapshot.parquet (có cột label + split) + ledger JSON
   ▼  M3 — LeakageAuditor + DataQualityGate
   │                                chứng minh snapshot sạch → PASS mới được dùng
   ▼
snapshot.parquet ──► M4 (ML owner): train / tune / eval theo cột `split`
```

**Triết lý cốt lõi:** pipeline chỉ **validate + quarantine + ghi vết**, KHÔNG bao giờ impute/dọn hộ. Dòng dữ liệu sai → vào `quarantine/`, không tự biến mất. Mọi con số đều truy vết được qua hash trong ledger.

---

## 2. Cài đặt môi trường

```bash
cd ml-data

# Tạo venv (Python ≥ 3.11)
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # runtime: pandas 3.1.0, pyarrow 26.0.0
.venv/bin/pip install -r requirements-dev.txt    # nếu cần chạy test/notebook (pytest, jupyter...)
```

**Kiểm tra cài đặt:**

```bash
.venv/bin/python -m pytest tests/ -q    # phải thấy: 56 passed
```

**Dữ liệu đầu vào:** 7 file CSV gốc của OULAD (không sửa tên/không sửa nội dung), ví dụ đặt tại `data/oulad/archive/`. Tất cả lệnh dưới đây chạy với **cwd = `ml-data/`**.

---

## 3. Chạy pipeline — 3 bước

### Bước 1 — M1: Ingestion (CSV → Parquet canonical)

```bash
.venv/bin/python -c "
from pathlib import Path
from learnguard.data.ingestion import ingest_oulad

report = ingest_oulad(Path('data/oulad/archive'), Path('artifacts/ingestion_m1_verify'))
print('status:', report.status)
"
```

Kết quả trong `artifacts/ingestion_m1_verify/`:
- `canonical/*.parquet` — dữ liệu đã qua kiểm định (đầu vào DUY NHẤT hợp lệ của M2)
- `quarantine/` — dòng vi phạm schema (kèm lý do), không rơi vào canonical
- `ingestion_report.json` — số liệu từng bảng, số dòng quarantine

**Đọc report thế nào:** `status = ok` và `quarantined_rows = 0` là sạch. Quarantine > 0 KHÔNG phải lỗi — đó là thiết kế (dòng xấu bị cách ly thay vì âm thầm trôi vào model). Xem lý do trong `quarantine/`, quyết định xử lý thuộc về data engineer.

### Bước 2 — M2: Feature snapshot (cutoff ngày 27)

```bash
.venv/bin/python -c "
from pathlib import Path
from learnguard.data.features import build_feature_snapshot

r = build_feature_snapshot(
    Path('artifacts/ingestion_m1_verify'),
    Path('artifacts/feature_snapshot_m2/snapshot.parquet'),
)
print('snapshot_version:', r.snapshot_version)
"
```

Kết quả trong `artifacts/feature_snapshot_m2/`:
- `snapshot.parquet` — ~27,535 dòng × 21 cột (schema đầy đủ ở mục 5.1)
- `snapshot.ledger.json` — giấy khai sinh của snapshot: `cutoff_day`, `source_hash`, `feature_rules`, `cohort_rules`, `split_rules`, `input_hashes`, `feature_list_hash`, `snapshot_version`

**Tại sao cutoff = ngày 27?** Snapshot mô phỏng thời điểm cố vấn can thiệp sớm trong kỳ (tuần ~4/8 kịch bản spec): mọi feature chỉ được tính từ sự kiện xảy ra **trước hoặc bằng** ngày 27. Đổi `cutoff_day` là đổi hợp đồng feature → phải qua data engineer (xem mục 7).

### Bước 3 — M3: Leakage audit + Quality gates (bắt buộc trước khi dùng)

```bash
.venv/bin/python -c "
from pathlib import Path
from learnguard.data.leakage import LeakageAuditor
from learnguard.data.quality import validate_snapshot

snap = Path('artifacts/feature_snapshot_m2/snapshot.parquet')
audit = LeakageAuditor().audit_snapshot(snap, Path('artifacts/ingestion_m1_verify'), cutoff_day=27)
gate  = validate_snapshot(snap)
print('AUDIT:', audit.status, '| violations:', len(audit.violations))
print('GATES:', gate.status, '| errors:', gate.errors or 'không')
"
```

**Điều kiện sử dụng snapshot:** cả hai đều `ok`. Bất kỳ violation/error nào → **KHÔNG train model**, báo data engineer. Audit gồm 3 lớp (ledger hash, recompute từng dòng, 16 injection probes cắm sự kiện tương lai) — snapshot không qua audit thì chưa chứng minh không leakage.

---

## 4. Quickstart — toàn bộ pipeline trong 1 lệnh

```bash
.venv/bin/python -c "
from pathlib import Path
from learnguard.data.ingestion import ingest_oulad
from learnguard.data.features import build_feature_snapshot
from learnguard.data.leakage import LeakageAuditor
from learnguard.data.quality import validate_snapshot

ing = ingest_oulad(Path('data/oulad/archive'), Path('artifacts/ingestion_m1_verify'))
snap = build_feature_snapshot(Path('artifacts/ingestion_m1_verify'),
                              Path('artifacts/feature_snapshot_m2/snapshot.parquet'))
audit = LeakageAuditor().audit_snapshot(snap.snapshot_path, Path('artifacts/ingestion_m1_verify'), 27)
gate = validate_snapshot(snap.snapshot_path)
assert ing.status == 'ok' and audit.status == 'ok' and gate.status == 'ok', \
    f'pipeline FAILED: {ing.status} / {audit.status} / {gate.status}'
print('PIPELINE OK —', snap.snapshot_version)
"
```

---

## 5. Hợp đồng tiêu thụ (ML owner đọc kỹ)

### 5.1 Schema `snapshot.parquet`

| Nhóm cột | Cột | Ghi chú |
|---|---|---|
| Khóa chính | `code_module`, `code_presentation`, `id_student` | PK composite — mỗi SV 1 dòng/1 kỳ |
| Target | `final_result`, `label` | `label = 1` nếu Fail/Withdrawn (at-risk) |
| **Split** | `split` | `train` / `validation` / `holdout` — gán theo kỳ học, KHÔNG random |
| Static | `gender`, `region`, `highest_education`, `imd_band`, `age_band`, `num_of_prev_attempts`, `studied_credits`, `disability`, `module_presentation` | `imd_band` có thể null — null là nghiệp vụ (kèm cờ `imd_band_missing`) |
| Behavioral (as-of ≤ 27) | `n_clicks_before_cutoff`, `click_trend_w3_w4`, `n_submitted_before_cutoff`, `avg_score_submitted` | `avg_score_submitted` null = SV chưa có điểm nào — **xử lý tường minh ở M4, không fillna mù** |
| Cờ hệ thống | `insufficient_data` | SV ít tương tác — cân nhắc khi phân tích |

### 5.2 Cách load đúng

```python
import pandas as pd

snap = pd.read_parquet("artifacts/feature_snapshot_m2/snapshot.parquet")

train = snap[snap["split"] == "train"]        # 11,811 dòng (2013B+2013J) → model.fit()
val   = snap[snap["split"] == "validation"]   #  6,510 dòng (2014B)       → tuning THRESHOLD/hyperparam
test  = snap[snap["split"] == "holdout"]      #  9,214 dòng (2014J)       → CHẠM 1 LẦN khi báo cáo
```

### 5.3 Ba luật vàng (vi phạm = mất giá trị khoa học của toàn pipeline)

1. **Tune CHỈ trên `validation`** — chọn threshold, hyperparameter, model. Không bao giờ tune trên holdout.
2. **`holdout` chạm đúng 1 LẦN** — chạy evaluate sau khi chốt config, lấy số báo cáo. Muốn thay đổi gì → sửa config → chu kỳ sau mới được chạm lại.
3. **Không tạo file train/test riêng** — luôn lọc từ cột `split` của snapshot. Copy tách file = mất audit trail.

### 5.4 Snapshot version & ledger

`snapshot_version = snapshot-v1:cutoff{day}:{source_hash[:8]}:{feature_list_hash[:8]}`

- `source_hash` đổi → **dữ liệu nguồn** đã thay đổi
- `feature_list_hash` đổi → **hợp đồng rule** (feature hoặc split) đã thay đổi
- Mỗi lần train nên **ghi `snapshot_version` vào model card/registry** — model phải gắn với đúng phiên bản snapshot đã train. Version hiện tại: `snapshot-v1:cutoff27:35de8103:39e10cfa`

---

## 6. Sự cố thường gặp

| Triệu chứng | Nghĩa là | Xử lý |
|---|---|---|
| `ingestion_report.json` có dòng quarantine | Dữ liệu vi phạm schema bị cách ly (thiết kế đúng) | Đọc lý do trong `quarantine/`; quyết định thuộc data engineer |
| `ValueError: unknown_presentation_for_split` | Có kỳ học mới chưa có rule split | **Đúng hành vi**. Báo data engineer chốt rule trong `SPLIT_RULES` rồi rebuild |
| Audit: `source_hash_mismatch` | Snapshot sinh từ dữ liệu nguồn khác hiện tại | Rebuild snapshot từ `canonical/` hiện tại |
| Audit: `snapshot_recompute_mismatch` | Snapshot bị sửa tay hoặc lệch nguồn | **Không bao giờ sửa snapshot trực tiếp** — rebuild từ pipeline |
| Gate: `missingness_exceeded` / `psi_exceeded` | Null quá 70% hoặc drift PSI > 0.2 so với baseline | PSI bỏ qua khi mẫu < 200/bên (thiết kế — tránh false alarm); còn lại báo data engineer |
| Audit: `future_event_leak:*` | Feature "nhìn thấy tương lai" — lỗi builder | DỪNG. Báo data engineer — đây là lỗi nghiêm trọng nhất |

---

## 7. Ranh giới trách nhiệm

| Thuộc data engineer | Thuộc ML owner |
|---|---|
| Định nghĩa feature (`FEATURE_CATALOG`), cutoff_day, split rules | Chọn model, hyperparameter, threshold (trên validation) |
| Chạy pipeline, xử lý quarantine, quyết định khi gates fail | Fairness slices, calibration, metrics Recall@top-k/AUC/ECE |
| Mọi thay đổi hợp đồng (feature/split/cutoff) | Ghi `snapshot_version` + config vào model card/registry |
| M3 gates + audit là ĐIỂM CHẶN — pass mới giao data | **Không sửa snapshot, không đi vòng gates** |

Thay đổi bất kỳ hợp đồng nào (thêm feature, đổi split, đổi cutoff) → đề xuất với data engineer → sửa code + test + rebuild + audit lại → version mới. Không có nhánh "sửa nhanh tay".

---

## 8. Tài liệu nền (đọc thêm khi cần)

| Tài liệu | Nội dung |
|---|---|
| `docs/M1_REPORT.md` | 8 quyết định thiết kế EDA (null nghiệp vụ, dedupe studentVle...) |
| `docs/M2_REPORT.md` | Cohort, 5 features, **phụ lục split + trade-off + phương án dự phòng** |
| `docs/M3_REPORT.md` | 3 lớp leakage audit, 4 quality gates, MIN_PSI_SAMPLE=200 |
| `docs/REPOSITORY_LAYOUT.md` | Ranh giới repo + spec `prediction-batch.v1` (M5 → backend) |
| `docs/IMPLEMENTATION_TODO.md` | Checklist tiến độ M1–M5 |
