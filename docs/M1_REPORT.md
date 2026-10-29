# Báo cáo M1 — OULAD Ingestion & Schema Contracts

**Dự án:** LearnGuard ML/Data · **Giai đoạn:** M1 · **Ngày hoàn thành:** 2026-10-26
**Trạng thái:** ✅ HOÀN THÀNH — commit `870f002` · Tests 23/23 pass · Verify dữ liệu thật: `ok`, 0 quarantine

---

## 1. Phạm vi công việc

M1 chuyển 7 file CSV OULAD thô (44MB) thành dữ liệu chuẩn hóa đã kiểm định, làm nền cho toàn bộ pipeline M2–M5:

| Hạng mục | Kết quả |
|---|---|
| EDA 7 bảng | `notebooks/01_oulad_eda.ipynb` — 10 sections, schema 43 trường có mô tả, chạy sạch |
| Schema contracts | `src/learnguard/data/schema.py` — dtype + enum + range + null policy cho 43 trường |
| Ingestion service | `src/learnguard/data/ingestion.py` — validate → quarantine → hash → dedupe → Parquet |
| Bộ test | `tests/` — 23 test pytest trên fixture nhỏ |
| Môi trường | `.venv` + `requirements.txt` / `requirements-dev.txt` / pyproject đồng bộ |
| Baseline commit | `870f002`, working tree sạch |

---

## 2. Kết quả EDA — đầu vào của schema

EDA phát hiện **1 quyết định cấu trúc lớn + 8 quyết định thiết kế**, tất cả đã được mã hóa vào `schema.py`:

| # | Phát hiện | Quyết định |
|---|---|---|
| 1 | `id_student` không unique trong `studentInfo` (3.538 SV học nhiều kỳ) | PK composite `(code_module, code_presentation, id_student)` |
| 2 | `imd_band` null 3.41% | Giữ NA + cờ `imd_band_missing` ở M2, không drop/impute |
| 3 | `score` null 173 dòng; `date_unregistration` null 69% | Null = hợp lệ nghiệp vụ (rút lui/chưa rút lui) — không quarantine |
| 4 | `assessments.date` null cho 11 đề Exam | Exam không ngày → không tính vào feature trước cutoff (M2) |
| 5 | `studentVle` trùng key 5 cột | **Bản sửa sau verify:** exact-duplicate → dedupe ở M1; sự kiện trùng key khác `sum_click` → SUM ở M2 |
| 6 | `studentVle` có click trước khai giảng (date âm, min −25) | Hợp lệ — 82% SV có click sớm; feature tính cả date âm |
| 7 | `studentVle` 10.6M dòng CSV | Ingestion xuất Parquet, giữ long format |
| 8 | `vle.week_from/week_to` null 82% (MNAR) | Loại khỏi feature catalog, vẫn ingest đầy đủ |

**Mâu thuẫn đã giải quyết:** feature catalog cũ dùng `0 <= date <= cutoff` — cắt mất 25 ngày click trước khai giảng (6.5% dòng, ảnh hưởng 82% SV, 558 SV mất sạch feature). Đã sửa thành `date <= cutoff`, đổi tên `n_clicks_w1_w4` → `n_clicks_before_cutoff`, đồng bộ `features.py` + `docs/system-design.md`.

---

## 3. Schema contracts — `schema.py`

- **43 cột / 7 bảng**, mỗi cột có `ColumnSpec`: dtype (`str`/`int64`/`float64`), `nullable`, `allowed_values` (enum), `min_value`/`max_value`
- Miền giá trị lấy từ quan sát thực tế (section 2–3 EDA), ví dụ: `gender ∈ {M,F}`, `score ∈ [0,100] nullable`, `studentVle.date ∈ [−25, 269]`
- PK nghiêm ngặt cho 6 bảng; `studentVle` là event log — không PK (quyết định #5)
- `TARGET_LABELS`: `Pass/Distinction = 0`, `Fail/Withdrawn = 1` — nhãn at-risk cho M4

## 4. Ingestion pipeline — `ingestion.py`

Quy trình `ingest()` 5 pha:

1. **Đọc + SHA-256** từng file nguồn (provenance — 7 hash ghi trong report)
2. **Contract cột**: thiếu/thừa cột, sai dtype → *lỗi nghiêm trọng*, bảng bị loại khỏi pipeline
3. **Contract dòng** (vector hóa, chịu được 10.6M dòng): null policy, enum, range, PK trùng → vi phạm bị **quarantine kèm lý do** (`_quarantine_reason`), không xóa im lặng
4. **Referential integrity**: 9 quan hệ FK — orphan bị quarantine
5. **Xuất**: `canonical/*.parquet` + `quarantine/*.csv` + `ingestion_report.json` (status, hashes, counts, quarantine, errors)

Triết lý thiết kế: **validate + tách, không sửa** — không có `fillna`/`dropna` nào trong code; null nghiệp vụ được giữ nguyên.

---

## 5. Kết quả verify trên OULAD thật

Chạy: `ingest_oulad('data/oulad/archive', 'artifacts/ingestion_m1_verify')`

| Bảng | Raw | Clean | Quarantine | Deduped | Khớp EDA |
|---|---:|---:|---:|---:|:---:|
| courses | 22 | 22 | 0 | 0 | ✅ |
| assessments | 206 | 206 | 0 | 0 | ✅ |
| vle | 6,364 | 6,364 | 0 | 0 | ✅ |
| studentInfo | 32,593 | 32,593 | 0 | 0 | ✅ |
| studentRegistration | 32,593 | 32,593 | 0 | 0 | ✅ |
| studentAssessment | 173,912 | 173,912 | 0 | 0 | ✅ |
| studentVle | 10,655,280 | 9,868,110 | 0 | **787,170** | (xem §6) |

- **Status: `ok` — 0 quarantine, 0 orphan FK** → null policy khớp hoàn toàn với EDA
- Output: 7 file Parquet (29MB, xuống từ 44MB CSV)
- Source hash mẫu (`studentInfo`): `7e6f3e474a5eee00639d2a414a6c7e928745823c2d2c2563ca1780145f99b0d6`

## 6. Phát hiện mới: exact-duplicate trong `studentVle`

Verify lần đầu bắt được 3,810,465 dòng "trùng PK" — điều tra kỹ thấy dữ liệu **trộn 2 bản chất**:

- **787,170 dòng = bản sao y hệt toàn cột** (artifact xuất trùng) → dedupe ở M1, đếm minh bạch trong `deduped_row_counts`
- Phần còn lại trùng key nhưng `sum_click` **khác nhau** = sự kiện click riêng biệt → **giữ nguyên**, M2 sẽ SUM

→ Quyết định EDA #5 được sửa có bằng chứng và đồng bộ vào notebook + docs. Nếu SUM thẳng không dedupe, click bị **đếm đôi ~7.4%**.

## 7. Test suite — 23/23 pass

`tests/test_schema.py` (10 test) — khóa quyết định schema: PK composite, null policy, enum, range biên.

`tests/test_ingestion.py` (13 test) — hành vi pipeline: quarantine có lý do, FK orphan, PK dup (cả 2 bản sao bị tách), hash đổi khi file đổi, fatal cases (thiếu cột/file, cột thừa), dedupe vs distinct events, report JSON.

Nguyên tắc: **fixture nhỏ nhân tạo, không test trên OULAD thật** — nhanh (1.2s), deterministic, lỗi cấy chủ đích.

**2 bug bắt được nhờ TDD:** (1) early-return bỏ qua kiểm PK; (2) bảng fatal vẫn bị ghi Parquet.

---

## 8. artifacts & cách chạy lại

```
artifacts/ingestion_m1_verify/
├── canonical/          # 7 Parquet — M2 chỉ được đọc từ đây, KHÔNG đọc lại CSV
├── quarantine/         # rỗng (dữ liệu thật sạch)
└── ingestion_report.json
```

```bash
.venv/bin/python -m pytest tests/ -q                                  # 23/23 pass
.venv/bin/python -c "from learnguard.data.ingestion import ingest_oulad; \
    ingest_oulad('data/oulad/archive', 'artifacts/ingestion')"
```

## 9. Việc tiếp theo — M2

1. Cohort + feature snapshot tới cutoff day-27 với as-of ledger/version
2. Feature table: 1 dòng/SV/khóa học, aggregate từ Parquet canonical (SUM `sum_click` cho `date <= 27`, gồm date âm)
3. Áp dụng quyết định #4: Exam không ngày không được tính vào feature trước cutoff
4. Cờ `imd_band_missing` cho null `imd_band`
5. Sau M2: leakage audit (M3) — future-event injection test
