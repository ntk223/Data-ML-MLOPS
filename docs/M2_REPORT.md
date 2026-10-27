# Báo cáo M2 — Cohort & Point-in-Time Feature Snapshot (cutoff day-27)

**Dự án:** LearnGuard ML/Data · **Giai đoạn:** M2 · **Ngày hoàn thành:** 2026-10-27
**Trạng thái:** ✅ HOÀN THÀNH · Tests 36/36 pass (13 test M2) · Verify dữ liệu thật: snapshot 27,535 dòng

---

## 1. Mục tiêu & sản phẩm

M2 trả lời câu hỏi pipeline: **"Tại ngày 27 của kỳ học, dựa CHỈ vào dữ liệu có trước thời điểm đó, mỗi sinh viên trông thế nào?"**

Sản phẩm:

```
artifacts/feature_snapshot_m2/
├── snapshot.parquet        # 27,535 dòng × 20 cột — đầu vào duy nhất của M4/M5
└── snapshot.ledger.json    # AsOfLedger — bằng chứng chống leakage
```

Snapshot version: `snapshot-v1:cutoff27:35de81037b78` (deterministic từ input hash — cùng dữ liệu luôn ra cùng version).

## 2. Cohort — chọn "dân" dự đoán

Quy tắc (ghi trong ledger `cohort_rules`):

| Quy tắc | Số dòng |
|---|---:|
| Đăng ký vào học: `date_registration <= 27` (hoặc null — 45 dòng, giữ + ghi minh bạch) | — |
| **Loại:** đã rút lui tính tới cutoff (`date_unregistration <= 27`) | **−5,040** |
| **Cohort cuối** | **27,535** |

Lý do loại 5,040: họ đã "có đáp án" (đã bỏ học) **trước** thời điểm dự đoán — giữ lại là target leakage. Rút lui *sau* cutoff (5,119 SV) được giữ vì tại ngày 27 chưa ai biết.

## 3. Feature table — 20 cột

| Nhóm | Cột |
|---|---|
| Khóa (PK từ M1) | `code_module`, `code_presentation`, `id_student` |
| Static (có trước khai giảng) | `gender`, `region`, `highest_education`, `imd_band` + `imd_band_missing` (1,028 dòng), `age_band`, `num_of_prev_attempts`, `studied_credits`, `disability` |
| Tương tác VLE | `n_clicks_before_cutoff` (SUM, **gồm date âm** — quyết định #6), `click_trend_w3_w4` |
| Bài nộp | `n_submitted_before_cutoff`, `avg_score_submitted` (**loại Exam không ngày** — quyết định #4) |
| Metadata | `module_presentation`, `insufficient_data` (796 SV) |
| Target | `final_result`, `label` (0 = Pass/Distinction, 1 = Fail/Withdrawn) — **nằm ngoài feature catalog** |

### Thống kê mô tả (cutoff day-27)

| Feature | mean | median | min | max | Ghi chú |
|---|---:|---:|---:|---:|---|
| `n_clicks_before_cutoff` | 340 | 214 | 0 | 7,477 | 96.7% SV có click; 922 SV = 0 |
| `click_trend_w3_w4` | — | — | — | — | tuần 3–4 − tuần 1–2 |
| `n_submitted_before_cutoff` | 0.76 | 1 | 0 | 2 | TMA đầu thường d19–25 |
| `avg_score_submitted` | 72.7 | 77 | 0 | 100 | 20,007/27,535 SV có điểm |

**Phân bố nhãn trong cohort:** Pass 12,356 · Fail 7,037 · Withdrawn 5,119 · Distinction 3,023 → at-risk = **44.2%** — cân tốt cho baseline M4.

**Độ phủ khóa học:** 20/22 presentation có cohort. AAA-2013B/AAA-2014B vắng (toàn bộ SV đã rút lui trước cutoff) — điểm cần nhớ khi đọc kết quả M4 theo module.

## 4. Chống leakage — cơ chế 3 lớp

1. **Cohort rule**: loại rút-lui-trước-cutoff (§2)
2. **As-of rule từng feature**: mọi aggregate chỉ dùng `date <= 27`; Exam không ngày không đủ căn cứ
3. **AsOfLedger**: `source_hash` (hash 6 file Parquet M1) + `input_hashes` + `feature_rules` + `cohort_rules` + `feature_list_hash` — snapshot nào sinh từ dữ liệu nào, truy vết được tuyệt đối

**Bằng chứng test:** future-event injection — cắm click `date=60` (sum 5,000) vào `studentVle` → snapshot **không đổi một bit**. Cutoff boundary: `date=27` được tính, `date=28` bị loại.

## 5. Test suite — 36/36 pass

13 test M2 mới (`tests/test_features.py`):

| Nhóm | Test |
|---|---|
| Cohort | rút lui d25 bị loại / d30 được giữ; đếm exclusions |
| Cutoff boundary | inclusive day-27; date âm được tính |
| **Leakage** | future-event injection → snapshot bất biến |
| Feature | trend w3−w4; Exam-không-ngày loại; score null không vào avg nhưng vẫn count; SV 0-hoạt-động = 0 + cờ insufficient |
| Ledger | end-to-end snapshot+ledger; version đổi khi input đổi; deterministic cùng input |

**Bug bắt được nhờ test:** merge feature khóa-học (`module_presentation`, 2 cột khóa) với khóa 3-cột làm mất dòng → tách key merge theo loại feature.

## 6. Quyết định thiết kế đã chốt trong M2

| # | Quyết định | Lý do |
|---|---|---|
| 1 | Rút lui `<= cutoff` bị loại (biên inclusive) | Tại thời điểm dự đoán, thông tin này đã có |
| 2 | `date_registration` null → giữ trong cohort (45 dòng) | Không có bằng chứng đăng ký sau cutoff; đếm minh bạch trong ledger |
| 3 | Count/trend feature: thiếu tương tác = 0; thiếu tổng thể = `insufficient_data` | Phân biệt "không click" vs "không có dữ liệu" |
| 4 | `avg_score_submitted` = NaN khi chưa có điểm eligible | Không impute — M4 sẽ xử lý NaN tường minh |
| 5 | `snapshot_version` = f(cutoff, source_hash) | Deterministic — tái lập được, không phụ thuộc thời điểm chạy |

## 7. Cách chạy lại

```bash
.venv/bin/python -m pytest tests/ -q   # 36/36 pass
.venv/bin/python -c "
from learnguard.data.features import build_feature_snapshot
build_feature_snapshot('artifacts/ingestion_m1_verify', 'artifacts/feature_snapshot_m2/snapshot.parquet')
"
```

## 8. Việc tiếp theo — M3

1. **Leakage audit suite**: future-event injection chạy như CI gate trên toàn pipeline
2. **Quality gates**: schema validation, missingness, volume, PSI (phân bố trôi giữa snapshot)
3. Gate fail → chặn pipeline, không xuất batch

---

## Phụ lục (2026-10-27) — Temporal split `train/validation/holdout`

Đáp ứng yêu cầu của M4: cột **`split`** được thêm vào snapshot, gán **deterministic theo kỳ học** (không random — mô phỏng kịch bản production "train kỳ cũ → dự đoán kỳ mới"). Rule nằm trong `SPLIT_RULES` (`features.py`) + ghi vào ledger `split_rules`; kỳ lạ → `unknown_presentation_for_split` fail lớn.

| Split | Presentation | Số dòng | At-risk % | Dùng để |
|---|---|---:|---:|---|
| `train` | 2013B | 4,127 | 49.3% | `model.fit()` |
| `validation` | 2013J | 7,684 | 41.8% | Tuning threshold/hyperparameter — **chỉ** ở đây |
| `holdout` | 2014B + 2014J | 15,724 | 44.0% | Đo 1 lần duy nhất khi báo cáo |

Nguyên tắc: holdout chỉ chạm 1 lần; trùng SV giữa 2013/2014 là hiện tượng thật (không leakage — nhãn 2014 không nằm trong row 2013); M5 vẫn chấm điểm TOÀN BỘ cohort bất kể split.

`snapshot_version` nâng cấp thành `cutoff:source_hash:rules_hash` — version đổi khi **cả** dữ liệu **hoặc** hợp đồng rule đổi. Snapshot mới: `snapshot-v1:cutoff27:35de8103:c593b405`, audit + gates đều `ok`.
