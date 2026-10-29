# Báo cáo M3 — Leakage Audit & Quality Gates

**Dự án:** LearnGuard ML/Data · **Giai đoạn:** M3 · **Ngày hoàn thành:** 2026-10-27
**Trạng thái:** ✅ HOÀN THÀNH · Tests 53/53 pass (17 test M3) · Verify dữ liệu thật: audit `ok` 0 violation, 4/4 gates pass

---

## 1. Vai trò của M3

M3 không tạo dữ liệu mới — nó là **cửa ải chặn**: chứng minh M1+M2 đã đúng, và **fail thì chặn pipeline** (không xuất batch, theo TODO Orchestration). Vai trò tương đương CI gate cho dữ liệu.

```
M2 snapshot ──► [M3: LeakageAuditor + DataQualityGate] ──PASS──► M4 train/eval
                          │
                          └──FAIL──► chặn + báo cáo lỗi rõ ràng
```

## 2. Leakage Auditor — 3 lớp phòng thủ (`leakage.py`)

| Lớp | Cơ chế | Bắt loại lỗi nào |
|---|---|---|
| **0 · Ledger check** | `source_hash` trong ledger phải khớp hash Parquet nguồn hiện tại | Snapshot sinh từ dữ liệu khác processed_dir |
| **1 · Recompute** | Tính lại 5 features từ nguồn, đối chiếu từng dòng với snapshot trên đĩa | Snapshot bị sửa tay, lệch nguồn, thiếu cột |
| **2 · Injection probes** | Cắm 4 loại sự kiện "tương lai/mô hồ" × 4 SV → tính lại → **phải bất biến** | Bất kỳ kênh nào cho tương lai lọt vào feature |

### 4 loại injection probes (thiết kế bám 2 nguồn rủi ro lớn nhất của OULAD)

| Probe | Sự kiện cấy | Phải vô hình vì |
|---|---|---|
| `vle_future_click` | Click 999,999 lượt ngày `cutoff+30` | VLE chỉ dùng `date <= 27` |
| `future_assessment_date` | Bài thi diễn ra d57, nộp d10, điểm 100 | Phải thỏa CẢ HAI: nộp sớm + bài diễn ra trước cutoff |
| `future_submission` | Nộp d57 (bài diễn ra d20) | Nộp sau cutoff không được tính |
| `dateless_exam` | Exam không ngày + điểm 100 + nộp d10 | Exam không ngày không đủ căn cứ (quyết định #4) |

Ngoài ra `audit_catalog` từ chối: as-of rule lạ, tham chiếu target/identity (`final_result`, `label`...) trong catalog.

## 3. Data Quality Gate — 4 cánh cửa (`quality.py`)

| Gate | Kiểm tra | Ngưỡng mặc định |
|---|---|---|
| **Schema** | Đủ 20 cột expected, `label ∈ {0,1}`, PK composite unique | — |
| **Missingness** | Null fraction từng cột | ≤ 0.70 |
| **Volume** | Tỷ lệ dòng so với snapshot baseline | 0.10 – 10.0 |
| **PSI (drift)** | Trôi phân bố 6 numeric + 7 categorical vs baseline | ≤ 0.20 |

**Quyết định thiết kế quan trọng — PSI cần mẫu tối thiểu:** thực nghiệm cho thấy 2 mẫu **cùng phân bố** n≈100 cho PSI = 1.2 (nhiễu lấy mẫu lấn át tín hiệu). → Gate chỉ tính PSI khi cả hai bên ≥ **200 mẫu**; nhỏ hơn thì bỏ qua (kích thước đã do volume gate quản). Không có rule này, gate sẽ false-alarm liên tục.

## 4. Kết quả verify trên OULAD thật

```
LEAKAGE AUDIT:  status = ok · 5 features · 16 injection probes · 0 violation
CATALOG AUDIT:  ok (5 features)
QUALITY GATES:  status = ok · 27,535 dòng · 0 errors
  — null cao nhất: avg_score_submitted 27.3% (hợp lệ — SV chưa đủ điểm), imd_band 3.7%
  — PSI demo (snapshot vs chính nó): volume_ratio = 1.0, mọi PSI = 0
```

→ Snapshot M2 được chứng minh **không leakage** và **đạt đủ 4 gates** — sẵn sàng cấp cho M4.

## 5. Test suite — 53/53 pass (17 test M3 mới)

**Triết lý test: gate phải bắt được lỗi cấy chủ đích** — gate không bắt được lỗi cứng nhắc thì vô dụng:

- ✅ **Leaky-builder test** (quan trọng nhất): monkeypatch builder "nhìn thấy tương lai" (`date <= cutoff+30`) → injection probe **bắt được** `future_event_leak:vle_future_click`
- ✅ Snapshot bị sửa tay 1 dòng → `snapshot_recompute_mismatch`
- ✅ Ledger hash bị giả mạo → `source_hash_mismatch`; thiếu ledger → `missing_ledger`
- ✅ Catalog cắm rule lạ / tham chiếu target → từ chối
- ✅ Quality: thiếu cột label, label=5, PK trùng, null 100%, volume sụp (ratio 0.05), PSI drift (scale ×100 / region nuốt trọn), custom thresholds

**3 bug bắt được nhờ test:** (1) `_frames_equal` merge sai suffix; (2) auditor so sánh thiếu ngữ nghĩa "không tương tác = 0" của M2 → 12,471 dòng báo nhầm; (3) PSI false-alarm ở mẫu nhỏ (nới thành MIN_PSI_SAMPLE=200).

## 6. Cách chạy lại

```bash
.venv/bin/python -m pytest tests/ -q   # 53/53 pass
.venv/bin/python -c "
from learnguard.data.leakage import LeakageAuditor
from learnguard.data.quality import validate_snapshot
audit = LeakageAuditor().audit_snapshot('artifacts/feature_snapshot_m2/snapshot.parquet', 'artifacts/ingestion_m1_verify', 27)
gate = validate_snapshot('artifacts/feature_snapshot_m2/snapshot.parquet')
print(audit.status, gate.status)
"
```

## 7. Việc tiếp theo — M4

1. Baseline Logistic Regression + candidate model trên snapshot M2
2. **Temporal split**: train 2013B+J → validation 2014B → holdout 2014J (không random split — tránh trùng SV giữa train/test)
3. Metrics: Recall@top-k, AUC, ECE, fairness slices (gender/region/disability/age), false negatives
4. Tuning threshold CHỈ trên validation; NaN `avg_score_submitted` xử lý tường minh
5. Model card + registry metadata
