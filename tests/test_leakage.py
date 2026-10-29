"""Test M3 — leakage auditor: catalog rules, recompute, injection probes, ledger."""

import json

import pandas as pd

from learnguard.data import leakage as leakage_mod
from learnguard.data.features import (
    KEYS,
    PointInTimeFeatureBuilder,
    build_feature_snapshot,
)
from learnguard.data.leakage import LeakageAuditor
from tests.test_features import base_tables, write_canonical

def build_snapshot(tmp_path):
    """Snapshot sạch từ fixture nhỏ (end-to-end qua M2 builder)."""
    processed = write_canonical(tmp_path, base_tables())
    return build_feature_snapshot(processed, tmp_path / "snapshot.parquet", cutoff_day=27), processed


def test_catalog_audit_passes_current_catalog():
    report = LeakageAuditor().audit_catalog()
    assert report.status == "ok", report.violations
    assert report.checked_features == 6


def test_catalog_audit_rejects_unknown_rule_and_target_reference(monkeypatch):
    from learnguard.data.features import FeatureDefinition

    tampered = (
        FeatureDefinition("f_bad_rule", "studentVle", "sum_click", "whatever_i_want", "numeric"),
        FeatureDefinition("f_target", "studentInfo", "final_result as feature", "static_before_start", "categorical"),
    )
    monkeypatch.setattr(leakage_mod, "FEATURE_CATALOG", tampered)
    report = LeakageAuditor().audit_catalog()
    assert report.status == "violated"
    rules = {v.rule for v in report.violations}
    assert "unknown_as_of_rule" in rules          # f_bad_rule
    assert "target_or_identity_reference" in rules  # f_target
    # Lỗi "aggregation không tôn trọng as-of rule" (vd: 'sum_click all rows')
    # được bắt ở lớp 2 — injection probe — chứ không phải bằng phân tích văn bản.


def test_audit_clean_snapshot_passes(tmp_path):
    result, processed = build_snapshot(tmp_path)
    report = LeakageAuditor().audit_snapshot(result.snapshot_path, processed, cutoff_day=27)
    assert report.status == "ok", [v.rule for v in report.violations]
    assert report.checked_features == 6
    assert report.source_checks > 0  # injection probes đã chạy


def test_audit_detects_hand_tampered_snapshot(tmp_path):
    result, processed = build_snapshot(tmp_path)
    snap = pd.read_parquet(result.snapshot_path)
    snap.loc[0, "n_clicks_before_cutoff"] += 1  # sửa tay 1 dòng
    tampered_path = tmp_path / "tampered.parquet"
    snap.to_parquet(tampered_path, index=False)
    # ledger phải đi kèm để không rơi vào missing_ledger trước
    (tmp_path / "tampered.ledger.json").write_text(result.ledger_path.read_text())

    report = LeakageAuditor().audit_snapshot(tampered_path, processed, cutoff_day=27)
    rules = {v.rule for v in report.violations}
    assert "snapshot_recompute_mismatch" in rules
    assert report.status == "violated"


def test_audit_detects_tampered_label_and_static_field(tmp_path):
    result, processed = build_snapshot(tmp_path)
    snap = pd.read_parquet(result.snapshot_path)
    snap.loc[0, "label"] = 1 - int(snap.loc[0, "label"])
    snap.loc[0, "region"] = "Wales"
    snap.to_parquet(result.snapshot_path, index=False)
    report = LeakageAuditor().audit_snapshot(result.snapshot_path, processed, cutoff_day=27)
    rules = {v.rule for v in report.violations}
    assert "snapshot_content_hash_mismatch" in rules
    assert "snapshot_recompute_mismatch" in rules


def test_audit_detects_ledger_hash_mismatch(tmp_path):
    result, processed = build_snapshot(tmp_path)
    ledger = json.loads(result.ledger_path.read_text())
    ledger["source_hash"] = "0" * 64
    result.ledger_path.write_text(json.dumps(ledger))

    report = LeakageAuditor().audit_snapshot(result.snapshot_path, processed, cutoff_day=27)
    assert any(v.rule == "source_hash_mismatch" for v in report.violations)


def test_audit_catches_leaky_builder_via_injection_probe(tmp_path, monkeypatch):
    """Cốt lõi M3: builder 'nhìn thấy tương lai' (date <= cutoff+30) → probe P1 phải bắt."""

    class LeakyBuilder(PointInTimeFeatureBuilder):
        def build_feature(self, feature, tables, cutoff_day):
            if feature.name == "n_clicks_before_cutoff":
                sv = tables["studentVle"]
                eligible = sv[sv["date"] <= cutoff_day + 30]  # LỖ: cho qua tương lai
                agg = eligible.groupby(list(KEYS), as_index=False)["sum_click"].sum()
                return agg.rename(columns={"sum_click": feature.name})
            return super().build_feature(feature, tables, cutoff_day)

    monkeypatch.setattr(leakage_mod, "PointInTimeFeatureBuilder", LeakyBuilder)
    result, processed = build_snapshot(tmp_path)

    report = LeakageAuditor().audit_snapshot(result.snapshot_path, processed, cutoff_day=27)
    assert report.status == "violated"
    rules = {v.rule for v in report.violations}
    assert any(r.startswith("future_event_leak:vle_future_click") for r in rules), rules


def test_audit_catches_future_submission_leak_in_same_offering(tmp_path, monkeypatch):
    class SubmissionLeakyBuilder(PointInTimeFeatureBuilder):
        def build_feature(self, feature, tables, cutoff_day):
            if feature.name in {"n_submitted_before_cutoff", "avg_score_submitted"}:
                submissions = tables["studentAssessment"]
                assessments = tables["assessments"]
                joined = submissions.merge(
                    assessments[["id_assessment", "code_module", "code_presentation", "date"]],
                    on="id_assessment", how="left",
                )
                eligible = joined["date"].notna() & joined["date"].le(cutoff_day)
                rows = joined[eligible]  # deliberately ignores submission cutoff
                if feature.name == "n_submitted_before_cutoff":
                    return rows.groupby(list(KEYS)).size().rename(feature.name).reset_index()
                return rows.dropna(subset=["score"]).groupby(list(KEYS))["score"].mean().rename(feature.name).reset_index()
            return super().build_feature(feature, tables, cutoff_day)

    monkeypatch.setattr(leakage_mod, "PointInTimeFeatureBuilder", SubmissionLeakyBuilder)
    result, processed = build_snapshot(tmp_path)
    report = LeakageAuditor().audit_snapshot(result.snapshot_path, processed, 27)
    assert any(v.rule == "future_event_leak:future_submission" for v in report.violations)


def test_audit_requires_ledger(tmp_path):
    result, processed = build_snapshot(tmp_path)
    result.ledger_path.unlink()
    report = LeakageAuditor().audit_snapshot(result.snapshot_path, processed, cutoff_day=27)
    assert any(v.rule == "missing_ledger" for v in report.violations)
