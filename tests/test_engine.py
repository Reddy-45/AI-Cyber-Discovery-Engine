"""
tests/test_engine.py — Critical path tests for the AI Cyber Discovery Engine.

Covers the pipeline end-to-end plus key unit tests for each stage.
Not a full test pyramid — targeted sanity checks that confirm the
demo will work and the AI reasoning produces sensible outputs.

Run: pytest tests/ -v
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.models import (
    CanonicalEvent,
    EventType,
    Severity,
    EnrichedEvent,
    IOCMatch,
    IOCType,
    MITRETechnique,
    RiskScore,
    ThreatAlert,
    AnalysisResult,
)
from engine.config import load_config
from engine.ingest import load_and_normalize, _normalize_event, _parse_timestamp
from engine.enrich import enrich_events, _extract_iocs, _map_mitre_techniques
from engine.analyze import (
    run_analysis,
    _classify_threat,
    _compute_anomaly_score,
    _compute_risk_score,
    _generate_explanation,
    _cluster_by_time_window,
)


# ── Fixtures ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "data" / "sample_logs"


@pytest.fixture
def sample_event() -> CanonicalEvent:
    return CanonicalEvent(
        timestamp=datetime(2025, 6, 15, 2, 11, 0, tzinfo=timezone.utc),
        source_ip="203.0.113.42",
        dest_ip="10.0.0.5",
        dest_port=22,
        protocol="TCP",
        event_type=EventType.AUTH,
        severity=Severity.HIGH,
        description="Failed password for admin from 203.0.113.42 port 54201 ssh2",
        raw="{}",
    )


@pytest.fixture
def brute_force_cluster(sample_event) -> list[EnrichedEvent]:
    """Simulate a brute-force cluster with 9 events spread over 4 minutes."""
    from datetime import timedelta
    base = datetime(2025, 6, 15, 2, 11, 0, tzinfo=timezone.utc)
    events = []
    for i in range(9):
        ev = CanonicalEvent(
            timestamp=base + timedelta(seconds=i * 25),  # 25s apart — stays within 5-min window
            source_ip="203.0.113.42",
            dest_ip="10.0.0.5",
            dest_port=22,
            protocol="TCP",
            event_type=EventType.AUTH,
            severity=Severity.MEDIUM if i < 8 else Severity.HIGH,
            description="Failed password for admin from 203.0.113.42",
            raw="{}",
        )
        events.append(EnrichedEvent(event=ev, reputation_score=0.05))
    return events


# ── Config Tests ──────────────────────────────────────────────────────

class TestConfig:
    def test_loads_successfully(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "pipeline" in cfg
        assert "risk_weights" in cfg

    def test_alert_threshold_present(self):
        cfg = load_config()
        assert "alert_threshold" in cfg["pipeline"]
        assert 0 <= cfg["pipeline"]["alert_threshold"] <= 100

    def test_risk_weights_sum_to_one(self):
        cfg = load_config()
        w = cfg["risk_weights"]
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"


# ── Ingestion Tests ───────────────────────────────────────────────────

class TestIngestion:
    def test_parse_iso_timestamp(self):
        dt = _parse_timestamp("2025-06-15T02:11:03Z")
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.tzinfo is not None

    def test_normalize_event_maps_severity(self):
        raw = {
            "timestamp": "2025-06-15T10:00:00Z",
            "event_type": "auth",
            "severity": "high",
            "description": "Test",
        }
        event = _normalize_event(raw)
        assert event.severity == Severity.HIGH
        assert event.event_type == EventType.AUTH

    def test_normalize_handles_missing_fields(self):
        raw = {"timestamp": "2025-06-15T10:00:00Z"}
        event = _normalize_event(raw)
        assert event.source_ip is None
        assert event.severity == Severity.INFO

    def test_load_sample_logs(self):
        if not LOG_DIR.exists():
            pytest.skip("Sample logs not found — run from project root")
        events = load_and_normalize(LOG_DIR)
        assert len(events) > 0
        # Verify chronological sort
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_events_are_immutable(self, sample_event):
        with pytest.raises(Exception):
            sample_event.source_ip = "evil"  # type: ignore


# ── Enrichment Tests ──────────────────────────────────────────────────

class TestEnrichment:
    def test_extracts_known_bad_ip(self, sample_event):
        from engine.enrich import _load_threat_intel
        threat_intel = _load_threat_intel()
        iocs, reputation = _extract_iocs(sample_event, threat_intel)
        known_bad = [i for i in iocs if i.is_known_bad]
        assert len(known_bad) > 0, "203.0.113.42 should be flagged as known bad"
        assert reputation < 0.5, "Reputation should be low for known-bad source"

    def test_maps_brute_force_to_mitre(self, sample_event):
        from engine.enrich import _load_mitre_db
        mitre_db = _load_mitre_db()
        techniques = _map_mitre_techniques(sample_event, mitre_db)
        tech_ids = [t.technique_id for t in techniques]
        assert "T1110" in tech_ids, "Brute force event should map to T1110"

    def test_enrich_returns_same_count(self, sample_event):
        events = [sample_event]
        enriched = enrich_events(events)
        assert len(enriched) == len(events)


# ── AI Reasoning Tests ────────────────────────────────────────────────

class TestAnalysis:
    def test_classify_brute_force(self, brute_force_cluster):
        threat_type, severity = _classify_threat(brute_force_cluster)
        assert "Brute Force" in threat_type or "Authentication" in threat_type
        assert severity in (Severity.HIGH, Severity.CRITICAL, Severity.MEDIUM)

    def test_anomaly_score_in_range(self, brute_force_cluster):
        score = _compute_anomaly_score(brute_force_cluster)
        assert 0.0 <= score <= 1.0

    def test_anomaly_score_single_event(self, brute_force_cluster):
        score = _compute_anomaly_score([brute_force_cluster[0]])
        assert 0.0 <= score <= 1.0

    def test_risk_score_in_range(self, brute_force_cluster):
        cfg = load_config()
        weights = cfg.get("risk_weights", {})
        risk = _compute_risk_score(brute_force_cluster, 0.85, [], weights)
        assert 0.0 <= risk.total <= 100.0

    def test_risk_score_components_sum(self, brute_force_cluster):
        cfg = load_config()
        weights = cfg.get("risk_weights", {})
        risk = _compute_risk_score(brute_force_cluster, 0.85, [], weights)
        component_sum = (
            risk.severity_component + risk.anomaly_component +
            risk.mitre_stage_component + risk.frequency_component +
            risk.asset_criticality_component
        )
        assert abs(component_sum - risk.total) < 1.0, "Components should sum to total"

    def test_explanation_contains_key_fields(self, brute_force_cluster):
        risk = RiskScore(total=85.0, severity_component=25.0, anomaly_component=22.0,
                         mitre_stage_component=18.0, frequency_component=15.0,
                         asset_criticality_component=5.0)
        exp = _generate_explanation(brute_force_cluster, "SSH Brute Force", risk, [], 0.9)
        assert "SSH Brute Force" in exp
        assert "RISK SCORE" in exp.upper()
        assert "85.0" in exp

    def test_full_pipeline_on_sample_data(self):
        if not LOG_DIR.exists():
            pytest.skip("Sample logs not found — run from project root")
        events = load_and_normalize(LOG_DIR)
        enriched = enrich_events(events)
        result = run_analysis(enriched)

        assert isinstance(result, AnalysisResult)
        assert result.total_events > 0
        # Expect at least one alert from our simulated attack scenarios
        assert len(result.alerts) >= 1, "Should detect at least one threat in sample data"
        # Top alert should have a non-trivial risk score
        assert result.alerts[0].risk_score.total > 40.0

    def test_alerts_sorted_by_risk_descending(self):
        if not LOG_DIR.exists():
            pytest.skip("Sample logs not found — run from project root")
        events = load_and_normalize(LOG_DIR)
        enriched = enrich_events(events)
        result = run_analysis(enriched)

        scores = [a.risk_score.total for a in result.alerts]
        assert scores == sorted(scores, reverse=True), "Alerts must be sorted highest risk first"
