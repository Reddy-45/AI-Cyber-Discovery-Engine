"""
tests/test_api.py — API layer tests for the AI Cyber Discovery Engine.

Uses FastAPI's built-in TestClient (backed by httpx) for synchronous
in-process testing — no real HTTP server needed.

Tests are structured around two phases:
    Phase A — Pre-analysis (no results in DB yet)
        GET endpoints return 404 with a helpful message.

    Phase B — Post-analysis (after POST /analyze runs the pipeline)
        All GET endpoints return valid, schema-compliant JSON.

The engine pipeline is exercised once via POST /analyze?source=sample,
and its output is verified across all five endpoints.

Existing engine tests (test_engine.py) remain completely unchanged.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app, raise_server_exceptions=True)


# ── Health endpoints ──────────────────────────────────────────────────

class TestHealth:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "version" in data

    def test_health_probe(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Pre-analysis: GET endpoints return 404 ────────────────────────────

class TestPreAnalysis:
    """
    Before any analysis is run the DB is empty.
    Every GET endpoint must return 404 (not 500) with a clear message.
    We clear the DB before each test to guarantee a clean state.
    """

    @pytest.fixture(autouse=True)
    def clear_results(self):
        """Wipe the DB before each pre-analysis test."""
        from engine.store import clear_db
        clear_db()

    def test_alerts_404_when_empty(self):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 404
        assert "No analysis results" in resp.json()["detail"]

    def test_summary_404_when_empty(self):
        resp = client.get("/api/v1/summary")
        assert resp.status_code == 404

    def test_mitre_404_when_empty(self):
        resp = client.get("/api/v1/mitre")
        assert resp.status_code == 404

    def test_graph_404_when_empty(self):
        resp = client.get("/api/v1/graph")
        assert resp.status_code == 404


# ── POST /analyze ─────────────────────────────────────────────────────

class TestAnalyzeEndpoint:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def run_analysis(cls):
        """Run the pipeline once for the whole test class."""
        resp = client.post("/api/v1/analyze?source=sample")
        assert resp.status_code == 200, f"Analysis failed: {resp.text}"
        cls._response = resp.json()

    def test_status_is_ok(self):
        assert self._response["status"] == "ok"

    def test_total_events_positive(self):
        assert self._response["total_events"] > 0

    def test_alerts_present(self):
        assert self._response["total_alerts"] >= 1

    def test_alerts_list_matches_total(self):
        assert len(self._response["alerts"]) == self._response["total_alerts"]

    def test_high_risk_count_non_negative(self):
        assert self._response["high_risk_count"] >= 0

    def test_analysed_at_present(self):
        assert "analysed_at" in self._response

    def test_alert_has_required_fields(self):
        alert = self._response["alerts"][0]
        required = {
            "alert_id", "title", "threat_type", "severity",
            "risk_score", "explanation", "mitigation",
            "source_ips", "dest_ips", "event_ids",
        }
        for field in required:
            assert field in alert, f"Missing field: {field}"

    def test_risk_score_has_all_components(self):
        rs = self._response["alerts"][0]["risk_score"]
        components = {
            "total", "severity_component", "anomaly_component",
            "mitre_stage_component", "frequency_component",
            "asset_criticality_component",
        }
        for c in components:
            assert c in rs, f"Missing risk_score component: {c}"

    def test_risk_score_total_in_range(self):
        for alert in self._response["alerts"]:
            score = alert["risk_score"]["total"]
            assert 0 <= score <= 100, f"Score out of range: {score}"

    def test_alerts_sorted_by_risk_descending(self):
        scores = [a["risk_score"]["total"] for a in self._response["alerts"]]
        assert scores == sorted(scores, reverse=True)

    def test_severity_values_valid(self):
        valid = {"info", "low", "medium", "high", "critical"}
        for alert in self._response["alerts"]:
            assert alert["severity"] in valid


# ── GET /alerts ───────────────────────────────────────────────────────

class TestAlertsEndpoint:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def ensure_results(cls):
        client.post("/api/v1/analyze?source=sample")

    def test_returns_200(self):
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200

    def test_total_matches_list_length(self):
        resp = client.get("/api/v1/alerts")
        data = resp.json()
        assert data["total"] == len(data["alerts"])

    def test_min_score_filter(self):
        resp = client.get("/api/v1/alerts?min_score=90")
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["risk_score"]["total"] >= 90

    def test_severity_filter(self):
        resp = client.get("/api/v1/alerts?severity=critical")
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["severity"] == "critical"

    def test_combined_filter(self):
        resp = client.get("/api/v1/alerts?min_score=50&severity=high")
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["severity"] == "high"
            assert alert["risk_score"]["total"] >= 50


# ── GET /summary ──────────────────────────────────────────────────────

class TestSummaryEndpoint:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def ensure_results(cls):
        client.post("/api/v1/analyze?source=sample")

    def test_returns_200(self):
        resp = client.get("/api/v1/summary")
        assert resp.status_code == 200

    def test_required_fields(self):
        data = client.get("/api/v1/summary").json()
        required = {
            "total_alerts", "high_risk_alerts", "critical_alerts",
            "unique_source_ips", "unique_dest_ips",
            "total_events", "mitre_tactic_count", "last_analysed_at",
        }
        for field in required:
            assert field in data, f"Missing: {field}"

    def test_total_alerts_positive(self):
        data = client.get("/api/v1/summary").json()
        assert data["total_alerts"] >= 1

    def test_total_events_positive(self):
        data = client.get("/api/v1/summary").json()
        assert data["total_events"] > 0

    def test_counts_non_negative(self):
        data = client.get("/api/v1/summary").json()
        for key in ("high_risk_alerts", "critical_alerts", "unique_source_ips",
                    "unique_dest_ips", "mitre_tactic_count"):
            assert data[key] >= 0, f"{key} should not be negative"


# ── GET /mitre ────────────────────────────────────────────────────────

class TestMITREEndpoint:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def ensure_results(cls):
        client.post("/api/v1/analyze?source=sample")

    def test_returns_200(self):
        assert client.get("/api/v1/mitre").status_code == 200

    def test_techniques_present(self):
        data = client.get("/api/v1/mitre").json()
        assert data["total_techniques"] >= 1
        assert len(data["techniques"]) == data["total_techniques"]

    def test_technique_fields(self):
        data = client.get("/api/v1/mitre").json()
        for t in data["techniques"]:
            assert "technique_id" in t
            assert "technique_name" in t
            assert "tactic" in t
            assert "tactic_order" in t

    def test_techniques_deduplicated(self):
        data = client.get("/api/v1/mitre").json()
        ids = [t["technique_id"] for t in data["techniques"]]
        assert len(ids) == len(set(ids)), "Duplicate technique IDs found"

    def test_sorted_by_tactic_order(self):
        data = client.get("/api/v1/mitre").json()
        orders = [t["tactic_order"] for t in data["techniques"]]
        assert orders == sorted(orders)


# ── GET /graph ────────────────────────────────────────────────────────

class TestGraphEndpoint:
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def ensure_results(cls):
        client.post("/api/v1/analyze?source=sample")

    def test_returns_200(self):
        assert client.get("/api/v1/graph").status_code == 200

    def test_node_edge_counts(self):
        data = client.get("/api/v1/graph").json()
        assert data["node_count"] == len(data["nodes"])
        assert data["edge_count"] == len(data["edges"])

    def test_nodes_have_required_fields(self):
        data = client.get("/api/v1/graph").json()
        assert data["node_count"] >= 1
        for node in data["nodes"]:
            assert "id" in node
            assert "risk_level" in node
            assert node["risk_level"] in ("high", "medium", "low")
            assert "alert_count" in node
            assert "alerts" in node

    def test_edges_have_required_fields(self):
        data = client.get("/api/v1/graph").json()
        for edge in data["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge
            assert edge["weight"] > 0
            assert "threat_types" in edge

    def test_no_self_loops(self):
        data = client.get("/api/v1/graph").json()
        for edge in data["edges"]:
            assert edge["source"] != edge["target"], "Self-loop found in graph"

    def test_graph_is_json_serializable(self):
        """Ensure nothing in the graph response breaks JSON serialization."""
        import json
        resp = client.get("/api/v1/graph")
        # If this doesn't raise, it's valid JSON
        json.loads(resp.text)
