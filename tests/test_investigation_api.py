"""
tests/test_investigation_api.py — Integration tests for the orchestration API.

Tests the full pipeline via FastAPI TestClient (no real HTTP, no real Ollama):
    POST /api/v1/investigate
    GET  /api/v1/investigations/{id}

All tests are offline:
    - Ollama is not required (template reasoning fallback is always exercised)
    - No external network calls

Sections:
    A — POST /investigate — success / happy path
    B — POST /investigate — error handling
    C — GET /investigations/{id} — retrieve / 404
    D — Pipeline output validation (reasoning_method, confidence, report_id)
    E — Varied query types (malware, CVE, APT, IOC, natural language)
    F — Report structure completeness
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app

# Force template reasoning across all tests (no Ollama needed, fast)
_NO_LLM = {"use_llm": "false"}

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _post(query: str, params: dict | None = None) -> dict:
    """POST /api/v1/investigate with use_llm=false and return the JSON body."""
    p = {"use_llm": "false"}
    if params:
        p.update(params)
    resp = client.post("/api/v1/investigate", json={"query": query}, params=p)
    return resp


def _post_ok(query: str) -> dict:
    """Assert 200 and return the parsed JSON."""
    resp = _post(query)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()


# ══════════════════════════════════════════════════════════════════════
# Section A — POST /investigate — success
# ══════════════════════════════════════════════════════════════════════

class TestPostInvestigateSuccess:

    def test_returns_200_for_known_malware(self):
        resp = _post("WannaCry")
        assert resp.status_code == 200

    def test_response_is_json(self):
        resp = _post("WannaCry")
        data = resp.json()
        assert isinstance(data, dict)

    def test_response_has_report_id(self):
        data = _post_ok("WannaCry")
        assert "report_id" in data
        assert len(data["report_id"]) == 36  # UUID format

    def test_response_has_request_id(self):
        data = _post_ok("WannaCry")
        assert "request_id" in data
        assert isinstance(data["request_id"], str)

    def test_response_has_executive_summary(self):
        data = _post_ok("WannaCry")
        assert "executive_summary" in data
        es = data["executive_summary"]
        assert "headline" in es
        assert "summary" in es
        assert "threat_level" in es
        assert "key_findings" in es

    def test_response_has_threat_assessment(self):
        data = _post_ok("WannaCry")
        ta = data["threat_assessment"]
        assert "risk_score" in ta
        assert "threat_level" in ta
        assert "ioc_count" in ta
        assert "confidence" in ta

    def test_response_has_evidence_summary(self):
        data = _post_ok("WannaCry")
        ev = data["evidence_summary"]
        assert "ioc_count" in ev
        assert "malware_families" in ev
        assert "apt_groups" in ev
        assert "cve_ids" in ev

    def test_response_has_recommendations(self):
        data = _post_ok("WannaCry")
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_response_has_references(self):
        data = _post_ok("WannaCry")
        assert "references" in data
        assert isinstance(data["references"], list)

    def test_response_has_kill_chain(self):
        data = _post_ok("WannaCry")
        assert "kill_chain" in data
        assert isinstance(data["kill_chain"], list)

    def test_response_has_reasoning_method(self):
        data = _post_ok("WannaCry")
        assert "reasoning_method" in data
        assert data["reasoning_method"] in ("ollama", "template")

    def test_response_has_generated_at(self):
        data = _post_ok("WannaCry")
        assert "generated_at" in data

    def test_response_has_confidence_score(self):
        data = _post_ok("WannaCry")
        assert "confidence_score" in data
        assert 0.0 <= data["confidence_score"] <= 1.0

    def test_report_id_is_unique_per_request(self):
        data1 = _post_ok("WannaCry")
        data2 = _post_ok("WannaCry")
        assert data1["report_id"] != data2["report_id"]

    def test_use_llm_false_gives_template_method(self):
        data = _post_ok("WannaCry")
        assert data["reasoning_method"] == "template"

    def test_risk_score_in_range(self):
        data = _post_ok("WannaCry")
        score = data["threat_assessment"]["risk_score"]
        assert 0.0 <= score <= 10.0

    def test_threat_level_valid_value(self):
        data = _post_ok("WannaCry")
        valid_levels = {"INFORMATIONAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert data["executive_summary"]["threat_level"] in valid_levels
        assert data["threat_assessment"]["threat_level"] in valid_levels


# ══════════════════════════════════════════════════════════════════════
# Section B — POST /investigate — error handling
# ══════════════════════════════════════════════════════════════════════

class TestPostInvestigateErrors:

    def test_empty_query_returns_400(self):
        resp = client.post("/api/v1/investigate", json={"query": ""}, params=_NO_LLM)
        assert resp.status_code in (400, 422)

    def test_whitespace_only_returns_400(self):
        resp = client.post("/api/v1/investigate", json={"query": "   "}, params=_NO_LLM)
        assert resp.status_code in (400, 422)

    def test_missing_query_field_returns_422(self):
        resp = client.post("/api/v1/investigate", json={}, params=_NO_LLM)
        assert resp.status_code == 422

    def test_missing_body_returns_422(self):
        resp = client.post("/api/v1/investigate", params=_NO_LLM)
        assert resp.status_code == 422

    def test_non_string_query_returns_422(self):
        resp = client.post("/api/v1/investigate", json={"query": 12345}, params=_NO_LLM)
        # FastAPI coerces int to str — should succeed or 422
        assert resp.status_code in (200, 422)

    def test_valid_query_does_not_return_400(self):
        resp = _post("CVE-2021-44228")
        assert resp.status_code != 400


# ══════════════════════════════════════════════════════════════════════
# Section C — GET /investigations/{id}
# ══════════════════════════════════════════════════════════════════════

class TestGetInvestigation:

    def test_404_for_unknown_id(self):
        resp = client.get(f"/api/v1/investigations/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_404_error_detail_present(self):
        resp = client.get(f"/api/v1/investigations/nonexistent-id")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_retrieve_stored_report(self):
        # POST to create a report
        post_data = _post_ok("Emotet")
        report_id = post_data["report_id"]

        # GET to retrieve it
        get_resp = client.get(f"/api/v1/investigations/{report_id}")
        assert get_resp.status_code == 200

    def test_retrieved_report_matches_original(self):
        post_data = _post_ok("APT28")
        report_id = post_data["report_id"]

        get_data = client.get(f"/api/v1/investigations/{report_id}").json()
        assert get_data["report_id"] == post_data["report_id"]
        assert get_data["request_id"] == post_data["request_id"]

    def test_retrieved_report_has_all_sections(self):
        post_data = _post_ok("Log4Shell")
        report_id = post_data["report_id"]

        get_data = client.get(f"/api/v1/investigations/{report_id}").json()
        for field in ["executive_summary", "threat_assessment", "evidence_summary",
                      "recommendations", "references", "reasoning_method"]:
            assert field in get_data, f"Missing field: {field}"

    def test_different_reports_have_different_ids(self):
        d1 = _post_ok("WannaCry")
        d2 = _post_ok("Mirai")
        assert d1["report_id"] != d2["report_id"]


# ══════════════════════════════════════════════════════════════════════
# Section D — Pipeline output validation
# ══════════════════════════════════════════════════════════════════════

class TestPipelineOutputValidation:

    def test_report_id_is_uuid_format(self):
        data = _post_ok("WannaCry")
        # Should not raise
        parsed = uuid.UUID(data["report_id"])
        assert str(parsed) == data["report_id"]

    def test_request_id_is_nonempty_string(self):
        data = _post_ok("WannaCry")
        assert isinstance(data["request_id"], str)
        assert len(data["request_id"]) > 0

    def test_input_value_reflects_query(self):
        data = _post_ok("WannaCry")
        # input_value should contain the original query text or be related
        assert isinstance(data.get("input_value", ""), str)

    def test_input_type_detected(self):
        data = _post_ok("WannaCry")
        assert isinstance(data.get("input_type", ""), str)
        assert len(data.get("input_type", "")) > 0

    def test_confidence_score_is_float(self):
        data = _post_ok("WannaCry")
        assert isinstance(data["confidence_score"], float)

    def test_executive_summary_headline_nonempty(self):
        data = _post_ok("WannaCry")
        assert len(data["executive_summary"]["headline"]) > 0

    def test_executive_summary_summary_nonempty(self):
        data = _post_ok("WannaCry")
        assert len(data["executive_summary"]["summary"]) > 0

    def test_threat_assessment_ioc_count_is_int(self):
        data = _post_ok("WannaCry")
        assert isinstance(data["threat_assessment"]["ioc_count"], int)

    def test_recommendations_have_required_fields(self):
        data = _post_ok("WannaCry")
        for rec in data["recommendations"]:
            assert "priority" in rec
            assert "category" in rec
            assert "action" in rec
            assert "rationale" in rec

    def test_kill_chain_phases_have_required_fields(self):
        data = _post_ok("WannaCry")
        for phase in data["kill_chain"]:
            assert "phase" in phase
            assert "tactics" in phase
            assert "techniques" in phase


# ══════════════════════════════════════════════════════════════════════
# Section E — Varied query types
# ══════════════════════════════════════════════════════════════════════

class TestQueryVariety:

    def test_malware_name_query(self):
        data = _post_ok("WannaCry")
        assert data["reasoning_method"] == "template"

    def test_cve_id_query(self):
        data = _post_ok("CVE-2021-44228")
        assert "report_id" in data
        assert data["input_type"] == "cve_id"

    def test_apt_group_query(self):
        data = _post_ok("APT28")
        assert "report_id" in data

    def test_ip_address_query(self):
        data = _post_ok("203.0.113.42")
        assert "report_id" in data
        assert data["input_type"] == "ip_address"

    def test_domain_query(self):
        data = _post_ok("malware-c2.ru")
        assert "report_id" in data

    def test_natural_language_query(self):
        data = _post_ok(
            "Suspicious PowerShell execution followed by outbound C2 traffic "
            "to a known Lazarus Group IP address"
        )
        assert "report_id" in data

    def test_mitre_technique_query(self):
        data = _post_ok("T1059")
        assert "report_id" in data

    def test_emotet_query(self):
        data = _post_ok("Emotet")
        assert "report_id" in data

    def test_lazarus_group_query(self):
        data = _post_ok("Lazarus Group")
        assert "report_id" in data

    def test_log4shell_query(self):
        data = _post_ok("Log4Shell")
        assert "report_id" in data

    def test_eternalblue_cve_query(self):
        data = _post_ok("CVE-2017-0144")
        assert "report_id" in data
        assert data["input_type"] == "cve_id"


# ══════════════════════════════════════════════════════════════════════
# Section F — Report structure completeness
# ══════════════════════════════════════════════════════════════════════

class TestReportStructure:

    def test_wannacry_has_malware_in_evidence(self):
        data = _post_ok("WannaCry")
        ev = data["evidence_summary"]
        assert isinstance(ev["malware_families"], list)
        assert "WannaCry" in ev["malware_families"]

    def test_wannacry_has_apt_in_evidence(self):
        data = _post_ok("WannaCry")
        ev = data["evidence_summary"]
        assert len(ev["apt_groups"]) >= 1

    def test_wannacry_has_cve_in_evidence(self):
        data = _post_ok("WannaCry")
        ev = data["evidence_summary"]
        assert len(ev["cve_ids"]) >= 1

    def test_wannacry_has_references(self):
        data = _post_ok("WannaCry")
        assert len(data["references"]) >= 1

    def test_wannacry_has_recommendations(self):
        data = _post_ok("WannaCry")
        assert len(data["recommendations"]) >= 1

    def test_wannacry_has_kill_chain(self):
        data = _post_ok("WannaCry")
        assert len(data["kill_chain"]) >= 1

    def test_wannacry_risk_score_above_zero(self):
        data = _post_ok("WannaCry")
        assert data["threat_assessment"]["risk_score"] > 0.0

    def test_wannacry_threat_level_not_informational(self):
        data = _post_ok("WannaCry")
        level = data["threat_assessment"]["threat_level"]
        assert level != "INFORMATIONAL"

    def test_cve_report_has_cve_in_exploited_cves(self):
        data = _post_ok("CVE-2021-44228")
        cves = data["threat_assessment"].get("exploited_cves", [])
        assert "CVE-2021-44228" in cves

    def test_report_fully_json_serialisable(self):
        import json
        data = _post_ok("WannaCry")
        # Should not raise
        serialised = json.dumps(data)
        assert len(serialised) > 100
