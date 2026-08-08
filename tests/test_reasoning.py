"""
tests/test_reasoning.py — Comprehensive unit tests for the AI Reasoning Layer.

Test sections:
    A — reasoning/models.py         Pydantic model construction & serialisation
    B — reasoning/prompts.py        Prompt template rendering & dispatcher
    C — OllamaClient                HTTP interaction, availability check, generate
    D — LLMReasoner (template path) Offline fallback behaviour
    E — LLMReasoner (Ollama path)   Mocked Ollama success & mid-call failure
    F — TemplateReasoner            Deterministic field generation
    G — ReportBuilder               Full build() on both paths
    H — build_report()              Module-level convenience function
    I — Integration                 End-to-end with EnrichedInvestigationContext

All tests are offline — no real Ollama, no real embeddings, no network calls.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# ── Shared engine / discovery types ─────────────────────────────────────
from engine.models import MITRETechnique
from discovery.models import DiscoveryRequest, InputType
from discovery.providers import APTMatch, CVEMatch, IOCIntelMatch, MalwareMatch
from discovery.aggregator import InvestigationContext
from semantic.retriever import EnrichedInvestigationContext, SemanticSearchResults, SemanticHit

# ── Reasoning layer ──────────────────────────────────────────────────────
from reasoning.models import (
    EvidenceSummary,
    ExecutiveSummary,
    InvestigationReport,
    KillChainPhase,
    Recommendation,
    ReasoningMethod,
    RecommendationCategory,
    RecommendationPriority,
    ThreatAssessment,
    ThreatLevel,
)
from reasoning.prompts import PromptBuilder
from reasoning.llm_reasoner import (
    LLMReasoner,
    OllamaClient,
    ReasoningOutput,
    _extract_json,
    OLLAMA_BASE_URL,
    DEFAULT_MODEL,
)
from reasoning.report_builder import (
    ReportBuilder,
    TemplateReasoner,
    build_report,
    _compute_risk_score,
    _score_to_level,
)


# ══════════════════════════════════════════════════════════════════════
# Fixtures — shared test objects
# ══════════════════════════════════════════════════════════════════════

def _make_mitre(technique_id: str = "T1059", name: str = "Command Scripting",
                tactic: str = "execution", order: int = 4) -> MITRETechnique:
    return MITRETechnique(
        technique_id=technique_id,
        technique_name=name,
        tactic=tactic,
        tactic_order=order,
    )


def _make_malware(name: str = "WannaCry", family: str = "ransomware") -> MalwareMatch:
    return MalwareMatch(
        name=name,
        family_type=family,
        attributed_to=["Lazarus Group"],
        cves_exploited=["CVE-2017-0144"],
        kill_chain=["exploitation", "installation"],
        aliases=["WCry"],
        mitre_techniques=["T1486", "T1059"],
        description="WannaCry ransomware worm.",
        confidence=0.95,
    )


def _make_apt(name: str = "APT28") -> APTMatch:
    return APTMatch(
        name=name,
        country="Russia",
        mitre_group_id="G0007",
        targeted_sectors=["government", "energy"],
        malware_used=["Sofacy", "X-Agent"],
        mitre_techniques=["T1566", "T1059", "T1078"],
        description="Russian state-sponsored APT.",
        confidence=0.9,
    )


def _make_cve(cve_id: str = "CVE-2021-44228", cvss: float = 10.0,
              kev: bool = True) -> CVEMatch:
    return CVEMatch(
        cve_id=cve_id,
        name="Log4Shell",
        cvss_score=cvss,
        severity="critical",
        cisa_kev=kev,
        exploit_public=True,
        exploitation_status="Actively Exploited",
        affected_products=["Apache Log4j 2.x"],
        related_malware=["Mirai"],
        description="Remote code execution in Apache Log4j.",
        confidence=1.0,
    )


def _make_ioc(value: str = "203.0.113.42", ioc_type: str = "ip",
              is_bad: bool = True) -> IOCIntelMatch:
    return IOCIntelMatch(
        value=value,
        ioc_type=ioc_type,
        tags=["c2", "botnet"],
        malware_family="Mirai",
        confidence=0.9,
        source="abuse_ch",
        is_known_bad=is_bad,
    )


def _make_context(
    *,
    malware: list[MalwareMatch] | None = None,
    apts: list[APTMatch] | None = None,
    cves: list[CVEMatch] | None = None,
    iocs: list[IOCIntelMatch] | None = None,
    techniques: list[MITRETechnique] | None = None,
    confidence: float = 0.85,
    input_type: str = InputType.MALWARE_NAME.value,
) -> InvestigationContext:
    return InvestigationContext(
        request_id=str(uuid.uuid4()),
        input_type=input_type,
        query_summary="Test investigation context",
        matched_techniques=techniques or [],
        matched_malware=malware or [],
        matched_cves=cves or [],
        matched_apt_groups=apts or [],
        matched_iocs=iocs or [],
        confidence_score=confidence,
        evidence=["Evidence string 1", "Evidence string 2"],
        provider_hits={"MITREProvider": 2, "MalwareProvider": 1},
        has_findings=True,
    )


def _make_enriched(ctx: InvestigationContext | None = None) -> EnrichedInvestigationContext:
    """Wrap an InvestigationContext in an EnrichedInvestigationContext."""
    if ctx is None:
        ctx = _make_context(
            malware=[_make_malware()],
            apts=[_make_apt()],
            cves=[_make_cve()],
            iocs=[_make_ioc()],
            techniques=[_make_mitre()],
        )
    semantic = SemanticSearchResults(
        hits=[
            SemanticHit(
                doc_id="hit1",
                text="WannaCry ransomware uses EternalBlue exploit",
                score=0.91,
                collection="malware",
                entity_type="malware",
            )
        ],
        query_text="wannacry ransomware",
        total_hits=1,
        collections_searched=["malware"],
        top_score=0.91,
        has_results=True,
    )
    return EnrichedInvestigationContext(
        **ctx.model_dump(),
        semantic_results=semantic,
        pipeline_stages=["aggregation", "semantic"],
    )


# ══════════════════════════════════════════════════════════════════════
# Section A — reasoning/models.py
# ══════════════════════════════════════════════════════════════════════

class TestModels:
    """Section A: Pydantic model construction and serialisation."""

    def test_threat_level_values(self):
        levels = [ThreatLevel.INFORMATIONAL, ThreatLevel.LOW, ThreatLevel.MEDIUM,
                  ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        assert len(levels) == 5

    def test_recommendation_priority_enum(self):
        assert RecommendationPriority.IMMEDIATE.value == "IMMEDIATE"
        assert RecommendationPriority.SHORT_TERM.value == "SHORT_TERM"
        assert RecommendationPriority.LONG_TERM.value == "LONG_TERM"

    def test_recommendation_category_enum(self):
        cats = [RecommendationCategory.DETECTION, RecommendationCategory.MITIGATION,
                RecommendationCategory.RESPONSE, RecommendationCategory.HARDENING,
                RecommendationCategory.MONITORING]
        assert len(cats) == 5

    def test_executive_summary_defaults(self):
        es = ExecutiveSummary(
            headline="Test headline",
            summary="Test summary.",
        )
        assert es.threat_level == ThreatLevel.INFORMATIONAL
        assert es.key_findings == []

    def test_threat_assessment_bounds(self):
        ta = ThreatAssessment(risk_score=7.5, confidence=0.8)
        assert 0.0 <= ta.risk_score <= 10.0
        assert 0.0 <= ta.confidence <= 1.0

    def test_threat_assessment_invalid_risk_score(self):
        with pytest.raises(Exception):
            ThreatAssessment(risk_score=11.0)

    def test_investigation_report_auto_id(self):
        report = InvestigationReport(
            executive_summary=ExecutiveSummary(headline="h", summary="s"),
        )
        assert report.report_id != ""
        assert len(report.report_id) == 36  # UUID format

    def test_investigation_report_json_serialisable(self):
        report = InvestigationReport(
            executive_summary=ExecutiveSummary(
                headline="WannaCry detected",
                summary="Ransomware found.",
                key_findings=["Finding 1", "Finding 2"],
            ),
            threat_assessment=ThreatAssessment(risk_score=8.5, confidence=0.9),
            kill_chain=[KillChainPhase(phase="Exploitation", techniques=["T1059"])],
            recommendations=[Recommendation(
                action="Block IOC",
                rationale="Known bad",
                priority=RecommendationPriority.IMMEDIATE,
                category=RecommendationCategory.MITIGATION,
            )],
        )
        data = report.model_dump(mode="json")
        # Must be JSON-serialisable
        serialised = json.dumps(data)
        assert isinstance(serialised, str)
        # Round-trip
        loaded = json.loads(serialised)
        assert loaded["executive_summary"]["headline"] == "WannaCry detected"

    def test_reasoning_method_enum(self):
        assert ReasoningMethod.OLLAMA.value == "ollama"
        assert ReasoningMethod.TEMPLATE.value == "template"

    def test_evidence_summary_defaults(self):
        ev = EvidenceSummary()
        assert ev.ioc_count == 0
        assert ev.malware_families == []
        assert ev.semantic_hits == 0

    def test_kill_chain_phase_defaults(self):
        kcp = KillChainPhase(phase="Delivery")
        assert kcp.tactics == []
        assert kcp.techniques == []
        assert kcp.evidence == []


# ══════════════════════════════════════════════════════════════════════
# Section B — reasoning/prompts.py
# ══════════════════════════════════════════════════════════════════════

class TestPrompts:
    """Section B: Prompt template rendering and dispatcher."""

    def test_prompt_builder_returns_two_strings(self):
        ctx = _make_context(malware=[_make_malware()])
        system, user = PromptBuilder.build(ctx)
        assert isinstance(system, str) and len(system) > 50
        assert isinstance(user, str) and len(user) > 50

    def test_system_prompt_contains_json_instruction(self):
        ctx = _make_context()
        system, _ = PromptBuilder.build(ctx)
        assert "JSON" in system or "json" in system

    def test_ioc_preamble_for_ip_address(self):
        ctx = _make_context(input_type=InputType.IP_ADDRESS.value, iocs=[_make_ioc()])
        _, user = PromptBuilder.build(ctx)
        assert "IOC" in user or "indicator" in user.lower() or "ip" in user.lower()

    def test_cve_preamble_for_cve_id(self):
        ctx = _make_context(input_type=InputType.CVE_ID.value, cves=[_make_cve()])
        _, user = PromptBuilder.build(ctx)
        assert "CVE" in user or "vulnerability" in user.lower()

    def test_malware_preamble_for_malware_name(self):
        ctx = _make_context(input_type=InputType.MALWARE_NAME.value, malware=[_make_malware()])
        _, user = PromptBuilder.build(ctx)
        assert "Malware" in user or "malware" in user

    def test_apt_preamble_for_apt_group(self):
        ctx = _make_context(input_type=InputType.APT_GROUP.value, apts=[_make_apt()])
        _, user = PromptBuilder.build(ctx)
        assert "APT" in user or "threat actor" in user.lower()

    def test_incident_preamble_for_natural_language(self):
        ctx = _make_context(input_type=InputType.NATURAL_LANGUAGE.value)
        _, user = PromptBuilder.build(ctx)
        assert "Incident" in user or "investigation" in user.lower()

    def test_incident_preamble_for_stix(self):
        ctx = _make_context(input_type=InputType.STIX_BUNDLE.value)
        _, user = PromptBuilder.build(ctx)
        assert len(user) > 10

    def test_data_section_includes_technique_ids(self):
        ctx = _make_context(techniques=[_make_mitre("T1059", "Command Scripting")])
        _, user = PromptBuilder.build(ctx)
        assert "T1059" in user

    def test_data_section_includes_malware_name(self):
        ctx = _make_context(malware=[_make_malware("Emotet", "trojan")])
        _, user = PromptBuilder.build(ctx)
        assert "Emotet" in user

    def test_data_section_includes_cve_id(self):
        ctx = _make_context(cves=[_make_cve("CVE-2021-44228")])
        _, user = PromptBuilder.build(ctx)
        assert "CVE-2021-44228" in user

    def test_data_section_includes_apt_name(self):
        ctx = _make_context(apts=[_make_apt("APT28")])
        _, user = PromptBuilder.build(ctx)
        assert "APT28" in user

    def test_build_flat_is_single_string(self):
        ctx = _make_context()
        flat = PromptBuilder.build_flat(ctx)
        assert isinstance(flat, str)
        assert "SYSTEM" in flat
        assert "USER" in flat

    def test_empty_context_prompt_still_valid(self):
        ctx = _make_context()  # no entities
        system, user = PromptBuilder.build(ctx)
        assert len(system) > 0
        assert len(user) > 0


# ══════════════════════════════════════════════════════════════════════
# Section C — OllamaClient
# ══════════════════════════════════════════════════════════════════════

class TestOllamaClient:
    """Section C: HTTP interaction via OllamaClient."""

    def test_is_available_returns_false_on_connection_error(self):
        """Simulate a connection error without making a real network call."""
        client = OllamaClient()
        with patch("httpx.get", side_effect=Exception("connection refused")):
            assert client.is_available() is False

    def test_is_available_returns_true_with_mock(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "llama3.2"}]}
        with patch("httpx.get", return_value=mock_resp):
            assert client.is_available() is True

    def test_is_available_returns_false_no_models(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": []}
        with patch("httpx.get", return_value=mock_resp):
            assert client.is_available() is False

    def test_is_available_returns_false_on_http_error(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {}
        with patch("httpx.get", return_value=mock_resp):
            assert client.is_available() is False

    def test_generate_returns_text_on_success(self):
        client = OllamaClient()
        expected_json = {"headline": "Test", "threat_level": "HIGH"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": json.dumps(expected_json)}
        with patch("httpx.post", return_value=mock_resp):
            result = client.generate("system", "user")
        assert result is not None
        assert "headline" in result

    def test_generate_returns_none_on_http_error(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("httpx.post", return_value=mock_resp):
            result = client.generate("system", "user")
        assert result is None

    def test_generate_returns_none_on_exception(self):
        client = OllamaClient()
        with patch("httpx.post", side_effect=Exception("connection refused")):
            result = client.generate("system", "user")
        assert result is None

    def test_list_models_returns_empty_on_error(self):
        """Simulate a connection error without making a real network call."""
        client = OllamaClient()
        with patch("httpx.get", side_effect=Exception("connection refused")):
            models = client.list_models()
        assert isinstance(models, list)
        assert len(models) == 0

    def test_list_models_returns_names(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3.2"}, {"name": "mistral"}]
        }
        with patch("httpx.get", return_value=mock_resp):
            models = client.list_models()
        assert "llama3.2" in models
        assert "mistral" in models

    def test_best_available_model_prefers_llama32(self):
        client = OllamaClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "mistral"}, {"name": "llama3.2"}]
        }
        with patch("httpx.get", return_value=mock_resp):
            model = client.best_available_model()
        assert model.startswith("llama3.2")

    def test_best_available_model_falls_back_to_default(self):
        client = OllamaClient(model="my-model")
        with patch.object(client, "list_models", return_value=[]):
            model = client.best_available_model()
        assert model == "my-model"


# ══════════════════════════════════════════════════════════════════════
# Section D — LLMReasoner (template / fallback path)
# ══════════════════════════════════════════════════════════════════════

class TestLLMReasonerTemplatePath:
    """Section D: Offline fallback when Ollama is unavailable."""

    def _make_unavailable_client(self) -> OllamaClient:
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = False
        return client

    def test_reason_returns_template_when_unavailable(self):
        reasoner = LLMReasoner(client=self._make_unavailable_client())
        output = reasoner.reason("system", "user")
        assert output.method == "template"
        assert output.parsed is False

    def test_reason_template_error_message_set(self):
        reasoner = LLMReasoner(client=self._make_unavailable_client())
        output = reasoner.reason("system", "user")
        assert len(output.error) > 0

    def test_reason_returns_template_on_empty_generate(self):
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = True
        client.best_available_model.return_value = "llama3.2"
        client.generate.return_value = None  # empty response
        reasoner = LLMReasoner(client=client)
        output = reasoner.reason("system", "user")
        assert output.method == "template"
        assert output.parsed is False

    def test_reason_returns_template_on_invalid_json(self):
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = True
        client.best_available_model.return_value = "llama3.2"
        client.generate.return_value = "This is not JSON at all!!!"
        reasoner = LLMReasoner(client=client)
        output = reasoner.reason("system", "user")
        assert output.method == "template"
        assert output.parsed is False


# ══════════════════════════════════════════════════════════════════════
# Section E — LLMReasoner (Ollama success path)
# ══════════════════════════════════════════════════════════════════════

class TestLLMReasonerOllamaPath:
    """Section E: Mocked successful Ollama call."""

    def _make_good_llm_response(self) -> str:
        return json.dumps({
            "headline": "WannaCry ransomware detected — HIGH risk",
            "threat_level": "HIGH",
            "summary": "WannaCry exploits EternalBlue for lateral movement.",
            "key_findings": ["WannaCry detected", "EternalBlue exploit used"],
            "risk_score": 8.5,
            "threat_actors": ["Lazarus Group"],
            "attack_techniques": ["T1486: Data Encrypted for Impact"],
            "exploited_cves": ["CVE-2017-0144"],
            "kill_chain": [
                {"phase": "Exploitation", "tactics": ["execution"],
                 "techniques": ["T1059"], "evidence": ["cmd.exe spawned"]}
            ],
            "recommendations": [
                {"priority": "IMMEDIATE", "category": "MITIGATION",
                 "action": "Block EternalBlue SMB ports", "rationale": "Prevents spread"},
            ],
        })

    def _make_ollama_client(self, response: str) -> OllamaClient:
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = True
        client.best_available_model.return_value = "llama3.2"
        client.generate.return_value = response
        return client

    def test_reason_returns_ollama_on_valid_json(self):
        client = self._make_ollama_client(self._make_good_llm_response())
        reasoner = LLMReasoner(client=client)
        output = reasoner.reason("system", "user")
        assert output.method == "ollama"
        assert output.parsed is True
        assert output.model == "llama3.2"

    def test_reason_data_contains_expected_keys(self):
        client = self._make_ollama_client(self._make_good_llm_response())
        reasoner = LLMReasoner(client=client)
        output = reasoner.reason("system", "user")
        assert "headline" in output.data
        assert "threat_level" in output.data
        assert "risk_score" in output.data

    def test_reason_raw_text_stored(self):
        response = self._make_good_llm_response()
        client = self._make_ollama_client(response)
        reasoner = LLMReasoner(client=client)
        output = reasoner.reason("system", "user")
        assert output.raw_text == response

    def test_extract_json_plain(self):
        raw = '{"key": "value"}'
        data = _extract_json(raw)
        assert data["key"] == "value"

    def test_extract_json_with_markdown_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        data = _extract_json(raw)
        assert data["key"] == "value"

    def test_extract_json_with_leading_garbage(self):
        raw = 'Here is the result: {"key": "value"} end.'
        data = _extract_json(raw)
        assert data["key"] == "value"

    def test_extract_json_raises_on_no_json(self):
        with pytest.raises(ValueError):
            _extract_json("No JSON here at all.")

    def test_extract_json_raises_on_unclosed(self):
        with pytest.raises(ValueError):
            _extract_json('{"key": "value"')

    # ── Qwen3 <think> block stripping ─────────────────────────────────

    def test_extract_json_strips_think_block_before_json(self):
        """Qwen3 emits <think>…</think> before the JSON object."""
        raw = '<think>Let me reason about this...</think>\n{"key": "value"}'
        data = _extract_json(raw)
        assert data["key"] == "value"

    def test_extract_json_strips_think_block_after_json(self):
        """<think> block appearing after JSON should still parse correctly."""
        raw = '{"key": "value"}<think>extra reasoning</think>'
        data = _extract_json(raw)
        assert data["key"] == "value"

    def test_extract_json_strips_multiline_think_block(self):
        """Multi-line <think> blocks are stripped."""
        raw = '<think>\nStep 1: analyse...\nStep 2: conclude...\n</think>\n{"result": 42}'
        data = _extract_json(raw)
        assert data["result"] == 42

    def test_extract_json_strips_think_and_markdown_fence(self):
        """Both <think> and markdown fences present simultaneously."""
        raw = '<think>thinking...</think>\n```json\n{"key": "value"}\n```'
        data = _extract_json(raw)
        assert data["key"] == "value"

    def test_extract_json_think_block_case_insensitive(self):
        """<THINK> upper-case variant is also stripped."""
        raw = '<THINK>Reasoning...</THINK>{"key": "ok"}'
        data = _extract_json(raw)
        assert data["key"] == "ok"

    def test_extract_json_think_only_raises(self):
        """A response that is ONLY a <think> block (no JSON) must raise."""
        with pytest.raises(ValueError):
            _extract_json("<think>I could not produce any JSON.</think>")

    def test_extract_json_think_with_nested_json_content(self):
        """JSON-like text inside <think> is not extracted."""
        raw = '<think>{"fake": "json inside think"}</think>{"real": "json"}'
        data = _extract_json(raw)
        assert data["real"] == "json"
        assert "fake" not in data


# ══════════════════════════════════════════════════════════════════════
# Section F — TemplateReasoner
# ══════════════════════════════════════════════════════════════════════

class TestTemplateReasoner:
    """Section F: Deterministic field generation from InvestigationContext."""

    def setup_method(self):
        self.tr = TemplateReasoner()
        self.ctx = _make_context(
            malware=[_make_malware()],
            apts=[_make_apt()],
            cves=[_make_cve()],
            iocs=[_make_ioc()],
            techniques=[_make_mitre()],
        )

    def test_headline_contains_malware_name(self):
        level = ThreatLevel.HIGH
        headline = self.tr._build_headline(self.ctx, level)
        assert "WannaCry" in headline
        assert "HIGH" in headline

    def test_headline_contains_risk_level(self):
        headline = self.tr._build_headline(self.ctx, ThreatLevel.CRITICAL)
        assert "CRITICAL" in headline

    def test_summary_is_non_empty_string(self):
        summary = self.tr._build_summary(self.ctx, 8.5, ThreatLevel.HIGH)
        assert isinstance(summary, str)
        assert len(summary) > 20

    def test_summary_mentions_query(self):
        summary = self.tr._build_summary(self.ctx, 7.0, ThreatLevel.HIGH)
        assert "Test investigation context" in summary or "8.5" in summary or "7.0" in summary

    def test_key_findings_non_empty(self):
        findings = self.tr._build_key_findings(self.ctx)
        assert len(findings) >= 1

    def test_key_findings_mention_malware(self):
        findings = self.tr._build_key_findings(self.ctx)
        assert any("WannaCry" in f for f in findings)

    def test_key_findings_mention_cve(self):
        findings = self.tr._build_key_findings(self.ctx)
        assert any("CVE-2021-44228" in f for f in findings)

    def test_key_findings_mention_ioc(self):
        findings = self.tr._build_key_findings(self.ctx)
        assert any("203.0.113.42" in f for f in findings)

    def test_executive_summary_returns_model(self):
        exec_sum = self.tr.build_executive_summary(self.ctx, 8.5, ThreatLevel.HIGH)
        assert isinstance(exec_sum, ExecutiveSummary)
        assert exec_sum.headline != ""
        assert exec_sum.summary != ""

    def test_threat_assessment_risk_score_in_range(self):
        ta = self.tr.build_threat_assessment(self.ctx, 7.0, ThreatLevel.HIGH)
        assert 0.0 <= ta.risk_score <= 10.0

    def test_threat_assessment_includes_apt_name(self):
        ta = self.tr.build_threat_assessment(self.ctx, 7.0, ThreatLevel.HIGH)
        assert "APT28" in ta.threat_actors

    def test_threat_assessment_includes_cve(self):
        ta = self.tr.build_threat_assessment(self.ctx, 7.0, ThreatLevel.HIGH)
        assert "CVE-2021-44228" in ta.exploited_cves

    def test_threat_assessment_ioc_count(self):
        ta = self.tr.build_threat_assessment(self.ctx, 7.0, ThreatLevel.HIGH)
        assert ta.ioc_count == 1

    def test_evidence_summary_populated(self):
        ev = self.tr.build_evidence_summary(self.ctx)
        assert ev.ioc_count == 1
        assert "WannaCry" in ev.malware_families
        assert "APT28" in ev.apt_groups
        assert "CVE-2021-44228" in ev.cve_ids

    def test_evidence_summary_techniques(self):
        ev = self.tr.build_evidence_summary(self.ctx)
        assert len(ev.techniques) >= 1
        assert any("T1059" in t for t in ev.techniques)

    def test_kill_chain_from_techniques(self):
        kc = self.tr.build_kill_chain(self.ctx)
        assert len(kc) >= 1
        phases = [p.phase for p in kc]
        assert any(p in ("Exploitation", "Installation", "Command & Control") for p in phases)

    def test_kill_chain_technique_ids_present(self):
        kc = self.tr.build_kill_chain(self.ctx)
        all_techs = [tid for p in kc for tid in p.techniques]
        assert "T1059" in all_techs

    def test_kill_chain_empty_context_returns_empty_or_malware(self):
        ctx_empty = _make_context()
        kc = self.tr.build_kill_chain(ctx_empty)
        assert isinstance(kc, list)

    def test_recommendations_non_empty(self):
        recs = self.tr.build_recommendations(self.ctx)
        assert len(recs) >= 1

    def test_recommendations_include_cisa_kev_patch(self):
        recs = self.tr.build_recommendations(self.ctx)
        immediate = [r for r in recs if r.priority == RecommendationPriority.IMMEDIATE]
        assert len(immediate) >= 1
        assert any("CVE-2021-44228" in r.action for r in immediate)

    def test_recommendations_include_ioc_block(self):
        recs = self.tr.build_recommendations(self.ctx)
        assert any("203.0.113.42" in r.action for r in recs)

    def test_recommendations_include_mfa(self):
        recs = self.tr.build_recommendations(self.ctx)
        assert any("multi-factor" in r.action.lower() or "MFA" in r.action for r in recs)

    def test_recommendations_include_ransomware_isolation(self):
        recs = self.tr.build_recommendations(self.ctx)
        assert any("ransomware" in r.action.lower() or "Isolate" in r.action for r in recs)

    def test_references_include_nvd_link(self):
        refs = self.tr.build_references(self.ctx)
        assert any("nvd.nist.gov" in r for r in refs)

    def test_references_include_mitre_link(self):
        refs = self.tr.build_references(self.ctx)
        assert any("attack.mitre.org" in r for r in refs)

    def test_references_include_cisa_kev(self):
        refs = self.tr.build_references(self.ctx)
        assert any("cisa.gov" in r for r in refs)

    def test_references_deduplicated(self):
        refs = self.tr.build_references(self.ctx)
        assert len(refs) == len(set(refs))


# ══════════════════════════════════════════════════════════════════════
# Section G — ReportBuilder
# ══════════════════════════════════════════════════════════════════════

class TestReportBuilder:
    """Section G: Full build() on both LLM and template paths."""

    def setup_method(self):
        self.ctx = _make_context(
            malware=[_make_malware()],
            apts=[_make_apt()],
            cves=[_make_cve()],
            iocs=[_make_ioc()],
            techniques=[_make_mitre()],
        )

    def _make_unavailable_builder(self) -> ReportBuilder:
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = False
        reasoner = LLMReasoner(client=client)
        return ReportBuilder(llm_reasoner=reasoner)

    def test_build_returns_investigation_report(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert isinstance(report, InvestigationReport)

    def test_build_template_path_reasoning_method(self):
        builder = ReportBuilder(use_llm=False)
        report = builder.build(self.ctx)
        assert report.reasoning_method == ReasoningMethod.TEMPLATE

    def test_build_report_id_is_uuid(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert len(report.report_id) == 36

    def test_build_request_id_matches_context(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert report.request_id == self.ctx.request_id

    def test_build_confidence_score_propagated(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert report.confidence_score == self.ctx.confidence_score

    def test_build_executive_summary_non_empty(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert report.executive_summary.headline != ""
        assert report.executive_summary.summary != ""

    def test_build_threat_level_reflects_risk(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        # WannaCry + CISA KEV + APT28 + IOC → should be >= MEDIUM
        assert report.threat_assessment.threat_level in (
            ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL
        )

    def test_build_risk_score_in_range(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert 0.0 <= report.threat_assessment.risk_score <= 10.0

    def test_build_kill_chain_non_empty(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert len(report.kill_chain) >= 1

    def test_build_recommendations_non_empty(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert len(report.recommendations) >= 1

    def test_build_references_non_empty(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        assert len(report.references) >= 1

    def test_build_with_llm_response(self):
        llm_data = {
            "headline": "LLM Headline",
            "threat_level": "CRITICAL",
            "summary": "LLM generated summary.",
            "key_findings": ["LLM finding 1"],
            "risk_score": 9.5,
            "threat_actors": ["LLM Actor"],
            "attack_techniques": ["T1059: Scripting"],
            "exploited_cves": ["CVE-2021-44228"],
            "kill_chain": [
                {"phase": "Exploitation", "tactics": ["execution"],
                 "techniques": ["T1059"], "evidence": ["evidence"]}
            ],
            "recommendations": [
                {"priority": "IMMEDIATE", "category": "RESPONSE",
                 "action": "LLM Recommended Action", "rationale": "because risk"}
            ],
        }
        client = MagicMock(spec=OllamaClient)
        client.is_available.return_value = True
        client.best_available_model.return_value = "llama3.2"
        client.generate.return_value = json.dumps(llm_data)
        llm_reasoner = LLMReasoner(client=client)
        builder = ReportBuilder(llm_reasoner=llm_reasoner)
        report = builder.build(self.ctx)
        # LLM data should influence the report
        assert report.reasoning_method == ReasoningMethod.OLLAMA
        assert report.llm_model == "llama3.2"
        assert "LLM Headline" in report.executive_summary.headline

    def test_build_enriched_context(self):
        enriched = _make_enriched(self.ctx)
        builder = self._make_unavailable_builder()
        report = builder.build(enriched)
        assert isinstance(report, InvestigationReport)
        # Semantic hits should be reflected in evidence summary
        assert report.evidence_summary.semantic_hits >= 0

    def test_build_empty_context(self):
        ctx_empty = _make_context(confidence=0.0)
        builder = ReportBuilder(use_llm=False)
        report = builder.build(ctx_empty)
        assert isinstance(report, InvestigationReport)
        assert report.threat_assessment.threat_level == ThreatLevel.INFORMATIONAL

    def test_build_report_fully_json_serialisable(self):
        builder = self._make_unavailable_builder()
        report = builder.build(self.ctx)
        data = report.model_dump(mode="json")
        serialised = json.dumps(data)
        assert len(serialised) > 100


# ══════════════════════════════════════════════════════════════════════
# Section H — build_report() module-level convenience
# ══════════════════════════════════════════════════════════════════════

class TestBuildReport:
    """Section H: Module-level convenience function."""

    def test_build_report_no_llm(self):
        ctx = _make_context(malware=[_make_malware()], cves=[_make_cve()])
        report = build_report(ctx, use_llm=False)
        assert isinstance(report, InvestigationReport)
        assert report.reasoning_method == ReasoningMethod.TEMPLATE

    def test_build_report_default_uses_ollama_attempt(self):
        """build_report(ctx) should attempt Ollama but fall back gracefully.
        We mock httpx.get so there is no real network call and no timeout wait.
        """
        ctx = _make_context()
        # Simulate Ollama unreachable (connection refused) without a real network call
        with patch("httpx.get", side_effect=Exception("connection refused")):
            report = build_report(ctx, use_llm=True)
        assert isinstance(report, InvestigationReport)
        # Template fallback must produce a complete report
        assert report.executive_summary.headline != ""
        assert report.reasoning_method == ReasoningMethod.TEMPLATE

    def test_build_report_with_enriched_context(self):
        enriched = _make_enriched()
        report = build_report(enriched, use_llm=False)
        assert isinstance(report, InvestigationReport)


# ══════════════════════════════════════════════════════════════════════
# Section I — Risk scoring helpers
# ══════════════════════════════════════════════════════════════════════

class TestRiskScoring:
    """Section I: Risk score computation and level mapping."""

    def test_compute_risk_score_zero_for_empty(self):
        ctx = _make_context(confidence=0.0)
        score = _compute_risk_score(ctx)
        assert score == 0.0

    def test_compute_risk_score_cisa_kev_adds_bonus(self):
        ctx_kev = _make_context(cves=[_make_cve(kev=True)], confidence=1.0)
        ctx_no_kev = _make_context(cves=[_make_cve(kev=False)], confidence=1.0)
        assert _compute_risk_score(ctx_kev) >= _compute_risk_score(ctx_no_kev)

    def test_compute_risk_score_capped_at_10(self):
        ctx = _make_context(
            malware=[_make_malware()] * 3,
            apts=[_make_apt()] * 5,
            cves=[_make_cve()] * 5,
            iocs=[_make_ioc()] * 5,
            confidence=1.0,
        )
        score = _compute_risk_score(ctx)
        assert score <= 10.0

    def test_compute_risk_score_malware_floor(self):
        ctx = _make_context(malware=[_make_malware()], confidence=0.1)
        score = _compute_risk_score(ctx)
        assert score >= 3.0

    def test_score_to_level_critical(self):
        assert _score_to_level(9.5) == ThreatLevel.CRITICAL

    def test_score_to_level_high(self):
        assert _score_to_level(7.5) == ThreatLevel.HIGH

    def test_score_to_level_medium(self):
        assert _score_to_level(5.5) == ThreatLevel.MEDIUM

    def test_score_to_level_low(self):
        assert _score_to_level(3.0) == ThreatLevel.LOW

    def test_score_to_level_informational(self):
        assert _score_to_level(0.0) == ThreatLevel.INFORMATIONAL

    def test_score_to_level_boundary_9(self):
        assert _score_to_level(9.0) == ThreatLevel.CRITICAL

    def test_score_to_level_boundary_7(self):
        assert _score_to_level(7.0) == ThreatLevel.HIGH

    def test_score_to_level_boundary_5(self):
        assert _score_to_level(5.0) == ThreatLevel.MEDIUM

    def test_score_to_level_boundary_2(self):
        assert _score_to_level(2.0) == ThreatLevel.LOW


# ══════════════════════════════════════════════════════════════════════
# Section J — Package-level imports from reasoning/__init__.py
# ══════════════════════════════════════════════════════════════════════

class TestPackageImports:
    """Section J: Verify all public symbols are importable from reasoning."""

    def test_import_build_report(self):
        from reasoning import build_report as br
        assert callable(br)

    def test_import_report_builder(self):
        from reasoning import ReportBuilder as RB
        assert RB is ReportBuilder

    def test_import_llm_reasoner(self):
        from reasoning import LLMReasoner as LR
        assert LR is LLMReasoner

    def test_import_investigation_report(self):
        from reasoning import InvestigationReport as IR
        assert IR is InvestigationReport

    def test_import_threat_level(self):
        from reasoning import ThreatLevel as TL
        assert TL is ThreatLevel

    def test_import_prompt_builder(self):
        from reasoning import PromptBuilder as PB
        assert PB is PromptBuilder

    def test_import_template_reasoner(self):
        from reasoning import TemplateReasoner as TR
        assert TR is TemplateReasoner


# ══════════════════════════════════════════════════════════════════════
# Section K — Qwen3 model preference
# ══════════════════════════════════════════════════════════════════════

class TestQwen3ModelPreference:
    """Section K: Verify qwen3:8b is prioritised in best_available_model()."""

    def _client_with_models(self, models: list[str]) -> OllamaClient:
        client = OllamaClient()
        with patch.object(client, "list_models", return_value=models):
            best = client.best_available_model()
        return best

    def test_prefers_qwen3_8b_over_llama(self):
        models = ["llama3.2:latest", "qwen3:8b", "mistral"]
        best = self._client_with_models(models)
        assert best == "qwen3:8b"

    def test_prefers_qwen3_8b_over_mistral(self):
        models = ["mistral:latest", "qwen3:8b"]
        best = self._client_with_models(models)
        assert best == "qwen3:8b"

    def test_prefers_qwen3_variant_over_llama(self):
        """Any qwen3 variant beats llama3.2."""
        models = ["llama3.2:latest", "qwen3:14b"]
        best = self._client_with_models(models)
        assert best.startswith("qwen3")

    def test_falls_back_to_llama_when_no_qwen3(self):
        models = ["llama3.2:latest", "mistral"]
        best = self._client_with_models(models)
        assert best.startswith("llama3.2")

    def test_falls_back_to_mistral_when_no_qwen3_or_llama(self):
        models = ["mistral:latest", "gemma2:2b"]
        best = self._client_with_models(models)
        assert best.startswith("mistral")

    def test_falls_back_to_default_when_no_models(self):
        client = OllamaClient()
        with patch.object(client, "list_models", return_value=[]):
            best = client.best_available_model()
        assert best == DEFAULT_MODEL

    def test_default_model_is_qwen3_8b(self):
        assert DEFAULT_MODEL == "qwen3:8b"

    def test_best_available_model_prefers_llama32(self):
        """Backward-compat alias — now qwen3:8b wins over llama3.2."""
        client = OllamaClient()
        with patch.object(client, "list_models", return_value=["llama3.2", "qwen3:8b"]):
            best = client.best_available_model()
        assert best == "qwen3:8b"

    def test_best_available_model_falls_back_to_default(self):
        client = OllamaClient()
        with patch.object(client, "list_models", return_value=[]):
            best = client.best_available_model()
        assert best == DEFAULT_MODEL
