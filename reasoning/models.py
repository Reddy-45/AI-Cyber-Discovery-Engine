"""
reasoning/models.py — Pydantic models for the AI Reasoning Layer.

These are the output contracts produced by the Reasoning Layer.

Pipeline position:
    InvestigationContext  (discovery/aggregator.py)
    ↓ enriched by semantic/retriever.py → EnrichedInvestigationContext
    ↓ reasoned over by reasoning/report_builder.py
    → InvestigationReport  ← this module's output type

All models are Pydantic v2, fully JSON-serialisable (model_dump(mode='json')).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════

class ThreatLevel(str, Enum):
    """Overall severity of the investigation finding."""
    INFORMATIONAL = "INFORMATIONAL"
    LOW           = "LOW"
    MEDIUM        = "MEDIUM"
    HIGH          = "HIGH"
    CRITICAL      = "CRITICAL"


class RecommendationPriority(str, Enum):
    IMMEDIATE  = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    LONG_TERM  = "LONG_TERM"


class RecommendationCategory(str, Enum):
    DETECTION   = "DETECTION"
    MITIGATION  = "MITIGATION"
    RESPONSE    = "RESPONSE"
    HARDENING   = "HARDENING"
    MONITORING  = "MONITORING"


class ReasoningMethod(str, Enum):
    """Which reasoning engine produced this report."""
    OLLAMA   = "ollama"
    TEMPLATE = "template"


# ══════════════════════════════════════════════════════════════════════
# Sub-models
# ══════════════════════════════════════════════════════════════════════

class ExecutiveSummary(BaseModel):
    """
    High-level, human-readable overview of the investigation result.

    Designed for a non-technical stakeholder audience.
    headline    — one sentence, e.g. "WannaCry ransomware detected, HIGH risk"
    threat_level — severity classification
    summary     — 2-5 sentence paragraph
    key_findings — bullet-point list of the most important discoveries
    """
    headline:     str
    threat_level: ThreatLevel   = ThreatLevel.INFORMATIONAL
    summary:      str
    key_findings: list[str]     = Field(default_factory=list)


class ThreatAssessment(BaseModel):
    """
    Technical scoring of the threat posed by this investigation.

    risk_score        — 0.0-10.0, same scale as CVSS
    threat_actors     — known APT groups or threat actor names
    attack_techniques — MITRE ATT&CK technique IDs
    exploited_cves    — CVE identifiers exploited in this campaign
    ioc_count         — how many malicious indicators were found
    confidence        — 0.0-1.0 overall confidence in the assessment
    """
    risk_score:        float           = Field(ge=0.0, le=10.0, default=0.0)
    threat_level:      ThreatLevel     = ThreatLevel.INFORMATIONAL
    threat_actors:     list[str]       = Field(default_factory=list)
    attack_techniques: list[str]       = Field(default_factory=list)
    exploited_cves:    list[str]       = Field(default_factory=list)
    ioc_count:         int             = 0
    confidence:        float           = Field(ge=0.0, le=1.0, default=0.0)


class EvidenceSummary(BaseModel):
    """
    Structured enumeration of all evidence found.

    Covers every entity type returned by the Knowledge Aggregation and
    Semantic layers so the report is self-contained.
    """
    ioc_count:       int         = 0
    malware_families: list[str]  = Field(default_factory=list)
    apt_groups:      list[str]   = Field(default_factory=list)
    cve_ids:         list[str]   = Field(default_factory=list)
    techniques:      list[str]   = Field(default_factory=list)
    semantic_hits:   int         = 0
    semantic_top_score: float    = 0.0
    sources:         list[str]   = Field(default_factory=list)


class KillChainPhase(BaseModel):
    """
    One phase of the Cyber Kill Chain or MITRE ATT&CK kill chain.

    phase      — kill chain phase name (e.g. "Exploitation")
    tactics    — MITRE tactic(s) in this phase
    techniques — technique IDs observed in this phase
    evidence   — supporting evidence strings for this phase
    """
    phase:      str
    tactics:    list[str]  = Field(default_factory=list)
    techniques: list[str]  = Field(default_factory=list)
    evidence:   list[str]  = Field(default_factory=list)


class Recommendation(BaseModel):
    """
    A single actionable security recommendation.

    priority — how urgently this should be addressed
    category — type of action (DETECTION, MITIGATION, etc.)
    action   — imperative sentence describing what to do
    rationale — why this is recommended (references findings)
    """
    priority:  RecommendationPriority  = RecommendationPriority.SHORT_TERM
    category:  RecommendationCategory  = RecommendationCategory.MITIGATION
    action:    str
    rationale: str


# ══════════════════════════════════════════════════════════════════════
# Top-level report
# ══════════════════════════════════════════════════════════════════════

class InvestigationReport(BaseModel):
    """
    Structured investigation report produced by the AI Reasoning Layer.

    This is the final output of the complete pipeline:
        Input → Normalise → Aggregate → Semantic → Reason → InvestigationReport

    report_id          — unique UUID for this report
    request_id         — links back to the originating DiscoveryRequest
    input_value        — the raw analyst query
    input_type         — detected input type
    executive_summary  — non-technical overview
    threat_assessment  — technical risk scoring
    evidence_summary   — all discovered entities
    kill_chain         — ordered attack phases observed
    recommendations    — prioritised action list
    references         — source strings and evidence links
    reasoning_method   — which engine produced the narrative ("ollama" or "template")
    llm_model          — Ollama model name if used, else ""
    generated_at       — UTC timestamp
    confidence_score   — overall pipeline confidence (from InvestigationContext)
    """
    report_id:          str             = Field(
                            default_factory=lambda: str(uuid.uuid4())
                        )
    request_id:         str             = ""
    input_value:        str             = ""
    input_type:         str             = ""

    executive_summary:  ExecutiveSummary
    threat_assessment:  ThreatAssessment = Field(default_factory=ThreatAssessment)
    evidence_summary:   EvidenceSummary  = Field(default_factory=EvidenceSummary)
    kill_chain:         list[KillChainPhase]   = Field(default_factory=list)
    recommendations:    list[Recommendation]   = Field(default_factory=list)
    references:         list[str]              = Field(default_factory=list)

    reasoning_method:   ReasoningMethod = ReasoningMethod.TEMPLATE
    llm_model:          str             = ""
    generated_at:       datetime        = Field(
                            default_factory=lambda: datetime.now(tz=timezone.utc)
                        )
    confidence_score:   float           = Field(ge=0.0, le=1.0, default=0.0)
