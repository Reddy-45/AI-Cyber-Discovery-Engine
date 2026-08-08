# reasoning/__init__.py
"""AI Reasoning Layer — deterministic template + optional Ollama LLM reasoning.

Public API
----------
    from reasoning import build_report, ReportBuilder, LLMReasoner, InvestigationReport

Pipeline position:
    EnrichedInvestigationContext (semantic/retriever.py)
        ↓
    ReportBuilder.build(ctx)  or  build_report(ctx)
        → InvestigationReport
"""

from reasoning.llm_reasoner import LLMReasoner, OllamaClient, ReasoningOutput
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
from reasoning.report_builder import ReportBuilder, TemplateReasoner, build_report

__all__ = [
    # Top-level convenience
    "build_report",
    # Classes
    "ReportBuilder",
    "TemplateReasoner",
    "LLMReasoner",
    "OllamaClient",
    "ReasoningOutput",
    "PromptBuilder",
    # Pydantic models
    "InvestigationReport",
    "ExecutiveSummary",
    "ThreatAssessment",
    "EvidenceSummary",
    "KillChainPhase",
    "Recommendation",
    # Enums
    "ThreatLevel",
    "ReasoningMethod",
    "RecommendationPriority",
    "RecommendationCategory",
]
