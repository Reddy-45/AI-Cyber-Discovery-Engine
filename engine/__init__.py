"""
engine/__init__.py — Public API for the engine package.

Downstream code (app.py, tests) imports from here.
"""

from engine.models import (
    AnalysisResult,
    CanonicalEvent,
    EnrichedEvent,
    EventType,
    IOCMatch,
    IOCType,
    MITRETechnique,
    RiskScore,
    Severity,
    ThreatAlert,
)
from engine.config import load_config
from engine.ingest import load_and_normalize
from engine.enrich import enrich_events
from engine.analyze import run_analysis
from engine.store import init_db, save_results, load_results

__all__ = [
    # Models
    "AnalysisResult", "CanonicalEvent", "EnrichedEvent",
    "EventType", "IOCMatch", "IOCType", "MITRETechnique",
    "RiskScore", "Severity", "ThreatAlert",
    # Config
    "load_config",
    # Pipeline stages
    "load_and_normalize", "enrich_events", "run_analysis",
    # Storage
    "init_db", "save_results", "load_results",
]
