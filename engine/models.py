"""
engine/models.py — All data models for the AI Cyber Discovery Engine.

Every pipeline stage reads and writes these types.
Pydantic v2 ensures bad data fails loudly at construction,
not silently at demo time.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────

class EventType(str, Enum):
    AUTH        = "auth"         # Login / logout / authentication failures
    NETWORK     = "network"      # Connections, scans, traffic anomalies
    MALWARE     = "malware"      # Malware detections, suspicious executables
    POLICY      = "policy"       # Policy violations, unauthorized access
    SYSTEM      = "system"       # Process creation, file changes
    UNKNOWN     = "unknown"


class Severity(str, Enum):
    INFO     = "info"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> float:
        """Numeric weight used in risk scoring (0.0 – 1.0)."""
        return {
            Severity.INFO:     0.0,
            Severity.LOW:      0.25,
            Severity.MEDIUM:   0.50,
            Severity.HIGH:     0.75,
            Severity.CRITICAL: 1.0,
        }[self]


class IOCType(str, Enum):
    IP_ADDRESS  = "ip_address"
    DOMAIN      = "domain"
    FILE_HASH   = "file_hash"
    URL         = "url"
    EMAIL       = "email"
    CVE         = "cve"


# ── Core Event Models ─────────────────────────────────────────────────

class CanonicalEvent(BaseModel):
    """
    The normalized representation of any security event.

    Produced by ingest.py from raw log JSON. Every downstream
    stage operates exclusively on this type.
    """
    event_id:    str       = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:   datetime
    source_ip:   str | None = None
    dest_ip:     str | None = None
    source_port: int | None = None
    dest_port:   int | None = None
    protocol:    str | None = None
    event_type:  EventType  = EventType.UNKNOWN
    severity:    Severity   = Severity.INFO
    description: str        = ""
    raw:         str        = ""          # Original log line — preserved for audit
    metadata:    dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}       # Events are immutable after creation


class IOCMatch(BaseModel):
    """A single Indicator of Compromise found in an event."""
    ioc_type:   IOCType
    value:      str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    is_known_bad: bool = False            # True if found in threat intel list


class MITRETechnique(BaseModel):
    """A MITRE ATT&CK technique associated with a threat."""
    technique_id:   str           # e.g. "T1110"
    technique_name: str           # e.g. "Brute Force"
    tactic:         str           # e.g. "Credential Access"
    tactic_order:   int = 0       # Kill-chain position (lower = earlier in attack)


class EnrichedEvent(BaseModel):
    """
    A CanonicalEvent augmented with threat intelligence context.
    Produced by enrich.py.
    """
    event:              CanonicalEvent
    iocs:               list[IOCMatch]        = Field(default_factory=list)
    mitre_techniques:   list[MITRETechnique]  = Field(default_factory=list)
    reputation_score:   float                 = Field(ge=0.0, le=1.0, default=0.5)
    # 0.0 = known malicious, 1.0 = known clean, 0.5 = unknown


# ── AI Engine Output Models ───────────────────────────────────────────

class RiskScore(BaseModel):
    """
    Transparent, decomposed risk score.

    Every factor's contribution is stored separately so the
    dashboard can show judges exactly why a score is what it is.
    """
    total:                    float = Field(ge=0.0, le=100.0)
    severity_component:       float = 0.0
    anomaly_component:        float = 0.0
    mitre_stage_component:    float = 0.0
    frequency_component:      float = 0.0
    asset_criticality_component: float = 0.0


class ThreatAlert(BaseModel):
    """
    Primary output of the AI Reasoning Engine.

    One alert represents one correlated attack pattern —
    with risk score, MITRE mapping, plain-English explanation,
    and concrete mitigation steps.
    """
    alert_id:         str       = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at:       datetime  = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    title:            str
    severity:         Severity
    risk_score:       RiskScore
    event_ids:        list[str]           = Field(default_factory=list)
    source_ips:       list[str]           = Field(default_factory=list)
    dest_ips:         list[str]           = Field(default_factory=list)
    iocs:             list[IOCMatch]      = Field(default_factory=list)
    mitre_techniques: list[MITRETechnique] = Field(default_factory=list)
    anomaly_score:    float               = Field(ge=0.0, le=1.0, default=0.0)
    explanation:      str                 = ""   # Stage 5: plain English
    mitigation:       list[str]           = Field(default_factory=list)  # Stage 6
    threat_type:      str                 = ""   # e.g. "Brute Force", "Port Scan"


class AnalysisResult(BaseModel):
    """Container for everything produced by analyze.py."""
    alerts:       list[ThreatAlert]   = Field(default_factory=list)
    total_events: int                 = 0
    high_risk_count: int              = 0
    analysed_at:  datetime            = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
