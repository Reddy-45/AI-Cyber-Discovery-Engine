"""
api/schemas.py — API-facing Pydantic response models.

These are SEPARATE from the engine's internal models (engine/models.py).

Why separate schemas?
    The engine models are designed for internal pipeline use —
    frozen CanonicalEvents, nested EnrichedEvents, etc.
    The API schemas are flat, clean, and optimised for JSON consumers
    (Lovable frontend, Postman, external dashboards).

    This layer also means the engine's internal model structure can
    evolve without breaking the API contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Shared sub-schemas ────────────────────────────────────────────────

class IOCOut(BaseModel):
    ioc_type:     str
    value:        str
    confidence:   float
    is_known_bad: bool


class MITRETechniqueOut(BaseModel):
    technique_id:   str
    technique_name: str
    tactic:         str
    tactic_order:   int


class RiskScoreOut(BaseModel):
    total:                       float
    severity_component:          float
    anomaly_component:           float
    mitre_stage_component:       float
    frequency_component:         float
    asset_criticality_component: float


# ── Alert response ────────────────────────────────────────────────────

class AlertOut(BaseModel):
    """Single threat alert — returned by GET /alerts and POST /analyze."""
    alert_id:         str
    created_at:       datetime
    title:            str
    threat_type:      str
    severity:         str
    risk_score:       RiskScoreOut
    anomaly_score:    float
    source_ips:       list[str]
    dest_ips:         list[str]
    event_ids:        list[str]
    iocs:             list[IOCOut]
    mitre_techniques: list[MITRETechniqueOut]
    explanation:      str
    mitigation:       list[str]


# ── Endpoint response wrappers ────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    """Response for POST /analyze."""
    status:       str = "ok"
    total_events: int
    total_alerts: int
    high_risk_count: int
    analysed_at:  datetime
    alerts:       list[AlertOut]


class AlertsResponse(BaseModel):
    """Response for GET /alerts."""
    total: int
    alerts: list[AlertOut]


class SummaryResponse(BaseModel):
    """Response for GET /summary — executive overview."""
    total_alerts:       int
    high_risk_alerts:   int
    critical_alerts:    int
    unique_source_ips:  int
    unique_dest_ips:    int
    top_threat_type:    str | None
    top_risk_score:     float | None
    last_analysed_at:   datetime | None
    total_events:       int
    mitre_tactic_count: int


class MITREResponse(BaseModel):
    """Response for GET /mitre — all observed ATT&CK techniques."""
    total_techniques: int
    techniques: list[MITRETechniqueOut]


# ── Graph schemas ─────────────────────────────────────────────────────

class GraphNode(BaseModel):
    """A network entity (IP address) in the attack graph."""
    id:          str       # IP address — used as unique node identifier
    risk_level:  str       # "high", "medium", "low" — derived from connected alerts
    alert_count: int
    alerts:      list[str] # Alert titles this node appears in


class GraphEdge(BaseModel):
    """A directed relationship between two entities."""
    source:       str        # Source IP
    target:       str        # Destination IP
    weight:       float      # Combined risk score of all alerts on this edge
    threat_types: list[str]  # Alert titles on this edge


class GraphResponse(BaseModel):
    """Response for GET /graph — topology data for visualization."""
    node_count: int
    edge_count: int
    nodes:      list[GraphNode]
    edges:      list[GraphEdge]
