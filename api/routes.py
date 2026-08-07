"""
api/routes.py — All route handlers for the AI Cyber Discovery Engine API.

Every handler:
    1. Calls existing engine functions (never duplicates their logic)
    2. Converts engine output to API schemas
    3. Returns JSON-serializable Pydantic response models

The only "logic" in this file is data-shape translation —
mapping engine model fields to API schema fields.

Attack graph assembly (GET /graph) is treated as presentation logic,
not business logic — it transforms already-computed alert data into
a graph topology format. The actual correlation was done by analyze.py.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
import tempfile

from engine.config import load_config
from engine.ingest import load_and_normalize
from engine.enrich import enrich_events
from engine.analyze import run_analysis
from engine.store import init_db, save_results, load_results

from api.schemas import (
    AlertOut,
    AlertsResponse,
    AnalyzeResponse,
    GraphEdge,
    GraphNode,
    GraphResponse,
    IOCOut,
    MITREResponse,
    MITRETechniqueOut,
    RiskScoreOut,
    SummaryResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

init_db()  # Ensure DB exists when routes are loaded


# ── Converters ────────────────────────────────────────────────────────

def _alert_to_out(alert) -> AlertOut:
    """Convert a ThreatAlert engine model to an AlertOut API schema."""
    return AlertOut(
        alert_id=alert.alert_id,
        created_at=alert.created_at,
        title=alert.title,
        threat_type=alert.threat_type,
        severity=alert.severity.value,
        risk_score=RiskScoreOut(
            total=alert.risk_score.total,
            severity_component=alert.risk_score.severity_component,
            anomaly_component=alert.risk_score.anomaly_component,
            mitre_stage_component=alert.risk_score.mitre_stage_component,
            frequency_component=alert.risk_score.frequency_component,
            asset_criticality_component=alert.risk_score.asset_criticality_component,
        ),
        anomaly_score=alert.anomaly_score,
        source_ips=alert.source_ips,
        dest_ips=alert.dest_ips,
        event_ids=alert.event_ids,
        iocs=[
            IOCOut(
                ioc_type=ioc.ioc_type.value,
                value=ioc.value,
                confidence=ioc.confidence,
                is_known_bad=ioc.is_known_bad,
            )
            for ioc in alert.iocs
        ],
        mitre_techniques=[
            MITRETechniqueOut(
                technique_id=t.technique_id,
                technique_name=t.technique_name,
                tactic=t.tactic,
                tactic_order=t.tactic_order,
            )
            for t in alert.mitre_techniques
        ],
        explanation=alert.explanation,
        mitigation=alert.mitigation,
    )


def _load_or_404():
    """Load results from the DB; raise 404 if no analysis has been run yet."""
    result = load_results()
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No analysis results found. Run POST /analyze first.",
        )
    return result


# ── POST /analyze ─────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Run the full AI analysis pipeline",
    description=(
        "Runs Ingest → Enrich → AI Analyze on the selected log source. "
        "Pass `source=sample` to analyse the built-in sample logs, or upload "
        "a JSON log file. Results are persisted in SQLite and available via GET /alerts."
    ),
)
async def analyze(
    source: str = Query(
        default="sample",
        description="Log source: 'sample' uses built-in demo logs",
    ),
    file: UploadFile | None = File(
        default=None,
        description="Optional JSON log file upload (list of log event objects)",
    ),
) -> AnalyzeResponse:
    """
    Run the complete six-stage AI pipeline.

    - source=sample (default): reads data/sample_logs/
    - file upload: saves to a temp dir and reads from there
    """
    cfg = load_config()

    try:
        if file is not None:
            # Write uploaded file to a temp directory and run pipeline on it
            contents = await file.read()
            with tempfile.TemporaryDirectory() as tmp_dir:
                upload_path = Path(tmp_dir) / (file.filename or "uploaded_logs.json")
                upload_path.write_bytes(contents)
                events = load_and_normalize(tmp_dir)
                enriched = enrich_events(events)
                result = run_analysis(enriched)
        else:
            # Use built-in sample logs
            log_dir = Path(cfg["data"].get("sample_log_path", "data/sample_logs"))
            events = load_and_normalize(log_dir)
            enriched = enrich_events(events)
            result = run_analysis(enriched)

        save_results(result)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    return AnalyzeResponse(
        status="ok",
        total_events=result.total_events,
        total_alerts=len(result.alerts),
        high_risk_count=result.high_risk_count,
        analysed_at=result.analysed_at,
        alerts=[_alert_to_out(a) for a in result.alerts],
    )


# ── GET /alerts ───────────────────────────────────────────────────────

@router.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="Return all stored threat alerts",
    description="Returns every alert from the most recent analysis run, sorted by risk score descending.",
)
def get_alerts(
    min_score: float = Query(default=0.0, ge=0.0, le=100.0,
                             description="Filter: only return alerts with risk_score.total >= this value"),
    severity: str | None = Query(default=None,
                                  description="Filter: 'critical', 'high', 'medium', 'low', 'info'"),
) -> AlertsResponse:
    result = _load_or_404()
    alerts = result.alerts

    if min_score > 0:
        alerts = [a for a in alerts if a.risk_score.total >= min_score]
    if severity:
        alerts = [a for a in alerts if a.severity.value == severity.lower()]

    return AlertsResponse(
        total=len(alerts),
        alerts=[_alert_to_out(a) for a in alerts],
    )


# ── GET /summary ──────────────────────────────────────────────────────

@router.get(
    "/summary",
    response_model=SummaryResponse,
    summary="Return executive summary of the threat landscape",
    description="Aggregated statistics across all alerts — suitable for an executive dashboard header.",
)
def get_summary() -> SummaryResponse:
    result = _load_or_404()
    alerts = result.alerts

    all_source_ips: set[str] = set()
    all_dest_ips: set[str] = set()
    threat_type_counts: dict[str, int] = defaultdict(int)
    all_mitre_ids: set[str] = set()
    critical_count = 0

    for alert in alerts:
        all_source_ips.update(alert.source_ips)
        all_dest_ips.update(alert.dest_ips)
        threat_type_counts[alert.threat_type] += 1
        for t in alert.mitre_techniques:
            all_mitre_ids.add(t.technique_id)
        if alert.severity.value == "critical":
            critical_count += 1

    top_threat = max(threat_type_counts, key=threat_type_counts.get) if threat_type_counts else None
    top_score = alerts[0].risk_score.total if alerts else None

    return SummaryResponse(
        total_alerts=len(alerts),
        high_risk_alerts=result.high_risk_count,
        critical_alerts=critical_count,
        unique_source_ips=len(all_source_ips),
        unique_dest_ips=len(all_dest_ips),
        top_threat_type=top_threat,
        top_risk_score=top_score,
        last_analysed_at=result.analysed_at,
        total_events=result.total_events,
        mitre_tactic_count=len(all_mitre_ids),
    )


# ── GET /mitre ────────────────────────────────────────────────────────

@router.get(
    "/mitre",
    response_model=MITREResponse,
    summary="Return all MITRE ATT&CK techniques observed across alerts",
    description=(
        "Aggregates and deduplicates every ATT&CK technique mapped by the AI engine "
        "across all stored alerts. Sorted by kill-chain phase (tactic_order)."
    ),
)
def get_mitre() -> MITREResponse:
    result = _load_or_404()

    seen: dict[str, MITRETechniqueOut] = {}
    for alert in result.alerts:
        for t in alert.mitre_techniques:
            if t.technique_id not in seen:
                seen[t.technique_id] = MITRETechniqueOut(
                    technique_id=t.technique_id,
                    technique_name=t.technique_name,
                    tactic=t.tactic,
                    tactic_order=t.tactic_order,
                )

    techniques = sorted(seen.values(), key=lambda t: t.tactic_order)

    return MITREResponse(
        total_techniques=len(techniques),
        techniques=techniques,
    )


# ── GET /graph ────────────────────────────────────────────────────────

@router.get(
    "/graph",
    response_model=GraphResponse,
    summary="Return attack graph topology as JSON",
    description=(
        "Constructs a node/edge graph from the stored alert data. "
        "Each node is an IP entity. Each edge is a relationship between "
        "two IPs that appeared together in one or more alerts. "
        "No Plotly or HTML — pure JSON for any frontend to render."
    ),
)
def get_graph() -> GraphResponse:
    result = _load_or_404()

    # Build nodes — one per unique IP across all alerts
    node_map: dict[str, dict] = {}  # ip → {risk_level, alert_count, alerts}

    for alert in result.alerts:
        # Determine risk level label for this alert
        score = alert.risk_score.total
        risk_level = "high" if score >= 70 else ("medium" if score >= 40 else "low")
        all_ips = list(set(alert.source_ips + alert.dest_ips))

        for ip in all_ips:
            if ip not in node_map:
                node_map[ip] = {"risk_level": risk_level, "alert_count": 0, "alerts": []}
            # Escalate risk level if a higher-risk alert touches this node
            existing = node_map[ip]["risk_level"]
            if risk_level == "high" or (risk_level == "medium" and existing == "low"):
                node_map[ip]["risk_level"] = risk_level
            node_map[ip]["alert_count"] += 1
            if alert.title not in node_map[ip]["alerts"]:
                node_map[ip]["alerts"].append(alert.title)

    nodes = [
        GraphNode(
            id=ip,
            risk_level=data["risk_level"],
            alert_count=data["alert_count"],
            alerts=data["alerts"],
        )
        for ip, data in node_map.items()
    ]

    # Build edges — one per (source_ip, dest_ip) pair per alert
    edge_map: dict[tuple[str, str], dict] = {}

    for alert in result.alerts:
        for src in alert.source_ips:
            for dst in alert.dest_ips:
                if src == dst:
                    continue
                key = (src, dst)
                if key not in edge_map:
                    edge_map[key] = {"weight": 0.0, "threat_types": []}
                edge_map[key]["weight"] = round(
                    edge_map[key]["weight"] + alert.risk_score.total, 2
                )
                if alert.threat_type not in edge_map[key]["threat_types"]:
                    edge_map[key]["threat_types"].append(alert.threat_type)

    edges = [
        GraphEdge(
            source=src,
            target=dst,
            weight=data["weight"],
            threat_types=data["threat_types"],
        )
        for (src, dst), data in edge_map.items()
    ]

    return GraphResponse(
        node_count=len(nodes),
        edge_count=len(edges),
        nodes=nodes,
        edges=edges,
    )
