"""
api/investigation_routes.py — Full pipeline orchestration endpoint.

Exposes:
    POST /api/v1/investigate           → run complete pipeline, return report
    GET  /api/v1/investigations/{id}   → retrieve a stored report by report_id

Pipeline (orchestration only — no logic duplicated here):
    1. normalize_text(query)      → DiscoveryRequest          [discovery]
    2. investigate(request)       → EnrichedInvestigationContext [semantic]
       └─ internally calls aggregate() [knowledge aggregation]
    3. build_report(ctx, use_llm) → InvestigationReport       [reasoning]

Reports are held in a module-level in-memory store for the lifetime of the
server process. This is sufficient for the current scope.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from discovery.normalizer import normalize_text
from semantic.retriever import investigate
from reasoning.report_builder import build_report

logger = logging.getLogger(__name__)
router = APIRouter()

# ── In-memory report store ─────────────────────────────────────────────
# dict[report_id → serialised report dict]
# Storing the dict rather than the model avoids re-serialisation on GET.
_report_store: dict[str, dict[str, Any]] = {}


# ── Request schema ─────────────────────────────────────────────────────

class InvestigateRequest(BaseModel):
    """Body schema for POST /investigate."""
    query: str = Field(
        description=(
            "Any analyst input: IP, domain, URL, hash, CVE, malware name, "
            "APT group, MITRE technique, or natural-language description."
        ),
        min_length=1,
        max_length=10_000,
        examples=["WannaCry", "CVE-2021-44228", "APT28", "203.0.113.42"],
    )


# ── POST /investigate ──────────────────────────────────────────────────

@router.post(
    "/investigate",
    summary="Run the complete AI investigation pipeline",
    description=(
        "Accepts any analyst input, runs it through the full AI pipeline:\n\n"
        "1. **Normalize** — detect and canonicalize the input type\n"
        "2. **Aggregate** — query the knowledge base (MITRE, malware, CVEs, APT groups, IOCs)\n"
        "3. **Semantic** — vector similarity search across the threat intel corpus\n"
        "4. **Reason** — generate a structured `InvestigationReport` "
        "(via Ollama if available, template fallback otherwise)\n\n"
        "The returned `InvestigationReport` contains an executive summary, "
        "threat assessment, kill chain, evidence summary, and prioritised recommendations."
    ),
    responses={
        200: {"description": "Investigation complete — InvestigationReport returned"},
        400: {"description": "Query is empty or invalid"},
        500: {"description": "Pipeline error"},
    },
)
def post_investigate(
    body: InvestigateRequest,
    use_llm: bool = Query(
        default=True,
        description=(
            "If true (default), attempt Ollama LLM reasoning and fall back to "
            "template automatically. Set false to force offline template reasoning."
        ),
    ),
) -> dict[str, Any]:
    """
    Orchestrate the full discovery → aggregation → semantic → reasoning pipeline.
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        # Step 1: Normalize input
        request = normalize_text(query)

        # Steps 2 + 3: Aggregate knowledge + semantic enrichment (single call)
        enriched_ctx = investigate(request)

        # Step 4: AI reasoning → InvestigationReport
        report = build_report(enriched_ctx, use_llm=use_llm)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Investigation pipeline failed for query %r", query)
        raise HTTPException(
            status_code=500,
            detail=f"Investigation pipeline error: {exc}",
        ) from exc

    # Serialise once, store, return
    report_data = report.model_dump(mode="json")
    _report_store[report.report_id] = report_data

    logger.info(
        "Investigation complete: report_id=%s query=%r method=%s risk=%.1f/10",
        report.report_id,
        query,
        report_data.get("reasoning_method"),
        report_data.get("threat_assessment", {}).get("risk_score", 0.0),
    )

    return report_data


# ── GET /investigations/{id} ───────────────────────────────────────────

@router.get(
    "/investigations/{report_id}",
    summary="Retrieve a completed investigation report by ID",
    description=(
        "Returns the cached `InvestigationReport` for a previous call to "
        "`POST /investigate`. Reports are stored in memory for the lifetime "
        "of the server process. Returns 404 if the report_id is not found."
    ),
    responses={
        200: {"description": "Report found and returned"},
        404: {"description": "No report found with this ID"},
    },
)
def get_investigation(report_id: str) -> dict[str, Any]:
    """Return a stored InvestigationReport by its report_id."""
    report_data = _report_store.get(report_id)
    if report_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No investigation report found with id '{report_id}'.",
        )
    return report_data
