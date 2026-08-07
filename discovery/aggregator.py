"""
discovery/aggregator.py — Knowledge Aggregation Orchestrator.

This is the top-level coordinator for the Knowledge Aggregation layer.

Responsibilities:
    1. Accept a normalized DiscoveryRequest
    2. Fan-out queries to all relevant providers in parallel-safe order
    3. Merge all ProviderResult objects
    4. Deduplicate across providers
    5. Build the relationship graph via KnowledgeCorrelator
    6. Compute a weighted confidence score
    7. Return one InvestigationContext

What this does NOT do:
    - No embeddings
    - No vector search
    - No LLM calls
    - No external network requests
    - Does NOT call engine/analyze.py (that happens in a later phase when
      the input contains structured log data)

InvestigationContext is the output contract for this phase.
It is consumed by:
    - FastAPI routes (api/discovery_routes.py)
    - Future embedding phase
    - Future LLM reasoning phase
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from discovery.correlator import KnowledgeCorrelator, RelationshipGraph
from discovery.models import DiscoveryRequest, InputType
from discovery.providers import (
    APTGroupProvider,
    APTMatch,
    BaseProvider,
    CVEMatch,
    CVEProvider,
    IOCIntelMatch,
    MalwareMatch,
    MalwareProvider,
    MITREProvider,
    ProviderResult,
    STIXProvider,
    ThreatIntelProvider,
)
from engine.models import MITRETechnique

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Output model: InvestigationContext
# ══════════════════════════════════════════════════════════════════════

class InvestigationContext(BaseModel):
    """
    The complete output of the Knowledge Aggregation layer.

    One InvestigationContext is produced per analyst query.
    It aggregates all knowledge found across all providers into a single,
    structured, JSON-serialisable object.

    Fields:
        request_id          — links back to the originating DiscoveryRequest
        input_type          — detected type of the analyst input
        query_summary       — human-readable one-line summary of what was queried
        matched_techniques  — ATT&CK techniques from any provider
        matched_malware     — malware families from any provider
        matched_cves        — CVE entries from any provider
        matched_apt_groups  — APT group profiles from any provider
        matched_iocs        — IOC intelligence hits
        relationship_graph  — serialisable entity relationship graph
        confidence_score    — weighted aggregate confidence (0.0 – 1.0)
        evidence            — list of supporting evidence strings
        provider_hits       — breakdown by provider name
        has_findings        — True if any provider returned results
        analysed_at         — timestamp
    """

    request_id:         str
    input_type:         str
    query_summary:      str                     = ""

    matched_techniques: list[MITRETechnique]   = Field(default_factory=list)
    matched_malware:    list[MalwareMatch]      = Field(default_factory=list)
    matched_cves:       list[CVEMatch]          = Field(default_factory=list)
    matched_apt_groups: list[APTMatch]          = Field(default_factory=list)
    matched_iocs:       list[IOCIntelMatch]     = Field(default_factory=list)

    relationship_graph: RelationshipGraph       = Field(default_factory=RelationshipGraph)

    confidence_score:   float                   = Field(ge=0.0, le=1.0, default=0.0)
    evidence:           list[str]               = Field(default_factory=list)
    provider_hits:      dict[str, int]          = Field(default_factory=dict)
    has_findings:       bool                    = False
    analysed_at:        datetime                = Field(
                            default_factory=lambda: datetime.now(tz=timezone.utc)
                        )


# ══════════════════════════════════════════════════════════════════════
# Aggregator
# ══════════════════════════════════════════════════════════════════════

# All providers instantiated once at module level — they are stateless
_ALL_PROVIDERS: list[BaseProvider] = [
    MITREProvider(),
    ThreatIntelProvider(),
    MalwareProvider(),
    APTGroupProvider(),
    CVEProvider(),
    STIXProvider(),
]


def aggregate(request: DiscoveryRequest) -> InvestigationContext:
    """
    Run all providers against a DiscoveryRequest and aggregate results.

    Args:
        request: A fully-normalized DiscoveryRequest from the Input Layer.

    Returns:
        InvestigationContext with all matched entities, graph, and confidence.
    """
    if not request.is_valid:
        logger.warning(
            "Aggregating invalid request %s: %s",
            request.request_id,
            request.validation_errors,
        )

    logger.info(
        "Aggregating request %s [%s] value='%s'",
        request.request_id,
        request.input_type.value,
        request.raw_value[:60],
    )

    # ── Fan-out: query every provider ─────────────────────────────────
    results: list[ProviderResult] = []
    for provider in _ALL_PROVIDERS:
        try:
            result = provider.query(request)
            results.append(result)
            logger.debug(
                "Provider %s: techs=%d malware=%d apts=%d cves=%d iocs=%d",
                provider.name,
                len(result.matched_techniques),
                len(result.matched_malware),
                len(result.matched_apt_groups),
                len(result.matched_cves),
                len(result.matched_iocs),
            )
        except Exception:
            logger.exception("Provider %s raised an exception", provider.name)
            # Never let a single provider failure abort the whole query
            results.append(ProviderResult(source=provider.name))

    # ── Merge: pool all results ────────────────────────────────────────
    merged = _merge_results(results)

    # ── Correlate: build the relationship graph ────────────────────────
    correlator = KnowledgeCorrelator()
    graph = correlator.correlate(results)

    # ── Provider hit counts ────────────────────────────────────────────
    provider_hits = {
        r.source: (
            len(r.matched_techniques)
            + len(r.matched_malware)
            + len(r.matched_apt_groups)
            + len(r.matched_cves)
            + len(r.matched_iocs)
        )
        for r in results
    }

    # ── Confidence ────────────────────────────────────────────────────
    confidence = _compute_confidence(merged)

    # ── Summary ───────────────────────────────────────────────────────
    summary = _build_summary(request, merged)

    has_findings = (
        bool(merged.matched_techniques)
        or bool(merged.matched_malware)
        or bool(merged.matched_cves)
        or bool(merged.matched_apt_groups)
        or bool(merged.matched_iocs)
    )

    context = InvestigationContext(
        request_id=request.request_id,
        input_type=request.input_type.value,
        query_summary=summary,
        matched_techniques=merged.matched_techniques,
        matched_malware=merged.matched_malware,
        matched_cves=merged.matched_cves,
        matched_apt_groups=merged.matched_apt_groups,
        matched_iocs=merged.matched_iocs,
        relationship_graph=graph,
        confidence_score=round(confidence, 3),
        evidence=merged.evidence,
        provider_hits=provider_hits,
        has_findings=has_findings,
    )

    logger.info(
        "Aggregation complete: techs=%d malware=%d apts=%d cves=%d iocs=%d "
        "nodes=%d edges=%d confidence=%.2f",
        len(context.matched_techniques),
        len(context.matched_malware),
        len(context.matched_apt_groups),
        len(context.matched_cves),
        len(context.matched_iocs),
        graph.node_count,
        graph.edge_count,
        confidence,
    )

    return context


# ══════════════════════════════════════════════════════════════════════
# Merge helpers
# ══════════════════════════════════════════════════════════════════════

def _merge_results(results: list[ProviderResult]) -> ProviderResult:
    """
    Merge all ProviderResult objects into one, deduplicating each entity type.
    """
    seen_techs:   set[str] = set()
    seen_malware: set[str] = set()
    seen_apts:    set[str] = set()
    seen_cves:    set[str] = set()
    seen_iocs:    set[str] = set()

    merged_techs:   list[MITRETechnique]  = []
    merged_malware: list[MalwareMatch]    = []
    merged_apts:    list[APTMatch]        = []
    merged_cves:    list[CVEMatch]        = []
    merged_iocs:    list[IOCIntelMatch]   = []
    all_evidence:   list[str]             = []

    for r in results:
        for t in r.matched_techniques:
            if t.technique_id not in seen_techs:
                seen_techs.add(t.technique_id)
                merged_techs.append(t)

        for m in r.matched_malware:
            key = m.name.lower()
            if key not in seen_malware:
                seen_malware.add(key)
                merged_malware.append(m)

        for a in r.matched_apt_groups:
            key = a.name.lower()
            if key not in seen_apts:
                seen_apts.add(key)
                merged_apts.append(a)

        for c in r.matched_cves:
            if c.cve_id not in seen_cves:
                seen_cves.add(c.cve_id)
                merged_cves.append(c)

        for i in r.matched_iocs:
            key = i.value.lower()
            if key not in seen_iocs:
                seen_iocs.add(key)
                merged_iocs.append(i)

        all_evidence.extend(r.evidence)

    # Sort by confidence descending within each list
    merged_techs.sort(key=lambda t: t.tactic_order)
    merged_malware.sort(key=lambda m: -m.confidence)
    merged_apts.sort(key=lambda a: -a.confidence)
    merged_cves.sort(key=lambda c: -c.cvss_score)
    merged_iocs.sort(key=lambda i: -i.confidence)

    return ProviderResult(
        source="merged",
        matched_techniques=merged_techs,
        matched_malware=merged_malware,
        matched_apt_groups=merged_apts,
        matched_cves=merged_cves,
        matched_iocs=merged_iocs,
        evidence=all_evidence,
    )


def _compute_confidence(merged: ProviderResult) -> float:
    """
    Compute a weighted aggregate confidence score.

    Weights reflect the reliability of each entity type's match:
    - IOC hits are most definitive (0.35 weight)
    - Malware exact matches are very reliable (0.25)
    - CVE exact matches are reliable (0.20)
    - APT attribution is somewhat reliable (0.15)
    - Technique keyword matches are least specific (0.05)

    The score represents: "how confident are we in the overall findings?"
    """
    if not any([
        merged.matched_iocs, merged.matched_malware,
        merged.matched_cves, merged.matched_apt_groups,
        merged.matched_techniques,
    ]):
        return 0.0

    scores: list[tuple[float, float]] = []   # (confidence, weight)

    if merged.matched_iocs:
        avg = sum(i.confidence for i in merged.matched_iocs) / len(merged.matched_iocs)
        scores.append((avg, 0.35))

    if merged.matched_malware:
        avg = sum(m.confidence for m in merged.matched_malware) / len(merged.matched_malware)
        scores.append((avg, 0.25))

    if merged.matched_cves:
        avg = sum(c.confidence for c in merged.matched_cves) / len(merged.matched_cves)
        scores.append((avg, 0.20))

    if merged.matched_apt_groups:
        avg = sum(a.confidence for a in merged.matched_apt_groups) / len(merged.matched_apt_groups)
        scores.append((avg, 0.15))

    if merged.matched_techniques:
        # Techniques are always confidence=1.0 in MITRETechnique model
        scores.append((0.80, 0.05))

    total_weight = sum(w for _, w in scores)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(conf * w for conf, w in scores)
    return weighted_sum / total_weight


def _build_summary(request: DiscoveryRequest, merged: ProviderResult) -> str:
    """Build a one-line summary of what the aggregator found."""
    parts: list[str] = []

    ioc_count  = len(merged.matched_iocs)
    mal_count  = len(merged.matched_malware)
    apt_count  = len(merged.matched_apt_groups)
    cve_count  = len(merged.matched_cves)
    tech_count = len(merged.matched_techniques)

    if ioc_count:
        hit = merged.matched_iocs[0]
        parts.append(
            f"{ioc_count} IOC hit{'s' if ioc_count > 1 else ''} "
            f"({'known bad' if hit.is_known_bad else 'low confidence'})"
        )
    if mal_count:
        names = ", ".join(m.name for m in merged.matched_malware[:3])
        parts.append(f"{mal_count} malware: {names}")
    if apt_count:
        names = ", ".join(a.name for a in merged.matched_apt_groups[:3])
        parts.append(f"{apt_count} APT group{'s' if apt_count > 1 else ''}: {names}")
    if cve_count:
        ids = ", ".join(c.cve_id for c in merged.matched_cves[:3])
        parts.append(f"{cve_count} CVE{'s' if cve_count > 1 else ''}: {ids}")
    if tech_count:
        parts.append(
            f"{tech_count} ATT&CK technique{'s' if tech_count > 1 else ''}"
        )

    if not parts:
        return (
            f"No knowledge base matches found for "
            f"{request.input_type.value}: '{request.raw_value[:60]}'"
        )

    return (
        f"Query '{request.raw_value[:40]}' [{request.input_type.value}]: "
        + "; ".join(parts)
    )
