"""
semantic/retriever.py — Semantic Retrieval Pipeline.

This is the top-level entry point for the Semantic Layer.

Two modes of operation:
    1. retrieve_only(request) → SemanticSearchResults
       Pure semantic search — no knowledge aggregation.
       Use when you want raw vector similarity results.

    2. investigate(request) → EnrichedInvestigationContext
       Full discovery pipeline:
           Phase 1: Knowledge Aggregation  (discovery/aggregator.py)
           Phase 2: Semantic Retrieval     (this module)
       The semantic results ENRICH the aggregated InvestigationContext —
       they are stored alongside it, not replacing it.

EnrichedInvestigationContext:
    Extends InvestigationContext (from aggregator.py) with a
    `semantic_results` field. The aggregator is never modified.

Routing strategy:
    The query's input_type determines which collections are searched first:
    - IOC types  → threat_intel + all
    - CVE        → cve + mitre
    - Malware    → malware + apt
    - APT        → apt + malware
    - MITRE      → mitre + apt + malware
    - NL / report → all collections (broad search)
    - STIX/JSON  → all collections
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from discovery.aggregator import InvestigationContext, aggregate
from discovery.models import DiscoveryRequest, InputType
from semantic.embedding import EmbeddingEngine
from semantic.vector_store import (
    KnowledgeVectorStore,
    SearchHit,
    COL_MITRE,
    COL_MALWARE,
    COL_CVE,
    COL_APT,
    COL_THREAT_INTEL,
    ALL_COLLECTIONS,
)

logger = logging.getLogger(__name__)

# Top-K default — sensible for a SOC analyst dashboard
DEFAULT_K = 5


# ══════════════════════════════════════════════════════════════════════
# Result models
# ══════════════════════════════════════════════════════════════════════

class SemanticHit(BaseModel):
    """
    A single result from a vector similarity search.

    score is cosine similarity [0, 1] — higher = more semantically similar.
    """
    doc_id:     str
    text:       str
    score:      float                       # cosine similarity, 0-1
    collection: str                         # which ChromaDB collection
    entity_type: str                        = ""   # mitre | malware | cve | apt | threat_intel
    metadata:   dict[str, Any]             = Field(default_factory=dict)


class SemanticSearchResults(BaseModel):
    """
    Complete output of the semantic retrieval phase.

    hits        — top-K results, sorted by score descending
    query_text  — the text that was embedded (for debugging/display)
    total_hits  — total results found (may be > len(hits) before slicing)
    collections_searched — which collections were queried
    """
    hits:                  list[SemanticHit]   = Field(default_factory=list)
    query_text:            str                 = ""
    total_hits:            int                 = 0
    collections_searched:  list[str]           = Field(default_factory=list)
    top_score:             float               = 0.0
    has_results:           bool               = False


# ══════════════════════════════════════════════════════════════════════
# EnrichedInvestigationContext — extends aggregation with semantic
# ══════════════════════════════════════════════════════════════════════

class EnrichedInvestigationContext(InvestigationContext):
    """
    InvestigationContext enriched with semantic retrieval results.

    This class extends (not replaces) InvestigationContext so all
    existing fields remain intact and the aggregator is never touched.

    New fields:
        semantic_results — hits from the vector similarity search
        pipeline_stages  — which stages ran ("aggregation", "semantic")
    """
    semantic_results: SemanticSearchResults = Field(
        default_factory=SemanticSearchResults
    )
    pipeline_stages: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════
# Routing map
# ══════════════════════════════════════════════════════════════════════

# Maps input_type → ordered list of collections to search (most relevant first)
_ROUTING: dict[str, list[str]] = {
    InputType.IP_ADDRESS.value:      [COL_THREAT_INTEL, COL_MALWARE, COL_APT],
    InputType.DOMAIN.value:          [COL_THREAT_INTEL, COL_MALWARE],
    InputType.URL.value:             [COL_THREAT_INTEL, COL_MALWARE],
    InputType.FILE_HASH.value:       [COL_THREAT_INTEL, COL_MALWARE],
    InputType.CVE_ID.value:          [COL_CVE, COL_MITRE, COL_MALWARE, COL_APT],
    InputType.MALWARE_NAME.value:    [COL_MALWARE, COL_APT, COL_CVE, COL_MITRE],
    InputType.APT_GROUP.value:       [COL_APT, COL_MALWARE, COL_CVE, COL_MITRE],
    InputType.MITRE_TECHNIQUE.value: [COL_MITRE, COL_APT, COL_MALWARE],
    InputType.NATURAL_LANGUAGE.value: ALL_COLLECTIONS,
    InputType.JSON_FILE.value:       ALL_COLLECTIONS,
    InputType.STIX_BUNDLE.value:     ALL_COLLECTIONS,
    InputType.CYBER_REPORT.value:    ALL_COLLECTIONS,
}


# ══════════════════════════════════════════════════════════════════════
# SemanticRetriever
# ══════════════════════════════════════════════════════════════════════

class SemanticRetriever:
    """
    Semantic search across the knowledge base vector store.

    Injecting `engine` and `store` makes this fully testable without
    real model downloads or disk I/O.

    Example (production):
        retriever = SemanticRetriever()             # uses defaults
        results   = retriever.retrieve(request)    # SemanticSearchResults
        context   = retriever.investigate(request) # EnrichedInvestigationContext

    Example (test):
        retriever = SemanticRetriever(engine=mock_engine, store=mock_store)
    """

    def __init__(
        self,
        engine: EmbeddingEngine | None = None,
        store: KnowledgeVectorStore | None = None,
    ) -> None:
        self._engine = engine
        self._store  = store

    @property
    def engine(self) -> EmbeddingEngine:
        """Lazy-load the embedding engine."""
        if self._engine is None:
            self._engine = EmbeddingEngine.get()
        return self._engine

    @property
    def store(self) -> KnowledgeVectorStore:
        """Lazy-load the vector store (creates persistent client)."""
        if self._store is None:
            self._store = KnowledgeVectorStore()
        return self._store

    def _ensure_indexed(self) -> None:
        """Index the knowledge base if not already indexed."""
        if self.store.total_documents() == 0:
            logger.info("Vector store is empty — indexing knowledge base…")
            self.store.index_knowledge_base(self.engine)

    def _query_text_for_request(self, request: DiscoveryRequest) -> str:
        """
        Build the text string that will be embedded for a given request.

        For most types: use the normalized value + metadata context.
        For NL / reports: use the full normalized value.
        """
        base = request.normalized_value.strip()

        if request.input_type == InputType.CVE_ID:
            return f"CVE {base} vulnerability"

        elif request.input_type == InputType.MITRE_TECHNIQUE:
            return f"MITRE ATT&CK technique {base} {request.metadata.get('id_type', '')}"

        elif request.input_type == InputType.MALWARE_NAME:
            return f"malware {base} ransomware trojan threat"

        elif request.input_type == InputType.APT_GROUP:
            return f"APT threat actor {base} nation state cyber espionage"

        elif request.input_type == InputType.IP_ADDRESS:
            tags = request.metadata.get("tags", [])
            return f"malicious IP address {base} {' '.join(tags) if tags else 'threat intel'}"

        elif request.input_type == InputType.DOMAIN:
            tld = request.metadata.get("tld", "")
            return f"malicious domain {base} {tld} threat"

        elif request.input_type == InputType.FILE_HASH:
            algo = request.metadata.get("algorithm", "hash")
            return f"malware {algo} hash {base}"

        elif request.input_type in (InputType.NATURAL_LANGUAGE, InputType.CYBER_REPORT):
            # Full text — truncate to 512 tokens worth (~400 words)
            words = base.split()
            return " ".join(words[:400])

        elif request.input_type == InputType.STIX_BUNDLE:
            return "STIX threat intelligence bundle indicator malware"

        else:
            return base[:500]

    def retrieve(
        self,
        request: DiscoveryRequest,
        k: int = DEFAULT_K,
    ) -> SemanticSearchResults:
        """
        Run semantic search for a DiscoveryRequest.

        Args:
            request: Normalized DiscoveryRequest from the Input Layer.
            k:       Number of results to return.

        Returns:
            SemanticSearchResults with ranked hits and metadata.
        """
        self._ensure_indexed()

        query_text = self._query_text_for_request(request)
        embedding  = self.engine.embed(query_text)

        # Determine which collections to search
        collections = _ROUTING.get(
            request.input_type.value,
            ALL_COLLECTIONS,
        )

        all_hits: list[SearchHit] = []
        for col_name in collections:
            try:
                hits = self.store.search(col_name, embedding, k=k)
                all_hits.extend(hits)
            except Exception:
                logger.exception("Error searching collection %s", col_name)

        # Global dedup + rank
        seen: set[str] = set()
        ranked: list[SearchHit] = []
        for h in sorted(all_hits, key=lambda x: -x.score):
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h)

        top_k = ranked[:k]

        # Convert to SemanticHit (adds entity_type from metadata)
        hits_out: list[SemanticHit] = [
            SemanticHit(
                doc_id=h.doc_id,
                text=h.text,
                score=h.score,
                collection=h.collection,
                entity_type=h.metadata.get("entity_type", ""),
                metadata=h.metadata,
            )
            for h in top_k
        ]

        return SemanticSearchResults(
            hits=hits_out,
            query_text=query_text,
            total_hits=len(ranked),
            collections_searched=list(dict.fromkeys(collections)),
            top_score=hits_out[0].score if hits_out else 0.0,
            has_results=bool(hits_out),
        )

    def investigate(
        self,
        request: DiscoveryRequest,
        k: int = DEFAULT_K,
    ) -> EnrichedInvestigationContext:
        """
        Run the complete discovery pipeline:
            Stage 1 — Knowledge Aggregation (knowledge base lookup)
            Stage 2 — Semantic Retrieval   (vector similarity search)

        Args:
            request: Normalized DiscoveryRequest from the Input Layer.
            k:       Number of semantic search results to attach.

        Returns:
            EnrichedInvestigationContext containing both aggregation and
            semantic results in a single, JSON-serialisable object.
        """
        # ── Stage 1: Knowledge Aggregation ────────────────────────────
        logger.info(
            "Starting full investigation pipeline for request %s [%s]",
            request.request_id,
            request.input_type.value,
        )
        ctx = aggregate(request)
        stages = ["aggregation"]

        # ── Stage 2: Semantic Retrieval ────────────────────────────────
        try:
            semantic = self.retrieve(request, k=k)
            stages.append("semantic")
        except Exception:
            logger.exception(
                "Semantic retrieval failed for request %s — "
                "returning aggregation-only result",
                request.request_id,
            )
            semantic = SemanticSearchResults()

        # ── Combine into EnrichedInvestigationContext ──────────────────
        enriched = EnrichedInvestigationContext(
            **ctx.model_dump(),
            semantic_results=semantic,
            pipeline_stages=stages,
        )

        logger.info(
            "Investigation complete: %d aggregation entities, "
            "%d semantic hits, top_score=%.3f",
            (len(ctx.matched_techniques) + len(ctx.matched_malware)
             + len(ctx.matched_cves) + len(ctx.matched_apt_groups)),
            semantic.total_hits,
            semantic.top_score,
        )

        return enriched


# ══════════════════════════════════════════════════════════════════════
# Module-level convenience functions
# ══════════════════════════════════════════════════════════════════════

_default_retriever: SemanticRetriever | None = None


def get_retriever() -> SemanticRetriever:
    """Return the module-level singleton retriever."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = SemanticRetriever()
    return _default_retriever


def retrieve(request: DiscoveryRequest, k: int = DEFAULT_K) -> SemanticSearchResults:
    """Convenience wrapper — semantic search only."""
    return get_retriever().retrieve(request, k=k)


def investigate(
    request: DiscoveryRequest,
    k: int = DEFAULT_K,
) -> EnrichedInvestigationContext:
    """Convenience wrapper — full pipeline (aggregation + semantic)."""
    return get_retriever().investigate(request, k=k)
