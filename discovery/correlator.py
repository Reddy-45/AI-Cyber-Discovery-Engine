"""
discovery/correlator.py — Relationship Graph Builder.

Takes the merged results from all providers and constructs a graph of
relationships between cybersecurity entities.

Node types:
    ip_address, domain, file_hash, malware, apt_group, cve, technique, tactic

Edge types:
    uses           malware/apt uses a technique
    attributed_to  malware attributed to an APT group
    exploits       malware/apt exploits a CVE
    drops          one malware drops another
    associated_ip  malware/apt associated with an IP
    associated_domain  malware associated with a domain
    related_cve    technique associated with a CVE
    targets        apt_group targets a sector

Output (RelationshipGraph) is deliberately simple — no networkx dependency here.
It is a pure-Python data structure that:
    - can be JSON-serialised directly by FastAPI
    - can be fed to networkx by the caller (graph visualisation layer)
    - can be read by the Lovable frontend

Why not networkx here?
    networkx is already used by engine/analyze.py for event correlation.
    The discovery layer returns a serialisable graph description.
    The visualisation layer (FastAPI / Lovable) converts it to networkx or D3.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from discovery.providers import (
    APTMatch,
    CVEMatch,
    IOCIntelMatch,
    MalwareMatch,
    ProviderResult,
)
from engine.models import MITRETechnique

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Graph data models
# ══════════════════════════════════════════════════════════════════════

class GraphNode(BaseModel):
    """A single entity node in the relationship graph."""
    id:           str               # Unique stable identifier
    label:        str               # Human-readable name
    node_type:    str               # ip_address | domain | hash | malware | apt_group | cve | technique | tactic
    properties:   dict[str, Any]    = Field(default_factory=dict)
    confidence:   float             = 1.0


class GraphEdge(BaseModel):
    """A directed relationship between two nodes."""
    source:       str               # Node id
    target:       str               # Node id
    relationship: str               # Edge type string
    confidence:   float             = 1.0
    evidence:     str               = ""


class RelationshipGraph(BaseModel):
    """
    A serialisable directed graph of cybersecurity entity relationships.

    Design goals:
        - JSON-serialisable with no post-processing
        - No cycles required (though they are allowed)
        - Stable node IDs (same entity always same ID)

    Consumers:
        - aggregator.py    wraps this in InvestigationContext
        - FastAPI routes   serve it directly
        - Lovable frontend renders it as a force-directed graph
        - NetworkX caller  can reconstruct: G.add_nodes_from / G.add_edges_from
    """
    nodes:       list[GraphNode]    = Field(default_factory=list)
    edges:       list[GraphEdge]    = Field(default_factory=list)
    node_count:  int                = 0
    edge_count:  int                = 0


# ══════════════════════════════════════════════════════════════════════
# Correlator
# ══════════════════════════════════════════════════════════════════════

class KnowledgeCorrelator:
    """
    Builds a relationship graph from merged ProviderResult data.

    Algorithm:
    1. Add one node per unique entity (malware, APT, CVE, technique, IOC)
    2. Iterate attribution links:  malware → APT via attributed_to
    3. Iterate technique links:    malware/APT → techniques they use
    4. Iterate CVE links:          malware/APT → CVEs they exploit
    5. Iterate IOC links:          IOC → malware family that owns it
    6. Remove duplicate edges (same src/rel/tgt)
    """

    def correlate(self, results: list[ProviderResult]) -> RelationshipGraph:
        """
        Build a relationship graph from a list of ProviderResult objects.

        Args:
            results: Merged output from all providers for a single query.

        Returns:
            RelationshipGraph with nodes and edges.
        """
        nodes: dict[str, GraphNode] = {}     # id → GraphNode
        edges: list[GraphEdge] = []
        edge_seen: set[tuple[str, str, str]] = set()

        # Collect all unique entities from all provider results
        all_malware: list[MalwareMatch]  = []
        all_apts:    list[APTMatch]      = []
        all_cves:    list[CVEMatch]      = []
        all_techs:   list[MITRETechnique] = []
        all_iocs:    list[IOCIntelMatch] = []

        for result in results:
            all_malware.extend(result.matched_malware)
            all_apts.extend(result.matched_apt_groups)
            all_cves.extend(result.matched_cves)
            all_techs.extend(result.matched_techniques)
            all_iocs.extend(result.matched_iocs)

        # Deduplicate by key
        malware_map  = _dedupe_by_key(all_malware, lambda m: m.name.lower())
        apt_map      = _dedupe_by_key(all_apts,    lambda a: a.name.lower())
        cve_map      = _dedupe_by_key(all_cves,    lambda c: c.cve_id.upper())
        tech_map     = _dedupe_by_key(all_techs,   lambda t: t.technique_id.upper())
        ioc_map      = _dedupe_by_key(all_iocs,    lambda i: i.value.lower())

        # ── Step 1: Build nodes ────────────────────────────────────────

        for m in malware_map.values():
            nid = _malware_id(m.name)
            nodes[nid] = GraphNode(
                id=nid, label=m.name, node_type="malware",
                properties={
                    "type": m.family_type,
                    "sectors": m.targeted_sectors,
                    "kill_chain": m.kill_chain,
                    "description": m.description[:200] if m.description else "",
                },
                confidence=m.confidence,
            )

        for a in apt_map.values():
            nid = _apt_id(a.name)
            nodes[nid] = GraphNode(
                id=nid, label=a.name, node_type="apt_group",
                properties={
                    "country": a.country,
                    "sponsor": a.sponsor,
                    "mitre_group_id": a.mitre_group_id,
                    "campaigns": a.known_campaigns,
                    "description": a.description[:200] if a.description else "",
                },
                confidence=a.confidence,
            )

        for c in cve_map.values():
            nid = _cve_id(c.cve_id)
            nodes[nid] = GraphNode(
                id=nid, label=c.cve_id, node_type="cve",
                properties={
                    "name": c.name,
                    "cvss_score": c.cvss_score,
                    "severity": c.severity,
                    "cisa_kev": c.cisa_kev,
                    "exploitation_status": c.exploitation_status,
                },
                confidence=c.confidence,
            )

        for t in tech_map.values():
            nid = _tech_id(t.technique_id)
            nodes[nid] = GraphNode(
                id=nid, label=t.technique_id, node_type="technique",
                properties={
                    "name": t.technique_name,
                    "tactic": t.tactic,
                    "tactic_order": t.tactic_order,
                },
                confidence=1.0,
            )

        for ioc in ioc_map.values():
            nid = _ioc_id(ioc.value, ioc.ioc_type)
            nodes[nid] = GraphNode(
                id=nid, label=ioc.value, node_type=ioc.ioc_type,
                properties={
                    "tags": ioc.tags,
                    "is_known_bad": ioc.is_known_bad,
                    "source": ioc.source,
                },
                confidence=ioc.confidence,
            )

        # ── Step 2: Malware → APT (attribution) ───────────────────────

        for m in malware_map.values():
            m_nid = _malware_id(m.name)
            for attr_name in m.attributed_to:
                # Find this APT in our graph by name/alias match
                a_nid = _find_apt_node(attr_name, apt_map)
                if a_nid:
                    _add_edge(edges, edge_seen, GraphEdge(
                        source=m_nid,
                        target=a_nid,
                        relationship="attributed_to",
                        confidence=0.85,
                        evidence=f"{m.name} attributed to {attr_name}",
                    ))

        # ── Step 3: Malware → Technique (uses) ────────────────────────

        for m in malware_map.values():
            m_nid = _malware_id(m.name)
            for tid in m.mitre_techniques:
                t_nid = _tech_id(tid.upper())
                if t_nid not in nodes:
                    # Technique referenced by malware but not fetched by MITRE provider
                    nodes[t_nid] = GraphNode(
                        id=t_nid, label=tid.upper(), node_type="technique",
                        properties={"name": "", "tactic": "", "tactic_order": 0},
                        confidence=0.70,
                    )
                _add_edge(edges, edge_seen, GraphEdge(
                    source=m_nid,
                    target=t_nid,
                    relationship="uses",
                    confidence=0.90,
                    evidence=f"{m.name} uses {tid}",
                ))

        # ── Step 4: APT → Technique (uses) ────────────────────────────

        for a in apt_map.values():
            a_nid = _apt_id(a.name)
            for tid in a.mitre_techniques:
                t_nid = _tech_id(tid.upper())
                if t_nid not in nodes:
                    nodes[t_nid] = GraphNode(
                        id=t_nid, label=tid.upper(), node_type="technique",
                        properties={"name": "", "tactic": "", "tactic_order": 0},
                        confidence=0.70,
                    )
                _add_edge(edges, edge_seen, GraphEdge(
                    source=a_nid,
                    target=t_nid,
                    relationship="uses",
                    confidence=0.88,
                    evidence=f"{a.name} uses {tid}",
                ))

        # ── Step 5: Malware → CVE (exploits) ──────────────────────────

        for m in malware_map.values():
            m_nid = _malware_id(m.name)
            for cve_str in m.cves_exploited:
                c_nid = _cve_id(cve_str.upper())
                if c_nid not in nodes:
                    nodes[c_nid] = GraphNode(
                        id=c_nid, label=cve_str.upper(), node_type="cve",
                        properties={"name": "", "cvss_score": 0.0,
                                    "severity": "", "cisa_kev": False,
                                    "exploitation_status": "unknown"},
                        confidence=0.70,
                    )
                _add_edge(edges, edge_seen, GraphEdge(
                    source=m_nid,
                    target=c_nid,
                    relationship="exploits",
                    confidence=0.90,
                    evidence=f"{m.name} exploits {cve_str}",
                ))

        # ── Step 6: APT → CVE (exploits) ──────────────────────────────

        for a in apt_map.values():
            a_nid = _apt_id(a.name)
            for cve_str in a.cves_exploited:
                c_nid = _cve_id(cve_str.upper())
                if c_nid not in nodes:
                    nodes[c_nid] = GraphNode(
                        id=c_nid, label=cve_str.upper(), node_type="cve",
                        properties={"name": "", "cvss_score": 0.0,
                                    "severity": "", "cisa_kev": False,
                                    "exploitation_status": "unknown"},
                        confidence=0.70,
                    )
                _add_edge(edges, edge_seen, GraphEdge(
                    source=a_nid,
                    target=c_nid,
                    relationship="exploits",
                    confidence=0.88,
                    evidence=f"{a.name} exploits {cve_str}",
                ))

        # ── Step 7: IOC → Malware family (associated) ─────────────────

        for ioc in ioc_map.values():
            i_nid = _ioc_id(ioc.value, ioc.ioc_type)
            if ioc.malware_family:
                m_nid = _malware_id(ioc.malware_family)
                if m_nid not in nodes:
                    nodes[m_nid] = GraphNode(
                        id=m_nid, label=ioc.malware_family,
                        node_type="malware",
                        properties={"type": "unknown", "sectors": [],
                                    "kill_chain": [], "description": ""},
                        confidence=0.75,
                    )
                _add_edge(edges, edge_seen, GraphEdge(
                    source=i_nid,
                    target=m_nid,
                    relationship="associated_malware",
                    confidence=ioc.confidence,
                    evidence=f"IOC {ioc.value[:20]} associated with {ioc.malware_family}",
                ))

        # ── Step 8: Malware IOC hashes/domains → malware node ─────────

        for m in malware_map.values():
            m_nid = _malware_id(m.name)
            for h in m.ioc_hashes:
                h_nid = _ioc_id(h, "hash")
                if h_nid not in nodes:
                    nodes[h_nid] = GraphNode(
                        id=h_nid, label=h[:16] + "...", node_type="file_hash",
                        properties={"tags": [], "is_known_bad": True, "source": "malware_db"},
                        confidence=0.90,
                    )
                _add_edge(edges, edge_seen, GraphEdge(
                    source=h_nid,
                    target=m_nid,
                    relationship="belongs_to",
                    confidence=0.90,
                    evidence=f"Hash is a known IOC for {m.name}",
                ))
            for d in m.ioc_domains:
                d_nid = _ioc_id(d, "domain")
                if d_nid not in nodes:
                    nodes[d_nid] = GraphNode(
                        id=d_nid, label=d, node_type="domain",
                        properties={"tags": [], "is_known_bad": True, "source": "malware_db"},
                        confidence=0.90,
                    )
                _add_edge(edges, edge_seen, GraphEdge(
                    source=d_nid,
                    target=m_nid,
                    relationship="belongs_to",
                    confidence=0.90,
                    evidence=f"Domain is a known IOC for {m.name}",
                ))

        node_list = list(nodes.values())
        return RelationshipGraph(
            nodes=node_list,
            edges=edges,
            node_count=len(node_list),
            edge_count=len(edges),
        )


# ══════════════════════════════════════════════════════════════════════
# Stable node ID helpers
# ══════════════════════════════════════════════════════════════════════

def _malware_id(name: str) -> str:
    return f"malware:{name.lower().replace(' ', '_')}"

def _apt_id(name: str) -> str:
    return f"apt:{name.lower().replace(' ', '_')}"

def _cve_id(cve: str) -> str:
    return f"cve:{cve.upper()}"

def _tech_id(tid: str) -> str:
    return f"technique:{tid.upper()}"

def _ioc_id(value: str, ioc_type: str) -> str:
    return f"{ioc_type}:{value.lower()}"


def _find_apt_node(name: str, apt_map: dict[str, APTMatch]) -> str | None:
    """Find an APT node ID by name or alias."""
    name_lower = name.lower()
    for key, apt in apt_map.items():
        if key == name_lower:
            return _apt_id(apt.name)
        if name_lower in [a.lower() for a in apt.aliases]:
            return _apt_id(apt.name)
    return None


def _add_edge(
    edges: list[GraphEdge],
    seen: set[tuple[str, str, str]],
    edge: GraphEdge,
) -> None:
    """Add edge only if (source, target, relationship) is not already present."""
    key = (edge.source, edge.target, edge.relationship)
    if key not in seen:
        seen.add(key)
        edges.append(edge)


def _dedupe_by_key(items: list, key_fn) -> dict:
    """Return a dict keyed by key_fn(item), keeping the first occurrence."""
    result: dict = {}
    for item in items:
        k = key_fn(item)
        if k not in result:
            result[k] = item
    return result
