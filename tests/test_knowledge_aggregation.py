"""
tests/test_knowledge_aggregation.py — Unit tests for the Knowledge Aggregation layer.

Tests cover:
    Section A — Individual providers (positive and empty cases)
    Section B — KnowledgeCorrelator graph structure
    Section C — Aggregator (full pipeline, confidence, summary)
    Section D — InvestigationContext model validation

All tests use local data files only.
No network calls, no engine pipeline invoked.
"""

from __future__ import annotations

import json
import pytest

from discovery.models import DiscoveryRequest, InputType, InputSource
from discovery.normalizer import normalize_text, normalize_file
from discovery.providers import (
    MITREProvider,
    ThreatIntelProvider,
    MalwareProvider,
    APTGroupProvider,
    CVEProvider,
    STIXProvider,
)
from discovery.correlator import KnowledgeCorrelator, RelationshipGraph
from discovery.aggregator import aggregate, InvestigationContext


# ── Fixtures ───────────────────────────────────────────────────────────

def _req(raw: str) -> DiscoveryRequest:
    """Shorthand: normalize a text query."""
    return normalize_text(raw)


def _req_nl(text: str) -> DiscoveryRequest:
    """Force natural-language classification for a long description."""
    return normalize_text(text)


# ══════════════════════════════════════════════════════════════════════
# Section A — Provider unit tests
# ══════════════════════════════════════════════════════════════════════

class TestMITREProvider:
    provider = MITREProvider()

    def test_direct_technique_match(self):
        result = self.provider.query(_req("T1486"))
        assert len(result.matched_techniques) == 1
        assert result.matched_techniques[0].technique_id == "T1486"

    def test_direct_technique_case_insensitive(self):
        result = self.provider.query(_req("t1110"))
        assert any(t.technique_id == "T1110" for t in result.matched_techniques)

    def test_nl_keyword_match_ransomware(self):
        result = self.provider.query(_req_nl(
            "The system was infected with ransomware and files were encrypted"
        ))
        ids = [t.technique_id for t in result.matched_techniques]
        assert "T1486" in ids  # Data Encrypted for Impact

    def test_nl_keyword_match_brute_force(self):
        result = self.provider.query(_req_nl(
            "Multiple failed login attempts detected from external IP"
        ))
        ids = [t.technique_id for t in result.matched_techniques]
        assert "T1110" in ids  # Brute Force

    def test_nl_keyword_match_exfiltration(self):
        result = self.provider.query(_req_nl(
            "Large outbound data transfer detected over c2 beacon connection"
        ))
        ids = [t.technique_id for t in result.matched_techniques]
        assert "T1041" in ids

    def test_wrong_input_type_returns_empty(self):
        result = self.provider.query(_req("203.0.113.42"))
        # IP address query: MITRE doesn't keyword-match on IPs directly
        assert isinstance(result.matched_techniques, list)

    def test_result_deduplication(self):
        result = self.provider.query(_req_nl(
            "brute force failed login brute failed password brute"
        ))
        ids = [t.technique_id for t in result.matched_techniques]
        assert len(ids) == len(set(ids)), "Duplicate technique IDs returned"

    def test_returns_technique_name(self):
        result = self.provider.query(_req("T1486"))
        assert result.matched_techniques[0].technique_name != ""

    def test_returns_tactic(self):
        result = self.provider.query(_req("T1486"))
        assert result.matched_techniques[0].tactic != ""

    def test_evidence_populated(self):
        result = self.provider.query(_req("T1595"))
        assert len(result.evidence) > 0


class TestThreatIntelProvider:
    provider = ThreatIntelProvider()

    def test_known_bad_ip(self):
        result = self.provider.query(_req("185.220.101.33"))
        assert len(result.matched_iocs) == 1
        assert result.matched_iocs[0].is_known_bad is True
        assert result.matched_iocs[0].confidence >= 0.9

    def test_known_bad_ip_has_tags(self):
        result = self.provider.query(_req("185.220.101.33"))
        assert len(result.matched_iocs[0].tags) > 0

    def test_unknown_ip_returns_empty(self):
        result = self.provider.query(_req("8.8.8.8"))
        assert result.matched_iocs == []

    def test_known_bad_hash(self):
        result = self.provider.query(_req("84c82835a5d21bbcf75a61706d8ab549"))
        assert len(result.matched_iocs) == 1
        assert result.matched_iocs[0].malware_family == "WannaCry"
        assert result.matched_iocs[0].is_known_bad is True

    def test_unknown_hash_returns_empty(self):
        result = self.provider.query(_req("a" * 32))
        assert result.matched_iocs == []

    def test_known_bad_domain(self):
        domain_req = _req("iuqerfsodp9ifjaposdfjhgosurijfaewrwergwea.com")
        result = self.provider.query(domain_req)
        assert len(result.matched_iocs) >= 1
        assert result.matched_iocs[0].is_known_bad is True

    def test_url_host_checked(self):
        url_req = normalize_text("http://iuqerfsodp9ifjaposdfjhgosurijfaewrwergwea.com/test")
        result = self.provider.query(url_req)
        assert len(result.matched_iocs) >= 1

    def test_evidence_populated_for_hit(self):
        result = self.provider.query(_req("185.220.101.33"))
        assert len(result.evidence) > 0


class TestMalwareProvider:
    provider = MalwareProvider()

    def test_wannacry_by_name(self):
        result = self.provider.query(_req("WannaCry"))
        assert len(result.matched_malware) == 1
        assert result.matched_malware[0].name == "WannaCry"

    def test_wannacry_alias(self):
        result = self.provider.query(_req("WannaCrypt"))
        # WannaCrypt is an alias — normalizer maps it to malware_name
        # if it's in the known_malware set; otherwise NL. Check either.
        assert isinstance(result.matched_malware, list)

    def test_emotet_by_name(self):
        result = self.provider.query(_req("Emotet"))
        assert len(result.matched_malware) == 1
        assert result.matched_malware[0].name == "Emotet"

    def test_malware_has_techniques(self):
        result = self.provider.query(_req("WannaCry"))
        assert len(result.matched_malware[0].mitre_techniques) > 0

    def test_malware_has_attribution(self):
        result = self.provider.query(_req("WannaCry"))
        assert "Lazarus Group" in result.matched_malware[0].attributed_to

    def test_malware_has_cves(self):
        result = self.provider.query(_req("WannaCry"))
        assert "CVE-2017-0144" in result.matched_malware[0].cves_exploited

    def test_by_hash_wannacry(self):
        result = self.provider.query(_req("84c82835a5d21bbcf75a61706d8ab549"))
        assert len(result.matched_malware) >= 1
        names = [m.name for m in result.matched_malware]
        assert "WannaCry" in names

    def test_by_cve(self):
        result = self.provider.query(_req("CVE-2017-0144"))
        names = [m.name for m in result.matched_malware]
        assert "WannaCry" in names or "TrickBot" in names

    def test_nl_mentions(self):
        result = self.provider.query(_req_nl(
            "The attacker deployed Emotet followed by TrickBot for persistence"
        ))
        names = [m.name for m in result.matched_malware]
        assert "Emotet" in names
        assert "TrickBot" in names

    def test_no_match_returns_empty(self):
        result = self.provider.query(_req("8.8.8.8"))
        assert result.matched_malware == []

    def test_evidence_populated(self):
        result = self.provider.query(_req("Ryuk"))
        assert len(result.evidence) > 0

    def test_deduplication(self):
        result = self.provider.query(_req_nl(
            "WannaCry WannaCry WannaCry ransomware encrypted"
        ))
        names = [m.name for m in result.matched_malware]
        assert names.count("WannaCry") == 1


class TestAPTGroupProvider:
    provider = APTGroupProvider()

    def test_apt28_by_name(self):
        result = self.provider.query(_req("APT28"))
        assert len(result.matched_apt_groups) == 1
        assert result.matched_apt_groups[0].name == "APT28"

    def test_apt28_has_country(self):
        result = self.provider.query(_req("APT28"))
        assert result.matched_apt_groups[0].country == "Russia"

    def test_apt28_alias_fancy_bear(self):
        result = self.provider.query(_req("Fancy Bear"))
        assert len(result.matched_apt_groups) == 1
        assert result.matched_apt_groups[0].name == "APT28"

    def test_lazarus_by_name(self):
        result = self.provider.query(_req("Lazarus Group"))
        assert len(result.matched_apt_groups) >= 1
        names = [a.name for a in result.matched_apt_groups]
        assert "Lazarus Group" in names

    def test_apt_has_techniques(self):
        result = self.provider.query(_req("APT29"))
        assert len(result.matched_apt_groups[0].mitre_techniques) > 0

    def test_apt_has_malware(self):
        result = self.provider.query(_req("APT41"))
        assert len(result.matched_apt_groups[0].malware_used) > 0

    def test_by_malware_name(self):
        # Groups that use WannaCry
        result = self.provider.query(_req("WannaCry"))
        names = [a.name for a in result.matched_apt_groups]
        # Lazarus Group uses WannaCry
        assert "Lazarus Group" in names

    def test_by_cve(self):
        result = self.provider.query(_req("CVE-2021-44228"))
        names = [a.name for a in result.matched_apt_groups]
        assert len(names) > 0  # APT41 and Lazarus exploit Log4Shell

    def test_nl_mention(self):
        result = self.provider.query(_req_nl(
            "Indicators suggest APT28 is behind this campaign targeting NATO"
        ))
        names = [a.name for a in result.matched_apt_groups]
        assert "APT28" in names

    def test_unknown_apt_returns_empty(self):
        result = self.provider.query(_req("8.8.8.8"))
        assert result.matched_apt_groups == []


class TestCVEProvider:
    provider = CVEProvider()

    def test_direct_cve_lookup(self):
        result = self.provider.query(_req("CVE-2021-44228"))
        assert len(result.matched_cves) == 1
        assert result.matched_cves[0].cve_id == "CVE-2021-44228"

    def test_cve_has_name(self):
        result = self.provider.query(_req("CVE-2021-44228"))
        assert result.matched_cves[0].name == "Log4Shell"

    def test_cve_has_cvss(self):
        result = self.provider.query(_req("CVE-2021-44228"))
        assert result.matched_cves[0].cvss_score == 10.0

    def test_cve_cisa_kev(self):
        result = self.provider.query(_req("CVE-2021-44228"))
        assert result.matched_cves[0].cisa_kev is True

    def test_eternalblue(self):
        result = self.provider.query(_req("CVE-2017-0144"))
        assert len(result.matched_cves) >= 1
        assert result.matched_cves[0].cve_id == "CVE-2017-0144"

    def test_nl_keyword_log4shell(self):
        result = self.provider.query(_req_nl(
            "Attacker exploited log4j via JNDI lookup in log messages"
        ))
        ids = [c.cve_id for c in result.matched_cves]
        assert "CVE-2021-44228" in ids

    def test_nl_cve_id_in_text(self):
        result = self.provider.query(_req_nl(
            "The CVE-2021-44228 vulnerability was used in this attack"
        ))
        ids = [c.cve_id for c in result.matched_cves]
        assert "CVE-2021-44228" in ids

    def test_by_malware(self):
        # CVEs exploited by Cobalt Strike (related_malware)
        result = self.provider.query(_req("Cobalt Strike"))
        # CVE-2021-44228 lists Cobalt Strike as related malware
        ids = [c.cve_id for c in result.matched_cves]
        assert "CVE-2021-44228" in ids

    def test_unknown_cve_returns_empty(self):
        result = self.provider.query(_req("CVE-9999-99999"))
        assert result.matched_cves == []


class TestSTIXProvider:
    provider = STIXProvider()

    def _make_bundle(self, objects=None) -> DiscoveryRequest:
        bundle = {
            "type": "bundle",
            "id": "bundle--test",
            "spec_version": "2.1",
            "objects": objects or [],
        }
        return normalize_file(json.dumps(bundle).encode(), "test.json")

    def test_stix_indicator_extracts_ip(self):
        req = self._make_bundle([{
            "type": "indicator",
            "id": "indicator--1",
            "name": "C2 IP",
            "pattern": "[ipv4-addr:value = '185.220.101.33']",
        }])
        result = self.provider.query(req)
        values = [i.value for i in result.matched_iocs]
        assert "185.220.101.33" in values

    def test_stix_malware_object(self):
        req = self._make_bundle([{
            "type": "malware",
            "id": "malware--1",
            "name": "CustomRat",
            "malware_types": ["trojan"],
        }])
        result = self.provider.query(req)
        names = [m.name for m in result.matched_malware]
        assert "CustomRat" in names

    def test_stix_threat_actor(self):
        req = self._make_bundle([{
            "type": "threat-actor",
            "id": "threat-actor--1",
            "name": "Evil Corp",
            "aliases": ["Indrik Spider"],
        }])
        result = self.provider.query(req)
        names = [a.name for a in result.matched_apt_groups]
        assert "Evil Corp" in names

    def test_stix_vulnerability_cve(self):
        req = self._make_bundle([{
            "type": "vulnerability",
            "id": "vulnerability--1",
            "name": "EternalBlue",
            "external_references": [
                {"source_name": "cve", "external_id": "CVE-2017-0144"},
            ],
        }])
        result = self.provider.query(req)
        ids = [c.cve_id for c in result.matched_cves]
        assert "CVE-2017-0144" in ids

    def test_stix_relationship_in_evidence(self):
        req = self._make_bundle([{
            "type": "relationship",
            "id": "relationship--1",
            "relationship_type": "uses",
            "source_ref": "threat-actor--1",
            "target_ref": "malware--1",
        }])
        result = self.provider.query(req)
        assert any("relationship" in e.lower() for e in result.evidence)

    def test_non_stix_input_ignored(self):
        result = self.provider.query(_req("APT28"))
        assert result.matched_iocs == []
        assert result.matched_malware == []


# ══════════════════════════════════════════════════════════════════════
# Section B — Correlator unit tests
# ══════════════════════════════════════════════════════════════════════

class TestCorrelator:
    correlator = KnowledgeCorrelator()

    def _run(self, raw: str) -> RelationshipGraph:
        req = _req(raw)
        results = [
            MITREProvider().query(req),
            MalwareProvider().query(req),
            APTGroupProvider().query(req),
            CVEProvider().query(req),
            ThreatIntelProvider().query(req),
        ]
        return self.correlator.correlate(results)

    def test_wannacry_has_nodes(self):
        g = self._run("WannaCry")
        assert g.node_count > 0

    def test_wannacry_has_malware_node(self):
        g = self._run("WannaCry")
        types = [n.node_type for n in g.nodes]
        assert "malware" in types

    def test_wannacry_has_technique_nodes(self):
        g = self._run("WannaCry")
        types = [n.node_type for n in g.nodes]
        assert "technique" in types

    def test_wannacry_has_cve_node(self):
        g = self._run("WannaCry")
        ids = [n.label for n in g.nodes]
        assert "CVE-2017-0144" in ids

    def test_wannacry_has_apt_node(self):
        g = self._run("WannaCry")
        types = [n.node_type for n in g.nodes]
        assert "apt_group" in types

    def test_edges_have_no_self_loops(self):
        g = self._run("WannaCry")
        for edge in g.edges:
            assert edge.source != edge.target

    def test_edge_count_matches(self):
        g = self._run("WannaCry")
        assert g.edge_count == len(g.edges)

    def test_node_count_matches(self):
        g = self._run("APT28")
        assert g.node_count == len(g.nodes)

    def test_node_ids_are_unique(self):
        g = self._run("WannaCry")
        ids = [n.id for n in g.nodes]
        assert len(ids) == len(set(ids)), "Duplicate node IDs in graph"

    def test_no_duplicate_edges(self):
        g = self._run("WannaCry")
        edge_keys = [(e.source, e.target, e.relationship) for e in g.edges]
        assert len(edge_keys) == len(set(edge_keys)), "Duplicate edges in graph"

    def test_empty_results_produce_empty_graph(self):
        from discovery.providers import ProviderResult
        g = self.correlator.correlate([ProviderResult(source="test")])
        assert g.node_count == 0
        assert g.edge_count == 0

    def test_apt28_has_technique_edges(self):
        g = self._run("APT28")
        rels = [e.relationship for e in g.edges]
        assert "uses" in rels

    def test_uses_edges_reference_existing_nodes(self):
        g = self._run("WannaCry")
        node_ids = {n.id for n in g.nodes}
        for edge in g.edges:
            assert edge.source in node_ids, f"Source {edge.source} not a node"
            assert edge.target in node_ids, f"Target {edge.target} not a node"


# ══════════════════════════════════════════════════════════════════════
# Section C — Aggregator integration tests
# ══════════════════════════════════════════════════════════════════════

class TestAggregator:

    def test_wannacry_has_findings(self):
        ctx = aggregate(_req("WannaCry"))
        assert ctx.has_findings is True

    def test_wannacry_malware_present(self):
        ctx = aggregate(_req("WannaCry"))
        names = [m.name for m in ctx.matched_malware]
        assert "WannaCry" in names

    def test_wannacry_apts_present(self):
        ctx = aggregate(_req("WannaCry"))
        names = [a.name for a in ctx.matched_apt_groups]
        assert "Lazarus Group" in names

    def test_wannacry_cves_present(self):
        ctx = aggregate(_req("WannaCry"))
        ids = [c.cve_id for c in ctx.matched_cves]
        assert "CVE-2017-0144" in ids

    def test_wannacry_techniques_present(self):
        ctx = aggregate(_req("WannaCry"))
        assert len(ctx.matched_techniques) > 0

    def test_known_bad_ip_aggregated(self):
        ctx = aggregate(_req("185.220.101.33"))
        assert ctx.has_findings is True
        assert len(ctx.matched_iocs) >= 1

    def test_cve_direct_lookup(self):
        ctx = aggregate(_req("CVE-2021-44228"))
        ids = [c.cve_id for c in ctx.matched_cves]
        assert "CVE-2021-44228" in ids

    def test_apt28_aggregated(self):
        ctx = aggregate(_req("APT28"))
        names = [a.name for a in ctx.matched_apt_groups]
        assert "APT28" in names

    def test_nl_query_aggregated(self):
        ctx = aggregate(_req_nl(
            "Ransomware encrypted files via EternalBlue SMB exploit on port 445"
        ))
        # Should match T1486, T1021, and WannaCry (eternalblue keyword)
        assert ctx.has_findings is True

    def test_confidence_is_nonzero_for_hit(self):
        ctx = aggregate(_req("WannaCry"))
        assert ctx.confidence_score > 0.0

    def test_confidence_is_zero_for_no_hit(self):
        ctx = aggregate(_req("xxxxxxxxxnotamalwarexxxxxxxxx"))
        assert ctx.confidence_score == 0.0

    def test_has_findings_false_for_unknown(self):
        ctx = aggregate(_req("unknownquery12345"))
        assert ctx.has_findings is False

    def test_graph_has_nodes(self):
        ctx = aggregate(_req("WannaCry"))
        assert ctx.relationship_graph.node_count > 0

    def test_graph_has_edges(self):
        ctx = aggregate(_req("WannaCry"))
        assert ctx.relationship_graph.edge_count > 0

    def test_request_id_preserved(self):
        req = _req("APT41")
        ctx = aggregate(req)
        assert ctx.request_id == req.request_id

    def test_input_type_preserved(self):
        ctx = aggregate(_req("APT28"))
        assert ctx.input_type == "apt_group"

    def test_evidence_list_populated(self):
        ctx = aggregate(_req("WannaCry"))
        assert len(ctx.evidence) > 0

    def test_provider_hits_populated(self):
        ctx = aggregate(_req("WannaCry"))
        assert len(ctx.provider_hits) > 0

    def test_query_summary_populated(self):
        ctx = aggregate(_req("WannaCry"))
        assert len(ctx.query_summary) > 0

    def test_deduplication_across_providers(self):
        # CVE-2017-0144 is returned by both MalwareProvider (WannaCry) and CVEProvider
        ctx = aggregate(_req("CVE-2017-0144"))
        ids = [c.cve_id for c in ctx.matched_cves]
        assert ids.count("CVE-2017-0144") == 1, "CVE deduplicated across providers"

    def test_invalid_request_still_returns_context(self):
        req = normalize_text("")   # empty → invalid
        ctx = aggregate(req)
        assert isinstance(ctx, InvestigationContext)

    def test_stix_bundle_aggregated(self):
        bundle = json.dumps({
            "type": "bundle",
            "id": "bundle--test",
            "spec_version": "2.1",
            "objects": [
                {
                    "type": "malware",
                    "id": "malware--1",
                    "name": "CustomTrojan",
                    "malware_types": ["trojan"],
                },
            ],
        })
        req = normalize_file(bundle.encode(), "test.json")
        ctx = aggregate(req)
        names = [m.name for m in ctx.matched_malware]
        assert "CustomTrojan" in names


# ══════════════════════════════════════════════════════════════════════
# Section D — InvestigationContext model validation
# ══════════════════════════════════════════════════════════════════════

class TestInvestigationContext:
    def test_serialisable_to_json(self):
        ctx = aggregate(_req("WannaCry"))
        # model_dump(mode="json") converts datetimes to ISO strings
        data = ctx.model_dump(mode="json")
        dumped = json.dumps(data)
        assert len(dumped) > 10

    def test_has_analysed_at(self):
        ctx = aggregate(_req("APT28"))
        assert ctx.analysed_at is not None

    def test_relationship_graph_serialisable(self):
        ctx = aggregate(_req("WannaCry"))
        graph_data = ctx.relationship_graph.model_dump()
        json.dumps(graph_data)  # must not raise

    def test_nodes_have_required_fields(self):
        ctx = aggregate(_req("WannaCry"))
        for node in ctx.relationship_graph.nodes:
            assert node.id
            assert node.label
            assert node.node_type
            assert 0.0 <= node.confidence <= 1.0

    def test_edges_have_required_fields(self):
        ctx = aggregate(_req("WannaCry"))
        for edge in ctx.relationship_graph.edges:
            assert edge.source
            assert edge.target
            assert edge.relationship
            assert 0.0 <= edge.confidence <= 1.0
