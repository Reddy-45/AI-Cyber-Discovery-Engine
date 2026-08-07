"""
tests/test_semantic.py — Unit tests for the Semantic Layer.

Tests are split into five sections:

Section A — entity_to_text() helpers (pure Python, no model needed)
Section B — EmbeddingEngine with a mocked SentenceTransformer
Section C — KnowledgeVectorStore with persist=False (in-memory) + mock embeddings
Section D — SemanticRetriever with injected mock engine + in-memory store
Section E — Output model validation

All tests use mocked or in-memory components:
    - EmbeddingEngine: mocked SentenceTransformer (no real model download)
    - KnowledgeVectorStore: persist=False (no disk I/O)
    - SemanticRetriever: injected engine + store

Existing tests (test_normalizer, test_knowledge_aggregation) are NOT affected.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from semantic.embedding import (
    EMBEDDING_DIM,
    EmbeddingEngine,
    entity_to_text,
)
from semantic.vector_store import (
    ALL_COLLECTIONS,
    COL_APT,
    COL_CVE,
    COL_MALWARE,
    COL_MITRE,
    COL_THREAT_INTEL,
    CollectionStats,
    KnowledgeVectorStore,
    SearchHit,
)
from semantic.retriever import (
    DEFAULT_K,
    EnrichedInvestigationContext,
    SemanticHit,
    SemanticRetriever,
    SemanticSearchResults,
)

from discovery.normalizer import normalize_text
from discovery.models import InputType


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def _fake_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    """Unit-length vector for testing (all equal values)."""
    v = 1.0 / math.sqrt(dim)
    return [v] * dim


def _make_mock_engine(dim: int = EMBEDDING_DIM) -> MagicMock:
    """Mock EmbeddingEngine returning deterministic unit vectors."""
    engine = MagicMock(spec=EmbeddingEngine)
    engine.dimension = dim
    engine.embed.return_value = _fake_embedding(dim)
    engine.embed_batch.side_effect = lambda texts: [_fake_embedding(dim)] * len(texts)
    engine.embed_entity.return_value = _fake_embedding(dim)
    return engine


@pytest.fixture()
def in_memory_store() -> KnowledgeVectorStore:
    """KnowledgeVectorStore with no disk I/O (persist=False)."""
    return KnowledgeVectorStore(persist=False, dim=EMBEDDING_DIM)


@pytest.fixture()
def mock_engine() -> MagicMock:
    return _make_mock_engine()


@pytest.fixture()
def indexed_store(in_memory_store, mock_engine) -> KnowledgeVectorStore:
    """In-memory store with the knowledge base already indexed."""
    in_memory_store.index_knowledge_base(mock_engine)
    return in_memory_store


@pytest.fixture()
def retriever(mock_engine, indexed_store) -> SemanticRetriever:
    """SemanticRetriever with injected mock engine and in-memory store."""
    return SemanticRetriever(engine=mock_engine, store=indexed_store)


# ══════════════════════════════════════════════════════════════════════
# Section A — entity_to_text() (pure Python)
# ══════════════════════════════════════════════════════════════════════

class TestEntityToText:

    def test_mitre_contains_id(self):
        text = entity_to_text("mitre", {
            "technique_id": "T1486",
            "technique_name": "Data Encrypted for Impact",
            "tactic": "Impact",
            "keywords": ["ransomware", "encryption"],
        })
        assert "T1486" in text
        assert "Data Encrypted for Impact" in text
        assert "Impact" in text

    def test_mitre_contains_keywords(self):
        text = entity_to_text("mitre", {
            "technique_id": "T1110",
            "technique_name": "Brute Force",
            "tactic": "Credential Access",
            "keywords": ["failed login", "brute"],
        })
        assert "failed login" in text
        assert "brute" in text

    def test_malware_contains_name(self):
        text = entity_to_text("malware", {
            "name": "WannaCry",
            "aliases": ["WannaCrypt"],
            "type": "ransomware",
            "attributed_to": ["Lazarus Group"],
            "description": "WannaCry is ransomware.",
        })
        assert "WannaCry" in text
        assert "WannaCrypt" in text
        assert "Lazarus Group" in text

    def test_malware_contains_cves(self):
        text = entity_to_text("malware", {
            "name": "WannaCry",
            "cves_exploited": ["CVE-2017-0144"],
            "description": "",
        })
        assert "CVE-2017-0144" in text

    def test_apt_contains_group_name(self):
        text = entity_to_text("apt", {
            "name": "APT28",
            "aliases": ["Fancy Bear"],
            "country": "Russia",
            "description": "Russian state actor.",
        })
        assert "APT28" in text
        assert "Fancy Bear" in text
        assert "Russia" in text

    def test_cve_contains_id_and_name(self):
        text = entity_to_text("cve", {
            "cve_id": "CVE-2021-44228",
            "name": "Log4Shell",
            "cvss_score": 10.0,
            "severity": "critical",
            "description": "Remote code execution via JNDI.",
            "keywords": ["log4j", "jndi"],
        })
        assert "CVE-2021-44228" in text
        assert "Log4Shell" in text
        assert "log4j" in text

    def test_ip_entity(self):
        text = entity_to_text("ip", {
            "ip": "185.220.101.33",
            "tags": ["tor_exit_node", "c2"],
        })
        assert "185.220.101.33" in text
        assert "tor_exit_node" in text

    def test_domain_entity(self):
        text = entity_to_text("domain", {
            "domain": "evil.ru",
            "tags": ["phishing"],
        })
        assert "evil.ru" in text
        assert "phishing" in text

    def test_hash_entity(self):
        text = entity_to_text("hash", {
            "hash": "84c82835a5d21bbcf75a61706d8ab549",
            "malware_family": "WannaCry",
            "type": "ransomware",
        })
        assert "84c82835a5d21bbcf75a61706d8ab549" in text
        assert "WannaCry" in text

    def test_nl_entity_returns_text(self):
        text = entity_to_text("nl", {"text": "Suspicious lateral movement via SMB"})
        assert "Suspicious" in text

    def test_unknown_entity_type_doesnt_crash(self):
        text = entity_to_text("xyzfoo", {"description": "something"})
        assert isinstance(text, str)

    def test_malware_description_truncated_at_300(self):
        long_desc = "A" * 1000
        text = entity_to_text("malware", {"name": "Test", "description": long_desc})
        assert long_desc[:300] in text
        assert long_desc[300:] not in text

    def test_mitre_returns_nonempty_string(self):
        text = entity_to_text("mitre", {
            "technique_id": "T1041",
            "technique_name": "Exfiltration Over C2 Channel",
            "tactic": "Exfiltration",
            "keywords": [],
        })
        assert len(text) > 5

    def test_cve_cvss_in_text(self):
        text = entity_to_text("cve", {
            "cve_id": "CVE-2021-44228",
            "name": "Log4Shell",
            "cvss_score": 10.0,
            "severity": "critical",
            "description": "",
        })
        assert "10.0" in text

    def test_apt_country_in_text(self):
        text = entity_to_text("apt", {
            "name": "APT28",
            "country": "Russia",
            "sponsor": "GRU",
            "description": "",
        })
        assert "Russia" in text
        assert "GRU" in text


# ══════════════════════════════════════════════════════════════════════
# Section B — EmbeddingEngine (mocked SentenceTransformer)
# ══════════════════════════════════════════════════════════════════════

class TestEmbeddingEngine:

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        EmbeddingEngine.reset()
        yield
        EmbeddingEngine.reset()

    def _make_mock_st(self, dim: int = EMBEDDING_DIM):
        mock = MagicMock()
        mock.get_sentence_embedding_dimension.return_value = dim
        mock.encode.side_effect = lambda texts, **kw: np.array(
            [_fake_embedding(dim)] * len(texts)
        )
        return mock

    def test_embed_returns_list(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            result = EmbeddingEngine().embed("test text")
        assert isinstance(result, list)

    def test_embed_correct_dimension(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            result = EmbeddingEngine().embed("test text")
        assert len(result) == EMBEDDING_DIM

    def test_embed_returns_floats(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            result = EmbeddingEngine().embed("test")
        assert all(isinstance(v, float) for v in result)

    def test_embed_empty_string_returns_zeros(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            result = EmbeddingEngine().embed("")
        assert result == [0.0] * EMBEDDING_DIM

    def test_embed_whitespace_returns_zeros(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            result = EmbeddingEngine().embed("   ")
        assert result == [0.0] * EMBEDDING_DIM

    def test_embed_batch_returns_list_of_lists(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            results = EmbeddingEngine().embed_batch(["text one", "text two", "text three"])
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(len(r) == EMBEDDING_DIM for r in results)

    def test_embed_batch_empty_input(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            results = EmbeddingEngine().embed_batch([])
        assert results == []

    def test_embed_entity_returns_vector(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            result = EmbeddingEngine().embed_entity("mitre", {
                "technique_id": "T1486",
                "technique_name": "Data Encrypted for Impact",
                "tactic": "Impact",
                "keywords": [],
            })
        assert len(result) == EMBEDDING_DIM

    def test_dimension_property(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st(128)):
            assert EmbeddingEngine().dimension == 128

    def test_singleton_returns_same_instance(self):
        with patch("semantic.embedding.SentenceTransformer", return_value=self._make_mock_st()):
            a = EmbeddingEngine.get()
            b = EmbeddingEngine.get()
        assert a is b

    def test_missing_library_raises_import_error(self):
        with patch("semantic.embedding.SentenceTransformer", None):
            with pytest.raises(ImportError):
                EmbeddingEngine(model_name="test")


# ══════════════════════════════════════════════════════════════════════
# Section C — KnowledgeVectorStore (in-memory, no disk I/O)
# ══════════════════════════════════════════════════════════════════════

class TestKnowledgeVectorStore:

    def test_initial_count_is_zero(self, in_memory_store):
        assert in_memory_store.total_documents() == 0

    def test_not_already_indexed_initially(self, in_memory_store):
        assert in_memory_store._already_indexed(COL_MITRE) is False

    def test_index_knowledge_base_returns_dict(self, in_memory_store, mock_engine):
        counts = in_memory_store.index_knowledge_base(mock_engine)
        assert isinstance(counts, dict)

    def test_index_returns_all_collection_names(self, in_memory_store, mock_engine):
        counts = in_memory_store.index_knowledge_base(mock_engine)
        for name in ALL_COLLECTIONS:
            assert name in counts

    def test_mitre_collection_indexed(self, indexed_store):
        assert indexed_store._already_indexed(COL_MITRE) is True

    def test_malware_collection_indexed(self, indexed_store):
        assert indexed_store._already_indexed(COL_MALWARE) is True

    def test_cve_collection_indexed(self, indexed_store):
        assert indexed_store._already_indexed(COL_CVE) is True

    def test_apt_collection_indexed(self, indexed_store):
        assert indexed_store._already_indexed(COL_APT) is True

    def test_threat_intel_indexed(self, indexed_store):
        assert indexed_store._already_indexed(COL_THREAT_INTEL) is True

    def test_mitre_doc_count(self, indexed_store):
        assert indexed_store._col(COL_MITRE).count() == 9  # 9 techniques in JSON

    def test_malware_doc_count(self, indexed_store):
        assert indexed_store._col(COL_MALWARE).count() == 6  # 6 families

    def test_cve_doc_count(self, indexed_store):
        assert indexed_store._col(COL_CVE).count() == 6   # 6 CVEs

    def test_apt_doc_count(self, indexed_store):
        assert indexed_store._col(COL_APT).count() == 6   # 6 APT groups

    def test_total_documents_nonzero(self, indexed_store):
        assert indexed_store.total_documents() > 0

    def test_search_returns_list(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=3)
        assert isinstance(hits, list)

    def test_search_respects_k(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=3)
        assert len(hits) <= 3

    def test_search_returns_search_hit_objects(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=5)
        for h in hits:
            assert isinstance(h, SearchHit)

    def test_search_hit_score_in_range(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=5)
        for h in hits:
            assert 0.0 <= h.score <= 1.0

    def test_search_hit_has_doc_id(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=5)
        for h in hits:
            assert h.doc_id != ""

    def test_search_hit_has_text(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=5)
        for h in hits:
            assert h.text != ""

    def test_search_hit_has_collection_name(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=5)
        for h in hits:
            assert h.collection == COL_MITRE

    def test_search_hit_has_metadata(self, indexed_store, mock_engine):
        hits = indexed_store.search(COL_MITRE, mock_engine.embed("test"), k=5)
        for h in hits:
            assert isinstance(h.metadata, dict)

    def test_search_all_returns_results(self, indexed_store, mock_engine):
        hits = indexed_store.search_all(mock_engine.embed("wannacry ransomware"), k=5)
        assert len(hits) <= 5

    def test_search_all_no_duplicate_ids(self, indexed_store, mock_engine):
        hits = indexed_store.search_all(mock_engine.embed("malware attack"), k=15)
        ids = [h.doc_id for h in hits]
        assert len(ids) == len(set(ids))

    def test_search_empty_collection_returns_empty(self, in_memory_store, mock_engine):
        hits = in_memory_store.search(COL_MITRE, mock_engine.embed("test"), k=5)
        assert hits == []

    def test_get_stats_returns_list_of_stats(self, indexed_store):
        stats = indexed_store.get_stats()
        assert isinstance(stats, list)
        assert all(isinstance(s, CollectionStats) for s in stats)

    def test_get_stats_covers_all_collections(self, indexed_store):
        stats = indexed_store.get_stats()
        names = {s.name for s in stats}
        assert set(ALL_COLLECTIONS) == names

    def test_get_stats_counts_positive(self, indexed_store):
        stats = indexed_store.get_stats()
        assert all(s.count >= 0 for s in stats)

    def test_second_index_skipped(self, in_memory_store, mock_engine):
        in_memory_store.index_knowledge_base(mock_engine)
        count_first = in_memory_store.total_documents()
        in_memory_store.index_knowledge_base(mock_engine)
        count_second = in_memory_store.total_documents()
        assert count_first == count_second

    def test_search_collection_type_mitre_only(self, indexed_store, mock_engine):
        hits = indexed_store.search_collection_type(
            mock_engine.embed("technique"), ["mitre"], k=3
        )
        for h in hits:
            assert h.collection == COL_MITRE

    def test_clear_collection(self, indexed_store):
        indexed_store.clear_collection(COL_MITRE)
        assert not indexed_store._already_indexed(COL_MITRE)

    def test_add_document_manually(self, in_memory_store, mock_engine):
        col = in_memory_store._col(COL_MITRE)
        col.add(
            ids=["T9999"],
            embeddings=[_fake_embedding()],
            texts=["test technique text"],
            metadatas=[{"entity_type": "mitre", "technique_id": "T9999"}],
        )
        assert col.count() == 1

    def test_metadata_preserved_in_hits(self, in_memory_store, mock_engine):
        col = in_memory_store._col(COL_MITRE)
        col.add(
            ids=["T9999"],
            embeddings=[_fake_embedding()],
            texts=["data encrypted for impact"],
            metadatas=[{"entity_type": "mitre", "technique_id": "T9999",
                        "technique_name": "Data Encrypted for Impact"}],
        )
        hits = in_memory_store.search(COL_MITRE, _fake_embedding(), k=1)
        assert len(hits) == 1
        assert hits[0].metadata["technique_id"] == "T9999"

    def test_mitre_doc_ids_are_technique_ids(self, indexed_store):
        hits = indexed_store.search(COL_MITRE, _fake_embedding(), k=9)
        ids = {h.doc_id for h in hits}
        # All MITRE doc IDs start with T
        assert all(i.startswith("T") for i in ids)

    def test_malware_doc_ids_have_malware_prefix(self, indexed_store):
        hits = indexed_store.search(COL_MALWARE, _fake_embedding(), k=6)
        ids = {h.doc_id for h in hits}
        assert all(i.startswith("malware:") for i in ids)

    def test_cve_doc_ids_start_with_cve(self, indexed_store):
        hits = indexed_store.search(COL_CVE, _fake_embedding(), k=6)
        ids = {h.doc_id for h in hits}
        assert all(i.startswith("CVE-") for i in ids)

    def test_apt_doc_ids_have_apt_prefix(self, indexed_store):
        hits = indexed_store.search(COL_APT, _fake_embedding(), k=6)
        ids = {h.doc_id for h in hits}
        assert all(i.startswith("apt:") for i in ids)


# ══════════════════════════════════════════════════════════════════════
# Section D — SemanticRetriever (injected engine + store)
# ══════════════════════════════════════════════════════════════════════

class TestSemanticRetriever:

    def test_retrieve_returns_results_object(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        assert isinstance(result, SemanticSearchResults)

    def test_retrieve_has_hits(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        assert isinstance(result.hits, list)

    def test_retrieve_respects_k(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"), k=3)
        assert len(result.hits) <= 3

    def test_retrieve_has_query_text(self, retriever):
        result = retriever.retrieve(normalize_text("APT28"))
        assert len(result.query_text) > 0

    def test_retrieve_total_hits_is_int(self, retriever):
        result = retriever.retrieve(normalize_text("APT28"))
        assert isinstance(result.total_hits, int)
        assert result.total_hits >= 0

    def test_retrieve_collections_searched_nonempty(self, retriever):
        result = retriever.retrieve(normalize_text("CVE-2021-44228"))
        assert len(result.collections_searched) > 0

    def test_retrieve_hits_are_semantic_hit_objects(self, retriever):
        result = retriever.retrieve(normalize_text("Emotet"))
        for h in result.hits:
            assert isinstance(h, SemanticHit)

    def test_semantic_hit_doc_id_nonempty(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        for h in result.hits:
            assert h.doc_id != ""

    def test_semantic_hit_text_nonempty(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        for h in result.hits:
            assert h.text != ""

    def test_semantic_hit_collection_nonempty(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        for h in result.hits:
            assert h.collection != ""

    def test_semantic_hit_score_in_bounds(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        for h in result.hits:
            assert 0.0 <= h.score <= 1.0

    def test_has_results_flag_true(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        assert result.has_results is True

    def test_cve_routes_to_cve_collection(self, retriever):
        result = retriever.retrieve(normalize_text("CVE-2021-44228"))
        assert COL_CVE in result.collections_searched

    def test_apt_routes_to_apt_collection(self, retriever):
        result = retriever.retrieve(normalize_text("APT28"))
        assert COL_APT in result.collections_searched

    def test_malware_routes_to_malware_collection(self, retriever):
        result = retriever.retrieve(normalize_text("Emotet"))
        assert COL_MALWARE in result.collections_searched

    def test_mitre_routes_to_mitre_collection(self, retriever):
        result = retriever.retrieve(normalize_text("T1486"))
        assert COL_MITRE in result.collections_searched

    def test_ip_routes_to_threat_intel(self, retriever):
        result = retriever.retrieve(normalize_text("185.220.101.33"))
        assert COL_THREAT_INTEL in result.collections_searched

    def test_nl_routes_to_all_collections(self, retriever):
        result = retriever.retrieve(normalize_text(
            "Attacker used PowerShell to download a beacon and moved laterally"
        ))
        assert len(result.collections_searched) == len(ALL_COLLECTIONS)

    def test_top_score_matches_max_hit_score(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry"))
        if result.hits:
            assert result.top_score == max(h.score for h in result.hits)
        else:
            assert result.top_score == 0.0

    def test_hits_sorted_by_score_descending(self, retriever):
        result = retriever.retrieve(normalize_text("ransomware lateral movement"), k=10)
        scores = [h.score for h in result.hits]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_doc_ids(self, retriever):
        result = retriever.retrieve(normalize_text("WannaCry ransomware"), k=15)
        ids = [h.doc_id for h in result.hits]
        assert len(ids) == len(set(ids))

    def test_investigate_returns_enriched_context(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert isinstance(ctx, EnrichedInvestigationContext)

    def test_investigate_preserves_request_id(self, retriever):
        req = normalize_text("APT28")
        ctx = retriever.investigate(req)
        assert ctx.request_id == req.request_id

    def test_investigate_has_aggregation_stage(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert "aggregation" in ctx.pipeline_stages

    def test_investigate_has_semantic_stage(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert "semantic" in ctx.pipeline_stages

    def test_investigate_has_semantic_results_field(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert isinstance(ctx.semantic_results, SemanticSearchResults)

    def test_investigate_preserves_matched_malware(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert isinstance(ctx.matched_malware, list)

    def test_investigate_preserves_matched_cves(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert isinstance(ctx.matched_cves, list)

    def test_investigate_preserves_matched_apts(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert isinstance(ctx.matched_apt_groups, list)

    def test_investigate_preserves_matched_techniques(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert isinstance(ctx.matched_techniques, list)

    def test_investigate_wannacry_finds_malware(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        names = [m.name for m in ctx.matched_malware]
        assert "WannaCry" in names

    def test_investigate_wannacry_finds_apt(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        names = [a.name for a in ctx.matched_apt_groups]
        assert "Lazarus Group" in names

    def test_investigate_has_graph(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        assert ctx.relationship_graph.node_count > 0

    def test_enriched_context_serialisable_to_json(self, retriever):
        ctx = retriever.investigate(normalize_text("WannaCry"))
        data = ctx.model_dump(mode="json")
        json.dumps(data)  # must not raise

    def test_investigate_unknown_query_no_crash(self, retriever):
        ctx = retriever.investigate(normalize_text("xxxxxxxxxnotamalwarexxxxxxxxx"))
        assert isinstance(ctx, EnrichedInvestigationContext)

    def test_semantic_failure_fallback(self, indexed_store):
        """Semantic failure must not kill aggregation."""
        broken = MagicMock(spec=EmbeddingEngine)
        broken.dimension = EMBEDDING_DIM
        broken.embed.side_effect = RuntimeError("model crash")
        broken.embed_batch.return_value = [_fake_embedding()] * 100

        r = SemanticRetriever(engine=broken, store=indexed_store)
        ctx = r.investigate(normalize_text("WannaCry"))
        # Aggregation data must be present
        names = [m.name for m in ctx.matched_malware]
        assert "WannaCry" in names
        # Semantic results should be gracefully empty
        assert ctx.semantic_results.has_results is False


# ══════════════════════════════════════════════════════════════════════
# Section E — Output model validation
# ══════════════════════════════════════════════════════════════════════

class TestSemanticSearchResults:

    def test_default_is_empty(self):
        r = SemanticSearchResults()
        assert r.hits == []
        assert r.has_results is False
        assert r.total_hits == 0
        assert r.top_score == 0.0

    def test_serialisable_to_json(self):
        r = SemanticSearchResults(
            hits=[SemanticHit(
                doc_id="T1486",
                text="ransomware",
                score=0.92,
                collection=COL_MITRE,
                entity_type="mitre",
            )],
            query_text="ransomware",
            total_hits=1,
            top_score=0.92,
            has_results=True,
        )
        json.dumps(r.model_dump())

    def test_hit_has_all_fields(self):
        h = SemanticHit(
            doc_id="T1486",
            text="Data Encrypted for Impact",
            score=0.85,
            collection=COL_MITRE,
            entity_type="mitre",
            metadata={"tactic": "impact"},
        )
        assert h.doc_id == "T1486"
        assert h.score == 0.85
        assert h.entity_type == "mitre"
        assert h.metadata["tactic"] == "impact"

    def test_collections_searched_is_list(self):
        r = SemanticSearchResults(collections_searched=[COL_MITRE, COL_MALWARE])
        assert isinstance(r.collections_searched, list)
        assert len(r.collections_searched) == 2


class TestEnrichedInvestigationContext:

    def test_is_subclass_of_investigation_context(self):
        from discovery.aggregator import InvestigationContext
        assert issubclass(EnrichedInvestigationContext, InvestigationContext)

    def test_has_semantic_results_field(self):
        ctx = EnrichedInvestigationContext(request_id="x", input_type="malware_name")
        assert hasattr(ctx, "semantic_results")
        assert isinstance(ctx.semantic_results, SemanticSearchResults)

    def test_has_pipeline_stages_field(self):
        ctx = EnrichedInvestigationContext(request_id="x", input_type="malware_name")
        assert isinstance(ctx.pipeline_stages, list)

    def test_serialisable_empty(self):
        ctx = EnrichedInvestigationContext(request_id="x", input_type="cve_id")
        json.dumps(ctx.model_dump(mode="json"))

    def test_aggregation_fields_present(self):
        ctx = EnrichedInvestigationContext(request_id="x", input_type="cve_id")
        assert hasattr(ctx, "matched_malware")
        assert hasattr(ctx, "matched_cves")
        assert hasattr(ctx, "matched_apt_groups")
        assert hasattr(ctx, "matched_techniques")
        assert hasattr(ctx, "relationship_graph")
