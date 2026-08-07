"""
semantic/vector_store.py — Local HNSW Vector Store.

Implements a persistent, file-based vector store using:
    - hnswlib  : Hierarchical Navigable Small World graph (HNSW) for
                 approximate nearest-neighbour search (ANN)
    - JSON     : Lightweight metadata + document storage per collection

Why NOT ChromaDB?
    ChromaDB depends on Pydantic v1's BaseSettings which is incompatible
    with Python 3.14. hnswlib is a single C++ extension, minimal deps,
    and teaches the underlying ANN algorithm directly — better for a capstone.

Collections (one HNSW index + JSON sidecar per entity type):
    mitre_techniques — 9 ATT&CK technique entries
    malware          — 6 malware family profiles
    cve              — 6 CVE entries with CVSS data
    apt_groups       — 6 APT group profiles
    threat_intel     — known-bad IPs, hashes, domains

Storage layout (data/vector_store/):
    {collection}.bin   — serialised HNSW index
    {collection}.json  — parallel doc array (text + metadata per vector)

Design:
    - Index is built on first query (lazy, one-time cost)
    - In-memory mode supported for tests (no disk I/O)
    - Cosine similarity via L2-normalised vectors + inner product space
    - The embedding engine is injected — fully testable without real model
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from semantic.embedding import EmbeddingEngine, entity_to_text

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
_ROOT  = Path(__file__).resolve().parent.parent
_DATA  = _ROOT / "data"
_VS_PATH = _DATA / "vector_store"

# ── Collection names ───────────────────────────────────────────────────
COL_MITRE        = "mitre_techniques"
COL_MALWARE      = "malware"
COL_CVE          = "cve"
COL_APT          = "apt_groups"
COL_THREAT_INTEL = "threat_intel"

ALL_COLLECTIONS = [COL_MITRE, COL_MALWARE, COL_CVE, COL_APT, COL_THREAT_INTEL]

# HNSW build parameters
_HNSW_M   = 16    # number of bi-directional links per element
_HNSW_EF_CONSTRUCTION = 200


# ══════════════════════════════════════════════════════════════════════
# Result models
# ══════════════════════════════════════════════════════════════════════

class SearchHit(BaseModel):
    """A single result from a vector similarity search."""
    doc_id:     str
    text:       str
    score:      float = Field(ge=0.0, le=1.0)   # cosine similarity [0, 1]
    collection: str
    metadata:   dict[str, Any] = Field(default_factory=dict)


class CollectionStats(BaseModel):
    name:  str
    count: int


# ══════════════════════════════════════════════════════════════════════
# Per-collection HNSW index
# ══════════════════════════════════════════════════════════════════════

class HNSWCollection:
    """
    One HNSW index + parallel JSON sidecar.

    Documents are stored as:
        _docs[i] = {"id": str, "text": str, "metadata": dict}

    HNSW label for document i is simply i (int).
    Cosine similarity is achieved by storing L2-normalised vectors
    in an inner-product (ip) space: sim = dot(a, b) since |a|=|b|=1.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        persist_path: Path | None = None,
    ) -> None:
        self.name = name
        self.dim  = dim
        self._persist_path = persist_path
        self._docs: list[dict[str, Any]] = []
        self._index = None   # hnswlib.Index, created on first add
        self._built = False

    # ── Persistence ─────────────────────────────────────────────────

    @property
    def _index_file(self) -> Path | None:
        return (self._persist_path / f"{self.name}.bin") if self._persist_path else None

    @property
    def _meta_file(self) -> Path | None:
        return (self._persist_path / f"{self.name}.json") if self._persist_path else None

    def _save(self) -> None:
        if self._persist_path is None or self._index is None:
            return
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self._index_file))
        self._meta_file.write_text(
            json.dumps(self._docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Saved collection '%s' (%d docs)", self.name, len(self._docs))

    def _load(self) -> bool:
        """Return True if loaded from disk successfully."""
        if self._persist_path is None:
            return False
        if not (self._index_file.exists() and self._meta_file.exists()):
            return False
        try:
            import hnswlib
            idx = hnswlib.Index(space="ip", dim=self.dim)
            idx.load_index(str(self._index_file), max_elements=100_000)
            self._docs = json.loads(self._meta_file.read_text(encoding="utf-8"))
            self._index = idx
            self._built = True
            logger.debug("Loaded collection '%s' (%d docs)", self.name, len(self._docs))
            return True
        except Exception as exc:
            logger.warning("Could not load collection '%s': %s", self.name, exc)
            return False

    # ── Document management ──────────────────────────────────────────

    def count(self) -> int:
        return len(self._docs)

    def _init_index(self) -> None:
        """Initialise HNSW index if not already done."""
        if self._index is None:
            import hnswlib
            self._index = hnswlib.Index(space="ip", dim=self.dim)
            self._index.init_index(
                max_elements=10_000,
                ef_construction=_HNSW_EF_CONSTRUCTION,
                M=_HNSW_M,
            )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents to the collection."""
        assert len(ids) == len(embeddings) == len(texts) == len(metadatas)
        if not ids:
            return

        self._init_index()

        # L2-normalise each vector for cosine similarity via ip space
        vecs = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        start_label = len(self._docs)
        labels = list(range(start_label, start_label + len(ids)))

        self._index.add_items(vecs, labels)
        for doc_id, text, meta in zip(ids, texts, metadatas):
            self._docs.append({"id": doc_id, "text": text, "metadata": meta})

        self._built = True
        self._save()

    def query(self, embedding: list[float], k: int = 5) -> list[SearchHit]:
        """Return top-k nearest neighbours."""
        if not self._built or self._index is None or len(self._docs) == 0:
            return []

        k = min(k, len(self._docs))
        # L2-normalise query
        vec = np.array([embedding], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # Set ef (query-time parameter) — higher = more accurate, slower
        self._index.set_ef(max(k * 10, 50))
        labels, distances = self._index.knn_query(vec, k=k)

        hits: list[SearchHit] = []
        for label, dist in zip(labels[0], distances[0]):
            label = int(label)
            if label >= len(self._docs):
                continue
            doc = self._docs[label]
            # For ip-space with L2-normalised vectors:
            # inner product = cosine similarity ∈ [-1, 1]
            # hnswlib returns negative inner product as distance
            similarity = float(max(0.0, min(1.0, 1.0 - dist)))
            hits.append(SearchHit(
                doc_id=doc["id"],
                text=doc["text"],
                score=round(similarity, 4),
                collection=self.name,
                metadata=doc["metadata"],
            ))

        return sorted(hits, key=lambda h: -h.score)

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        self._docs = []
        self._index = None
        self._built = False
        if self._persist_path:
            if self._index_file and self._index_file.exists():
                self._index_file.unlink()
            if self._meta_file and self._meta_file.exists():
                self._meta_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# KnowledgeVectorStore
# ══════════════════════════════════════════════════════════════════════

class KnowledgeVectorStore:
    """
    Local HNSW-backed vector store for the AI Cyber Discovery Engine.

    Usage (production — persistent to disk):
        store = KnowledgeVectorStore()
        store.index_knowledge_base(engine)          # one-time; skipped if loaded from disk
        hits  = store.search_all(embedding, k=5)

    Usage (testing — in-memory only):
        store = KnowledgeVectorStore(persist=False)
    """

    def __init__(
        self,
        persist: bool = True,
        persist_path: Path | str | None = None,
        dim: int = 384,
    ) -> None:
        """
        Args:
            persist:      If True, save/load indices to/from disk.
            persist_path: Override directory. Defaults to data/vector_store/.
            dim:          Embedding dimension (must match EmbeddingEngine).
        """
        self._dim = dim
        self._persist_path: Path | None = None
        if persist:
            self._persist_path = Path(persist_path) if persist_path else _VS_PATH

        self._collections: dict[str, HNSWCollection] = {
            name: HNSWCollection(name, dim, self._persist_path)
            for name in ALL_COLLECTIONS
        }
        self._indexed = False

        # Try loading from disk
        if persist:
            self._try_load_from_disk()

    def _try_load_from_disk(self) -> None:
        all_loaded = all(
            col._load() for col in self._collections.values()
        )
        if all_loaded:
            self._indexed = True

    def _col(self, name: str) -> HNSWCollection:
        return self._collections[name]

    # ── Knowledge base indexing ────────────────────────────────────────

    def index_knowledge_base(self, engine: EmbeddingEngine) -> dict[str, int]:
        """
        Index all local knowledge base JSON files.

        Idempotent — skips collections that already have documents.

        Returns:
            {collection_name: docs_indexed_this_call}
        """
        counts: dict[str, int] = {}
        counts[COL_MITRE]        = self._index_mitre(engine)
        counts[COL_MALWARE]      = self._index_malware(engine)
        counts[COL_CVE]          = self._index_cve(engine)
        counts[COL_APT]          = self._index_apt(engine)
        counts[COL_THREAT_INTEL] = self._index_threat_intel(engine)
        self._indexed = True
        logger.info("Knowledge base indexing complete: %s", counts)
        return counts

    def _index_mitre(self, engine: EmbeddingEngine) -> int:
        col = self._col(COL_MITRE)
        if col.count() > 0:
            return 0
        data = _load_json(_DATA / "mitre_attack.json").get("techniques", [])
        if not data:
            return 0
        texts = [entity_to_text("mitre", t) for t in data]
        embeddings = engine.embed_batch(texts)
        ids = [t["technique_id"] for t in data]
        metadatas = [
            {
                "technique_id": t["technique_id"],
                "technique_name": t["technique_name"],
                "tactic": t["tactic"],
                "tactic_order": t.get("tactic_order", 0),
                "entity_type": "mitre",
            }
            for t in data
        ]
        col.add(ids=ids, embeddings=embeddings, texts=texts, metadatas=metadatas)
        logger.info("Indexed %d MITRE techniques.", len(ids))
        return len(ids)

    def _index_malware(self, engine: EmbeddingEngine) -> int:
        col = self._col(COL_MALWARE)
        if col.count() > 0:
            return 0
        families = _load_json(_DATA / "malware_families.json").get("families", [])
        if not families:
            return 0
        texts = [entity_to_text("malware", f) for f in families]
        embeddings = engine.embed_batch(texts)
        ids = [f"malware:{f['name'].lower().replace(' ', '_')}" for f in families]
        metadatas = [
            {
                "name": f["name"],
                "type": f.get("type", ""),
                "attributed_to": json.dumps(f.get("attributed_to", [])),
                "targeted_sectors": json.dumps(f.get("targeted_sectors", [])),
                "mitre_techniques": json.dumps(f.get("mitre_techniques", [])),
                "cves_exploited": json.dumps(f.get("cves_exploited", [])),
                "entity_type": "malware",
            }
            for f in families
        ]
        col.add(ids=ids, embeddings=embeddings, texts=texts, metadatas=metadatas)
        logger.info("Indexed %d malware families.", len(ids))
        return len(ids)

    def _index_cve(self, engine: EmbeddingEngine) -> int:
        col = self._col(COL_CVE)
        if col.count() > 0:
            return 0
        cves = _load_json(_DATA / "cve_db.json").get("cves", [])
        if not cves:
            return 0
        texts = [entity_to_text("cve", c) for c in cves]
        embeddings = engine.embed_batch(texts)
        ids = [c["cve_id"] for c in cves]
        metadatas = [
            {
                "cve_id": c["cve_id"],
                "name": c.get("name", ""),
                "cvss_score": c.get("cvss_score", 0.0),
                "severity": c.get("severity", ""),
                "cisa_kev": c.get("cisa_kev", False),
                "exploitation_status": c.get("exploitation_status", ""),
                "related_malware": json.dumps(c.get("related_malware", [])),
                "related_apt_groups": json.dumps(c.get("related_apt_groups", [])),
                "entity_type": "cve",
            }
            for c in cves
        ]
        col.add(ids=ids, embeddings=embeddings, texts=texts, metadatas=metadatas)
        logger.info("Indexed %d CVEs.", len(ids))
        return len(ids)

    def _index_apt(self, engine: EmbeddingEngine) -> int:
        col = self._col(COL_APT)
        if col.count() > 0:
            return 0
        groups = _load_json(_DATA / "apt_groups.json").get("groups", [])
        if not groups:
            return 0
        texts = [entity_to_text("apt", g) for g in groups]
        embeddings = engine.embed_batch(texts)
        ids = [f"apt:{g['name'].lower().replace(' ', '_')}" for g in groups]
        metadatas = [
            {
                "name": g["name"],
                "country": g.get("country", ""),
                "sponsor": g.get("sponsor", ""),
                "mitre_group_id": g.get("mitre_group_id", ""),
                "targeted_sectors": json.dumps(g.get("targeted_sectors", [])),
                "malware_used": json.dumps(g.get("malware_used", [])),
                "entity_type": "apt",
            }
            for g in groups
        ]
        col.add(ids=ids, embeddings=embeddings, texts=texts, metadatas=metadatas)
        logger.info("Indexed %d APT groups.", len(ids))
        return len(ids)

    def _index_threat_intel(self, engine: EmbeddingEngine) -> int:
        col = self._col(COL_THREAT_INTEL)
        if col.count() > 0:
            return 0
        data = _load_json(_DATA / "threat_intel.json")
        all_texts: list[str] = []
        all_ids: list[str] = []
        all_meta: list[dict] = []

        for entry in data.get("malicious_ips", []):
            txt = entity_to_text("ip", entry)
            all_texts.append(txt)
            all_ids.append(f"ip:{entry['ip']}")
            all_meta.append({
                "value": entry["ip"], "ioc_type": "ip",
                "tags": json.dumps(entry.get("tags", [])),
                "confidence": entry.get("confidence", 0.5),
                "entity_type": "threat_intel",
            })
        for entry in data.get("malicious_hashes", []):
            txt = entity_to_text("hash", entry)
            all_texts.append(txt)
            all_ids.append(f"hash:{entry['hash'][:16]}")
            all_meta.append({
                "value": entry["hash"], "ioc_type": "hash",
                "malware_family": entry.get("malware_family", ""),
                "confidence": entry.get("confidence", 0.5),
                "entity_type": "threat_intel",
            })
        for entry in data.get("malicious_domains", []):
            txt = entity_to_text("domain", entry)
            all_texts.append(txt)
            all_ids.append(f"domain:{entry['domain'][:30]}")
            all_meta.append({
                "value": entry["domain"], "ioc_type": "domain",
                "tags": json.dumps(entry.get("tags", [])),
                "confidence": entry.get("confidence", 0.5),
                "entity_type": "threat_intel",
            })

        if not all_texts:
            return 0
        embeddings = engine.embed_batch(all_texts)
        col.add(ids=all_ids, embeddings=embeddings, texts=all_texts, metadatas=all_meta)
        logger.info("Indexed %d threat intel entries.", len(all_ids))
        return len(all_ids)

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        collection_name: str,
        embedding: list[float],
        k: int = 5,
    ) -> list[SearchHit]:
        """Search a single named collection."""
        return self._col(collection_name).query(embedding, k=k)

    def search_all(
        self,
        embedding: list[float],
        k: int = 5,
    ) -> list[SearchHit]:
        """Search all collections, return globally ranked top-k (deduped)."""
        all_hits: list[SearchHit] = []
        for name in ALL_COLLECTIONS:
            try:
                all_hits.extend(self._col(name).query(embedding, k=k))
            except Exception:
                logger.exception("Error searching collection %s", name)

        all_hits.sort(key=lambda h: -h.score)
        seen: set[str] = set()
        deduped: list[SearchHit] = []
        for h in all_hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                deduped.append(h)
        return deduped[:k]

    def search_collection_type(
        self,
        embedding: list[float],
        entity_types: list[str],
        k: int = 5,
    ) -> list[SearchHit]:
        """Search only collections matching the given type strings."""
        type_to_col = {
            "mitre": COL_MITRE, "malware": COL_MALWARE,
            "cve": COL_CVE, "apt": COL_APT, "threat_intel": COL_THREAT_INTEL,
        }
        all_hits: list[SearchHit] = []
        for et in entity_types:
            col_name = type_to_col.get(et)
            if col_name:
                all_hits.extend(self._col(col_name).query(embedding, k=k))
        all_hits.sort(key=lambda h: -h.score)
        return all_hits[:k]

    # ── Stats / maintenance ────────────────────────────────────────────

    def get_stats(self) -> list[CollectionStats]:
        return [
            CollectionStats(name=name, count=self._col(name).count())
            for name in ALL_COLLECTIONS
        ]

    def total_documents(self) -> int:
        return sum(self._col(n).count() for n in ALL_COLLECTIONS)

    def clear_collection(self, collection_name: str) -> None:
        """Delete all documents in a collection (used in tests)."""
        self._col(collection_name).clear()

    def _already_indexed(self, collection_name: str) -> bool:
        return self._col(collection_name).count() > 0


# ══════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return {}
