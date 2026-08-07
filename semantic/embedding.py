"""
semantic/embedding.py — Local Embedding Engine.

Converts cybersecurity entities and raw analyst text into dense vector
representations using a locally-cached sentence-transformers model.

Model: all-MiniLM-L6-v2
    - 22M parameters, 80 MB on disk
    - 384-dimensional output vectors
    - Cosine similarity compatible
    - Excellent at short/medium text semantic matching
    - Downloads once to the HuggingFace cache (~/.cache/huggingface/)

Key responsibilities:
    1. Provide a singleton model (load once per process)
    2. Convert any entity type → embeddable text string
    3. Embed single texts or batches efficiently
    4. Zero external API calls — everything runs on local CPU

Design choices:
    - SentenceTransformer is imported at module level so unittest.mock.patch()
      can target it as `semantic.embedding.SentenceTransformer`.
    - `entity_to_text()` is pure Python — can be tested without the model.
    - Batch embedding is always used internally for efficiency.
    - Embeddings are returned as list[float] for JSON-serialisability.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Module-level import so patch("semantic.embedding.SentenceTransformer") works.
# If sentence-transformers is not installed the attribute is None and
# _load_model() raises a helpful ImportError at runtime.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment, misc]

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384

# Default model — small, fast, high quality for semantic search
DEFAULT_MODEL = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════
# Entity → Text conversion
# ══════════════════════════════════════════════════════════════════════

def entity_to_text(entity_type: str, data: dict[str, Any]) -> str:
    """
    Convert a cybersecurity entity to a rich text string for embedding.

    The text combines all relevant fields so that semantic search can
    match across names, descriptions, techniques, sectors, and keywords.

    Args:
        entity_type: One of "mitre", "malware", "apt", "cve",
                     "ip", "domain", "hash", "nl"
        data: Entity fields dict (matches knowledge base JSON schemas)

    Returns:
        Human-readable text string optimised for embedding.
    """
    if entity_type == "mitre":
        tid   = data.get("technique_id", "")
        tname = data.get("technique_name", "")
        tact  = data.get("tactic", "")
        kws   = " ".join(data.get("keywords", []))
        return (
            f"MITRE ATT&CK {tid} {tname} | Tactic: {tact} | "
            f"Keywords: {kws}"
        )

    elif entity_type == "malware":
        name    = data.get("name", "")
        aliases = " ".join(data.get("aliases", []))
        ftype   = data.get("type", "")
        attrs   = " ".join(data.get("attributed_to", []))
        sectors = " ".join(data.get("targeted_sectors", []))
        techs   = " ".join(data.get("mitre_techniques", []))
        cves    = " ".join(data.get("cves_exploited", []))
        desc    = data.get("description", "")[:300]
        kill    = " | ".join(data.get("kill_chain", []))
        return (
            f"Malware: {name} | Aliases: {aliases} | Type: {ftype} | "
            f"Attributed to: {attrs} | Targets: {sectors} | "
            f"Techniques: {techs} | CVEs: {cves} | "
            f"Kill chain: {kill} | {desc}"
        )

    elif entity_type == "apt":
        name      = data.get("name", "")
        aliases   = " ".join(data.get("aliases", []))
        country   = data.get("country", "")
        sponsor   = data.get("sponsor", "")
        sectors   = " ".join(data.get("targeted_sectors", []))
        techs     = " ".join(data.get("mitre_techniques", []))
        malware   = " ".join(data.get("malware_used", []))
        campaigns = " | ".join(data.get("known_campaigns", []))
        desc      = data.get("description", "")[:300]
        return (
            f"APT Group: {name} | Aliases: {aliases} | "
            f"Country: {country} | Sponsor: {sponsor} | "
            f"Targets: {sectors} | Techniques: {techs} | "
            f"Malware: {malware} | Campaigns: {campaigns} | {desc}"
        )

    elif entity_type == "cve":
        cve_id   = data.get("cve_id", "")
        name     = data.get("name", "")
        cvss     = data.get("cvss_score", 0)
        severity = data.get("severity", "")
        products = " ".join(data.get("affected_products", []))
        malware  = " ".join(data.get("related_malware", []))
        apts     = " ".join(data.get("related_apt_groups", []))
        kws      = " ".join(data.get("keywords", []))
        desc     = data.get("description", "")[:300]
        status   = data.get("exploitation_status", "")
        return (
            f"CVE: {cve_id} ({name}) | CVSS: {cvss} {severity} | "
            f"Products: {products} | Status: {status} | "
            f"Related malware: {malware} | APTs: {apts} | "
            f"Keywords: {kws} | {desc}"
        )

    elif entity_type == "ip":
        ip   = data.get("ip", "")
        tags = " ".join(data.get("tags", []))
        src  = data.get("source", "")
        return f"Malicious IP: {ip} | Tags: {tags} | Source: {src}"

    elif entity_type == "domain":
        dom  = data.get("domain", "")
        tags = " ".join(data.get("tags", []))
        src  = data.get("source", "")
        return f"Malicious domain: {dom} | Tags: {tags} | Source: {src}"

    elif entity_type == "hash":
        h    = data.get("hash", "")
        fam  = data.get("malware_family", "")
        htype = data.get("type", "")
        return f"Malicious hash: {h} | Malware family: {fam} | Type: {htype}"

    else:
        # Natural language / free-form — use as-is, truncated
        return str(data.get("text", data.get("description", str(data))))[:500]


# ══════════════════════════════════════════════════════════════════════
# Embedding Engine (singleton)
# ══════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Wraps sentence-transformers for local embedding generation.

    Usage:
        engine = EmbeddingEngine.get()          # singleton
        vec = engine.embed("Cobalt Strike C2")  # list[float], len=384

    The model is loaded on the first call to get() or embed().
    Subsequent calls use the cached instance.
    """

    _instance: "EmbeddingEngine | None" = None
    _model = None   # sentence_transformers.SentenceTransformer

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        """Load the sentence-transformers model (downloads once if needed)."""
        # Reference the module-level SentenceTransformer so that
        # unittest.mock.patch("semantic.embedding.SentenceTransformer", ...)
        # is picked up during tests without a real model download.
        import semantic.embedding as _self_module
        _ST = _self_module.SentenceTransformer
        if _ST is None:
            raise ImportError(
                "sentence-transformers is required for the semantic layer. "
                "Install with: pip install sentence-transformers"
            )
        logger.info("Loading embedding model '%s'…", self._model_name)
        self._model = _ST(self._model_name)
        logger.info(
            "Embedding model loaded. Dimension: %d",
            self._model.get_sentence_embedding_dimension(),
        )

    @classmethod
    def get(cls, model_name: str = DEFAULT_MODEL) -> "EmbeddingEngine":
        """Return the singleton EmbeddingEngine, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — used in tests to inject a mock."""
        cls._instance = None
        cls._model = None

    @property
    def dimension(self) -> int:
        """Output embedding dimension."""
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return EMBEDDING_DIM

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Any text — entity description, query, incident report…

        Returns:
            list[float] of length self.dimension (384 for all-MiniLM-L6-v2)
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        result = self._model.encode(
            [text],
            normalize_embeddings=True,   # L2-normalised → cosine sim = dot product
            show_progress_bar=False,
        )
        return result[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a single forward pass (faster than one-by-one).

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        results = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [r.tolist() for r in results]

    def embed_entity(self, entity_type: str, data: dict[str, Any]) -> list[float]:
        """
        Convert an entity dict to text, then embed it.

        Args:
            entity_type: See entity_to_text() for valid types.
            data: Entity fields dict.

        Returns:
            Embedding vector as list[float].
        """
        text = entity_to_text(entity_type, data)
        return self.embed(text)
