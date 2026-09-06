"""
Knowledge store for the agricultural RAG pipeline.

This is the MVP vector storage layer (Phase 6). MongoDB Atlas Vector Search is
NOT provisioned in this project (the app runs on the in-memory fallback), so per
the task's guidance we use the simplest reliable approach instead of standing up
a vector database:

    knowledge documents (data/agricultural_knowledge.json)
        -> ingest_knowledge.py embeds them
        -> a JSON index of {id, text, embedding, metadata} (data/knowledge_index.json)
        -> loaded once, cached in memory
        -> cosine similarity at query time -> top-k chunks

The store is deliberately provider-agnostic: it holds vectors and does the maths,
but knows nothing about DashScope. Swapping in a real vector DB later means
replacing this module, not the RAG route.

Nothing here needs the SDK or a key, so it imports safely everywhere.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# backend/app/services/knowledge_store.py -> parents[2] == backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _BACKEND_ROOT / "data"

# Module-level so tests can point these at fixtures via monkeypatch.
DOCUMENTS_PATH = _DATA_DIR / "agricultural_knowledge.json"
INDEX_PATH = _DATA_DIR / "knowledge_index.json"

# Rough chunk size for splitting long documents. The MVP corpus is already made
# of short passages, so most documents become a single chunk.
_CHUNK_MAX_CHARS = 600


@dataclass
class KnowledgeChunk:
    """A stored passage: its text, embedding vector, and citation metadata."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "KnowledgeChunk":
        return cls(
            id=raw["id"],
            text=raw["text"],
            metadata=raw.get("metadata", {}),
            embedding=list(raw.get("embedding", [])),
        )


# ── Document loading + chunking ─────────────────────────────────────────────


def load_documents() -> list[dict]:
    """
    Load raw knowledge documents from the JSON corpus.

    Returns the `documents` list (each a dict with content + metadata). Missing
    file or malformed JSON yields an empty list with a logged warning rather than
    an exception, so a broken corpus never crashes the app.
    """
    try:
        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logger.warning("Knowledge corpus not found at %s", DOCUMENTS_PATH)
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read knowledge corpus: %s", exc)
        return []

    documents = data.get("documents", []) if isinstance(data, dict) else []
    return [d for d in documents if isinstance(d, dict) and d.get("content")]


def chunk_text(text: str, max_chars: int = _CHUNK_MAX_CHARS) -> list[str]:
    """
    Split text into chunks no larger than `max_chars`, breaking on sentence
    boundaries so a chunk is never cut mid-sentence.

    Short text (the common case for this corpus) returns a single chunk.
    """
    text = " ".join((text or "").split())  # collapse whitespace
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = text.replace("! ", ". ").replace("? ", ". ").split(". ")
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        piece = sentence.strip()
        if not piece:
            continue
        candidate = f"{current}. {piece}" if current else piece
        if len(candidate) > max_chars and current:
            chunks.append(current.strip(". ").strip() + ".")
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current.strip(". ").strip() + ".")
    return chunks


def documents_to_chunks(documents: list[dict]) -> list[KnowledgeChunk]:
    """
    Turn raw documents into embed-ready chunks (without embeddings yet).

    A document that splits into several chunks gets ids like `wheat-intro#1`.
    Metadata (title, crop, topic, language, source, source_url) is carried on
    every chunk so retrieval stays source-aware.
    """
    chunks: list[KnowledgeChunk] = []
    for doc in documents:
        pieces = chunk_text(doc.get("content", ""))
        for i, piece in enumerate(pieces):
            base_id = str(doc.get("id") or doc.get("title") or "doc")
            chunk_id = base_id if len(pieces) == 1 else f"{base_id}#{i + 1}"
            metadata = {
                "title": doc.get("title", ""),
                "crop": doc.get("crop", ""),
                "topic": doc.get("topic", ""),
                "language": doc.get("language", ""),
                "source": doc.get("source", ""),
                "source_url": doc.get("source_url", ""),
            }
            chunks.append(KnowledgeChunk(id=chunk_id, text=piece, metadata=metadata))
    return chunks


# ── Index persistence ───────────────────────────────────────────────────────


def save_index(chunks: list[KnowledgeChunk]) -> None:
    """Write the embedded chunks to the JSON index (used by ingestion)."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunks": [c.to_dict() for c in chunks]}
    with open(INDEX_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    reset_cache()
    logger.info("Wrote knowledge index: %d chunks -> %s", len(chunks), INDEX_PATH)


def index_exists() -> bool:
    return INDEX_PATH.exists()


def _load_index_from_disk() -> list[KnowledgeChunk]:
    if not INDEX_PATH.exists():
        return []
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read knowledge index: %s", exc)
        return []
    raw_chunks = data.get("chunks", []) if isinstance(data, dict) else []
    return [KnowledgeChunk.from_dict(c) for c in raw_chunks if c.get("embedding")]


# ── In-memory cache ─────────────────────────────────────────────────────────

_index_cache: list[KnowledgeChunk] | None = None


def get_index() -> list[KnowledgeChunk]:
    """
    Return the embedded chunks, loading from disk once and caching in memory.

    Caching (Phase 16) means the index is read from disk at most once per process
    rather than on every request.
    """
    global _index_cache
    if _index_cache is None:
        _index_cache = _load_index_from_disk()
    return _index_cache


def set_index(chunks: list[KnowledgeChunk]) -> None:
    """Replace the in-memory index directly (used by tests and ingestion)."""
    global _index_cache
    _index_cache = list(chunks)


def reset_cache() -> None:
    """Drop the cache so the next get_index() reloads from disk."""
    global _index_cache
    _index_cache = None


def indexed_count() -> int:
    """Number of embedded chunks currently available for retrieval."""
    return len(get_index())


# ── Similarity search ───────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity of two vectors, in pure Python (no numpy dependency).

    Returns 0.0 for a zero-length vector or a dimension mismatch rather than
    raising, so a malformed stored vector can never crash retrieval.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(query_vector: list[float], top_k: int, min_score: float) -> list[tuple[KnowledgeChunk, float]]:
    """
    Rank indexed chunks against `query_vector` by cosine similarity.

    Returns up to `top_k` (chunk, score) pairs whose score meets `min_score`,
    highest first. An empty index or an all-below-threshold result yields [].
    """
    scored = [
        (chunk, cosine_similarity(query_vector, chunk.embedding))
        for chunk in get_index()
        if chunk.embedding
    ]
    scored = [(c, s) for c, s in scored if s >= min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[: max(0, top_k)]
