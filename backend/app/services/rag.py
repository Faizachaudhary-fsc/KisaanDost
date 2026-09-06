"""
Agricultural RAG retrieval service.

Turns a farmer's question into a small set of grounding passages drawn from the
local agricultural knowledge base (data/agricultural_knowledge.json), embedded
with Alibaba Cloud's multilingual text-embedding model and matched by cosine
similarity.

Pipeline:
    query -> embed_query -> cosine similarity over the indexed chunks
          -> top-k above a relevance threshold -> RetrievalResult

Design guarantees the voice route depends on:
- The public interface is unchanged: `retrieve(query)` returns a RetrievalResult
  exposing `.context`, so app/routes/assistant.py needs no edits.
- retrieve() NEVER raises. A missing key, an unbuilt index, or a provider error
  all degrade to an EMPTY result (Phase 9 low-confidence / Phase 15 resilience),
  and the LLM stage then answers from general agronomy and states its
  uncertainty — the request is never failed because of RAG.
- Source metadata rides along on every chunk so answers can be made
  source-aware later without redesigning retrieval (Phase 11).
"""

import logging
import re
from dataclasses import dataclass, field

from app.config import settings
from app.services import knowledge_store

logger = logging.getLogger(__name__)

IMPLEMENTED = True

DEFAULT_TOP_K = settings.RAG_TOP_K


@dataclass
class RetrievedChunk:
    """A single grounding passage plus provenance for citation."""

    text: str
    source: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Everything the LLM stage needs to build a grounded prompt."""

    chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def context(self) -> str:
        """
        Chunks flattened into a single prompt-ready context block.

        Each passage is numbered and labelled with its crop/topic and source, so
        the LLM can tell grounding material apart from the question and cite where
        an answer came from (Phase 10).
        """
        blocks = []
        for i, c in enumerate(self.chunks, start=1):
            crop = c.metadata.get("crop", "")
            topic = c.metadata.get("topic", "")
            tag = " / ".join(p for p in (crop, topic) if p)
            header = f"[{i}]" + (f" ({tag})" if tag else "") + (f" — Source: {c.source}" if c.source else "")
            blocks.append(f"{header}\n{c.text}")
        return "\n\n".join(blocks)

    @property
    def sources(self) -> list[dict]:
        """Distinct source metadata, kept for a future optional 'sources' feature."""
        seen: list[dict] = []
        keys = set()
        for c in self.chunks:
            key = (c.source, c.metadata.get("source_url", ""))
            if key not in keys:
                keys.add(key)
                seen.append(
                    {
                        "title": c.metadata.get("title", ""),
                        "source": c.source,
                        "source_url": c.metadata.get("source_url", ""),
                    }
                )
        return seen

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def top_score(self) -> float:
        return max((c.score for c in self.chunks), default=0.0)


_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_ALIASES = {
    "گندم": {"wheat", "gandum", "گندم"},
    "پانی": {"water", "irrigation", "pani", "پانی"},
    "پیلا": {"yellow", "nitrogen", "yellowing", "پیلے", "پیلا"},
    "پتے": {"leaves", "leaf", "پتے", "پتا"},
    "کپاس": {"cotton", "کپاس"},
    "کیڑے": {"insect", "insects", "pest", "pests", "whitefly", "کیڑے"},
    "چاول": {"rice", "چاول"},
}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) > 1}


def _local_score(query_tokens: set[str], chunk: knowledge_store.KnowledgeChunk) -> float:
    content = f"{chunk.text} {chunk.metadata.get('crop', '')} {chunk.metadata.get('topic', '')}".lower()
    content_tokens = _tokens(content)
    expanded = set(query_tokens)
    for key, aliases in _ALIASES.items():
        if key in query_tokens or query_tokens.intersection(aliases):
            expanded.update(aliases)
    overlap = len(expanded.intersection(content_tokens))
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(expanded))


async def retrieve(query: str, top_k: int | None = None) -> RetrievalResult:
    """
    Fetch grounding passages relevant to `query`.

    Uses the local agricultural JSON corpus directly. This keeps voice RAG
    available without an external embedding provider or generated index.
    """
    if not query or not query.strip():
        return RetrievalResult(chunks=[])

    k = top_k if top_k is not None else settings.RAG_TOP_K

    try:
        chunks = knowledge_store.documents_to_chunks(knowledge_store.load_documents())
        query_tokens = _tokens(query)
        hits = [
            (chunk, _local_score(query_tokens, chunk))
            for chunk in chunks
        ]
        hits = [(chunk, score) for chunk, score in hits if score > 0]
        hits.sort(key=lambda pair: pair[1], reverse=True)
        hits = hits[:k]

        chunks = [
            RetrievedChunk(
                text=chunk.text,
                source=chunk.metadata.get("source", ""),
                score=round(score, 4),
                metadata=chunk.metadata,
            )
            for chunk, score in hits
        ]
        logger.info(
            "rag.retrieve: query_chars=%d matched=%d top_score=%.3f",
            len(query),
            len(chunks),
            chunks[0].score if chunks else 0.0,
        )
        return RetrievalResult(chunks=chunks)

    except Exception:  # noqa: BLE001 - retrieval must never fail the request
        # Logged with traceback for us; the pipeline continues without context.
        logger.exception("local rag.retrieve failed; degrading to empty context")
        return RetrievalResult(chunks=[])


def status() -> dict:
    """
    RAG readiness for /health (Phase 17). Never exposes credentials.

    - "unavailable": no embedding provider (SDK/key), so queries cannot be
      embedded and retrieval always returns empty.
    - "not_indexed": provider configured but the vector index has not been built
      yet (run scripts.ingest_knowledge).
    - "ready": provider configured and an index is loaded.
    """
    indexed = len(knowledge_store.documents_to_chunks(knowledge_store.load_documents()))
    state = "ready" if indexed else "unavailable"
    return {
        "status": state,
        "indexed_chunks": indexed,
        "embedding_model": "local-keyword-retrieval",
    }
