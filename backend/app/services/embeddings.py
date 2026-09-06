"""
Text embedding service — Alibaba Cloud DashScope (text-embedding-v4).

The RAG pipeline's only bridge to the embedding provider. Isolated here (behind
`embed_query` / `embed_documents`) so the vector storage and retrieval code never
import the SDK directly and the provider can be swapped without touching them.

Why text-embedding-v4: unlike Alibaba's ASR/TTS, the text embedding models
genuinely document 100+ languages including Urdu (Qwen3-Embedding series), so an
Urdu question can be matched against English or Urdu knowledge cross-lingually.

Notes:
- The `dashscope` import is guarded via dashscope_client, so importing this module
  never crashes a machine without the SDK or a key.
- `text_type` is set per the DashScope docs: 'document' when embedding stored
  knowledge, 'query' when embedding a user question — this improves retrieval.
- v4 accepts at most 10 texts per call, so batches are chunked accordingly.
- The API key is never logged.
"""

import logging

from app.config import settings
from app.services import dashscope_client

logger = logging.getLogger(__name__)

# text-embedding-v4 batch limit (max texts per call). See Model Studio docs.
_MAX_BATCH = 10


def is_ready() -> bool:
    """True when embeddings can actually be generated (SDK present AND key set)."""
    return dashscope_client.is_configured()


def _extract_vectors(response, expected: int) -> list[list[float]]:
    """
    Pull ordered embedding vectors out of a DashScope TextEmbedding response.

    The SDK returns output['embeddings'] as a list of
    {'text_index': i, 'embedding': [...]} — not necessarily in input order — so
    we sort by text_index to realign with the input texts.
    """
    output = getattr(response, "output", None)
    if output is None:
        raise RuntimeError("Embedding response had no output")

    items = output.get("embeddings") if isinstance(output, dict) else getattr(output, "embeddings", None)
    if not items:
        raise RuntimeError("Embedding response contained no vectors")

    ordered = sorted(items, key=lambda it: it.get("text_index", 0))
    vectors = [list(it["embedding"]) for it in ordered]
    if len(vectors) != expected:
        raise RuntimeError(f"Embedding count mismatch: got {len(vectors)}, expected {expected}")
    return vectors


def _call(texts: list[str], text_type: str) -> list[list[float]]:
    """One blocking DashScope embedding call for up to _MAX_BATCH texts."""
    dashscope_client.configure()
    import dashscope  # guarded/available: is_ready() gated the caller

    response = dashscope.TextEmbedding.call(
        model=settings.EMBEDDING_MODEL,
        input=texts,
        text_type=text_type,
        dimension=settings.EMBEDDING_DIMENSION,
    )

    status = getattr(response, "status_code", 200)
    if status != 200:
        # Provider code is logged for us; it is never surfaced to the caller.
        logger.error(
            "Embedding call failed: status=%s code=%s", status, getattr(response, "code", None)
        )
        raise RuntimeError(f"Embedding call failed with status {status}")

    return _extract_vectors(response, expected=len(texts))


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed knowledge documents for storage (text_type='document').

    Used at ingestion time. Batches into groups of _MAX_BATCH so a corpus larger
    than one batch is handled transparently.
    """
    if not texts:
        return []
    if not is_ready():
        raise RuntimeError("Embedding provider is not configured (no DashScope key)")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _MAX_BATCH):
        batch = texts[start : start + _MAX_BATCH]
        vectors.extend(_call(batch, text_type="document"))
    return vectors


def embed_query(text: str) -> list[float]:
    """
    Embed a single user question for retrieval (text_type='query').

    Raises RuntimeError when the provider is not configured — callers in the RAG
    layer catch this and degrade to an empty retrieval rather than failing the
    request.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed an empty query")
    if not is_ready():
        raise RuntimeError("Embedding provider is not configured (no DashScope key)")

    return _call([text], text_type="query")[0]
