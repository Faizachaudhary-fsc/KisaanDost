"""
Knowledge-base ingestion for the agricultural RAG pipeline.

Turns the local knowledge corpus into an embedded, searchable index:

    data/agricultural_knowledge.json   (the source documents)
        -> load + clean + chunk
        -> embed each chunk with Alibaba Cloud text-embedding-v4
        -> data/knowledge_index.json    (chunks + vectors + metadata)

Run it from the backend/ directory, with a real DASHSCOPE_API_KEY in .env:

    python -m scripts.ingest_knowledge          # build / rebuild the index
    python -m scripts.ingest_knowledge --dry-run  # chunk only, no API calls

Idempotency (Phase 7): the index is rebuilt in full from the canonical corpus on
every run, keyed by chunk id, so running it again simply overwrites the previous
index with the same content — it never appends duplicates.

Safety:
- The API key is read from .env via app.config and is NEVER printed or logged.
- With no key/SDK the script exits WITHOUT writing an index and WITHOUT
  pretending to have succeeded.
- pytest never imports or runs this module, so the test suite makes no live
  embedding calls.
"""

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/ingest_knowledge.py) too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import embeddings, knowledge_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the agricultural knowledge index.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk the corpus but make no embedding API calls.",
    )
    args = parser.parse_args()

    print("KisaanDost — agricultural knowledge ingestion")
    print("-" * 48)

    documents = knowledge_store.load_documents()
    if not documents:
        print(f"ERROR: no documents found at {knowledge_store.DOCUMENTS_PATH}")
        return 1

    chunks = knowledge_store.documents_to_chunks(documents)
    crops = sorted({c.metadata.get("crop", "") for c in chunks if c.metadata.get("crop")})
    print(f"Documents loaded     : {len(documents)}")
    print(f"Chunks produced      : {len(chunks)}")
    print(f"Crops covered        : {', '.join(crops)}")

    # Guard against accidental duplicate ids in the corpus.
    ids = [c.id for c in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        print(f"ERROR: duplicate chunk ids in corpus: {sorted(duplicates)}")
        return 1

    if args.dry_run:
        print("\nDRY RUN: skipping embeddings and not writing an index.")
        return 0

    print(f"Embedding model      : {embeddings.settings.EMBEDDING_MODEL}")
    print(f"SDK installed        : {embeddings.dashscope_client.is_available()}")
    print(f"API key configured   : {embeddings.settings.dashscope_configured}")  # bool only

    if not embeddings.is_ready():
        print("\nRESULT: embedding provider not configured (no DashScope key/SDK).")
        print("        Set DASHSCOPE_API_KEY in backend/.env, then run again.")
        print("        No index was written.")
        return 2

    print(f"\nEmbedding {len(chunks)} chunks (batched)...")
    try:
        vectors = embeddings.embed_documents([c.text for c in chunks])
    except Exception as exc:  # noqa: BLE001 - surface a safe summary only
        print(f"RESULT: embedding FAILED ({type(exc).__name__}). See logs; no index written.")
        return 1

    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    knowledge_store.save_index(chunks)
    print(f"\nRESULT: wrote {len(chunks)} embedded chunks to {knowledge_store.INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
