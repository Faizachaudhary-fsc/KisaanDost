"""
Agricultural RAG pipeline tests.

None of these need a real DashScope key, a real MongoDB, or live embedding calls:

  * Document loading and chunking are tested against the real corpus file.
  * The embedding SDK is monkeypatched, so embeddings.* parsing is covered
    without a network call (the tests importorskip the SDK).
  * Retrieval, threshold, and the full route are tested with a deterministic
    local *stand-in* embedder. It maps a small multilingual keyword set (English
    AND Urdu) to vector dimensions, so cosine similarity genuinely ranks wheat
    docs for a wheat question and cotton docs for a cotton-pest question — the
    same cross-lingual behaviour the real multilingual model provides, but
    offline and reproducible.

Covers the twelve required scenarios (loading, chunking, mocked embedding,
wheat/cotton/rice retrieval, no unrelated hits, empty KB, no relevant result,
duplicate-ingestion protection, RAG failure handling, LLM receiving context)
plus the five representative Urdu questions and the complete mocked pipeline.
"""

import io
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import dashscope_client, embeddings, knowledge_store, llm, rag, speech_to_text, text_to_speech


# ── Deterministic offline stand-in embedder ────────────────────────────────
# Each concept is one vector dimension; a text's value on that dimension is how
# many of the concept's terms (English or Urdu) it contains. This is ONLY for
# tests — production uses the real multilingual text-embedding-v4.
_CONCEPTS = [
    ("wheat", ["wheat", "gandum", "گندم"]),
    ("rice", ["rice", "chawal", "basmati", "چاول"]),
    ("cotton", ["cotton", "kapas", "کپاس"]),
    ("maize", ["maize", "makai", "مکئی"]),
    ("potato", ["potato", "aalu", "آلو"]),
    ("yellow", ["yellow", "peelay", "پیلے"]),
    ("leaf", ["leaf", "leaves", "pattay", "پتے"]),
    ("pest", ["pest", "insect", "whitefly", "bollworm", "borer", "aphid", "کیڑے"]),
    ("irrigation", ["irrigation", "water", "aabpashi", "آبپاشی", "پانی"]),
    ("sowing", ["sow", "sowing", "کاشت", "کب کاشت"]),
    ("disease", ["disease", "virus", "blight", "بیماری"]),
    ("symptom", ["symptom", "sign", "spots", "علامات"]),
    ("fertilizer", ["fertiliser", "fertilizer", "nitrogen", "urea", "کھاد"]),
    ("harvest", ["harvest", "picking", "کٹائی"]),
]


def _fake_vector(text: str) -> list[float]:
    t = (text or "").lower()
    return [float(sum(term in t for term in terms)) for _, terms in _CONCEPTS]


@pytest.fixture(autouse=True)
def _clean_rag_cache():
    """Isolate the module-level index cache around every test."""
    knowledge_store.reset_cache()
    yield
    knowledge_store.reset_cache()


def _install_fake_rag(monkeypatch, documents=None):
    """
    Build an in-memory index from the (real, unless overridden) corpus using the
    stand-in embedder, and make the RAG layer think the provider is ready.
    """
    docs = documents if documents is not None else knowledge_store.load_documents()
    chunks = knowledge_store.documents_to_chunks(docs)
    for c in chunks:
        c.embedding = _fake_vector(c.text)
    knowledge_store.set_index(chunks)
    monkeypatch.setattr(embeddings, "is_ready", lambda: True)
    monkeypatch.setattr(embeddings, "embed_query", lambda q: _fake_vector(q))
    return chunks


# ── 1. Knowledge document loading ───────────────────────────────────────────


def test_documents_load_with_required_metadata():
    documents = knowledge_store.load_documents()
    assert len(documents) >= 25  # MVP target is ~25-50
    crops = {d["crop"] for d in documents}
    assert {"wheat", "rice", "cotton", "maize", "potato"}.issubset(crops)
    for d in documents:
        # Source metadata must ride along for later source-aware answers.
        for key in ("title", "crop", "topic", "language", "source", "source_url", "content"):
            assert key in d and d[key] != ""


# ── 2. Document chunking ────────────────────────────────────────────────────


def test_short_text_is_a_single_chunk():
    assert knowledge_store.chunk_text("Wheat is a Rabi crop.") == ["Wheat is a Rabi crop."]


def test_long_text_splits_on_sentence_boundaries():
    long_text = ". ".join(f"Sentence number {i} about the crop" for i in range(60))
    chunks = knowledge_store.chunk_text(long_text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # ~max_chars, never wildly over


def test_documents_to_chunks_carries_metadata_and_unique_ids():
    docs = knowledge_store.load_documents()
    chunks = knowledge_store.documents_to_chunks(docs)
    assert len(chunks) >= len(docs)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))  # no duplicate ids
    assert all(c.metadata.get("crop") for c in chunks)


# ── 3. Embedding service mocked successfully ────────────────────────────────


def _configure(monkeypatch):
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(dashscope_client, "configure", lambda: None)


def test_embed_query_parses_mocked_sdk_response(monkeypatch):
    dashscope = pytest.importorskip("dashscope")
    _configure(monkeypatch)
    monkeypatch.setattr(
        dashscope.TextEmbedding,
        "call",
        staticmethod(
            lambda **_: SimpleNamespace(
                status_code=200,
                output={"embeddings": [{"text_index": 0, "embedding": [0.1, 0.2, 0.3]}]},
            )
        ),
    )
    assert embeddings.embed_query("gandum peelay") == [0.1, 0.2, 0.3]


def test_embed_documents_reorders_by_text_index(monkeypatch):
    dashscope = pytest.importorskip("dashscope")
    _configure(monkeypatch)
    # Return the two vectors out of order; the service must realign by text_index.
    monkeypatch.setattr(
        dashscope.TextEmbedding,
        "call",
        staticmethod(
            lambda **_: SimpleNamespace(
                status_code=200,
                output={
                    "embeddings": [
                        {"text_index": 1, "embedding": [9.0]},
                        {"text_index": 0, "embedding": [1.0]},
                    ]
                },
            )
        ),
    )
    assert embeddings.embed_documents(["a", "b"]) == [[1.0], [9.0]]


def test_embed_query_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "")
    with pytest.raises(RuntimeError):
        embeddings.embed_query("koi sawal")


# ── 4-6. Relevant retrieval per crop ────────────────────────────────────────


@pytest.mark.anyio
async def test_retrieval_returns_wheat_info_for_wheat_question(monkeypatch):
    _install_fake_rag(monkeypatch)
    result = await rag.retrieve("My wheat leaves are turning yellow, what is the cause?")
    assert not result.is_empty
    assert result.chunks[0].metadata["crop"] == "wheat"


@pytest.mark.anyio
async def test_retrieval_returns_cotton_info_for_cotton_pest_question(monkeypatch):
    _install_fake_rag(monkeypatch)
    result = await rag.retrieve("There are insects and whitefly attacking my cotton crop")
    assert not result.is_empty
    assert result.chunks[0].metadata["crop"] == "cotton"


@pytest.mark.anyio
async def test_retrieval_returns_rice_info_for_rice_question(monkeypatch):
    _install_fake_rag(monkeypatch)
    result = await rag.retrieve("How much irrigation and water does a rice crop need?")
    assert not result.is_empty
    assert result.chunks[0].metadata["crop"] == "rice"


# ── 7. No unrelated documents ───────────────────────────────────────────────


@pytest.mark.anyio
async def test_retrieval_does_not_return_unrelated_documents(monkeypatch):
    _install_fake_rag(monkeypatch)
    # Nothing agricultural here — every concept dimension is zero.
    result = await rag.retrieve("How do I repair the engine of my motorcycle?")
    assert result.is_empty


# ── 8. Empty knowledge base ─────────────────────────────────────────────────


@pytest.mark.anyio
async def test_empty_knowledge_base_returns_no_context(monkeypatch):
    knowledge_store.set_index([])
    monkeypatch.setattr(embeddings, "is_ready", lambda: True)
    monkeypatch.setattr(embeddings, "embed_query", lambda q: _fake_vector(q))
    result = await rag.retrieve("wheat yellow leaves")
    assert result.is_empty


# ── 9. No relevant result (threshold) ───────────────────────────────────────


@pytest.mark.anyio
async def test_threshold_filters_low_similarity(monkeypatch):
    _install_fake_rag(monkeypatch)
    # A high threshold makes even a partial match fall through: low confidence.
    monkeypatch.setattr(settings, "RAG_MIN_SCORE", 0.999)
    result = await rag.retrieve("wheat")  # single concept, imperfect match
    assert result.is_empty


# ── 10. Duplicate ingestion protection ──────────────────────────────────────


def test_reingestion_is_idempotent_not_duplicative(monkeypatch):
    docs = knowledge_store.load_documents()
    first = knowledge_store.documents_to_chunks(docs)
    second = knowledge_store.documents_to_chunks(docs)  # "run ingestion again"
    # Rebuilding from the canonical corpus yields the same ids, not duplicates.
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


# ── 11. RAG failure handling (never raises) ─────────────────────────────────


@pytest.mark.anyio
async def test_retrieve_degrades_to_empty_on_embedding_failure(monkeypatch):
    _install_fake_rag(monkeypatch)

    def boom(_q):
        raise RuntimeError("simulated embedding outage")

    monkeypatch.setattr(embeddings, "embed_query", boom)
    result = await rag.retrieve("wheat yellow leaves")  # must NOT raise
    assert result.is_empty


@pytest.mark.anyio
async def test_retrieve_returns_empty_when_provider_unavailable(monkeypatch):
    _install_fake_rag(monkeypatch)
    monkeypatch.setattr(embeddings, "is_ready", lambda: False)  # no key
    result = await rag.retrieve("wheat yellow leaves")
    assert result.is_empty


# ── 12. LLM receives the retrieved context ──────────────────────────────────


@pytest.mark.anyio
async def test_llm_prompt_includes_retrieved_context(monkeypatch):
    _install_fake_rag(monkeypatch)
    result = await rag.retrieve("My wheat leaves are turning yellow")
    prompt = llm._build_user_prompt("My wheat leaves are turning yellow", result.context)
    assert "FARMER QUESTION" in prompt and "RETRIEVED AGRICULTURAL CONTEXT" in prompt
    assert "wheat" in prompt.lower()
    assert "Source:" in prompt  # provenance travels into the prompt


# ── 13. Representative Urdu questions ───────────────────────────────────────

_URDU_QUESTIONS = [
    ("میری گندم کے پتے پیلے ہو رہے ہیں، کیا وجہ ہو سکتی ہے؟", "wheat"),
    ("کپاس کی فصل میں کیڑے لگ گئے ہیں، کیا کروں؟", "cotton"),
    ("چاول کی فصل کو کتنی آبپاشی کی ضرورت ہوتی ہے؟", "rice"),
    ("مکئی کی فصل کب کاشت کرنی چاہیے؟", "maize"),
    ("آلو میں بیماری کی عام علامات کیا ہیں؟", "potato"),
]


@pytest.mark.anyio
@pytest.mark.parametrize("question,expected_crop", _URDU_QUESTIONS)
async def test_urdu_question_retrieves_the_right_crop(monkeypatch, question, expected_crop):
    _install_fake_rag(monkeypatch)
    result = await rag.retrieve(question)
    assert not result.is_empty, f"no retrieval for: {question}"
    assert result.chunks[0].metadata["crop"] == expected_crop


# ── 14. Complete pipeline through the voice route (all external AI mocked) ───


def _audio_upload():
    return {"audio": ("q.m4a", io.BytesIO(b"fake-audio-bytes"), "audio/m4a")}


def test_full_pipeline_urdu_question_reaches_llm_with_context(client, monkeypatch):
    _install_fake_rag(monkeypatch)
    captured = {}

    async def fake_transcribe(audio_bytes, filename):
        return speech_to_text.TranscriptionResult(
            "میری گندم کے پتے پیلے ہو رہے ہیں", "urdu", True
        )

    async def fake_answer(question, context=""):
        captured["question"] = question
        captured["context"] = context
        return "Yeh nitrogen ki kami ho sakti hai."

    async def fake_synth(text):
        return "QUJDRA=="  # valid base64

    monkeypatch.setattr(speech_to_text, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm, "generate_answer", fake_answer)
    monkeypatch.setattr(text_to_speech, "synthesize", fake_synth)

    response = client.post("/api/assistant/voice", files=_audio_upload(), data={"farmerId": "f1"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "transcription", "language", "answer", "audio_base64"}
    assert body["language"] == "urdu"
    assert body["answer"] == "Yeh nitrogen ki kami ho sakti hai."
    # The RAG context genuinely reached the LLM and mentions wheat.
    assert captured["context"].strip() != ""
    assert "wheat" in captured["context"].lower()


# ── Health status (Phase 17) ────────────────────────────────────────────────


def test_rag_status_unavailable_without_key(monkeypatch):
    monkeypatch.setattr(embeddings, "is_ready", lambda: False)
    assert rag.status()["status"] == "unavailable"


def test_rag_status_not_indexed_when_key_present_but_no_index(monkeypatch):
    monkeypatch.setattr(embeddings, "is_ready", lambda: True)
    knowledge_store.set_index([])
    status = rag.status()
    assert status["status"] == "not_indexed"
    assert status["indexed_chunks"] == 0


def test_rag_status_ready_when_indexed(monkeypatch):
    monkeypatch.setattr(embeddings, "is_ready", lambda: True)
    _install_fake_rag(monkeypatch)
    status = rag.status()
    assert status["status"] == "ready"
    assert status["indexed_chunks"] > 0
    assert status["embedding_model"] == settings.EMBEDDING_MODEL
