"""
Service layer: one module per AI pipeline stage.

All four stages now have real (Alibaba Cloud DashScope) implementations behind
clean provider abstractions: speech_to_text, rag, llm and text_to_speech. Each
stage degrades safely when DashScope is not configured (rag returns no context,
llm a stub answer, tts a placeholder), so routes are written against final
signatures and the app boots and tests run with no key.

Pipeline order for POST /api/assistant/voice:
    audio -> speech_to_text -> rag -> llm -> text_to_speech -> audio_base64
"""

from app.services import llm, rag, speech_to_text, text_to_speech

__all__ = ["llm", "rag", "speech_to_text", "text_to_speech", "pipeline_status", "rag_status"]


def pipeline_status() -> dict[str, bool]:
    """Which stages are real vs still stubbed. Surfaced by /health."""
    return {
        "speech_to_text": speech_to_text.IMPLEMENTED,
        "rag": rag.IMPLEMENTED,
        "llm": llm.IMPLEMENTED,
        "text_to_speech": text_to_speech.IMPLEMENTED,
    }


def rag_status() -> dict:
    """
    RAG readiness detail for /health: ready / not_indexed / unavailable, plus the
    indexed chunk count and embedding model. Additive to pipeline_status(); never
    exposes credentials.
    """
    return rag.status()
