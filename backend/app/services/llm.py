"""Gemini agricultural answer generation service."""

import asyncio
import logging

from app.config import settings
from app.services.gemini_client import generate_answer as gemini_generate_answer

logger = logging.getLogger(__name__)

IMPLEMENTED = True

# System instruction for the agricultural assistant. Kept here so prompt
# iteration does not touch route code, and never exposed to the frontend.
SYSTEM_PROMPT = (
    "You are KisaanDost, an agricultural assistant for Pakistani farmers.\n"
    "Answer in simple Urdu script when the farmer speaks Urdu, as if speaking "
    "aloud to a farmer. Avoid complicated technical terminology.\n"
    "Give practical, understandable farming guidance.\n"
    "PRIORITISE the retrieved agricultural context provided with the question. "
    "Base your answer on it whenever it is relevant.\n"
    "If the retrieved context is empty or does not actually cover the question, "
    "do NOT pretend it does: give only safe general guidance and say plainly (in "
    "Urdu) that you do not have reliable specific information on this point.\n"
    "Do not invent agricultural facts. Do not give specific pesticide or "
    "fertiliser dosages unless the supplied context supports them.\n"
    "If the question is not about farming, politely say you focus on agriculture.\n"
    "You are an AI assistant, not a human agricultural officer.\n"
    "Keep the answer to two or three short sentences, because it will be read "
    "aloud."
)

# Spoken back to the farmer when speech could not be transcribed. The contract
# pairs this with `transcription: ""` and `language: "unrecognized"`.
FALLBACK_ANSWER = "Maazrat, samajh nahi aaya. Dobara Urdu mein poochain."


def _build_user_prompt(question: str, context: str) -> str:
    """
    Compose the user turn, keeping the farmer's question and the retrieved
    context (with its source labels) clearly separated so the model never
    confuses grounding material for the question itself (Phase 10).
    """
    grounding = context.strip() if context and context.strip() else "(none available)"
    return (
        "FARMER QUESTION:\n"
        f"{question}\n\n"
        "RETRIEVED AGRICULTURAL CONTEXT (each passage is numbered and shows its "
        "source; may be empty):\n"
        f"{grounding}"
    )


def _extract_text(response) -> str:
    """
    Pull the reply text out of a DashScope Generation response.

    Uses result_format="message", so the text is at
    output.choices[0].message.content. Kept defensive because the SDK returns a
    dict-like object whose exact shape varies by version.
    """
    output = getattr(response, "output", None)
    if output is None:
        raise RuntimeError("Qwen response had no output")

    choices = getattr(output, "choices", None) or (
        output.get("choices") if isinstance(output, dict) else None
    )
    if choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else choices[0]["message"]
        content = message["content"] if isinstance(message, dict) else message.content
        if isinstance(content, list):
            # Some models return a list of content parts; join their text.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content).strip()

    # Older/simple result formats expose `output.text` directly.
    text = getattr(output, "text", None) or (output.get("text") if isinstance(output, dict) else None)
    if text:
        return str(text).strip()

    raise RuntimeError("Qwen response contained no answer text")


async def generate_answer(question: str, context: str = "") -> str:
    """
    Produce the assistant's reply text.

    Args:
        question: Transcribed farmer question.
        context: RAG grounding passages; empty string is valid and means
            "answer without retrieval".

    Returns:
        Reply text in Roman Urdu, ready to be sent to text_to_speech.

    Raises:
        Exception: on a real provider/API error, so the route returns the
            contract's 500. Missing configuration is NOT an error — it returns
            the stub answer instead.
    """
    if not question.strip():
        return FALLBACK_ANSWER

    logger.info("llm.generate_answer: calling Qwen3.5-Omni, context_chars=%d", len(context))
    answer = await asyncio.wait_for(
        asyncio.to_thread(gemini_generate_answer, question, context),
        timeout=settings.VOICE_STAGE_TIMEOUT_SECONDS,
    )
    if not answer:
        raise RuntimeError("Qwen returned an empty answer")
    return answer
