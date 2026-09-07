"""Google Gemini client for the voice assistant."""

import base64
import io
import logging
import wave
from typing import Any

from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini voice configuration is incomplete."""


class GeminiProviderError(RuntimeError):
    """Raised when Gemini returns an unusable response."""


def _client() -> genai.Client:
    if not settings.gemini_configured:
        raise GeminiConfigurationError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _mime_type(filename: str | None, content_type: str | None) -> str:
    if content_type and content_type.startswith("audio/"):
        return content_type
    extension = (filename.rsplit(".", 1)[-1] if filename and "." in filename else "").lower()
    return {
        "webm": "audio/webm",
        "wav": "audio/wav",
        "m4a": "audio/m4a",
        "mp3": "audio/mp3",
        "mpeg": "audio/mpeg",
        "ogg": "audio/ogg",
        "opus": "audio/opus",
    }.get(extension, "audio/webm")


def transcribe_audio(audio_bytes: bytes, filename: str | None, content_type: str | None) -> str:
    """Ask Gemini to return only the Urdu transcription."""
    client = _client()
    prompt = (
        "Transcribe the attached farmer voice recording exactly. "
        "The expected language is Urdu as spoken in Pakistan (ur-PK). "
        "Return ONLY the spoken Urdu transcription text. Do not answer the "
        "question, summarize it, translate it, or add commentary."
    )
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_TEXT_MODEL,
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=_mime_type(filename, content_type),
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=512,
            ),
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        status_code = (
            getattr(exc, "status_code", None)
            or getattr(response, "status_code", None)
            or getattr(exc, "code", None)
        )
        logger.exception(
            "Gemini audio understanding failed: exception_type=%s status_code=%s message=%s",
            type(exc).__name__,
            status_code,
            str(exc),
        )
        raise GeminiProviderError("Gemini audio understanding failed") from exc

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise GeminiProviderError("Gemini returned no transcription")
    return text


def generate_answer(question: str, context: str) -> str:
    """Generate a concise, grounded Urdu agricultural answer."""
    client = _client()
    system_instruction = (
        "KisaanDost is an agricultural assistant for Pakistani farmers. "
        "Answer farmers in simple, natural Urdu. Use the supplied agricultural "
        "knowledge as the primary source. Do not invent agricultural facts when "
        "relevant knowledge is unavailable. Give practical, concise advice that "
        "a Pakistani farmer can understand. If the question is unrelated to "
        "agriculture, politely explain that KisaanDost is focused on agriculture."
    )
    user_prompt = (
        f"Farmer question:\n{question}\n\n"
        f"Retrieved agricultural knowledge:\n{context or '(No matching local knowledge found.)'}\n\n"
        "Answer in Urdu script, in two or three short sentences."
    )
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_TEXT_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                max_output_tokens=512,
            ),
        )
    except Exception as exc:
        logger.exception("Gemini answer generation failed")
        raise GeminiProviderError("Gemini answer generation failed") from exc

    answer = (getattr(response, "text", None) or "").strip()
    if not answer:
        raise GeminiProviderError("Gemini returned no answer")
    return answer


def _audio_bytes_from_response(response: Any) -> bytes:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None) if inline_data else None
            if data:
                if isinstance(data, str):
                    return base64.b64decode(data)
                return bytes(data)
    return b""


def _pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    if not pcm:
        raise GeminiProviderError("Gemini TTS returned no audio")
    if pcm.startswith(b"RIFF"):
        return pcm
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def synthesize_urdu(text: str) -> str:
    """Generate real Urdu speech and return base64-encoded playable WAV."""
    client = _client()
    speech_config = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=settings.GEMINI_TTS_VOICE,
            )
        )
    )
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_TTS_MODEL,
            contents=f"Speak this Urdu answer naturally and clearly:\n{text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=speech_config,
            ),
        )
    except Exception as exc:
        logger.exception("Gemini TTS failed")
        raise GeminiProviderError("Gemini TTS failed") from exc
    return base64.b64encode(_pcm_to_wav(_audio_bytes_from_response(response))).decode("ascii")
