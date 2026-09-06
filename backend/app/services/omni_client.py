"""Official OpenAI-compatible Qwen3.5-Omni client helpers."""

import base64
import io
import logging
import mimetypes
import wave
from typing import Any

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class OmniConfigurationError(RuntimeError):
    """Raised when the real Omni provider is not configured."""


class OmniProviderError(RuntimeError):
    """Raised when a real Omni request fails or returns unusable data."""


def _client() -> OpenAI:
    if not settings.dashscope_configured:
        raise OmniConfigurationError("DASHSCOPE_API_KEY is not configured")
    base_url = settings.OMNI_BASE_URL.strip() or settings.DASHSCOPE_BASE_URL.strip()
    if not base_url:
        raise OmniConfigurationError("OMNI_BASE_URL is not configured")
    return OpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=base_url,
        timeout=settings.VOICE_STAGE_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _audio_format(filename: str | None, content_type: str | None) -> str:
    """Return the source format token expected by OpenAI-compatible Omni."""
    extension = (filename.rsplit(".", 1)[-1] if filename and "." in filename else "").lower()
    by_mime = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/mp4": "m4a",
        "audio/m4a": "m4a",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
    }
    return extension or by_mime.get((content_type or "").lower(), "wav")


def audio_message(audio_bytes: bytes, filename: str | None, content_type: str | None) -> dict[str, Any]:
    """Build a data-URL audio message without writing user audio to disk."""
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    mime = content_type or mimetypes.guess_type(filename or "")[0] or "audio/wav"
    return {
        "type": "input_audio",
        "input_audio": {
            "data": f"data:{mime};base64,{encoded}",
            "format": _audio_format(filename, content_type),
        },
    }


def _text_from_chunk(chunk: Any) -> str:
    if not getattr(chunk, "choices", None):
        return ""
    delta = getattr(chunk.choices[0], "delta", None)
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


def _audio_from_chunk(chunk: Any) -> str:
    if not getattr(chunk, "choices", None):
        return ""
    delta = getattr(chunk.choices[0], "delta", None)
    audio = getattr(delta, "audio", None)
    if not audio:
        return ""
    if isinstance(audio, dict):
        return str(audio.get("data") or "")
    return str(getattr(audio, "data", "") or "")


def stream_completion(messages: list[dict[str, Any]], *, with_audio: bool) -> tuple[str, bytes]:
    """Run one documented streaming Omni request and collect text/audio."""
    client = _client()
    try:
        completion = client.chat.completions.create(
            model=settings.OMNI_MODEL,
            messages=messages,
            modalities=["text", "audio"] if with_audio else ["text"],
            **({"audio": {"voice": settings.OMNI_VOICE, "format": "wav"}} if with_audio else {}),
            stream=True,
            stream_options={"include_usage": True},
        )
        text_parts: list[str] = []
        audio_parts: list[str] = []
        for chunk in completion:
            text = _text_from_chunk(chunk)
            if text:
                text_parts.append(text)
            audio = _audio_from_chunk(chunk)
            if audio:
                audio_parts.append(audio)
    except Exception as exc:  # provider details stay server-side
        logger.exception("Qwen3.5-Omni request failed")
        raise OmniProviderError("Qwen3.5-Omni request failed") from exc

    text = "".join(text_parts).strip()
    raw_audio = b""
    if audio_parts:
        try:
            raw_audio = base64.b64decode("".join(audio_parts), validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise OmniProviderError("Qwen3.5-Omni returned invalid audio") from exc
    return text, raw_audio


def pcm_to_wav(raw_audio: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap Omni's documented 16-bit mono PCM audio in a playable WAV file."""
    if not raw_audio:
        raise OmniProviderError("Qwen3.5-Omni returned no audio")
    if raw_audio.startswith(b"RIFF"):
        return raw_audio
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw_audio)
    return output.getvalue()
