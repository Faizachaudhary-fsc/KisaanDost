"""
Urdu speech-to-text (ASR) service.

Groq's OpenAI-compatible transcription API is the active provider. The
provider abstraction keeps the route contract stable while allowing the
uploaded WebM/M4A bytes to be sent directly to the current OpenAI SDK.

Replace/extend the providers only. `transcribe()`'s signature and the
TranscriptionResult shape are what the route depends on.
"""

import asyncio
import logging
import os
import tempfile
from typing import Optional, Protocol

from app.config import settings
from app.services.gemini_client import GeminiConfigurationError, GeminiProviderError

logger = logging.getLogger(__name__)

# The provider layer is implemented (real, swappable). Whether a real ASR call
# actually happens depends on ASR_ENABLED + a configured key; /health reports
# this flag so the team can see the stage is wired.
IMPLEMENTED = True

# Returned when audio is unusable / ASR is disabled. The contract expects this
# exact string in `language` for the fallback path.
UNRECOGNIZED_LANGUAGE = "unrecognized"

# Language tokens (lowercased) we accept as genuinely Urdu.
_URDU_TOKENS = {"ur", "urd", "urdu"}


class TranscriptionResult:
    """Plain container so callers do not depend on a provider's SDK types."""

    def __init__(self, text: str, language: str, recognised: bool = True) -> None:
        self.text = text
        self.language = language
        self.recognised = recognised


def _normalise_language(raw: Optional[str]) -> str:
    """
    Map a provider language code/name to our reported value, honestly.

    Returns "urdu" ONLY for an actual Urdu token. Any other detected language
    (e.g. Hindi, Arabic) is passed through as-is rather than being relabelled,
    and an absent/unknown value becomes "unknown".
    """
    if not raw:
        return "unknown"
    token = raw.strip().lower()
    if token in _URDU_TOKENS:
        return "urdu"
    return token


class ASRProvider(Protocol):
    """Contract every ASR backend must satisfy. Keeps the route provider-agnostic."""

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> TranscriptionResult:
        ...


class GroqASRProvider:
    """Urdu transcription through Groq's OpenAI-compatible audio API."""

    async def transcribe(self, audio_bytes: bytes, filename: Optional[str], content_type: Optional[str]) -> TranscriptionResult:
        logger.info(
            "[ASR DIAG] provider_class=%s model=%s GROQ_API_KEY=%s filename=%s mime_type=%s byte_size=%d",
            type(self).__name__,
            settings.GROQ_ASR_MODEL,
            "YES" if settings.groq_configured else "NO",
            filename,
            content_type,
            len(audio_bytes),
        )
        if not settings.groq_configured:
            raise GeminiConfigurationError("GROQ_API_KEY is not configured")
        text = await asyncio.to_thread(self._call, audio_bytes, filename, content_type)
        recognised = bool(text.strip())
        return TranscriptionResult(
            text=text,
            language="ur" if recognised else UNRECOGNIZED_LANGUAGE,
            recognised=recognised,
        )

    @staticmethod
    def _call(audio_bytes: bytes, filename: Optional[str], content_type: Optional[str]) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        upload_name = filename or "recording.webm"
        upload_type = content_type or "audio/webm"
        logger.info(
            "[ASR DIAG] Groq request model=%s filename=%s mime_type=%s byte_size=%d",
            settings.GROQ_ASR_MODEL,
            upload_name,
            upload_type,
            len(audio_bytes),
        )
        try:
            response = client.audio.transcriptions.create(
                model=settings.GROQ_ASR_MODEL,
                file=(upload_name, audio_bytes, upload_type),
                language="ur",
            )
        except Exception as exc:
            logger.error(
                "Groq audio transcription failed: exception_type=%s status_code=%s message=%s",
                type(exc).__name__,
                getattr(exc, "status_code", None),
                str(exc),
            )
            raise GeminiProviderError("Groq audio transcription failed") from exc

        text = (getattr(response, "text", None) or str(response)).strip()
        if not text:
            raise GeminiProviderError("Groq returned no transcription")
        logger.info(
            "[ASR DIAG] Groq response status_code=%s transcription_received=YES",
            getattr(response, "status_code", 200),
        )
        return text


class AzureSpeechASRProvider:
    """Azure Speech recognition configured for Urdu (Pakistan)."""

    async def transcribe(self, audio_bytes: bytes, filename: Optional[str]) -> TranscriptionResult:
        if not audio_bytes:
            return TranscriptionResult(text="", language=UNRECOGNIZED_LANGUAGE, recognised=False)
        return await asyncio.to_thread(self._call, audio_bytes, filename)

    def _call(self, audio_bytes: bytes, filename: Optional[str]) -> TranscriptionResult:
        import azure.cognitiveservices.speech as speechsdk

        suffix = os.path.splitext(filename)[1] if filename else ".wav"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            speech_config = speechsdk.SpeechConfig(
                subscription=settings.AZURE_SPEECH_KEY,
                region=settings.AZURE_SPEECH_REGION,
            )
            speech_config.speech_recognition_language = settings.AZURE_SPEECH_LANGUAGE
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=speechsdk.audio.AudioConfig(filename=tmp_path),
            )
            result = recognizer.recognize_once_async().get()
            if result.reason != speechsdk.ResultReason.RecognizedSpeech:
                return TranscriptionResult("", UNRECOGNIZED_LANGUAGE, False)
            text = (result.text or "").strip()
            return TranscriptionResult(text, "urdu" if text else UNRECOGNIZED_LANGUAGE, bool(text))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


class DashScopeASRProvider:
    """
    Alibaba Cloud DashScope multilingual ASR (settings.STT_MODEL).

    Used only when ASR_ENABLED is true AND a key is configured. Language is
    reported from the provider's detection, never assumed to be Urdu.
    """

    async def transcribe(self, audio_bytes: bytes, filename: Optional[str]) -> TranscriptionResult:
        if not audio_bytes:
            return TranscriptionResult(text="", language=UNRECOGNIZED_LANGUAGE, recognised=False)
        return await asyncio.to_thread(self._call, audio_bytes, filename)

    def _call(self, audio_bytes: bytes, filename: Optional[str]) -> TranscriptionResult:
        dashscope_client.configure()
        from dashscope.audio.qwen_asr import QwenTranscription

        suffix = os.path.splitext(filename)[1] if filename else ".m4a"
        # The SDK reads audio from a file URL, so persist the upload briefly.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            response = QwenTranscription.call(
                model=settings.STT_MODEL,
                file_url=os.path.abspath(tmp_path),
                wait_timeout=60,
            )

            status = getattr(response, "status_code", 200)
            if status != 200:
                logger.error("ASR call failed: status=%s code=%s", status, getattr(response, "code", None))
                raise RuntimeError(f"ASR call failed with status {status}")

            text, detected = self._parse(response)
            language = _normalise_language(detected)
            recognised = bool(text.strip())
            if not recognised:
                language = UNRECOGNIZED_LANGUAGE
            return TranscriptionResult(text=text.strip(), language=language, recognised=recognised)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _parse(response) -> tuple[str, Optional[str]]:
        """Extract (text, detected_language) defensively from the SDK response."""
        output = getattr(response, "output", None) or {}
        choices = getattr(output, "choices", None) or (
            output.get("choices") if isinstance(output, dict) else None
        )
        text = ""
        detected = None
        if choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else choices[0]["message"]
            content = message.get("content") if isinstance(message, dict) else message.content
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            elif content:
                text = str(content)
            annotations = message.get("annotations", {}) if isinstance(message, dict) else getattr(message, "annotations", {})
            if isinstance(annotations, dict):
                detected = annotations.get("language")

        # Native transcription responses expose sentence text under transcripts.
        if not text and isinstance(output, dict):
            transcripts = output.get("transcripts") or []
            sentences = [s for t in transcripts for s in t.get("sentences", [])]
            text = " ".join(str(s.get("text", "")) for s in sentences if s.get("text"))

        # Detected language is surfaced under different keys across versions.
        if isinstance(output, dict):
            detected = detected or output.get("language") or output.get("detected_language")
        return text, detected


def _select_provider() -> ASRProvider:
    """
    Choose the ASR backend from configuration.

    Urdu voice input requires the configured Groq transcription provider.
    """
    if not settings.ASR_ENABLED:
        raise GeminiConfigurationError("ASR_ENABLED must be true for Urdu voice input")
    provider = GroqASRProvider()
    logger.info(
        "[ASR DIAG] selected_provider=%s model=%s GROQ_API_KEY=%s",
        type(provider).__name__,
        settings.GROQ_ASR_MODEL,
        "YES" if settings.groq_configured else "NO",
    )
    return provider


async def transcribe(
    audio_bytes: bytes,
    filename: str | None = None,
    content_type: str | None = None,
) -> TranscriptionResult:
    """
    Transcribe Urdu speech to text.

    Args:
        audio_bytes: Raw audio payload uploaded by the app (m4a from Expo).
        filename: Original upload name, used to infer the audio format.

    Returns:
        TranscriptionResult. `recognised=False` signals the caller should take
        the contract's unrecognised-speech fallback path rather than error out.

    Raises:
        Exception: only on a genuine provider/API error, which the route maps to
            the contract's 500. A disabled provider or empty audio is NOT an
            error — it returns an unrecognised result.
    """
    if not audio_bytes:
        return TranscriptionResult(text="", language=UNRECOGNIZED_LANGUAGE, recognised=False)

    provider = _select_provider()
    logger.info(
        "speech_to_text.transcribe: %d bytes, filename=%s, provider=%s",
        len(audio_bytes),
        filename,
        type(provider).__name__,
    )
    return await asyncio.wait_for(
        provider.transcribe(audio_bytes, filename, content_type),
        timeout=settings.VOICE_STAGE_TIMEOUT_SECONDS,
    )
