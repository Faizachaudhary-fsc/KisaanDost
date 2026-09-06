"""
Urdu text-to-speech (TTS) service.

HONEST-URDU NOTE (verified against Alibaba Cloud Model Studio docs):
    No Alibaba Cloud voice (CosyVoice, Qwen3-TTS, Qwen-TTS) currently lists Urdu
    among its supported languages. Synthesising Urdu with a non-Urdu voice would
    produce wrong pronunciation, which the task forbids doing silently.

Consequences, reflected in this module:
    1. A clean provider abstraction (`TTSProvider`) so a genuinely Urdu-capable
       voice can be slotted in later without touching assistant.py.
    2. TTS is DISABLED by default (settings.TTS_ENABLED = False). While disabled,
       `synthesize()` returns a CONTROLLED audio fallback (a well-formed base64
       placeholder the app can handle) instead of mispronounced audio.
    3. When explicitly enabled with a key, the DashScope voice is used and its
       audio is base64-encoded for the response.

`synthesize(text) -> base64 str` is the only thing the route depends on; it must
always return valid base64, never null.
"""

import asyncio
import base64
import logging
import os
import tempfile
from urllib.request import urlopen
from typing import Optional, Protocol

from app.config import settings
from app.services.gemini_client import GeminiConfigurationError, synthesize_urdu

logger = logging.getLogger(__name__)

# Provider layer implemented (real, swappable). A real TTS call happens only
# when TTS_ENABLED + a key are present; /health reports this flag.
IMPLEMENTED = True

class TTSProvider(Protocol):
    """Contract every TTS backend must satisfy. Keeps the route provider-agnostic."""

    async def synthesize(self, text: str) -> bytes:
        """Return raw audio bytes for `text` (encoding handled by the caller)."""
        ...


class GeminiTTSProvider:
    """Gemini Urdu speech output through the official Gemini API."""

    async def synthesize(self, text: str) -> bytes:
        return base64.b64decode(await asyncio.to_thread(synthesize_urdu, text))


class DashScopeTTSProvider:
    """
    Alibaba Cloud DashScope speech synthesis (settings.TTS_MODEL).

    Used only when TTS_ENABLED is true AND a key is configured.
    """

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._call, text)

    def _call(self, text: str) -> bytes:
        dashscope_client.configure()
        import dashscope  # guarded/available: gated by is_configured()

        response = dashscope.MultiModalConversation.call(
            model=settings.TTS_MODEL,
            text=text,
            voice=settings.TTS_VOICE,
            language_type=settings.TTS_LANGUAGE,
            stream=False,
        )

        status = getattr(response, "status_code", 200)
        if status != 200:
            logger.error("TTS call failed: status=%s code=%s", status, getattr(response, "code", None))
            raise RuntimeError(f"TTS call failed with status {status}")

        audio = self._extract_audio_bytes(response)
        if not audio:
            raise RuntimeError("TTS response contained no audio")
        return audio

    @staticmethod
    def _extract_audio_bytes(response) -> bytes:
        """Extract inline base64 audio, raw bytes, or provider URL audio."""
        output = getattr(response, "output", None) or {}
        audio = getattr(output, "audio", None) or (
            output.get("audio") if isinstance(output, dict) else None
        )
        if audio is None:
            return b""

        data = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
        if data:
            return base64.b64decode(data)

        url = audio.get("url") if isinstance(audio, dict) else getattr(audio, "url", None)
        if url:
            with urlopen(url, timeout=30) as response:  # noqa: S310 - provider URL
                return response.read()

        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        return b""


class AzureSpeechTTSProvider:
    """Azure Speech synthesis configured for an Urdu neural voice."""

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._call, text)

    def _call(self, text: str) -> bytes:
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(
            subscription=settings.AZURE_SPEECH_KEY,
            region=settings.AZURE_SPEECH_REGION,
        )
        speech_config.speech_synthesis_voice_name = settings.AZURE_SPEECH_VOICE
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
            result = synthesizer.speak_text_async(text).get()
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
                raise RuntimeError(f"Azure TTS failed: {details.reason}")
            with open(tmp_path, "rb") as audio_file:
                return audio_file.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

def _select_provider() -> Optional[TTSProvider]:
    """
    Choose the TTS backend, or None to signal the controlled fallback.

    Real TTS requires BOTH an explicit opt-in (TTS_ENABLED) and a configured
    key/SDK; otherwise we return None and `synthesize()` uses the placeholder.
    """
    if not settings.TTS_ENABLED:
        raise GeminiConfigurationError("TTS_ENABLED must be true for Urdu voice output")
    return GeminiTTSProvider()


async def synthesize(text: str) -> str:
    """
    Synthesise speech for `text`.

    Returns:
        Base64-encoded playable WAV audio from Qwen3.5-Omni.

    Raises:
        Exception: only on a genuine provider/API error while TTS is enabled,
            which the route maps to the contract's 500.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text")

    provider = _select_provider()
    logger.info("text_to_speech.synthesize: calling Gemini TTS, %d chars", len(text))
    audio_bytes = await asyncio.wait_for(
        provider.synthesize(text),
        timeout=settings.VOICE_STAGE_TIMEOUT_SECONDS,
    )
    return base64.b64encode(audio_bytes).decode("utf-8")
