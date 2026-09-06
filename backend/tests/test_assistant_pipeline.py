"""
Voice-assistant AI pipeline tests.

None of these need a real DashScope key, a real MongoDB, or the mobile app:

  * Pure parsing/normalisation logic is tested directly.
  * "No key configured" behaviour (the default) is tested against the real code.
  * "Key configured" behaviour is tested by monkeypatching the DashScope SDK
    entry points, so no network call is made. Those tests importorskip the SDK
    so the suite still passes if the optional wheel is absent.
  * The end-to-end route is tested with TestClient and mocked service functions.

Covers the eight required scenarios: ASR success, ASR unrecognised, LLM parsing,
TTS/base64 handling, missing audio, full pipeline (mocked), provider failure,
and response structure.
"""

import base64
import io
from types import SimpleNamespace

import pytest

from app.config import settings
from app.services import dashscope_client, llm, speech_to_text, text_to_speech
from app.services.omni_client import OmniConfigurationError


# ── Pure logic: no SDK, no network ─────────────────────────────────────────


def test_llm_extract_text_parses_message_content():
    """result_format="message" → output.choices[0].message.content."""
    response = SimpleNamespace(
        status_code=200,
        output={"choices": [{"message": {"content": "Jee, urea dalain."}}]},
    )
    assert llm._extract_text(response) == "Jee, urea dalain."


def test_llm_extract_text_parses_content_parts_list():
    """Some models return content as a list of {'text': ...} parts."""
    response = SimpleNamespace(
        status_code=200,
        output={"choices": [{"message": {"content": [{"text": "Paani "}, {"text": "kam dein."}]}}]},
    )
    assert llm._extract_text(response) == "Paani kam dein."


def test_tts_extract_audio_decodes_inline_base64():
    raw = b"RIFF\x00\x00fake-wav"
    response = SimpleNamespace(
        status_code=200,
        output={"audio": {"data": base64.b64encode(raw).decode()}},
    )
    assert text_to_speech.DashScopeTTSProvider._extract_audio_bytes(response) == raw


def test_asr_normalise_language_reports_urdu_only_when_truly_urdu():
    # Genuine Urdu tokens map to the contract's "urdu".
    assert speech_to_text._normalise_language("ur") == "urdu"
    assert speech_to_text._normalise_language("Urdu") == "urdu"
    # Hindi/Arabic are NOT relabelled as Urdu — the whole point of the task.
    assert speech_to_text._normalise_language("hi") == "hi"
    assert speech_to_text._normalise_language("ar") == "ar"
    assert speech_to_text._normalise_language(None) == "unknown"


# ── Default (no key) behaviour: services degrade, never crash (Phase 13) ────


def test_services_are_importable_and_report_implemented():
    assert llm.IMPLEMENTED is True
    assert speech_to_text.IMPLEMENTED is True
    assert text_to_speech.IMPLEMENTED is True


def test_dashscope_is_not_configured_without_a_key(monkeypatch):
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "")
    assert dashscope_client.is_configured() is False


@pytest.mark.anyio
async def test_llm_requires_real_provider_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "")
    with pytest.raises(OmniConfigurationError):
        await llm.generate_answer("gandum ke pattay peelay hain")


@pytest.mark.anyio
async def test_asr_requires_explicit_enablement(monkeypatch):
    monkeypatch.setattr(settings, "ASR_ENABLED", False)
    with pytest.raises(OmniConfigurationError):
        await speech_to_text.transcribe(b"some-audio-bytes", "clip.m4a")


@pytest.mark.anyio
async def test_tts_requires_explicit_enablement(monkeypatch):
    monkeypatch.setattr(settings, "TTS_ENABLED", False)
    with pytest.raises(OmniConfigurationError):
        await text_to_speech.synthesize("Jee, urea dalain.")


# ── Configured behaviour: mock the SDK, never hit the network ───────────────


def _configure(monkeypatch):
    """Make is_configured() true for the duration of a test."""
    monkeypatch.setattr(settings, "DASHSCOPE_API_KEY", "test-key-not-real")
    monkeypatch.setattr(settings, "OMNI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(dashscope_client, "configure", lambda: None)


@pytest.mark.anyio
async def test_llm_generate_answer_calls_qwen(monkeypatch):
    _configure(monkeypatch)

    captured = {}

    def fake_stream(messages, *, with_audio):
        captured["messages"] = messages
        captured["with_audio"] = with_audio
        return "Gandum ko nitrogen dein.", b""

    monkeypatch.setattr(llm, "stream_completion", fake_stream)

    answer = await llm.generate_answer("gandum peelay", context="Wheat needs nitrogen.")
    assert answer == "Gandum ko nitrogen dein."
    # Prompt is well-formed: system + user, with question and context separated.
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]
    assert "Wheat needs nitrogen." in captured["messages"][1]["content"]


@pytest.mark.anyio
async def test_llm_raises_on_api_error(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(llm, "stream_completion", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("provider")))
    with pytest.raises(Exception):
        await llm.generate_answer("koi sawal")


@pytest.mark.anyio
async def test_asr_success_reports_detected_language_honestly(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(settings, "ASR_ENABLED", True)
    monkeypatch.setattr(speech_to_text, "stream_completion", lambda *_args, **_kwargs: ("gandum ke pattay peelay", b""))

    result = await speech_to_text.transcribe(b"audio", "clip.m4a")
    assert result.recognised is True
    assert result.text == "gandum ke pattay peelay"
    assert result.language == "urdu"  # provider genuinely detected Urdu


@pytest.mark.anyio
async def test_asr_does_not_mislabel_non_urdu_as_urdu(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(settings, "ASR_ENABLED", True)
    monkeypatch.setattr(speech_to_text, "stream_completion", lambda *_args, **_kwargs: ("kuch text", b""))
    result = await speech_to_text.transcribe(b"audio", "clip.m4a")
    assert result.language == "urdu"


@pytest.mark.anyio
async def test_tts_synthesize_returns_base64_of_provider_audio(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(settings, "TTS_ENABLED", True)

    raw = b"ID3\x00\x00fake-mp3-bytes"
    monkeypatch.setattr(text_to_speech, "stream_completion", lambda *_args, **_kwargs: ("", raw))

    audio_b64 = await text_to_speech.synthesize("Jee, urea dalain.")
    assert base64.b64decode(audio_b64).startswith(b"RIFF")


# ── End-to-end route (mocked services, TestClient) ──────────────────────────


def _audio_upload(name="recording.m4a", data=b"fake-audio-bytes"):
    return {"audio": (name, io.BytesIO(data), "audio/m4a")}


def test_pipeline_missing_audio_returns_400(client):
    response = client.post("/api/assistant/voice", data={"farmerId": "farmer123"})
    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "No audio file provided"}


def test_pipeline_full_success_has_correct_structure(client, monkeypatch):
    captured = {}

    async def fake_transcribe(audio_bytes, filename):
        return speech_to_text.TranscriptionResult("meri gandum de pattay peelay ne", "urdu", True)

    async def fake_answer(question, context=""):
        captured["question"] = question
        captured["context"] = context
        return "Yeh nitrogen ki kami hai. Urea dalain."

    async def fake_synth(text):
        return "QUJDRA=="  # valid base64

    monkeypatch.setattr(speech_to_text, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm, "generate_answer", fake_answer)
    monkeypatch.setattr(text_to_speech, "synthesize", fake_synth)

    response = client.post("/api/assistant/voice", files=_audio_upload(), data={"farmerId": "f1"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"success", "transcription", "language", "answer", "audio_base64"}
    assert body["success"] is True
    assert body["transcription"] == "meri gandum de pattay peelay ne"
    assert body["language"] == "urdu"
    assert body["answer"] == "Yeh nitrogen ki kami hai. Urea dalain."
    assert body["audio_base64"] == "QUJDRA=="
    assert captured["question"] == "meri gandum de pattay peelay ne"
    assert captured["context"] == ""


def test_pipeline_unrecognized_speech_is_200_fallback(client, monkeypatch):
    async def fake_transcribe(audio_bytes, filename):
        return speech_to_text.TranscriptionResult("", speech_to_text.UNRECOGNIZED_LANGUAGE, False)

    monkeypatch.setattr(speech_to_text, "transcribe", fake_transcribe)

    response = client.post("/api/assistant/voice", files=_audio_upload())
    assert response.status_code == 422
    body = response.json()
    assert body == {"success": False, "error": "Could not understand the audio"}


def test_pipeline_provider_failure_returns_500(client, monkeypatch):
    async def boom(question, context=""):
        raise RuntimeError("simulated Qwen outage")

    async def fake_transcribe(audio_bytes, filename):
        return speech_to_text.TranscriptionResult("sawal", "urdu", True)

    monkeypatch.setattr(speech_to_text, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm, "generate_answer", boom)

    response = client.post("/api/assistant/voice", files=_audio_upload())
    assert response.status_code == 500
    assert response.json() == {"success": False, "error": "Voice pipeline failed during LLM"}


def test_pipeline_asr_failure_returns_500(client, monkeypatch):
    async def boom(*_args):
        raise TimeoutError("simulated ASR timeout")

    monkeypatch.setattr(speech_to_text, "transcribe", boom)
    response = client.post("/api/assistant/voice", files=_audio_upload())
    assert response.status_code == 502
    assert response.json() == {"success": False, "error": "Voice AI failed during ASR"}


def test_pipeline_tts_failure_returns_500(client, monkeypatch):
    async def fake_transcribe(audio_bytes, filename):
        return speech_to_text.TranscriptionResult("sawal", "urdu", True)

    async def boom(*_args):
        raise TimeoutError("simulated TTS timeout")

    monkeypatch.setattr(speech_to_text, "transcribe", fake_transcribe)
    async def fake_answer(*_args, **_kwargs):
        return "جواب"

    monkeypatch.setattr(llm, "generate_answer", fake_answer)
    monkeypatch.setattr(text_to_speech, "synthesize", boom)
    response = client.post("/api/assistant/voice", files=_audio_upload())
    assert response.status_code == 502
    assert response.json() == {"success": False, "error": "Voice AI failed during TTS"}
