"""
Voice assistant schemas.

Response shape is fixed by the contract in frontend/services/api.ts:

    {
      "success": true,
      "transcription": "meri gandum de pattay peelay ne",
      "language": "urdu",
      "answer": "...",
      "audio_base64": "..."
    }

Note `audio_base64` is snake_case in the contract while listing fields are
camelCase. That inconsistency is intentional here — the contract is the source
of truth, so it is reproduced verbatim rather than normalised.

The unrecognised-speech fallback uses the same 200 shape with
`transcription: ""` and `language: "unrecognized"`. It is a success, not an
error, so the UI can still play a spoken "please repeat" reply.
"""

from pydantic import BaseModel, Field


class VoiceResponse(BaseModel):
    """POST /api/assistant/voice success envelope."""

    success: bool = True
    transcription: str = Field(
        description="Urdu transcription of the audio; empty string when unrecognised.",
        examples=["meri gandum de pattay peelay ne"],
    )
    language: str = Field(
        description="Detected language, or the literal 'unrecognized' on fallback.",
        examples=["urdu"],
    )
    answer: str = Field(
        description="Assistant reply text, in Roman Urdu.",
        examples=["Yeh nitrogen ki kami ho sakti hai."],
    )
    audio_base64: str = Field(
        description="Base64-encoded TTS audio (m4a) of `answer`.",
    )


class VoicePipelineResult(BaseModel):
    """
    Internal hand-off between the service layer and the route.

    Kept separate from `VoiceResponse` so the AI pipeline can later attach
    diagnostics (latency, retrieved RAG chunks, model IDs) without changing
    the public contract.
    """

    transcription: str
    language: str
    answer: str
    audio_base64: str
    recognised: bool = True
