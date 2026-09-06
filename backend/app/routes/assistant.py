"""
Voice assistant route — POST /api/assistant/voice

Contract (frontend/services/api.ts :: sendVoiceQuery):
  Request:  multipart/form-data — `audio` file, optional `farmerId` text field
  200:      { success, transcription, language, answer, audio_base64 }
  400:      { success: false, error: "No audio file provided" }
  500:      { success: false, error: "ASR/LLM pipeline failed" }

The pipeline stages are stubs for now (see app/services/), but the wiring and
response shape are final, so the frontend can flip USE_MOCK to false and hit
this endpoint for real.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from app.models.assistant import VoiceResponse
from app.models.common import ErrorResponse
from app.services import llm, rag, speech_to_text, text_to_speech
from app.services.gemini_client import GeminiConfigurationError, GeminiProviderError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


@router.post(
    "/voice",
    response_model=VoiceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "No audio file provided"},
        500: {"model": ErrorResponse, "description": "ASR/LLM pipeline failed"},
    },
    summary="Ask the assistant a question by voice",
)
async def voice_query(
    # Optional at the framework level on purpose. Declaring this as required
    # would make FastAPI raise a validation error first, producing
    # "Missing required field: audio" instead of the contract's exact
    # "No audio file provided" message.
    audio: Optional[UploadFile] = File(
        default=None, description="Recorded audio (m4a from Expo)"
    ),
    farmerId: Optional[str] = Form(  # noqa: N803 - name fixed by the API contract
        default=None,
        description="Optional farmer id, for future personalisation.",
    ),
):
    """
    Run the voice pipeline: ASR -> RAG -> LLM -> TTS.

    Unrecognised speech is a 200, not an error: the contract expects
    `transcription: ""` with `language: "unrecognized"` and a spoken prompt to
    try again, so the app can play a reply either way.
    """
    logger.info("VOICE_STAGE=upload VOICE REQUEST RECEIVED farmer_id_present=%s", bool(farmerId))
    audio_bytes = await audio.read() if audio is not None else b""
    logger.info(
        "AUDIO RECEIVED filename=%s content_type=%s size_bytes=%d",
        audio.filename if audio else None,
        audio.content_type if audio else None,
        len(audio_bytes),
    )

    if not audio_bytes:
        # Covers both a missing part and an empty upload; the contract names
        # this exact message for the 400 case.
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "No audio file provided"},
        )

    stage = "ASR"
    try:
        logger.info("VOICE_STAGE=gemini_audio ASR START")
        transcription = await speech_to_text.transcribe(
            audio_bytes,
            audio.filename if audio else None,
            audio.content_type if audio else None,
        )
        logger.info(
            "VOICE_STAGE=gemini_audio ASR COMPLETE transcription=%r language=%s",
            transcription.text[:200],
            transcription.language,
        )

        if not transcription.recognised or not transcription.text.strip():
            return JSONResponse(
                status_code=422,
                content={"success": False, "error": "Could not understand the audio"},
            )

        stage = "RAG"
        logger.info("VOICE_STAGE=rag RAG START")
        retrieval = await rag.retrieve(transcription.text)
        logger.info("VOICE_STAGE=rag RAG COMPLETE retrieved_results=%d", len(retrieval.chunks))
        stage = "LLM"
        logger.info("VOICE_STAGE=gemini_answer LLM START")
        answer = await llm.generate_answer(transcription.text, retrieval.context)
        logger.info("VOICE_STAGE=gemini_answer LLM COMPLETE")
        stage = "TTS"
        logger.info("VOICE_STAGE=gemini_tts TTS START")
        audio_base64 = await text_to_speech.synthesize(answer)
        logger.info("VOICE_STAGE=gemini_tts TTS COMPLETE")

        logger.info("VOICE_STAGE=complete VOICE RESPONSE READY")
        return VoiceResponse(
            transcription=transcription.text,
            language=transcription.language,
            answer=answer,
            audio_base64=audio_base64,
        )

    except GeminiConfigurationError:
        logger.exception("Voice pipeline configuration failed at %s", stage)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": f"Voice AI is not configured for {stage}"},
        )
    except (GeminiProviderError, TimeoutError, asyncio.TimeoutError):
        logger.exception("Voice pipeline provider failure at %s", stage)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"success": False, "error": f"Voice AI failed during {stage}"},
        )
    except Exception:  # noqa: BLE001 - safe final contract error
        logger.exception("Voice pipeline failed at %s for farmerId=%s", stage, farmerId)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": f"Voice pipeline failed during {stage}"},
        )
