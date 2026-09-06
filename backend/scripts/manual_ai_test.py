"""
Manual, opt-in live test of the Alibaba Cloud AI pipeline.

This is NOT part of the automated suite (pytest never calls it). It performs a
real DashScope call so you can confirm your key and the selected model work.

Usage (from the backend/ directory, with .env holding a real key):

    python -m scripts.manual_ai_test

Safety:
- The API key is read from the environment / .env via app.config. It is NEVER
  printed or logged by this script.
- If no key is configured, the script says so and exits WITHOUT fabricating a
  successful result.

By default only the LLM (Qwen) is exercised, because it is the one stage with
documented Urdu support. ASR/TTS are attempted only if you have opted in via
ASR_ENABLED / TTS_ENABLED, and their lack of documented Urdu support is noted.
"""

import asyncio
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/manual_ai_test.py) too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services import dashscope_client, llm, text_to_speech  # noqa: E402

SAMPLE_QUESTION = "Meri gandum ke pattay peelay ho rahe hain, mein kya karoon?"


async def main() -> int:
    print("KisaanDost — manual Alibaba Cloud AI test")
    print("-" * 48)
    print(f"SDK installed        : {dashscope_client.is_available()}")
    print(f"API key configured   : {settings.dashscope_configured}")  # bool only, never the key
    print(f"LLM model            : {settings.LLM_MODEL}")
    print(f"ASR enabled / model  : {settings.ASR_ENABLED} / {settings.STT_MODEL}")
    print(f"TTS enabled / model  : {settings.TTS_ENABLED} / {settings.TTS_MODEL}")
    print("-" * 48)

    if not dashscope_client.is_available():
        print("RESULT: DashScope SDK is not installed. Live test NOT performed.")
        return 2
    if not settings.dashscope_configured:
        print("RESULT: No DASHSCOPE_API_KEY configured. Live test NOT performed.")
        print("        (Set DASHSCOPE_API_KEY in backend/.env to run this.)")
        return 2

    print(f"\nAsking Qwen: {SAMPLE_QUESTION!r}\n")
    try:
        answer = await llm.generate_answer(SAMPLE_QUESTION)
    except Exception as exc:  # noqa: BLE001 - surface a safe summary only
        print(f"RESULT: Qwen call FAILED ({type(exc).__name__}). See server logs for detail.")
        return 1

    print("Qwen answer (Roman Urdu):")
    print(f"  {answer}")

    if settings.TTS_ENABLED:
        print("\nRunning TTS (note: no Alibaba voice documents Urdu support)...")
        try:
            audio_b64 = await text_to_speech.synthesize(answer)
            print(f"  TTS returned {len(audio_b64)} base64 chars.")
        except Exception as exc:  # noqa: BLE001
            print(f"  TTS FAILED ({type(exc).__name__}).")

    print("\nRESULT: Live LLM test performed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
