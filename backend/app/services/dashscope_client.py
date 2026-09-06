"""
Central DashScope SDK access for the AI services.

One place owns the optional `dashscope` import and the credential/endpoint
wiring, so the ASR, LLM and TTS modules do not each repeat it. Key points:

- The import is GUARDED (like motor in database.py). If the wheel is missing,
  `DASHSCOPE_AVAILABLE` is False and callers degrade to their stubs/fallbacks
  instead of crashing at import time. This is what lets the backend boot, and
  the test suite run, on a machine without the SDK.
- The API key is read from settings and handed to the SDK lazily, only when a
  real call is about to happen. It is NEVER logged.
"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import dashscope  # type: ignore

    DASHSCOPE_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "SDK unavailable"
    dashscope = None  # type: ignore
    DASHSCOPE_AVAILABLE = False


def is_available() -> bool:
    """True when the DashScope SDK is importable in this environment."""
    return DASHSCOPE_AVAILABLE


def is_configured() -> bool:
    """
    True only when we can actually call Alibaba Cloud: SDK present AND a
    non-blank API key configured. Services check this to decide between a real
    call and their safe fallback.
    """
    return DASHSCOPE_AVAILABLE and settings.dashscope_configured


def configure() -> None:
    """
    Apply credentials/endpoint to the SDK immediately before a call.

    Safe to call repeatedly. Raises RuntimeError if invoked without the SDK or
    a key, so callers must gate on `is_configured()` first — this is defence in
    depth, not the primary guard.
    """
    if not DASHSCOPE_AVAILABLE:
        raise RuntimeError("DashScope SDK is not installed")
    if not settings.dashscope_configured:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    # Set on the module rather than passed per-call so every service shares one
    # configuration. The key is assigned here and never written to logs.
    dashscope.api_key = settings.DASHSCOPE_API_KEY
    base_url = settings.DASHSCOPE_BASE_URL.strip()
    if base_url:
        dashscope.base_http_api_url = base_url
