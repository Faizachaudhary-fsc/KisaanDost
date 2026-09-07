"""
Application configuration.

Every value has a safe default so the backend boots with no .env present.
Copy .env.example to .env and fill in real values as they become available.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "KisaanDost API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ── MongoDB ────────────────────────────────────────────────────────────
    # Empty on purpose: an empty URI means "no database configured", and the
    # app falls back to an in-memory store instead of failing to start.
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "kisaandost"

    # ── CORS ───────────────────────────────────────────────────────────────
    # Expo dev clients call from arbitrary LAN origins, so default to open.
    # Tighten this to the deployed frontend origin before production.
    CORS_ORIGINS: str = "*"

    # ── Google Gemini voice assistant ─────────────────────────────────────
    GEMINI_API_KEY: str = ""
    GEMINI_TEXT_MODEL: str = "gemini-2.5-flash"
    GEMINI_TTS_MODEL: str = "gemini-2.5-flash-preview-tts"
    GEMINI_TTS_VOICE: str = "Kore"

    @property
    def gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY.strip())

    # ── Legacy Alibaba settings (unused by the voice path) ────────────────
    # Empty key = "AI not configured": the pipeline degrades to safe stubs/
    # fallbacks rather than crashing, so the app boots and tests run with no key.
    DASHSCOPE_API_KEY: str = ""
    # Region endpoint override for the DashScope SDK. Leave blank to use the SDK
    # default (Beijing). Set to the international endpoint for a Singapore
    # account: https://dashscope-intl.aliyuncs.com/api/v1
    DASHSCOPE_BASE_URL: str = ""

    # Qwen3.5-Omni supports Urdu audio input and speech output through the
    # official OpenAI-compatible API.
    OMNI_MODEL: str = "qwen3.5-omni-plus"
    OMNI_VOICE: str = "Tina"
    OMNI_BASE_URL: str = ""

    # Kept as a compatibility alias for existing non-voice tooling/tests.
    LLM_MODEL: str = "qwen-plus"

    # Multilingual ASR model. NOTE: as of the current Model Studio docs, no
    # Alibaba ASR model lists Urdu among its supported languages (Hindi and
    # Arabic are, Urdu is not). ASR is therefore OFF by default so we never
    # mislabel Hindi/Arabic as Urdu — see app/services/speech_to_text.py.
    STT_MODEL: str = "qwen3.5-omni-plus"
    ASR_ENABLED: bool = True

    # TTS model and documented system voice. No Alibaba voice currently lists
    # Urdu, so TTS remains OFF by default; see text_to_speech.py.
    TTS_MODEL: str = "qwen3.5-omni-plus"
    TTS_VOICE: str = "Tina"
    TTS_LANGUAGE: str = "Urdu"
    TTS_ENABLED: bool = True

    # Maximum time an individual external AI stage may hold a voice request.
    VOICE_STAGE_TIMEOUT_SECONDS: float = 60.0

    # Optional Azure Speech configuration. Azure supports both Urdu (Pakistan)
    # recognition and Urdu neural voices; blank values keep the existing
    # graceful fallback behavior.
    AZURE_SPEECH_KEY: str = ""
    AZURE_SPEECH_REGION: str = ""
    AZURE_SPEECH_LANGUAGE: str = "ur-PK"
    AZURE_SPEECH_VOICE: str = "ur-PK-UzmaNeural"

    # ── RAG (agricultural knowledge base) ──────────────────────────────────
    # Embedding model for both ingestion and query embedding. Unlike ASR/TTS,
    # Alibaba's text-embedding-v4 (Qwen3-Embedding series) genuinely lists 100+
    # languages INCLUDING Urdu, so multilingual Urdu retrieval is real here.
    EMBEDDING_MODEL: str = "text-embedding-v4"
    # 1024 is the model default and the recommended balance of quality vs cost.
    EMBEDDING_DIMENSION: int = 1024
    # How many grounding chunks to feed the LLM. Small on purpose (Phase 8): we
    # never dump the whole knowledge base into the prompt.
    RAG_TOP_K: int = 4
    # Cosine-similarity floor. A query whose best match scores below this is
    # treated as "no relevant agricultural knowledge found" (Phase 9) rather
    # than forcing irrelevant context into the answer.
    RAG_MIN_SCORE: float = 0.30

    @property
    def dashscope_configured(self) -> bool:
        """True only when a non-blank DashScope key is present."""
        return bool(self.DASHSCOPE_API_KEY.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS_ORIGINS as a list, since env vars arrive as comma-separated text."""
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def mongodb_configured(self) -> bool:
        return bool(self.MONGODB_URI.strip())

    @property
    def azure_speech_configured(self) -> bool:
        return bool(self.AZURE_SPEECH_KEY.strip() and self.AZURE_SPEECH_REGION.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()


settings = get_settings()
