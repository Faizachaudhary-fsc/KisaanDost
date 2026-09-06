"""Schemas shared across routes."""

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    The contract's failure envelope: { "success": false, "error": "..." }

    The frontend never sees thrown exceptions — it reads this shape off the
    resolved response body, so every failure path must serialise to it.
    """

    success: bool = Field(default=False, examples=[False])
    error: str = Field(examples=["Listing not found"])


class HealthResponse(BaseModel):
    """Operational probe — not part of the frontend API contract."""

    success: bool = True
    status: str = Field(examples=["ok"])
    version: str
    database: dict[str, Any] = Field(
        description="Whether MongoDB is live, and the in-memory fallback state."
    )
    ai_pipeline: dict[str, bool] = Field(
        description="Per-stage flag: True once that stage is really implemented."
    )
    rag: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "RAG readiness detail: status (ready/not_indexed/unavailable), "
            "indexed_chunks, and embedding_model. Additive — never breaks the "
            "existing ai_pipeline shape."
        ),
    )
