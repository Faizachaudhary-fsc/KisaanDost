"""
FastAPI application entry point.

Run locally:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs: http://localhost:8000/docs

A note on error handling, which is the subtle part of this file: FastAPI's
default validation response is HTTP 422 with a `{"detail": [...]}` body. The
API contract requires `{"success": false, "error": "..."}` on every failure, so
the handlers below translate all error paths into that envelope. Without them
the frontend's `result.error` reads as undefined and its error UI breaks.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.database import (
    close_mongo_connection,
    connect_to_mongo,
    connection_status,
    is_connected,
)
from app.models.common import HealthResponse
from app.repositories import listing_repository
from app.routes import assistant, listings
from app.services import pipeline_status, rag_status

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown.

    A failed database connection is logged, not raised — the app must still
    serve requests while credentials are pending.
    """
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    await connect_to_mongo()
    if not is_connected():
        logger.warning(
            "Running WITHOUT MongoDB (%s). Listings are stored in memory and "
            "will be lost on restart.",
            connection_status(),
        )
    yield
    await close_mongo_connection()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Backend for KisaanDost — a voice-first Urdu agricultural assistant "
        "and crop marketplace for Pakistani farmers.\n\n"
        "Endpoint shapes follow the existing frontend API contract "
        "(`frontend/services/api.ts`), which is the source of truth."
    ),
    lifespan=lifespan,
)

# The Expo app calls from a device/emulator on the LAN, so its origin is not
# fixed during development. Narrow CORS_ORIGINS for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error handlers: force every failure into the contract's envelope ────────


def _field_name(loc: tuple) -> str:
    """
    Last meaningful segment of a Pydantic error location.

    Drops the leading "body"/"query" marker so the message names the field the
    way the client sent it (e.g. `farmerId`, not `body -> farmerId`).
    """
    parts = [str(p) for p in loc if p not in ("body", "query", "path")]
    return parts[-1] if parts else "request body"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Translate validation failures into `{ success: false, error }` with 400.

    Only the first error is reported: the contract's `error` is a single string,
    and the frontend maps it to one field at a time.
    """
    errors = exc.errors()
    if not errors:
        message = "Invalid request"
    else:
        first = errors[0]
        field = _field_name(first.get("loc", ()))
        if first.get("type") == "missing":
            message = f"Missing required field: {field}"
        else:
            reason = first.get("msg", "invalid value")
            # Field name stays in the string so the app can attribute the error
            # to the right input (see mapApiError in AddListingScreen.tsx).
            message = f"Invalid value for field: {field} ({reason})"

    logger.info("Validation failed on %s: %s", request.url.path, message)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": message},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Reshape HTTPException into the envelope.

    Registered against Starlette's HTTPException rather than FastAPI's subclass
    on purpose: unmatched routes and method-not-allowed are raised by Starlette
    itself, and a handler bound to the subclass would miss them, leaking the
    default `{"detail": "Not Found"}` body.
    """
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last-resort handler so an unexpected crash still returns valid JSON.

    The real cause is logged with a traceback; the client gets a generic message.
    """
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": "Internal server error"},
    )


# ── Routers ────────────────────────────────────────────────────────────────

app.include_router(assistant.router)
app.include_router(listings.router)


# ── Operational endpoints (not part of the frontend contract) ──────────────


@app.get("/", tags=["meta"], summary="Service banner")
async def root():
    return {
        "success": True,
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Health probe")
async def health():
    """Reports the active storage backend and which AI stages are still stubbed."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        database={
            "mongodb_connected": is_connected(),
            "detail": connection_status(),
            "using_memory_fallback": listing_repository.using_memory_store(),
            "listing_count": await listing_repository.count_listings(),
        },
        ai_pipeline=pipeline_status(),
        rag=rag_status(),
    )
