"""
MongoDB connection management.

Design constraint for the hackathon: the backend must start and serve requests
even when no MongoDB credentials exist yet. So this module never raises on
startup. It reports its state through `is_connected()`, and callers (see
app/repositories/listing_repository.py) fall back to an in-memory store when
the database is unavailable.

`motor` is imported defensively too, so a partially installed environment
still boots.
"""

import logging
import re
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Matches the credentials segment of a connection URI: scheme://user:pass@host
_CREDENTIALS_IN_URI = re.compile(r"//[^/\s]*:[^/\s]*@")

# /health is unauthenticated, so the driver's message is capped rather than
# passed through whole. ServerSelectionTimeoutError in particular appends a full
# topology dump that is long and of no use to a client.
_MAX_REASON_LENGTH = 200

try:  # pragma: no cover - depends on the installed environment
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

    MOTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncIOMotorClient = None  # type: ignore[assignment,misc]
    AsyncIOMotorDatabase = Any  # type: ignore[assignment,misc]
    MOTOR_AVAILABLE = False
    logger.warning("motor is not installed — MongoDB support is disabled.")


# Collection names, centralised so routes/repositories never hardcode strings.
LISTINGS_COLLECTION = "listings"


class _MongoState:
    """Holds the process-wide client so it is created exactly once."""

    client: Optional[Any] = None
    database: Optional[Any] = None
    connected: bool = False
    reason: str = "not initialised"


_state = _MongoState()


async def connect_to_mongo() -> bool:
    """
    Attempt a connection. Returns True on success, False otherwise.

    Never raises: a failure here is an expected, recoverable state.
    """
    if not MOTOR_AVAILABLE:
        _state.reason = "motor package not installed"
        logger.warning("MongoDB unavailable: %s. Using in-memory store.", _state.reason)
        return False

    if not settings.mongodb_configured:
        _state.reason = "MONGODB_URI is not set"
        logger.warning("MongoDB unavailable: %s. Using in-memory store.", _state.reason)
        return False

    try:
        client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=3000,
        )
        # ping forces real network validation; the constructor alone is lazy.
        await client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001 - any driver/network error is non-fatal
        _state.reason = f"{type(exc).__name__}: {exc}"
        logger.warning("MongoDB connection failed: %s. Using in-memory store.", _state.reason)
        return False

    _state.client = client
    _state.database = client[settings.MONGODB_DB_NAME]
    _state.connected = True
    _state.reason = "connected"
    logger.info("Connected to MongoDB database '%s'.", settings.MONGODB_DB_NAME)
    await _ensure_indexes()
    return True


async def _ensure_indexes() -> None:
    """
    Indexes backing GET /api/listings. Idempotent — create_index is a no-op when
    the index already exists, so this is safe on every startup.

    What each one is actually worth, having checked it against the queries the
    repository emits:

      * `crop` — used. The filter is an exact equality match, which is exactly
        what a single-field b-tree index serves.
      * `createdAt` descending — used, and the important one. Every listing
        query sorts newest-first, and a descending index lets MongoDB walk the
        results in order instead of doing a blocking in-memory sort.
      * `location` — only partially useful, and deliberately kept anyway. The
        repository filters location with a case-insensitive, non-anchored
        `$regex`, and MongoDB cannot seek a plain b-tree index for that shape;
        it scans. The index is retained because it is cheap at marketplace scale
        and still serves exact-match/prefix lookups. If location filtering ever
        becomes a bottleneck, the fix is a stored lowercase field or a collated
        index — NOT another regex index, which would not help.

    No further indexes are created on purpose: the remaining fields are never
    filtered or sorted on, so an index would only slow writes down.
    """
    if _state.database is None:
        return
    try:
        collection = _state.database[LISTINGS_COLLECTION]
        await collection.create_index("crop")
        await collection.create_index("location")
        await collection.create_index([("createdAt", -1)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create indexes: %s", exc)


async def close_mongo_connection() -> None:
    """Close the client on shutdown."""
    if _state.client is not None:
        _state.client.close()
        logger.info("MongoDB connection closed.")
    _state.client = None
    _state.database = None
    _state.connected = False
    _state.reason = "closed"


def is_connected() -> bool:
    return _state.connected


def _safe_reason(reason: str) -> str:
    """
    Strip anything credential-shaped out of a status string, then cap its length.

    Driver errors were checked against several failing URIs (bad SRV host,
    refused connection, malformed URI, wrong scheme, unescaped password) and
    pymongo redacted the password in every one. This is defence in depth: the
    reason text is raw third-party output on an unauthenticated endpoint, and a
    future driver version or URI shape could start echoing it back.
    """
    redacted = _CREDENTIALS_IN_URI.sub("//<credentials>@", reason)
    if len(redacted) > _MAX_REASON_LENGTH:
        return redacted[:_MAX_REASON_LENGTH].rstrip() + "..."
    return redacted


def connection_status() -> str:
    """
    Human-readable state, surfaced by the /health endpoint.

    Sanitised because /health is unauthenticated — see `_safe_reason`.
    """
    return _safe_reason(_state.reason)


def get_database() -> Optional[Any]:
    """
    The active database handle, or None when unavailable.

    Callers MUST handle None rather than assuming a live connection.
    """
    return _state.database


def get_listings_collection() -> Optional[Any]:
    database = get_database()
    if database is None:
        return None
    return database[LISTINGS_COLLECTION]
