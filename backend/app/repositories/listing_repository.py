"""
Listing persistence — the marketplace data layer.

Wraps the `listings` collection behind a small function-level interface and
transparently falls back to an in-memory store when MongoDB is unavailable, so
the whole marketplace API works before credentials exist.

Two invariants keep the two backing stores interchangeable from the API's point
of view:

  * ids are always exposed as 24-character hex **strings**, never raw ObjectIds;
  * listings always come back newest-first, with ties broken deterministically.

The connection itself is owned by app/database.py — this module never creates a
client of its own. The fallback store is per-process and NOT durable: data is
lost on restart, and `/health` reports which store is live so nobody mistakes it
for real persistence.
"""

import itertools
import logging
import re
import uuid
from typing import Any, Optional

from app.database import get_listings_collection
from app.models.listing import Listing, ListingCreate, utc_now

logger = logging.getLogger(__name__)

try:  # pragma: no cover - bson ships with pymongo, which motor requires
    from bson import ObjectId
    from bson.errors import InvalidId

    BSON_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Mirrors database.py's defensive motor import: the API must still work
    # (in-memory) in an environment where the Mongo stack is absent.
    ObjectId = None  # type: ignore[assignment,misc]
    InvalidId = Exception  # type: ignore[assignment,misc]
    BSON_AVAILABLE = False


# ── In-memory fallback store ───────────────────────────────────────────────

# Keyed by id string. Values are documents shaped exactly like Mongo's, plus a
# private "_seq" sort key that never leaves this module.
_memory_store: dict[str, dict[str, Any]] = {}

# Monotonic counter used only to break `createdAt` ties. Two listings created
# within the same microsecond would otherwise sort unpredictably, making
# "newest first" flaky under fast test runs.
_insertion_counter = itertools.count()


def reset_memory_store() -> None:
    """
    Clear the fallback store.

    Test-only seam. It lets each test start from a known-empty state without
    needing a real MongoDB server; application code never calls this.
    """
    _memory_store.clear()


def using_memory_store() -> bool:
    """
    True when reads and writes go to the in-memory fallback.

    Derived from the collection handle rather than the connection flag so it
    always reflects what this module will actually do.
    """
    return get_listings_collection() is None


# ── Document <-> model conversion ──────────────────────────────────────────


def _new_memory_id() -> str:
    """
    A 24-character hex id, matching the shape of a MongoDB ObjectId.

    Using the same shape in both modes means the frontend cannot accidentally
    depend on a difference between them.
    """
    return uuid.uuid4().hex[:24]


def _id_query(listing_id: str) -> dict[str, Any]:
    """
    Build a query that matches a listing by id in either stored form.

    Documents written by this backend get a real ObjectId `_id` from the driver,
    but the same collection can also hold string ids — from seed scripts,
    imports, or documents created while the in-memory fallback was active and
    later migrated. Matching both forms means a lookup never silently misses a
    document that plainly exists.
    """
    if BSON_AVAILABLE:
        try:
            # A valid 24-hex string could be stored either way, so match both.
            return {"_id": {"$in": [listing_id, ObjectId(listing_id)]}}
        except (InvalidId, TypeError):
            # Not a valid ObjectId (e.g. an arbitrary path segment); a plain
            # string match is both correct and all that is possible.
            pass
    return {"_id": listing_id}


def _to_listing(document: dict[str, Any]) -> Listing:
    """
    Convert a stored document into the API response model.

    Drops the internal sort key; `_id` stringification and `createdAt` timezone
    normalisation are handled by validators on the Listing model itself.
    """
    data = dict(document)
    data.pop("_seq", None)
    return Listing.model_validate(data)


# ── Commands ───────────────────────────────────────────────────────────────


async def create_listing(payload: ListingCreate) -> Listing:
    """
    Insert a listing and return it with its generated id and `createdAt`.

    In MongoDB mode the `_id` is left to the driver so stored documents carry a
    real ObjectId; the in-memory mode generates an equivalent hex string.
    """
    document: dict[str, Any] = {
        "farmerId": payload.farmer_id,
        "crop": payload.crop.value,
        "quantity": payload.quantity,
        "price": payload.price,
        "location": payload.location,
        "phone": payload.phone,
        "createdAt": utc_now(),
    }

    collection = get_listings_collection()
    if collection is not None:
        result = await collection.insert_one(document)
        document["_id"] = result.inserted_id
        return _to_listing(document)

    document["_id"] = _new_memory_id()
    document["_seq"] = next(_insertion_counter)
    _memory_store[document["_id"]] = document
    return _to_listing(document)


async def delete_listing(listing_id: str) -> bool:
    """Delete one listing. Returns False when the id was not found."""
    collection = get_listings_collection()

    if collection is not None:
        result = await collection.delete_one(_id_query(listing_id))
        return result.deleted_count > 0

    return _memory_store.pop(listing_id, None) is not None


# ── Queries ────────────────────────────────────────────────────────────────


async def list_listings(
    crop: Optional[str] = None,
    location: Optional[str] = None,
) -> list[Listing]:
    """
    Return listings newest-first, optionally filtered.

    `crop` is an exact match; `location` is a case-insensitive substring match,
    mirroring the frontend's own filter behaviour. Filters combine with AND. An
    empty result is a normal outcome, never an error.
    """
    collection = get_listings_collection()

    if collection is not None:
        query: dict[str, Any] = {}
        if crop:
            query["crop"] = crop
        if location:
            # Escaped so user input cannot inject regex syntax.
            query["location"] = {"$regex": re.escape(location), "$options": "i"}

        # createdAt is indexed descending (see database.py); the secondary _id
        # key only breaks ties so equal timestamps stay in a stable order.
        cursor = collection.find(query).sort([("createdAt", -1), ("_id", -1)])
        return [_to_listing(document) async for document in cursor]

    results = list(_memory_store.values())
    if crop:
        results = [d for d in results if d["crop"] == crop]
    if location:
        needle = location.lower()
        results = [d for d in results if needle in d["location"].lower()]
    results.sort(key=lambda d: (d["createdAt"], d["_seq"]), reverse=True)
    return [_to_listing(d) for d in results]


async def get_listing(listing_id: str) -> Optional[Listing]:
    """Fetch one listing, or None when it does not exist."""
    collection = get_listings_collection()

    if collection is not None:
        document = await collection.find_one(_id_query(listing_id))
        return _to_listing(document) if document else None

    document = _memory_store.get(listing_id)
    return _to_listing(document) if document else None


async def count_listings() -> int:
    """Total number of stored listings. Surfaced by /health."""
    collection = get_listings_collection()
    if collection is not None:
        return await collection.count_documents({})
    return len(_memory_store)
