"""
MongoDB-path tests for the listing repository.

Everything else in this suite runs against the in-memory fallback, which leaves
the MongoDB branch of the repository unverified. These tests drive that branch
through a small async test double that emulates just enough of Motor's surface
(`insert_one`, `find(...).sort(...)`, `find_one`, `delete_one`,
`count_documents`) to prove the real code paths behave.

Deliberately no live server and no new dependency — the task requires the suite
to run without MongoDB.

The double mimics two Mongo behaviours that matter for correctness:
  * it assigns a real ObjectId `_id` when the document does not carry one;
  * it returns `createdAt` as a naive datetime, exactly as BSON does.
"""

import re
from datetime import datetime, timezone

import pytest

from app.models.listing import ListingCreate
from app.repositories import listing_repository

pytest.importorskip("bson", reason="bson ships with pymongo/motor")

from bson import ObjectId  # noqa: E402  - imported after the availability check


# ── Motor test double ──────────────────────────────────────────────────────


def _matches(document, query):
    """Evaluate the small subset of Mongo query syntax this repository emits."""
    for field, condition in query.items():
        value = document.get(field)
        if isinstance(condition, dict):
            if "$in" in condition:
                if value not in condition["$in"]:
                    return False
            elif "$regex" in condition:
                flags = re.IGNORECASE if "i" in condition.get("$options", "") else 0
                if value is None or not re.search(condition["$regex"], value, flags):
                    return False
            else:  # pragma: no cover - no other operators are used
                raise AssertionError(f"unsupported operator in {condition!r}")
        elif value != condition:
            return False
    return True


class FakeCursor:
    def __init__(self, documents):
        self._documents = documents

    def sort(self, spec):
        # Applied right-to-left, mirroring how a compound sort composes.
        for field, direction in reversed(spec):
            self._documents.sort(
                key=lambda d: (d[field] is None, str(d[field])),
                reverse=direction < 0,
            )
        return self

    async def __aiter__(self):
        for document in self._documents:
            yield document


class FakeCollection:
    """Minimal stand-in for an AsyncIOMotorCollection."""

    def __init__(self):
        self.documents = []
        self.insert_calls = []

    async def insert_one(self, document):
        self.insert_calls.append(dict(document))
        stored = dict(document)
        if "_id" not in stored:
            # Mongo generates the id when the document omits it.
            stored["_id"] = ObjectId()
        # BSON has no timezone: datetimes come back naive, in UTC.
        if isinstance(stored.get("createdAt"), datetime):
            stored["createdAt"] = stored["createdAt"].replace(tzinfo=None)
        self.documents.append(stored)
        return type("InsertResult", (), {"inserted_id": stored["_id"]})()

    def find(self, query):
        return FakeCursor([d for d in self.documents if _matches(d, query)])

    async def find_one(self, query):
        for document in self.documents:
            if _matches(document, query):
                return document
        return None

    async def delete_one(self, query):
        for index, document in enumerate(self.documents):
            if _matches(document, query):
                del self.documents[index]
                return type("DeleteResult", (), {"deleted_count": 1})()
        return type("DeleteResult", (), {"deleted_count": 0})()

    async def count_documents(self, query):
        return len([d for d in self.documents if _matches(d, query)])


@pytest.fixture
def mongo(monkeypatch):
    """Point the repository at the fake collection instead of the fallback."""
    collection = FakeCollection()
    monkeypatch.setattr(
        listing_repository, "get_listings_collection", lambda: collection
    )
    return collection


def payload(**overrides):
    return ListingCreate.model_validate(
        {
            "farmerId": "farmer_001",
            "crop": "wheat",
            "quantity": 500,
            "price": 3200,
            "location": "Lahore",
            "phone": "03001234567",
            **overrides,
        }
    )


# ── Writes ─────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_create_lets_mongodb_generate_the_id(mongo):
    """The driver must own `_id` so stored documents carry a real ObjectId."""
    await listing_repository.create_listing(payload())

    assert "_id" not in mongo.insert_calls[0]
    assert isinstance(mongo.documents[0]["_id"], ObjectId)


@pytest.mark.anyio
async def test_create_returns_the_objectid_as_a_string(mongo):
    listing = await listing_repository.create_listing(payload())

    assert isinstance(listing.id, str)
    assert listing.id == str(mongo.documents[0]["_id"])


@pytest.mark.anyio
async def test_stored_document_uses_contract_field_names(mongo):
    await listing_repository.create_listing(payload())

    stored = mongo.documents[0]
    assert set(stored) == {
        "_id",
        "farmerId",
        "crop",
        "quantity",
        "price",
        "location",
        "phone",
        "createdAt",
    }
    # The private in-memory sort key must never be persisted.
    assert "_seq" not in stored


@pytest.mark.anyio
async def test_naive_mongo_datetime_is_returned_as_utc(mongo):
    """Guards the ISO-format difference between the two backing stores."""
    listing = await listing_repository.create_listing(payload())

    assert mongo.documents[0]["createdAt"].tzinfo is None  # as BSON stores it
    assert listing.created_at.tzinfo is not None
    assert listing.created_at.utcoffset() == timezone.utc.utcoffset(None)


@pytest.mark.anyio
async def test_using_memory_store_is_false_when_mongodb_is_active(mongo):
    assert listing_repository.using_memory_store() is False


# ── Reads ──────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_get_listing_finds_a_document_written_with_an_objectid(mongo):
    created = await listing_repository.create_listing(payload())

    found = await listing_repository.get_listing(created.id)

    assert found is not None
    assert found.id == created.id


@pytest.mark.anyio
async def test_get_listing_also_finds_string_ids(mongo):
    """
    Seed scripts and imports can leave string `_id`s in the collection. Matching
    only ObjectIds would make those documents invisible to the API.
    """
    mongo.documents.append(
        {
            "_id": "6f2a1c8b4e9d3a7b5c0d1e2f",
            "farmerId": "farmer_002",
            "crop": "rice",
            "quantity": 200,
            "price": 4500,
            "location": "Gujranwala",
            "phone": "03212345678",
            "createdAt": datetime(2026, 8, 30, 9, 0, 0),
        }
    )

    found = await listing_repository.get_listing("6f2a1c8b4e9d3a7b5c0d1e2f")

    assert found is not None
    assert found.id == "6f2a1c8b4e9d3a7b5c0d1e2f"
    assert found.crop.value == "rice"


@pytest.mark.anyio
async def test_get_missing_listing_returns_none(mongo):
    assert await listing_repository.get_listing(str(ObjectId())) is None


@pytest.mark.anyio
async def test_malformed_id_returns_none_rather_than_raising(mongo):
    """ObjectId() would raise InvalidId — the repository must absorb that."""
    assert await listing_repository.get_listing("not-an-objectid") is None


@pytest.mark.anyio
async def test_list_listings_returns_newest_first(mongo):
    await listing_repository.create_listing(payload(location="First"))
    await listing_repository.create_listing(payload(location="Second"))
    await listing_repository.create_listing(payload(location="Third"))

    listings = await listing_repository.list_listings()

    assert [item.location for item in listings] == ["Third", "Second", "First"]


@pytest.mark.anyio
async def test_list_listings_filters_by_crop(mongo):
    await listing_repository.create_listing(payload(crop="wheat"))
    await listing_repository.create_listing(payload(crop="rice"))

    listings = await listing_repository.list_listings(crop="rice")

    assert [item.crop.value for item in listings] == ["rice"]


@pytest.mark.anyio
async def test_list_listings_filters_by_location_case_insensitively(mongo):
    await listing_repository.create_listing(payload(location="Lahore"))
    await listing_repository.create_listing(payload(location="Multan"))

    listings = await listing_repository.list_listings(location="lahor")

    assert [item.location for item in listings] == ["Lahore"]


@pytest.mark.anyio
async def test_list_listings_escapes_regex_metacharacters(mongo):
    """`.*` must be matched literally, not as a wildcard pattern."""
    await listing_repository.create_listing(payload(location="Lahore"))

    assert await listing_repository.list_listings(location=".*") == []


@pytest.mark.anyio
async def test_list_listings_combines_filters(mongo):
    await listing_repository.create_listing(payload(crop="wheat", location="Lahore"))
    await listing_repository.create_listing(payload(crop="rice", location="Lahore"))
    await listing_repository.create_listing(payload(crop="wheat", location="Multan"))

    listings = await listing_repository.list_listings(crop="wheat", location="Lahore")

    assert len(listings) == 1
    assert listings[0].crop.value == "wheat"
    assert listings[0].location == "Lahore"


@pytest.mark.anyio
async def test_list_listings_on_empty_collection_returns_empty(mongo):
    assert await listing_repository.list_listings() == []


# ── Deletes and counts ─────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_delete_listing_removes_the_document(mongo):
    created = await listing_repository.create_listing(payload())

    assert await listing_repository.delete_listing(created.id) is True
    assert mongo.documents == []


@pytest.mark.anyio
async def test_delete_missing_listing_returns_false(mongo):
    assert await listing_repository.delete_listing(str(ObjectId())) is False


@pytest.mark.anyio
async def test_delete_with_malformed_id_returns_false(mongo):
    assert await listing_repository.delete_listing("not-an-objectid") is False


@pytest.mark.anyio
async def test_count_listings_counts_documents(mongo):
    assert await listing_repository.count_listings() == 0

    await listing_repository.create_listing(payload())
    await listing_repository.create_listing(payload())

    assert await listing_repository.count_listings() == 2


# ── Single active store ───────────────────────────────────────────────
#
# The fallback exists so the API works without MongoDB, but the two stores must
# never be live at once — that would let reads and writes disagree depending on
# which branch a given call happened to take.


@pytest.mark.anyio
async def test_writes_never_reach_the_fallback_while_mongodb_is_active(mongo):
    await listing_repository.create_listing(payload())
    await listing_repository.create_listing(payload(crop="rice"))

    assert listing_repository._memory_store == {}
    assert len(mongo.documents) == 2
    # Reads and counts resolve from MongoDB, not from the empty fallback.
    assert await listing_repository.count_listings() == 2
    assert len(await listing_repository.list_listings()) == 2


@pytest.mark.anyio
async def test_deletes_never_reach_the_fallback_while_mongodb_is_active(mongo):
    """A pre-existing fallback entry must be left strictly alone."""
    listing_repository._memory_store["left-over"] = {"_id": "left-over"}
    created = await listing_repository.create_listing(payload())

    assert await listing_repository.delete_listing(created.id) is True

    assert mongo.documents == []
    assert "left-over" in listing_repository._memory_store
