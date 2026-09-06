"""
Marketplace data layer tests.

Covers the listings API end-to-end through the repository, plus repository-level
unit tests for the MongoDB-specific conversion logic that cannot be reached
through HTTP without a live server.

Every test runs against the in-memory fallback (see conftest.py), so the suite
needs no MongoDB.

Run:  cd backend && pytest
"""

from datetime import datetime, timezone

import pytest

from app.models.listing import Listing
from app.repositories import listing_repository

VALID_LISTING = {
    "farmerId": "farmer_001",
    "crop": "wheat",
    "quantity": 500,
    "price": 3200,
    "location": "Lahore",
    "phone": "03001234567",
}

LISTING_FIELDS = {
    "_id",
    "farmerId",
    "crop",
    "quantity",
    "price",
    "location",
    "phone",
    "createdAt",
}


def create(client, **overrides):
    """Create a listing and return the response body's `listing` object."""
    response = client.post("/api/listings", json={**VALID_LISTING, **overrides})
    assert response.status_code == 201, response.text
    return response.json()["listing"]


# ── POST /api/listings — success ───────────────────────────────────────────


def test_create_listing_succeeds_with_201_and_envelope(client):
    response = client.post("/api/listings", json=VALID_LISTING)

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert set(body["listing"]) == LISTING_FIELDS


def test_created_listing_echoes_submitted_values(client):
    listing = create(client)

    assert listing["farmerId"] == "farmer_001"
    assert listing["crop"] == "wheat"
    assert listing["quantity"] == 500
    assert listing["price"] == 3200
    assert listing["location"] == "Lahore"
    assert listing["phone"] == "03001234567"


def test_created_listing_id_is_a_string_not_an_objectid(client):
    listing = create(client)

    assert isinstance(listing["_id"], str)
    assert listing["_id"]
    # Never the repr of a BSON ObjectId.
    assert "ObjectId" not in listing["_id"]


def test_created_at_is_iso_parseable_and_utc(client):
    listing = create(client)

    parsed = datetime.fromisoformat(listing["createdAt"])
    # Timezone-aware, so the value is unambiguous for the client.
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_each_created_listing_gets_a_unique_id(client):
    first = create(client)
    second = create(client)

    assert first["_id"] != second["_id"]


# ── POST /api/listings — validation ────────────────────────────────────────


@pytest.mark.parametrize(
    "missing_field",
    ["farmerId", "crop", "quantity", "price", "location", "phone"],
)
def test_missing_required_field_returns_contract_error(client, missing_field):
    payload = {k: v for k, v in VALID_LISTING.items() if k != missing_field}

    response = client.post("/api/listings", json=payload)

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"] == f"Missing required field: {missing_field}"
    # Must never be FastAPI's default validation shape.
    assert "detail" not in body


def test_empty_body_returns_contract_error(client):
    response = client.post("/api/listings", json={})

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert "detail" not in body


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantity", 0),
        ("quantity", -5),
        ("price", 0),
        ("price", -1),
        ("phone", "123"),
        ("phone", "abcdefghijk"),
        ("crop", "bananas"),
        ("location", "   "),
    ],
)
def test_invalid_field_value_is_rejected_naming_the_field(client, field, value):
    response = client.post("/api/listings", json={**VALID_LISTING, field: value})

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    # AddListingScreen's mapApiError attributes the error by finding the field
    # name in the message, so it has to appear.
    assert field in body["error"].lower()


def test_rejected_listing_is_not_persisted(client):
    client.post("/api/listings", json={**VALID_LISTING, "phone": "nope"})

    assert client.get("/api/listings").json()["listings"] == []


# ── GET /api/listings ──────────────────────────────────────────────────────


def test_empty_database_returns_empty_list_not_an_error(client):
    response = client.get("/api/listings")

    assert response.status_code == 200
    assert response.json() == {"success": True, "listings": []}


def test_get_all_listings_returns_every_listing(client):
    create(client, location="Lahore")
    create(client, location="Multan")
    create(client, location="Karachi")

    body = client.get("/api/listings").json()

    assert body["success"] is True
    assert len(body["listings"]) == 3
    assert all(set(item) == LISTING_FIELDS for item in body["listings"])


def test_listings_are_returned_newest_first(client):
    first = create(client, location="First")
    second = create(client, location="Second")
    third = create(client, location="Third")

    listings = client.get("/api/listings").json()["listings"]

    assert [item["_id"] for item in listings] == [
        third["_id"],
        second["_id"],
        first["_id"],
    ]


# ── GET /api/listings — filtering ──────────────────────────────────────────


def test_filter_by_crop_returns_only_that_crop(client):
    create(client, crop="wheat")
    create(client, crop="rice")
    create(client, crop="cotton")

    listings = client.get("/api/listings", params={"crop": "rice"}).json()["listings"]

    assert len(listings) == 1
    assert listings[0]["crop"] == "rice"


def test_filter_by_crop_with_no_matches_returns_empty_list(client):
    create(client, crop="wheat")

    response = client.get("/api/listings", params={"crop": "maize"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "listings": []}


def test_unknown_crop_filter_is_rejected_with_contract_error(client):
    response = client.get("/api/listings", params={"crop": "bananas"})

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert "detail" not in body


def test_filter_by_location_is_case_insensitive_substring(client):
    create(client, location="Lahore")
    create(client, location="Multan")

    for query in ("Lahore", "lahore", "LAHORE", "lahor", "hor"):
        listings = client.get("/api/listings", params={"location": query}).json()[
            "listings"
        ]
        assert len(listings) == 1, f"query={query!r}"
        assert listings[0]["location"] == "Lahore"


def test_filter_by_location_with_no_matches_returns_empty_list(client):
    create(client, location="Lahore")

    response = client.get("/api/listings", params={"location": "Atlantis"})

    assert response.status_code == 200
    assert response.json() == {"success": True, "listings": []}


def test_crop_and_location_filters_combine_with_and(client):
    create(client, crop="wheat", location="Lahore")
    create(client, crop="rice", location="Lahore")
    create(client, crop="wheat", location="Multan")

    listings = client.get(
        "/api/listings", params={"crop": "wheat", "location": "Lahore"}
    ).json()["listings"]

    assert len(listings) == 1
    assert listings[0]["crop"] == "wheat"
    assert listings[0]["location"] == "Lahore"


def test_location_filter_treats_input_as_literal_text(client):
    """A regex metacharacter must not be interpreted as a pattern."""
    create(client, location="Lahore")

    response = client.get("/api/listings", params={"location": ".*"})

    assert response.status_code == 200
    assert response.json()["listings"] == []


# ── GET /api/listings/{id} ─────────────────────────────────────────────────


def test_get_listing_by_id_returns_that_listing(client):
    created = create(client, location="Faisalabad")

    response = client.get(f"/api/listings/{created['_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["listing"] == created


def test_get_listing_by_id_selects_the_right_one(client):
    first = create(client, location="First")
    second = create(client, location="Second")

    body = client.get(f"/api/listings/{first['_id']}").json()

    assert body["listing"]["_id"] == first["_id"]
    assert body["listing"]["location"] == "First"
    assert body["listing"]["_id"] != second["_id"]


def test_get_missing_listing_returns_404_and_contract_error(client):
    response = client.get("/api/listings/6f2a1c8b4e9d3a7b5c0d1e2f")

    assert response.status_code == 404
    assert response.json() == {"success": False, "error": "Listing not found"}


def test_get_listing_with_malformed_id_returns_404_not_500(client):
    """An id that is not ObjectId-shaped must be a clean miss, never a crash."""
    response = client.get("/api/listings/not-a-real-id")

    assert response.status_code == 404
    assert response.json() == {"success": False, "error": "Listing not found"}


# ── DELETE /api/listings/{id} ──────────────────────────────────────────────


def test_delete_listing_returns_success_message(client):
    created = create(client)

    response = client.delete(f"/api/listings/{created['_id']}")

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "Listing deleted"}


def test_deleted_listing_is_actually_gone(client):
    created = create(client)
    client.delete(f"/api/listings/{created['_id']}")

    assert client.get(f"/api/listings/{created['_id']}").status_code == 404
    assert client.get("/api/listings").json()["listings"] == []


def test_delete_only_removes_the_targeted_listing(client):
    doomed = create(client, location="Doomed")
    survivor = create(client, location="Survivor")

    client.delete(f"/api/listings/{doomed['_id']}")

    listings = client.get("/api/listings").json()["listings"]
    assert len(listings) == 1
    assert listings[0]["_id"] == survivor["_id"]


def test_delete_missing_listing_returns_404_and_contract_error(client):
    response = client.delete("/api/listings/6f2a1c8b4e9d3a7b5c0d1e2f")

    assert response.status_code == 404
    assert response.json() == {"success": False, "error": "Listing not found"}


def test_deleting_twice_reports_the_second_attempt_as_missing(client):
    created = create(client)

    assert client.delete(f"/api/listings/{created['_id']}").status_code == 200
    second = client.delete(f"/api/listings/{created['_id']}")

    assert second.status_code == 404
    assert second.json() == {"success": False, "error": "Listing not found"}


# ── Repository unit tests ──────────────────────────────────────────────────
#
# These reach the MongoDB-specific logic that HTTP tests cannot exercise without
# a live server.


def test_count_listings_tracks_the_store(client):
    import anyio

    assert anyio.run(listing_repository.count_listings) == 0

    create(client)
    create(client)

    assert anyio.run(listing_repository.count_listings) == 2


def test_using_memory_store_is_true_without_mongodb():
    assert listing_repository.using_memory_store() is True


def test_reset_memory_store_empties_it(client):
    create(client)
    listing_repository.reset_memory_store()

    assert client.get("/api/listings").json()["listings"] == []


def test_id_query_matches_both_objectid_and_string_forms():
    """
    A 24-hex id may be stored as an ObjectId (documents this backend writes) or
    as a string (seed data, imports). Lookups must match either.
    """
    hex_id = "6f2a1c8b4e9d3a7b5c0d1e2f"

    query = listing_repository._id_query(hex_id)

    candidates = query["_id"]["$in"]
    assert hex_id in candidates
    if listing_repository.BSON_AVAILABLE:
        from bson import ObjectId

        assert ObjectId(hex_id) in candidates


def test_id_query_falls_back_to_plain_string_for_non_objectid():
    query = listing_repository._id_query("not-a-valid-objectid")

    assert query == {"_id": "not-a-valid-objectid"}


def test_model_stringifies_a_raw_objectid():
    """The API must expose a string id even when Mongo hands back an ObjectId."""
    if not listing_repository.BSON_AVAILABLE:
        pytest.skip("bson is not installed")

    from bson import ObjectId

    oid = ObjectId()
    listing = Listing.model_validate(
        {
            "_id": oid,
            "farmerId": "farmer_001",
            "crop": "wheat",
            "quantity": 500,
            "price": 3200,
            "location": "Lahore",
            "phone": "03001234567",
            "createdAt": datetime(2026, 9, 1, 12, 0, 0),
        }
    )

    assert listing.id == str(oid)
    assert isinstance(listing.id, str)


def test_model_attaches_utc_to_naive_mongo_datetimes():
    """
    MongoDB returns BSON datetimes without tzinfo. Without normalisation the same
    listing would serialise differently depending on the active backing store.
    """
    listing = Listing.model_validate(
        {
            "_id": "6f2a1c8b4e9d3a7b5c0d1e2f",
            "farmerId": "farmer_001",
            "crop": "wheat",
            "quantity": 500,
            "price": 3200,
            "location": "Lahore",
            "phone": "03001234567",
            "createdAt": datetime(2026, 9, 1, 12, 0, 0),  # naive, as Mongo returns
        }
    )

    assert listing.created_at.tzinfo is not None
    assert listing.created_at.isoformat() == "2026-09-01T12:00:00+00:00"


def test_document_is_serialised_with_contract_field_names(client):
    """Wire format stays camelCase with `_id`, matching frontend/services/api.ts."""
    listing = create(client)

    assert set(listing) == LISTING_FIELDS
    # snake_case internals must not leak.
    assert "farmer_id" not in listing
    assert "created_at" not in listing
    assert "id" not in listing
