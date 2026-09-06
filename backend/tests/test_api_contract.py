"""
Contract conformance tests.

These assert the response *shapes* the frontend depends on, not the stubbed AI
content. They should keep passing unchanged once the real pipeline lands — if
one fails after that, the contract has been broken.

Run:  cd backend && pytest -q
"""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    # The context manager form triggers lifespan, exercising DB startup too.
    with TestClient(app) as test_client:
        yield test_client


# ── Meta ───────────────────────────────────────────────────────────────────


def test_root_returns_service_banner(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["service"]
    assert body["version"]
    # Points callers at the interactive docs.
    assert body["docs"] == "/docs"


def test_health_reports_storage_and_pipeline(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["status"] == "ok"
    assert body["version"]

    # All six things /health is required to report.
    database = body["database"]
    assert "mongodb_connected" in database
    assert "using_memory_fallback" in database
    assert "detail" in database
    assert isinstance(database["listing_count"], int)

    # All four AI stages are now implemented (real providers, DashScope-backed).
    # /health reports which stages are wired, independently of whether a key is
    # configured.
    assert body["ai_pipeline"] == {
        "speech_to_text": True,
        "rag": True,
        "llm": True,
        "text_to_speech": True,
    }

    # Additive RAG detail (Phase 17): must not break the existing shape. The
    # suite runs without a key, so the embedding provider is unavailable and no
    # index is loaded.
    assert body["rag"]["status"] in {"unavailable", "not_indexed", "ready"}
    assert isinstance(body["rag"]["indexed_chunks"], int)
    assert body["rag"]["embedding_model"]


def test_health_listing_count_tracks_stored_listings(client):
    assert client.get("/health").json()["database"]["listing_count"] == 0

    client.post("/api/listings", json=VALID_LISTING)

    assert client.get("/health").json()["database"]["listing_count"] == 1


def test_health_reports_the_fallback_when_mongodb_is_absent(client):
    """The suite runs without MongoDB, so the fallback must be advertised."""
    database = client.get("/health").json()["database"]

    assert database["mongodb_connected"] is False
    assert database["using_memory_fallback"] is True


def test_docs_endpoints_are_served(client):
    """/docs is how this backend is verified by hand, so it must stay reachable."""
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/api/listings" in schema.json()["paths"]


# ── POST /api/listings ─────────────────────────────────────────────────────

VALID_LISTING = {
    "farmerId": "farmer_001",
    "crop": "wheat",
    "quantity": 500,
    "price": 3200,
    "location": "Lahore",
    "phone": "03001234567",
}


def test_create_listing_returns_contract_shape(client):
    response = client.post("/api/listings", json=VALID_LISTING)
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True

    listing = body["listing"]
    # Wire format must be camelCase with _id, exactly as the frontend types it.
    assert set(listing) == {
        "_id",
        "farmerId",
        "crop",
        "quantity",
        "price",
        "location",
        "phone",
        "createdAt",
    }
    assert listing["farmerId"] == "farmer_001"


def test_missing_field_uses_contract_error_message(client):
    payload = {k: v for k, v in VALID_LISTING.items() if k != "crop"}
    response = client.post("/api/listings", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    # The frontend matches on this exact prefix and the field name.
    assert body["error"] == "Missing required field: crop"


def test_invalid_phone_is_rejected_with_field_name(client):
    response = client.post("/api/listings", json={**VALID_LISTING, "phone": "123"})
    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    # AddListingScreen's mapApiError looks for the field name in the message.
    assert "phone" in body["error"].lower()


def test_non_positive_quantity_is_rejected(client):
    response = client.post("/api/listings", json={**VALID_LISTING, "quantity": 0})
    assert response.status_code == 400
    assert "quantity" in response.json()["error"].lower()


# ── GET /api/listings ──────────────────────────────────────────────────────


def test_get_listings_returns_list(client):
    client.post("/api/listings", json=VALID_LISTING)
    response = client.get("/api/listings")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert isinstance(body["listings"], list)


def test_filters_narrow_results_and_empty_is_success(client):
    client.post("/api/listings", json=VALID_LISTING)

    by_crop = client.get("/api/listings", params={"crop": "wheat"}).json()
    assert all(item["crop"] == "wheat" for item in by_crop["listings"])

    # Case-insensitive substring match on location.
    by_location = client.get("/api/listings", params={"location": "lahor"}).json()
    assert len(by_location["listings"]) >= 1

    # No matches must still be a success with an empty list, never a 404.
    empty = client.get("/api/listings", params={"location": "Atlantis"})
    assert empty.status_code == 200
    assert empty.json() == {"success": True, "listings": []}


def test_unknown_crop_filter_is_rejected(client):
    response = client.get("/api/listings", params={"crop": "bananas"})
    assert response.status_code == 400
    assert response.json()["success"] is False


# ── GET / DELETE /api/listings/{id} ────────────────────────────────────────


def test_get_and_delete_listing_by_id(client):
    created = client.post("/api/listings", json=VALID_LISTING).json()["listing"]
    listing_id = created["_id"]

    fetched = client.get(f"/api/listings/{listing_id}")
    assert fetched.status_code == 200
    assert fetched.json()["listing"]["_id"] == listing_id

    deleted = client.delete(f"/api/listings/{listing_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"success": True, "message": "Listing deleted"}

    # Second delete must report the miss rather than succeed again.
    assert client.delete(f"/api/listings/{listing_id}").status_code == 404


def test_missing_listing_returns_not_found_error(client):
    response = client.get("/api/listings/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"success": False, "error": "Listing not found"}


# ── POST /api/assistant/voice ──────────────────────────────────────────────


def test_voice_returns_all_contract_fields(client, monkeypatch):
    from app.services import llm, speech_to_text, text_to_speech

    async def fake_transcribe(audio_bytes, filename):
        return speech_to_text.TranscriptionResult("گندم کی فصل", "urdu", True)

    async def fake_answer(question, context=""):
        return "گندم کو مناسب وقت پر پانی دیں۔"

    async def fake_speech(text):
        return "UklGRg=="

    monkeypatch.setattr(speech_to_text, "transcribe", fake_transcribe)
    monkeypatch.setattr(llm, "generate_answer", fake_answer)
    monkeypatch.setattr(text_to_speech, "synthesize", fake_speech)

    response = client.post(
        "/api/assistant/voice",
        files={"audio": ("recording.m4a", io.BytesIO(b"fake-audio-bytes"), "audio/m4a")},
        data={"farmerId": "farmer_001"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "success",
        "transcription",
        "language",
        "answer",
        "audio_base64",
    }
    assert body["success"] is True
    assert body["audio_base64"]  # must never be empty — the app plays it


def test_voice_without_audio_returns_contract_error(client):
    response = client.post("/api/assistant/voice", data={"farmerId": "farmer_001"})
    assert response.status_code == 400
    # Must be the contract's wording, not FastAPI's generic validation message.
    assert response.json() == {"success": False, "error": "No audio file provided"}


def test_empty_audio_takes_unrecognized_fallback(client):
    response = client.post(
        "/api/assistant/voice",
        files={"audio": ("empty.m4a", io.BytesIO(b""), "audio/m4a")},
    )
    # Contract: an empty upload is the "No audio file provided" 400 case.
    assert response.status_code == 400
    assert response.json() == {"success": False, "error": "No audio file provided"}


# ── Error envelope ────────────────────────────────────────────────────


def test_unknown_route_uses_error_envelope(client):
    """Even framework-generated 404s must not leak Starlette's {"detail": ...}."""
    response = client.get("/api/definitely-not-a-route")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert "detail" not in body


def test_wrong_method_uses_error_envelope(client):
    response = client.put("/api/listings")
    assert response.status_code == 405
    body = response.json()
    assert body["success"] is False
    assert "detail" not in body
