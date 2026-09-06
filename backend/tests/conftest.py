"""
Shared test fixtures.

Two things every test in this suite gets, both aimed at the same goal — the
suite must never need a real MongoDB server, and no test may see another test's
data:

  * `force_memory_store` (autouse) pins the repository to its in-memory
    fallback and clears it around each test.
  * `client` provides a TestClient whose context manager runs the app lifespan.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.repositories import listing_repository


@pytest.fixture(autouse=True)
def force_memory_store(monkeypatch):
    """
    Pin the repository to the in-memory store and isolate each test.

    Patching the collection getter rather than the settings means the suite
    behaves identically whether or not the developer has a real MONGODB_URI in
    their local .env — tests never touch, or depend on, a live database.

    The reset is what makes assertions like "an empty database returns []"
    possible: `_memory_store` is module-global, so without it listings created by
    an earlier test leak into later ones.
    """
    monkeypatch.setattr(listing_repository, "get_listings_collection", lambda: None)
    listing_repository.reset_memory_store()
    yield
    listing_repository.reset_memory_store()


@pytest.fixture
def client():
    # The context manager form triggers lifespan, exercising DB startup too.
    with TestClient(app) as test_client:
        yield test_client
