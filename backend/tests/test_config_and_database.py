"""
MongoDB configuration and connection-lifecycle tests.

Covers the parts of TASK 2/3/9 that the HTTP contract tests cannot reach: that
configuration arrives through environment variables, that the connection helpers
degrade gracefully with no credentials, that shutdown really releases the client,
and that `/health` cannot leak a password.

Nothing here needs a MongoDB server. The one test that touches a real URI uses
an address that is guaranteed to fail, and asserts on the *reported reason*
rather than on any successful connection.
"""

import re
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.database import (
    LISTINGS_COLLECTION,
    _safe_reason,
    close_mongo_connection,
    connect_to_mongo,
    connection_status,
    get_database,
    get_listings_collection,
    is_connected,
)

BACKEND_DIR = Path(__file__).resolve().parent.parent


# ── TASK 2: configuration through environment variables ────────────────────


def test_mongodb_settings_come_from_environment(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://example-host:27017")
    monkeypatch.setenv("MONGODB_DB_NAME", "some_other_db")

    # A fresh Settings() rather than the cached singleton, so the env is re-read.
    settings = Settings()

    assert settings.MONGODB_URI == "mongodb://example-host:27017"
    assert settings.MONGODB_DB_NAME == "some_other_db"


def test_mongodb_uri_defaults_to_empty_so_the_app_can_boot_without_it(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)

    settings = Settings(_env_file=None)

    assert settings.MONGODB_URI == ""
    assert settings.mongodb_configured is False


def test_mongodb_configured_is_true_only_for_a_real_value(monkeypatch):
    assert Settings(MONGODB_URI="mongodb://host:27017").mongodb_configured is True
    # Whitespace is not a configuration.
    assert Settings(MONGODB_URI="   ").mongodb_configured is False


def test_default_database_name_is_the_project_database():
    assert Settings(_env_file=None).MONGODB_DB_NAME == "kisaandost"


# ── TASK 2: no secrets in tracked files ────────────────────────────────────


def test_env_example_contains_no_real_credentials():
    """
    .env.example is committed, so every value in it must be a placeholder.

    Guards against the classic mistake of pasting a working Atlas URI in while
    debugging and committing it.
    """
    text = (BACKEND_DIR / ".env.example").read_text(encoding="utf-8")

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "MONGODB_URI":
            assert value.strip() == "", "MONGODB_URI must be left empty"
        # No line may carry embedded credentials.
        assert not re.search(r"//[^/<\s]+:[^/<\s]+@", value), f"credentials in {key}"

    assert "DASHSCOPE_API_KEY=" in text
    # The API key placeholder must be empty too.
    assert re.search(r"^DASHSCOPE_API_KEY=\s*$", text, re.MULTILINE)


def test_no_connection_string_is_hardcoded_in_the_app_package():
    """The real URI must live in .env, never in a Python file."""
    offenders = []
    for path in (BACKEND_DIR / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"mongodb(\+srv)?://\S*", text):
            snippet = match.group(0)
            # Documentation examples use placeholder angle brackets or localhost.
            if "<" in snippet or "localhost" in snippet or "example" in snippet:
                continue
            offenders.append(f"{path.name}: {snippet}")

    assert offenders == [], f"hardcoded connection string: {offenders}"


def test_env_file_is_git_ignored():
    """A committed .env would leak credentials, so the ignore rule must exist."""
    ignore_rules = (BACKEND_DIR / ".gitignore").read_text(encoding="utf-8")

    assert re.search(r"^\.env$", ignore_rules, re.MULTILINE)
    assert re.search(r"^!\.env\.example$", ignore_rules, re.MULTILINE)


# ── TASK 3: graceful behaviour with no credentials ─────────────────────────


@pytest.mark.anyio
async def test_connect_returns_false_without_a_uri(monkeypatch):
    """Startup must not raise when MONGODB_URI is unset."""
    monkeypatch.setattr("app.database.settings.MONGODB_URI", "")

    assert await connect_to_mongo() is False
    assert is_connected() is False
    assert connection_status() == "MONGODB_URI is not set"


@pytest.mark.anyio
async def test_connect_returns_false_for_an_unreachable_server(monkeypatch):
    """
    A bad URI is a recoverable state, not a crash — and it must give up quickly.

    The elapsed-time assertion is the point of this test: pymongo's default
    serverSelectionTimeoutMS is 30s, which would stall application startup for
    half a minute behind a bad URI. database.py sets 3s, and this proves it.
    """
    monkeypatch.setattr(
        "app.database.settings.MONGODB_URI", "mongodb://127.0.0.1:27099"
    )

    started = time.monotonic()
    connected = await connect_to_mongo()
    elapsed = time.monotonic() - started

    assert connected is False
    assert is_connected() is False
    # The reason is reported, so /health can explain the fallback.
    assert connection_status() != "connected"
    assert elapsed < 10, f"startup blocked for {elapsed:.1f}s; timeout not applied"


@pytest.mark.anyio
async def test_close_releases_the_handles(monkeypatch):
    monkeypatch.setattr("app.database.settings.MONGODB_URI", "")
    await connect_to_mongo()

    await close_mongo_connection()

    assert get_database() is None
    assert get_listings_collection() is None
    assert is_connected() is False


def test_no_database_means_no_collection_handle():
    """The repository relies on this None to pick its fallback."""
    assert get_database() is None
    assert get_listings_collection() is None


def test_collection_name_is_centralised():
    """Routes and repositories must not hardcode the collection name."""
    assert LISTINGS_COLLECTION == "listings"


# ── TASK 9: /health must not expose secrets ────────────────────────────────


@pytest.mark.parametrize(
    "reason",
    [
        "ServerSelectionTimeoutError: mongodb://admin:SuperSecret123@host:27017 failed",
        "ConfigurationError: mongodb+srv://user:p4ssw0rd@cluster.mongodb.net",
        "InvalidURI: //root:letmein@localhost",
    ],
)
def test_safe_reason_redacts_credentials(reason):
    cleaned = _safe_reason(reason)

    assert "SuperSecret123" not in cleaned
    assert "p4ssw0rd" not in cleaned
    assert "letmein" not in cleaned
    assert "<credentials>" in cleaned


def test_safe_reason_truncates_a_verbose_driver_dump():
    """ServerSelectionTimeoutError appends a whole topology description."""
    cleaned = _safe_reason("x" * 5000)

    assert len(cleaned) < 250
    assert cleaned.endswith("...")


def test_safe_reason_leaves_an_ordinary_message_alone():
    assert _safe_reason("MONGODB_URI is not set") == "MONGODB_URI is not set"
    assert _safe_reason("connected") == "connected"


def test_health_response_carries_no_secret_values(client, monkeypatch):
    """End-to-end: nothing secret-shaped reaches the /health body."""
    monkeypatch.setattr("app.database._state.reason", "failed for mongodb://u:pw@h/db")

    body = client.get("/health").text

    assert "pw@h" not in body
    assert "<credentials>" in body
