"""
Data access layer.

Routes depend on these functions rather than on the database driver, so the
MongoDB-vs-in-memory decision stays in one place.
"""

from app.repositories import listing_repository

__all__ = ["listing_repository"]
