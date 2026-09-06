"""
API routers.

Each module owns one resource and declares its own `/api/...` prefix, so
main.py includes them without re-declaring paths.
"""

from app.routes import assistant, listings

__all__ = ["assistant", "listings"]
