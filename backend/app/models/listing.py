"""
Listing schemas.

Field names follow the API contract exactly (camelCase on the wire: `_id`,
`farmerId`, `createdAt`) while staying snake_case in Python. Aliases do the
translation, and FastAPI serialises responses by alias by default.

Validation rules are deliberately identical to the frontend's own checks in
screens/AddListingScreen.tsx, so the backend never rejects input the UI
considers valid:
  quantity > 0, price > 0, location non-empty, phone exactly 11 digits.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Crop(str, Enum):
    """Mirrors the frontend `Crop` union type."""

    WHEAT = "wheat"
    RICE = "rice"
    COTTON = "cotton"
    MAIZE = "maize"


class ListingCreate(BaseModel):
    """Request body for POST /api/listings."""

    model_config = ConfigDict(populate_by_name=True)

    farmer_id: str = Field(alias="farmerId", min_length=1, examples=["farmer_001"])
    crop: Crop = Field(examples=["wheat"])
    quantity: float = Field(gt=0, description="Kilograms", examples=[500])
    price: float = Field(gt=0, description="PKR per kg", examples=[3200])
    location: str = Field(min_length=1, examples=["Lahore"])
    phone: str = Field(examples=["03001234567"])

    @field_validator("location")
    @classmethod
    def location_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("location must not be blank")
        return cleaned

    @field_validator("phone")
    @classmethod
    def phone_is_eleven_digits(cls, v: str) -> str:
        cleaned = v.strip()
        if not (len(cleaned) == 11 and cleaned.isdigit()):
            raise ValueError("phone must be exactly 11 digits")
        return cleaned


class Listing(BaseModel):
    """A stored listing, as returned to the frontend."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id", examples=["6f2a1c8b4e9d3a7b5c0d1e2f"])
    farmer_id: str = Field(alias="farmerId", examples=["farmer_001"])
    crop: Crop
    quantity: float
    price: float
    location: str
    phone: str
    created_at: datetime = Field(alias="createdAt")

    @field_validator("id", mode="before")
    @classmethod
    def stringify_id(cls, v: object) -> str:
        """
        Accept a raw MongoDB ObjectId and expose it as a string.

        The contract types `_id` as a string, so an ObjectId must never reach the
        API surface. Coercing here means every construction path is covered, not
        just the repository's.
        """
        return v if isinstance(v, str) else str(v)

    @field_validator("created_at")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """
        Normalise `createdAt` to timezone-aware UTC.

        MongoDB returns BSON datetimes without tzinfo while the in-memory store
        keeps them aware, which would otherwise serialise to two different ISO
        shapes ('...+00:00' vs '...') depending on the active backing store.
        """
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class ListingResponse(BaseModel):
    """POST /api/listings and GET /api/listings/{id} success envelope."""

    success: bool = True
    listing: Listing


class ListingsResponse(BaseModel):
    """
    GET /api/listings success envelope.

    An empty list is a valid success, never an error — the frontend's
    ListingsScreen renders an empty state from it.
    """

    success: bool = True
    listings: list[Listing]


class DeleteListingResponse(BaseModel):
    """DELETE /api/listings/{id} success envelope."""

    success: bool = True
    message: str = Field(examples=["Listing deleted"])


class ListingFilters(BaseModel):
    """Optional query parameters for GET /api/listings."""

    crop: Optional[Crop] = None
    location: Optional[str] = None


def utc_now() -> datetime:
    """Timezone-aware creation timestamp, serialised as an ISO string."""
    return datetime.now(timezone.utc)
