"""
Marketplace listings routes.

Contract (frontend/services/api.ts):
  POST   /api/listings        -> { success: true, listing }
                                 { success: false, error: "Missing required field: x" }
  GET    /api/listings        -> { success: true, listings: [...] }   (empty list is OK)
  GET    /api/listings/{id}   -> { success: true, listing }
                                 { success: false, error: "Listing not found" }
  DELETE /api/listings/{id}   -> { success: true, message: "Listing deleted" }
                                 { success: false, error: "Listing not found" }

Persistence goes through app/repositories/listing_repository.py, which uses
MongoDB when configured and an in-memory store otherwise.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from app.models.common import ErrorResponse
from app.models.listing import (
    Crop,
    DeleteListingResponse,
    ListingCreate,
    ListingResponse,
    ListingsResponse,
)
from app.repositories import listing_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/listings", tags=["listings"])

# The contract uses this exact string for both GET-by-id and DELETE misses.
NOT_FOUND_ERROR = "Listing not found"


def _not_found() -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"success": False, "error": NOT_FOUND_ERROR},
    )


@router.post(
    "",
    response_model=ListingResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse, "description": "Validation failed"}},
    summary="Create a crop listing",
)
async def create_listing(payload: ListingCreate):
    """
    Create a listing.

    Field validation lives in ListingCreate; failures are converted into the
    contract's error shape by the RequestValidationError handler in main.py.
    """
    listing = await listing_repository.create_listing(payload)
    return ListingResponse(listing=listing)


@router.get(
    "",
    response_model=ListingsResponse,
    summary="List crop listings",
)
async def get_listings(
    crop: Optional[Crop] = Query(default=None, description="Exact crop match"),
    location: Optional[str] = Query(
        default=None, description="Case-insensitive substring match"
    ),
):
    """
    Return listings newest-first, optionally filtered by crop and/or location.

    No matches is a successful empty list — the app renders an empty state
    from it, so this must never 404.
    """
    listings = await listing_repository.list_listings(
        crop=crop.value if crop else None,
        location=location,
    )
    return ListingsResponse(listings=listings)


@router.get(
    "/{listing_id}",
    response_model=ListingResponse,
    responses={404: {"model": ErrorResponse, "description": NOT_FOUND_ERROR}},
    summary="Get one listing by id",
)
async def get_listing(listing_id: str):
    listing = await listing_repository.get_listing(listing_id)
    if listing is None:
        return _not_found()
    return ListingResponse(listing=listing)


@router.delete(
    "/{listing_id}",
    response_model=DeleteListingResponse,
    responses={404: {"model": ErrorResponse, "description": NOT_FOUND_ERROR}},
    summary="Delete a listing by id",
)
async def delete_listing(listing_id: str):
    deleted = await listing_repository.delete_listing(listing_id)
    if not deleted:
        return _not_found()
    return DeleteListingResponse(message="Listing deleted")
