"""Health check endpoint."""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str


@router.get("", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and version info.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="2.0.0"
    )
