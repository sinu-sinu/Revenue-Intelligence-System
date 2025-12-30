"""API route modules."""

from .deals import router as deals_router
from .forecast import router as forecast_router
from .health import router as health_router

__all__ = ["deals_router", "forecast_router", "health_router"]
