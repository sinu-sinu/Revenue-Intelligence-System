"""
FastAPI application for Revenue Intelligence System.

Provides REST API endpoints for the Vue.js frontend.
Run with: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import deals_router, forecast_router, health_router

# Create FastAPI app
app = FastAPI(
    title="Revenue Intelligence API",
    description="""
    REST API for the Revenue Intelligence System.

    Provides endpoints for:
    - **Deals**: List, filter, and view deal details with risk scores
    - **Forecasts**: Monte Carlo revenue forecasting with P10/P50/P90 bands
    - **Health**: Service health checks

    The API wraps the existing ML models and business logic from the core module.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for Vue.js development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Vue dev server
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api")
app.include_router(deals_router, prefix="/api")
app.include_router(forecast_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Revenue Intelligence API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "deals": "/api/deals",
            "forecast": "/api/forecast",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
