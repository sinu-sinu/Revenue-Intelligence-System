"""Forecast-related Pydantic schemas."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request parameters for forecast generation."""
    horizon_weeks: int = Field(12, ge=1, le=52, description="Forecast horizon in weeks")
    period: str = Field("week", description="Aggregation period (week/month)")
    accounts: Optional[list[str]] = Field(None, description="Filter by accounts")
    sales_agents: Optional[list[str]] = Field(None, description="Filter by sales agents")
    products: Optional[list[str]] = Field(None, description="Filter by products")


class ForecastPeriod(BaseModel):
    """Forecast data for a single time period."""
    period: str = Field(..., description="Period label (e.g., '2024-W01')")
    p10: float = Field(..., description="10th percentile (pessimistic)")
    p50: float = Field(..., description="50th percentile (expected)")
    p90: float = Field(..., description="90th percentile (optimistic)")


class DealForecast(BaseModel):
    """Deal-level forecast contribution."""
    opportunity_id: str
    account: str
    product: str
    win_probability: float
    product_sales_price: float
    expected_value: float = Field(..., description="win_prob * value")
    expected_close_days: int


class ForecastSummary(BaseModel):
    """Aggregate forecast statistics."""
    forecast_date: datetime = Field(..., description="When forecast was generated")
    horizon_days: int = Field(..., description="Forecast horizon in days")
    total_deals: int = Field(..., description="Deals included in forecast")
    total_pipeline: float = Field(..., description="Sum of all deal values")
    expected_revenue: float = Field(..., description="Probability-weighted revenue")
    p10_forecast: float = Field(..., description="Pessimistic total forecast")
    p50_forecast: float = Field(..., description="Expected total forecast")
    p90_forecast: float = Field(..., description="Optimistic total forecast")
    simulation_runs: int = Field(1000, description="Monte Carlo iterations")


class ForecastResponse(BaseModel):
    """Complete forecast response."""
    periods: list[ForecastPeriod] = Field(..., description="Time series forecast data")
    deal_forecasts: list[DealForecast] = Field(..., description="Per-deal contributions")
    summary: ForecastSummary = Field(..., description="Aggregate statistics")
