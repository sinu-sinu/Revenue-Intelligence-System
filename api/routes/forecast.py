"""Forecast API endpoints."""

import sys
from pathlib import Path
from fastapi import APIRouter

# Add parent directory to path to import core modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.data_loader import DataLoader
from core.forecasting.revenue_forecast import RevenueForecast
from api.schemas.forecast import (
    ForecastRequest,
    ForecastPeriod,
    ForecastResponse,
    ForecastSummary,
    DealForecast,
)

router = APIRouter(prefix="/forecast", tags=["forecast"])

# Initialize services
data_loader = DataLoader()


@router.post("", response_model=ForecastResponse)
async def generate_forecast(request: ForecastRequest):
    """
    Generate Monte Carlo revenue forecast.

    Returns P10/P50/P90 probability bands over the forecast horizon.
    Optionally filter deals by account, sales agent, or product.
    """
    # Get predictions (optionally filtered)
    df = data_loader.filter_predictions(
        accounts=request.accounts,
        sales_agents=request.sales_agents,
        products=request.products,
    )

    if df.empty:
        return ForecastResponse(
            periods=[],
            deal_forecasts=[],
            summary=ForecastSummary(
                forecast_date=__import__("datetime").datetime.now(),
                horizon_days=request.horizon_weeks * 7,
                total_deals=0,
                total_pipeline=0.0,
                expected_revenue=0.0,
                p10_forecast=0.0,
                p50_forecast=0.0,
                p90_forecast=0.0,
                simulation_runs=1000,
            ),
        )

    # Prepare data for forecast engine
    # Add default time-to-close distribution if not present
    if "days_p10" not in df.columns:
        df = df.copy()
        df["days_p10"] = 8
        df["days_p50"] = df["predicted_close_days"].fillna(45)
        df["days_p90"] = df["predicted_close_days"].fillna(45) + 40

    # Run forecast
    forecaster = RevenueForecast(n_simulations=1000)
    result = forecaster.generate_forecast(
        deals=df,
        horizon_weeks=request.horizon_weeks,
        period=request.period,
    )

    # Convert to response format
    periods = [
        ForecastPeriod(
            period=period,
            p10=result.p10[i],
            p50=result.p50[i],
            p90=result.p90[i],
        )
        for i, period in enumerate(result.periods)
    ]

    # Deal-level forecasts
    deal_forecasts = []
    for _, row in result.deal_forecasts.iterrows():
        # Get account and product from original df
        orig_row = df[df["opportunity_id"] == row["opportunity_id"]]
        if not orig_row.empty:
            deal_forecasts.append(
                DealForecast(
                    opportunity_id=str(row["opportunity_id"]),
                    account=str(orig_row.iloc[0]["account"]),
                    product=str(orig_row.iloc[0]["product"]),
                    win_probability=float(row["win_probability"]),
                    product_sales_price=float(row["product_sales_price"]),
                    expected_value=float(row["expected_value"]),
                    expected_close_days=int(row["expected_close_days"]),
                )
            )

    # Get summary
    summary_data = forecaster.get_forecast_summary(result)
    summary = ForecastSummary(
        forecast_date=result.forecast_date,
        horizon_days=result.horizon_days,
        total_deals=summary_data["total_deals"],
        total_pipeline=summary_data["total_pipeline"],
        expected_revenue=summary_data["expected_revenue"],
        p10_forecast=summary_data["p10_forecast"],
        p50_forecast=summary_data["p50_forecast"],
        p90_forecast=summary_data["p90_forecast"],
        simulation_runs=result.simulation_runs,
    )

    return ForecastResponse(
        periods=periods,
        deal_forecasts=deal_forecasts,
        summary=summary,
    )
