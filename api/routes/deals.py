"""Deal-related API endpoints."""

import sys
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

# Add parent directory to path to import app services
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.data_loader import DataLoader
from app.services.risk_calculator import RiskCalculator
from api.schemas.deal import (
    DealSummary,
    DealDetail,
    DealListResponse,
    SummaryStats,
    FilterOptions,
    RiskDriver,
    DealExplanation,
    FeatureContribution,
)
from api.services.explainer_service import get_explainer_service

router = APIRouter(prefix="/deals", tags=["deals"])

# Initialize services
data_loader = DataLoader()
risk_calculator = RiskCalculator()


def _row_to_summary(row) -> DealSummary:
    """Convert DataFrame row to DealSummary."""
    return DealSummary(
        opportunity_id=str(row["opportunity_id"]),
        account=str(row["account"]),
        sales_agent=str(row["sales_agent"]),
        product=str(row["product"]),
        deal_stage=str(row["deal_stage"]),
        win_probability=float(row["win_probability"]),
        risk_score=int(row["risk_score"]),
        risk_category=str(row["risk_category"]),
        product_sales_price=float(row["product_sales_price"]),
    )


def _row_to_detail(row) -> DealDetail:
    """Convert DataFrame row to DealDetail."""
    # Parse risk_drivers
    drivers = row.get("risk_drivers", [])
    if isinstance(drivers, str):
        import json
        try:
            drivers = json.loads(drivers.replace("'", '"'))
        except:
            drivers = []

    risk_drivers = [
        RiskDriver(
            driver=d.get("driver", "Unknown"),
            detail=d.get("detail", ""),
            impact=d.get("impact", ""),
            icon=d.get("icon", "warning"),
        )
        for d in (drivers or [])
    ]

    # Parse close range
    close_range = row.get("predicted_close_range", (0, 0))
    if isinstance(close_range, str):
        import json
        try:
            close_range = tuple(json.loads(close_range))
        except:
            close_range = (0, 0)

    return DealDetail(
        opportunity_id=str(row["opportunity_id"]),
        account=str(row["account"]),
        sales_agent=str(row["sales_agent"]),
        product=str(row["product"]),
        deal_stage=str(row["deal_stage"]),
        win_probability=float(row["win_probability"]),
        risk_score=int(row["risk_score"]),
        risk_category=str(row["risk_category"]),
        product_sales_price=float(row["product_sales_price"]),
        predicted_close_days=int(row.get("predicted_close_days", 0)),
        predicted_close_range=close_range,
        days_in_stage=int(row.get("days_in_stage", 0)),
        risk_drivers=risk_drivers,
    )


@router.get("", response_model=DealListResponse)
async def list_deals(
    account: Optional[list[str]] = Query(None, description="Filter by accounts"),
    sales_agent: Optional[list[str]] = Query(None, description="Filter by sales agents"),
    product: Optional[list[str]] = Query(None, description="Filter by products"),
    risk_category: Optional[list[str]] = Query(None, description="Filter by risk categories"),
    min_risk_score: int = Query(0, ge=0, le=100),
    max_risk_score: int = Query(100, ge=0, le=100),
    sort_by: str = Query("risk_score", description="Sort field"),
    sort_order: str = Query("desc", description="asc or desc"),
    limit: int = Query(50, ge=1, le=500),
):
    """
    List deals with optional filters.

    Supports filtering by account, sales agent, product, and risk category.
    Results are paginated and sorted.
    """
    # Get filtered predictions
    df = data_loader.filter_predictions(
        accounts=account,
        sales_agents=sales_agent,
        products=product,
        risk_categories=risk_category,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
    )

    total = len(df)

    # Sort
    if sort_by in df.columns:
        ascending = sort_order.lower() == "asc"
        df = df.sort_values(sort_by, ascending=ascending)

    # Limit
    df = df.head(limit)

    # Convert to response models
    deals = [_row_to_summary(row) for _, row in df.iterrows()]

    return DealListResponse(deals=deals, total=total, limit=limit)


@router.get("/summary", response_model=SummaryStats)
async def get_summary():
    """
    Get dashboard summary statistics.

    Returns aggregate metrics for all deals.
    """
    stats = data_loader.get_summary_stats()

    return SummaryStats(
        total_deals=stats["total_deals"],
        at_risk_revenue=stats["at_risk_revenue"],
        high_risk_count=stats["high_risk_count"],
        avg_win_probability=stats["avg_win_probability"],
        risk_distribution=stats.get("risk_distribution", {}),
    )


@router.get("/filters", response_model=FilterOptions)
async def get_filters():
    """
    Get available filter options.

    Returns lists of accounts, sales agents, products for filter dropdowns.
    """
    filters = data_loader.get_filters()

    return FilterOptions(
        accounts=filters["accounts"],
        sales_agents=filters["sales_agents"],
        products=filters["products"],
        risk_categories=filters["risk_categories"],
    )


@router.get("/{deal_id}", response_model=DealDetail)
async def get_deal(deal_id: str):
    """
    Get detailed information for a single deal.

    Includes risk drivers and time-to-close predictions.
    """
    deal = data_loader.get_deal(deal_id)

    if deal is None:
        raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")

    return _row_to_detail(deal)


@router.get("/{deal_id}/explanation", response_model=DealExplanation)
async def get_deal_explanation(deal_id: str):
    """
    Get SHAP-based model explanation for a deal prediction.

    Returns feature contributions showing which factors increase or decrease
    the predicted win probability. Calculations are performed on-demand.
    """
    # Get deal data
    deal = data_loader.get_deal(deal_id)
    if deal is None:
        raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")

    # Get explainer service
    explainer = get_explainer_service()

    # Check if available
    if not explainer.is_available():
        error = explainer.get_error()
        raise HTTPException(
            status_code=503,
            detail=f"SHAP explanations unavailable: {error}"
        )

    # Convert deal row to dict for explainer
    deal_data = {
        "opportunity_id": deal.get("opportunity_id"),
        "account": deal.get("account"),
        "sales_agent": deal.get("sales_agent"),
        "product": deal.get("product"),
        "deal_stage": deal.get("deal_stage", "Engaging"),
        "engage_date": deal.get("engage_date"),
    }

    # Generate explanation
    explanation = explainer.explain_deal(deal_data, deal_id)

    if explanation is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate explanation for this deal"
        )

    # Convert to response model
    return DealExplanation(
        opportunity_id=explanation.opportunity_id,
        win_probability=explanation.win_probability,
        base_value=explanation.base_value,
        top_positive=[
            FeatureContribution(
                feature=fc.feature,
                value=fc.value,
                contribution=fc.contribution,
                explanation=fc.explanation
            )
            for fc in explanation.top_positive
        ],
        top_negative=[
            FeatureContribution(
                feature=fc.feature,
                value=fc.value,
                contribution=fc.contribution,
                explanation=fc.explanation
            )
            for fc in explanation.top_negative
        ],
        summary_text=explanation.summary_text
    )
