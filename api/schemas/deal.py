"""Deal-related Pydantic schemas."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class RiskDriver(BaseModel):
    """Risk factor contributing to deal risk score."""
    driver: str = Field(..., description="Risk driver name")
    detail: str = Field(..., description="Detailed explanation")
    impact: str = Field(..., description="Impact on risk score (e.g., '+30%')")
    icon: str = Field(..., description="Icon identifier")


class DealSummary(BaseModel):
    """Summary view of a deal for list displays."""
    opportunity_id: str = Field(..., description="Unique deal identifier")
    account: str = Field(..., description="Account/company name")
    sales_agent: str = Field(..., description="Assigned sales representative")
    product: str = Field(..., description="Product name")
    deal_stage: str = Field(..., description="Current pipeline stage")
    win_probability: float = Field(..., ge=0, le=1, description="Predicted win probability")
    risk_score: int = Field(..., ge=0, le=100, description="Composite risk score")
    risk_category: Literal["Low", "Medium", "High", "Critical"] = Field(..., description="Risk level")
    product_sales_price: float = Field(..., description="Deal value estimate")


class DealDetail(DealSummary):
    """Full deal details including risk drivers."""
    predicted_close_days: int = Field(..., description="Expected days to close")
    predicted_close_range: tuple[int, int] = Field(..., description="Close date range (min, max)")
    days_in_stage: int = Field(..., description="Days in current stage")
    risk_drivers: list[RiskDriver] = Field(default_factory=list, description="Risk factors")


class DealFilters(BaseModel):
    """Query parameters for filtering deals."""
    accounts: Optional[list[str]] = Field(None, description="Filter by account names")
    sales_agents: Optional[list[str]] = Field(None, description="Filter by sales agents")
    products: Optional[list[str]] = Field(None, description="Filter by products")
    risk_categories: Optional[list[str]] = Field(None, description="Filter by risk categories")
    min_risk_score: int = Field(0, ge=0, le=100, description="Minimum risk score")
    max_risk_score: int = Field(100, ge=0, le=100, description="Maximum risk score")
    sort_by: str = Field("risk_score", description="Sort field")
    sort_order: Literal["asc", "desc"] = Field("desc", description="Sort direction")
    limit: int = Field(50, ge=1, le=500, description="Max results to return")


class DealListResponse(BaseModel):
    """Response for deal list endpoint."""
    deals: list[DealSummary]
    total: int = Field(..., description="Total deals matching filters")
    limit: int = Field(..., description="Limit applied")


class SummaryStats(BaseModel):
    """Dashboard summary statistics."""
    total_deals: int = Field(..., description="Total number of active deals")
    at_risk_revenue: float = Field(..., description="Total revenue at risk")
    high_risk_count: int = Field(..., description="Count of high/critical risk deals")
    avg_win_probability: float = Field(..., description="Average win probability")
    risk_distribution: dict[str, int] = Field(..., description="Count by risk category")


class FilterOptions(BaseModel):
    """Available filter options derived from data."""
    accounts: list[str] = Field(..., description="Available account names")
    sales_agents: list[str] = Field(..., description="Available sales agents")
    products: list[str] = Field(..., description="Available products")
    risk_categories: list[str] = Field(
        default=["Critical", "High", "Medium", "Low"],
        description="Risk category options"
    )
