"""
Forecast chart component with uncertainty bands.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def forecast_chart(
    forecast_data: pd.DataFrame,
    target: Optional[float] = None,
    title: str = "Revenue Forecast",
    height: int = 400
) -> None:
    """
    Display a forecast chart with confidence bands.
    
    Args:
        forecast_data: DataFrame with columns:
            - date: Date/period
            - p10: 10th percentile (conservative)
            - p50: 50th percentile (expected)
            - p90: 90th percentile (optimistic)
        target: Optional target line value
        title: Chart title
        height: Chart height in pixels
    """
    if not HAS_PLOTLY:
        st.error("Plotly is required for charts. Install with: pip install plotly")
        return
    
    fig = go.Figure()
    
    # P90 upper bound (invisible line for fill)
    fig.add_trace(go.Scatter(
        x=forecast_data["date"],
        y=forecast_data["p90"],
        mode="lines",
        line=dict(width=0),
        name="P90 (Optimistic)",
        showlegend=False,
    ))
    
    # P10 lower bound with fill to P90
    fig.add_trace(go.Scatter(
        x=forecast_data["date"],
        y=forecast_data["p10"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(79, 70, 229, 0.2)",
        line=dict(width=0),
        name="Confidence Band (P10-P90)",
    ))
    
    # P50 expected line
    fig.add_trace(go.Scatter(
        x=forecast_data["date"],
        y=forecast_data["p50"],
        mode="lines+markers",
        line=dict(color="#4F46E5", width=3),
        marker=dict(size=8),
        name="Expected (P50)",
    ))
    
    # Target line
    if target is not None:
        fig.add_hline(
            y=target,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Target: ${target:,.0f}",
            annotation_position="right",
        )
    
    # Styling
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Revenue ($)",
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            tickformat="$,.0f",
            gridcolor="rgba(255,255,255,0.1)",
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)",
        ),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def generate_forecast_data(
    predictions: pd.DataFrame,
    weeks: int = 8,
    base_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Generate forecast data from predictions.
    
    Args:
        predictions: Predictions DataFrame with win_probability and predicted_close_days
        weeks: Number of weeks to forecast
        base_date: Start date (defaults to today)
        
    Returns:
        DataFrame with forecast periods
    """
    if base_date is None:
        base_date = datetime.now()
    
    forecasts = []
    
    for week in range(weeks):
        week_start = base_date + timedelta(days=7 * week)
        week_end = week_start + timedelta(days=6)
        
        # Find deals expected to close in this period
        closing_mask = (
            (predictions["predicted_close_days"] >= 7 * week) &
            (predictions["predicted_close_days"] < 7 * (week + 1))
        )
        
        closing_deals = predictions[closing_mask]
        
        if len(closing_deals) > 0:
            # Expected revenue (weighted by win probability)
            if "product_sales_price" in closing_deals.columns:
                values = closing_deals["product_sales_price"].fillna(1000)
            else:
                values = 1000  # Default deal value
            
            probs = closing_deals["win_probability"]
            
            # Monte Carlo-ish estimation
            p50 = float((values * probs).sum())
            p10 = float(p50 * 0.6)  # Conservative
            p90 = float(p50 * 1.4)  # Optimistic
        else:
            p50 = 0
            p10 = 0
            p90 = 0
        
        forecasts.append({
            "date": week_start.strftime("%b %d"),
            "week": week + 1,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "n_deals": len(closing_deals),
        })
    
    return pd.DataFrame(forecasts)


def forecast_summary_table(forecast_data: pd.DataFrame) -> None:
    """
    Display a summary table of the forecast.
    
    Args:
        forecast_data: Forecast DataFrame
    """
    display_df = forecast_data[["date", "p10", "p50", "p90", "n_deals"]].copy()
    
    # Format currency columns
    for col in ["p10", "p50", "p90"]:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
    display_df = display_df.rename(columns={
        "date": "Week Of",
        "p10": "Conservative",
        "p50": "Expected",
        "p90": "Optimistic",
        "n_deals": "# Deals",
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


