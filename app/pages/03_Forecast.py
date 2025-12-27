"""
Forecast Page
Phase 1B: Streamlit UI

Revenue forecast with uncertainty bands.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_loader import DataLoader

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Page config
st.set_page_config(
    page_title="Revenue Forecast - Revenue Intelligence",
    page_icon="::chart_with_upwards_trend::",
    layout="wide"
)


def format_currency(value):
    """Format value as currency."""
    if pd.isna(value) or value == 0:
        return "$0"
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    return f"${value:.0f}"


def generate_forecast(predictions, weeks=8):
    """
    Generate forecast data from predictions.
    
    Args:
        predictions: Predictions DataFrame
        weeks: Number of weeks to forecast
        
    Returns:
        Forecast DataFrame
    """
    base_date = datetime.now()
    forecasts = []
    
    for week in range(weeks):
        week_start = base_date + timedelta(days=7 * week)
        
        # Find deals expected to close in this period
        min_days = 7 * week
        max_days = 7 * (week + 1)
        
        closing_mask = (
            (predictions["predicted_close_days"] >= min_days) &
            (predictions["predicted_close_days"] < max_days)
        )
        
        closing_deals = predictions[closing_mask]
        
        if len(closing_deals) > 0:
            # Get deal values
            if "product_sales_price" in closing_deals.columns:
                values = closing_deals["product_sales_price"].fillna(1000)
            else:
                values = pd.Series([1000] * len(closing_deals))
            
            probs = closing_deals["win_probability"]
            
            # Expected revenue (weighted by probability)
            p50 = float((values * probs).sum())
            
            # Uncertainty based on probability variance
            uncertainty = float((values * probs * (1 - probs)).sum() ** 0.5)
            
            p10 = max(0, p50 - 1.28 * uncertainty)  # 10th percentile
            p90 = p50 + 1.28 * uncertainty  # 90th percentile
            
            n_deals = len(closing_deals)
        else:
            p50 = 0
            p10 = 0
            p90 = 0
            n_deals = 0
        
        forecasts.append({
            "week": week + 1,
            "date": week_start.strftime("%b %d"),
            "date_full": week_start,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "n_deals": n_deals,
        })
    
    return pd.DataFrame(forecasts)


def main():
    """Main forecast page."""
    st.title("Revenue Forecast")
    st.caption("Pipeline forecast with uncertainty bands")
    
    # Initialize services
    loader = DataLoader()
    predictions = loader.load_predictions()
    
    if predictions.empty:
        st.warning("No predictions found. Please run: python models/inference/predict.py")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.header("Forecast Settings")
        
        # Time range
        weeks = st.selectbox(
            "Forecast Horizon:",
            options=[4, 8, 12],
            index=1,
            format_func=lambda x: f"{x} weeks",
            help="Number of weeks to forecast"
        )
        
        # Target
        target = st.number_input(
            "Revenue Target:",
            min_value=0,
            max_value=10_000_000,
            value=500_000,
            step=50_000,
            help="Target revenue for the period"
        )
        
        st.markdown("---")
        
        # Filters (optional)
        st.markdown("### Optional Filters")
        
        filters = loader.get_filters()
        
        product_filter = st.multiselect(
            "Product",
            options=filters["products"],
            help="Filter forecast by product"
        )
        
        agent_filter = st.multiselect(
            "Sales Rep",
            options=filters["sales_agents"],
            help="Filter forecast by rep"
        )
    
    # Apply filters if any
    filtered = predictions.copy()
    
    if product_filter:
        filtered = filtered[filtered["product"].isin(product_filter)]
    
    if agent_filter:
        filtered = filtered[filtered["sales_agent"].isin(agent_filter)]
    
    # Generate forecast
    forecast = generate_forecast(filtered, weeks=weeks)
    
    st.markdown("---")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_p50 = forecast["p50"].sum()
    total_p10 = forecast["p10"].sum()
    total_p90 = forecast["p90"].sum()
    total_deals = forecast["n_deals"].sum()
    
    with col1:
        st.metric(
            "Expected Revenue",
            format_currency(total_p50),
            help="Expected (P50) total revenue"
        )
    
    with col2:
        st.metric(
            "Conservative Estimate",
            format_currency(total_p10),
            help="Conservative (P10) total revenue"
        )
    
    with col3:
        st.metric(
            "Optimistic Estimate",
            format_currency(total_p90),
            help="Optimistic (P90) total revenue"
        )
    
    with col4:
        pct_of_target = (total_p50 / target * 100) if target > 0 else 0
        delta_color = "normal" if pct_of_target >= 100 else "inverse"
        st.metric(
            "vs Target",
            f"{pct_of_target:.0f}%",
            delta=f"{format_currency(total_p50 - target)}",
            delta_color=delta_color
        )
    
    st.markdown("---")
    
    # Forecast chart
    st.markdown("### Revenue Projection")
    
    if HAS_PLOTLY:
        fig = go.Figure()
        
        # Confidence band (P10 to P90)
        fig.add_trace(go.Scatter(
            x=forecast["date"],
            y=forecast["p90"],
            mode="lines",
            line=dict(width=0),
            name="P90 (Optimistic)",
            showlegend=False,
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast["date"],
            y=forecast["p10"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(79, 70, 229, 0.2)",
            line=dict(width=0),
            name="Confidence Band",
        ))
        
        # Expected (P50) line
        fig.add_trace(go.Scatter(
            x=forecast["date"],
            y=forecast["p50"],
            mode="lines+markers",
            line=dict(color="#4F46E5", width=3),
            marker=dict(size=8),
            name="Expected (P50)",
        ))
        
        # Target line
        fig.add_hline(
            y=target / weeks,  # Weekly target
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Weekly Target: {format_currency(target/weeks)}",
            annotation_position="right",
        )
        
        # Styling
        fig.update_layout(
            height=400,
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
                title="Weekly Revenue ($)",
                tickformat="$,.0f",
                gridcolor="rgba(255,255,255,0.1)",
            ),
            xaxis=dict(
                title="Week Of",
                gridcolor="rgba(255,255,255,0.1)",
            ),
            margin=dict(l=60, r=20, t=40, b=60),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly not installed. Install with: pip install plotly")
        st.dataframe(forecast)
    
    st.markdown("---")
    
    # Forecast breakdown table
    st.markdown("### Weekly Breakdown")
    
    display_df = forecast[["date", "p10", "p50", "p90", "n_deals"]].copy()
    
    # Format currency
    for col in ["p10", "p50", "p90"]:
        display_df[col] = display_df[col].apply(format_currency)
    
    display_df = display_df.rename(columns={
        "date": "Week Of",
        "p10": "Conservative",
        "p50": "Expected",
        "p90": "Optimistic",
        "n_deals": "# Deals",
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Deals contributing to forecast
    with st.expander("View Contributing Deals"):
        st.markdown("### Deals Expected to Close")
        
        # Show deals sorted by expected close time
        sorted_deals = filtered.sort_values("predicted_close_days")
        
        display_cols = ["opportunity_id", "account", "product", 
                       "win_probability", "predicted_close_days", "product_sales_price"]
        available = [c for c in display_cols if c in sorted_deals.columns]
        
        table_df = sorted_deals[available].head(20).copy()
        
        if "win_probability" in table_df.columns:
            table_df["win_probability"] = table_df["win_probability"].apply(lambda x: f"{x:.0%}")
        
        if "product_sales_price" in table_df.columns:
            table_df["product_sales_price"] = table_df["product_sales_price"].apply(format_currency)
        
        table_df = table_df.rename(columns={
            "opportunity_id": "Deal ID",
            "account": "Account",
            "product": "Product",
            "win_probability": "Win %",
            "predicted_close_days": "Days to Close",
            "product_sales_price": "Value",
        })
        
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    
    # Methodology note
    st.markdown("---")
    st.caption("""
    **Methodology:** Forecast uses predicted win probabilities and close dates from the ML model.
    The confidence band (P10-P90) represents uncertainty in both win probability and timing.
    This is not a guarantee - actual results depend on sales execution and market conditions.
    """)


if __name__ == "__main__":
    main()
