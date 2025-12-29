"""
Interactive deal table component.
"""

import streamlit as st
import pandas as pd
from typing import List, Optional, Callable

from app.components.risk_badge import risk_color


def deal_table(
    deals: pd.DataFrame,
    columns: Optional[List[str]] = None,
    on_select: Optional[Callable] = None,
    key: str = "deal_table"
) -> Optional[str]:
    """
    Display an interactive deal table.
    
    Args:
        deals: DataFrame of deals with predictions
        columns: Columns to display (default: standard set)
        on_select: Callback when a deal is selected
        key: Unique key for the component
        
    Returns:
        Selected opportunity_id if any
    """
    if deals.empty:
        st.info("No deals to display.")
        return None
    
    # Default columns
    if columns is None:
        columns = [
            "opportunity_id",
            "account",
            "sales_agent",
            "product",
            "win_probability",
            "risk_score",
            "risk_category",
            "days_in_stage",
        ]
    
    # Filter to available columns
    available_cols = [c for c in columns if c in deals.columns]
    display_df = deals[available_cols].copy()
    
    # Format columns
    if "win_probability" in display_df.columns:
        display_df["win_probability"] = display_df["win_probability"].apply(
            lambda x: f"{x:.0%}"
        )
    
    if "product_sales_price" in display_df.columns:
        display_df["product_sales_price"] = display_df["product_sales_price"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "-"
        )
    
    # Rename columns for display
    column_names = {
        "opportunity_id": "Deal ID",
        "account": "Account",
        "sales_agent": "Rep",
        "product": "Product",
        "win_probability": "Win %",
        "risk_score": "Risk",
        "risk_category": "Status",
        "days_in_stage": "Days Open",
        "product_sales_price": "Value",
    }
    
    display_df = display_df.rename(columns=column_names)
    
    # Style the dataframe
    def style_risk(val):
        """Color-code the risk column."""
        if val in ["Low", "Medium", "High", "Critical"]:
            color = risk_color(val)
            return f"color: {color}; font-weight: bold"
        return ""
    
    styled_df = display_df.style.map(
        style_risk,
        subset=["Status"] if "Status" in display_df.columns else []
    )
    
    # Display with selection
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=400,
    )
    
    # Selection via dropdown (since Streamlit doesn't have table click events)
    if on_select and "Deal ID" in display_df.columns:
        deal_ids = deals["opportunity_id"].tolist()
        selected = st.selectbox(
            "Select a deal to view details:",
            options=[""] + deal_ids,
            key=f"{key}_select"
        )
        
        if selected:
            return selected
    
    return None


def deal_table_with_sorting(
    deals: pd.DataFrame,
    key: str = "sorted_table"
) -> pd.DataFrame:
    """
    Display a deal table with sorting controls.
    
    Args:
        deals: DataFrame of deals
        key: Unique key prefix
        
    Returns:
        Sorted DataFrame
    """
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sort_col = st.selectbox(
            "Sort by:",
            options=["risk_score", "win_probability", "days_in_stage", "product_sales_price"],
            format_func=lambda x: {
                "risk_score": "Risk Score (highest first)",
                "win_probability": "Win Probability (lowest first)",
                "days_in_stage": "Days Open (longest first)",
                "product_sales_price": "Deal Value (highest first)",
            }.get(x, x),
            key=f"{key}_sort"
        )
    
    with col2:
        sort_order = st.radio(
            "Order:",
            options=["Descending", "Ascending"],
            horizontal=True,
            key=f"{key}_order"
        )
    
    ascending = sort_order == "Ascending"
    
    # For risk prioritization, we often want to reverse the natural order
    if sort_col == "win_probability":
        ascending = not ascending
    
    sorted_deals = deals.sort_values(sort_col, ascending=ascending)
    
    return sorted_deals


