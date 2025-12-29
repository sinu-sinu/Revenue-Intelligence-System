"""
Risk Dashboard Page
Phase 1B: Streamlit UI

Primary screen showing deals requiring attention, sorted by risk x value.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_loader import DataLoader
from app.services.risk_calculator import RiskCalculator
from app.components.risk_badge import risk_color


# Page config
st.set_page_config(
    page_title="Risk Dashboard - Revenue Intelligence",
    page_icon="🎯",
    layout="wide"
)


def format_currency(value):
    """Format value as currency."""
    if pd.isna(value) or value == 0:
        return "-"
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    return f"${value:.0f}"


def main():
    """Main dashboard page."""
    st.title("Risk Dashboard")
    st.caption("Deals requiring attention, sorted by risk x value")
    
    # Initialize services
    loader = DataLoader()
    calculator = RiskCalculator()
    
    # Load predictions
    predictions = loader.load_predictions()
    
    if predictions.empty:
        st.warning("No predictions found. Please run: python models/inference/predict.py")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        filters = loader.get_filters()
        
        # Risk category filter
        risk_filter = st.multiselect(
            "Risk Level",
            options=filters["risk_categories"],
            default=["Critical", "High"],
            help="Filter by risk category"
        )
        
        # Account filter
        account_filter = st.multiselect(
            "Account",
            options=filters["accounts"],
            help="Filter by account"
        )
        
        # Sales agent filter
        agent_filter = st.multiselect(
            "Sales Rep",
            options=filters["sales_agents"],
            help="Filter by sales representative"
        )
        
        # Product filter
        product_filter = st.multiselect(
            "Product",
            options=filters["products"],
            help="Filter by product"
        )
        
        # Risk score range
        st.markdown("---")
        risk_range = st.slider(
            "Risk Score Range",
            min_value=0,
            max_value=100,
            value=(0, 100),
            help="Filter by risk score"
        )
    
    # Apply filters
    filtered = loader.filter_predictions(
        accounts=account_filter if account_filter else None,
        sales_agents=agent_filter if agent_filter else None,
        products=product_filter if product_filter else None,
        risk_categories=risk_filter if risk_filter else None,
        min_risk_score=risk_range[0],
        max_risk_score=risk_range[1],
    )
    
    # Summary metrics
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    if "product_sales_price" in filtered.columns:
        at_risk_rev = filtered[filtered["risk_score"] > 50]["product_sales_price"].sum()
    else:
        at_risk_rev = len(filtered[filtered["risk_score"] > 50]) * 1000
    
    high_risk_count = len(filtered[filtered["risk_category"].isin(["High", "Critical"])])
    avg_win_prob = filtered["win_probability"].mean() if not filtered.empty else 0
    
    with col1:
        st.metric(
            "At-Risk Revenue",
            format_currency(at_risk_rev),
            help="Total value of deals with risk score > 50"
        )
    
    with col2:
        st.metric(
            "High Risk Deals",
            high_risk_count,
            help="Deals marked High or Critical"
        )
    
    with col3:
        st.metric(
            "Avg Win Probability",
            f"{avg_win_prob:.0%}" if not pd.isna(avg_win_prob) else "N/A",
            help="Average win probability of filtered deals"
        )
    
    with col4:
        st.metric(
            "Filtered Deals",
            len(filtered),
            help="Number of deals matching current filters"
        )
    
    st.markdown("---")
    
    # Sorting options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            options=[
                ("Risk x Value (Priority)", "priority"),
                ("Risk Score (Highest)", "risk_score"),
                ("Win Probability (Lowest)", "win_probability"),
                ("Days Open (Longest)", "days_in_stage"),
            ],
            format_func=lambda x: x[0],
            key="sort_select"
        )
    
    with col2:
        show_count = st.selectbox(
            "Show:",
            options=[10, 25, 50, 100],
            index=1,
            key="show_count"
        )
    
    # Sort the data
    if sort_by[1] == "priority":
        # Calculate priority score
        if "product_sales_price" in filtered.columns:
            filtered = filtered.copy()
            filtered["priority_score"] = (
                filtered["risk_score"] * 
                np.log1p(filtered["product_sales_price"].fillna(1000))
            )
            filtered = filtered.sort_values("priority_score", ascending=False)
        else:
            filtered = filtered.sort_values("risk_score", ascending=False)
    elif sort_by[1] == "win_probability":
        filtered = filtered.sort_values("win_probability", ascending=True)
    else:
        filtered = filtered.sort_values(sort_by[1], ascending=False)
    
    # Limit to show count
    display_df = filtered.head(show_count)
    
    # Display table
    st.markdown("### Deals Requiring Attention")
    
    if display_df.empty:
        st.info("No deals match the current filters.")
    else:
        # Prepare display columns
        display_cols = ["opportunity_id", "account", "sales_agent", "product", 
                       "win_probability", "risk_score", "risk_category", "days_in_stage"]
        
        # Filter to available columns
        available = [c for c in display_cols if c in display_df.columns]
        table_df = display_df[available].copy()
        
        # Format columns
        if "win_probability" in table_df.columns:
            table_df["win_probability"] = table_df["win_probability"].apply(lambda x: f"{x:.0%}")
        
        # Rename for display
        table_df = table_df.rename(columns={
            "opportunity_id": "Deal ID",
            "account": "Account",
            "sales_agent": "Rep",
            "product": "Product",
            "win_probability": "Win %",
            "risk_score": "Risk",
            "risk_category": "Status",
            "days_in_stage": "Days Open",
        })
        
        # Display with highlighting
        def highlight_risk(val):
            if val in ["Critical"]:
                return "background-color: #7C2D12; color: white; font-weight: bold"
            elif val in ["High"]:
                return "background-color: #DC2626; color: white; font-weight: bold"
            elif val in ["Medium"]:
                return "background-color: #F59E0B; color: black; font-weight: bold"
            elif val in ["Low"]:
                return "background-color: #10B981; color: white; font-weight: bold"
            return ""
        
        styled = table_df.style.map(
            highlight_risk, 
            subset=["Status"] if "Status" in table_df.columns else []
        )
        
        st.dataframe(styled, use_container_width=True, hide_index=True, height=500)
        
        # Deal selection for drill-down
        st.markdown("---")
        st.markdown("### View Deal Details")
        
        deal_options = display_df["opportunity_id"].tolist()
        selected_deal = st.selectbox(
            "Select a deal to view details:",
            options=[""] + deal_options,
            format_func=lambda x: f"{x} - {display_df[display_df['opportunity_id']==x]['account'].values[0]}" if x else "Choose a deal...",
            key="deal_select"
        )
        
        if selected_deal:
            # Show quick summary
            deal = display_df[display_df["opportunity_id"] == selected_deal].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **Account:** {deal.get('account', 'N/A')}  
                **Rep:** {deal.get('sales_agent', 'N/A')}  
                **Product:** {deal.get('product', 'N/A')}
                """)
            
            with col2:
                win_prob = deal.get('win_probability', 0)
                st.markdown(f"""
                **Win Probability:** {win_prob:.0%}  
                **Risk Score:** {deal.get('risk_score', 0)}  
                **Status:** {deal.get('risk_category', 'N/A')}
                """)
            
            with col3:
                st.markdown(f"""
                **Days Open:** {deal.get('days_in_stage', 0)}  
                **Deal Value:** {format_currency(deal.get('product_sales_price', 0))}
                """)
            
            # Link to detail page
            st.page_link(
                "pages/02_Deal_Detail.py",
                label=f"View Full Details for {selected_deal}",
                icon="➡️"
            )
            
            # Store selected deal in session state for detail page
            st.session_state["selected_deal"] = selected_deal


if __name__ == "__main__":
    main()
