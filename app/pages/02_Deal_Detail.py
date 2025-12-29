"""
Deal Detail Page
Phase 1B: Streamlit UI

Drill-down view for individual deals with risk explanations.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.services.data_loader import DataLoader
from app.components.risk_badge import risk_color, risk_badge_html


# Page config
st.set_page_config(
    page_title="Deal Detail - Revenue Intelligence",
    page_icon="🔍",
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


def get_icon(icon_name):
    """Get ASCII icon representation."""
    icons = {
        "chart_decreasing": "[v]",
        "hourglass": "[T]",
        "dollar": "[$]",
        "person": "[P]",
        "check": "[OK]",
    }
    return icons.get(icon_name, "[*]")


def main():
    """Main deal detail page."""
    st.title("Deal Detail")
    st.caption("In-depth analysis of individual deals")
    
    # Initialize services
    loader = DataLoader()
    predictions = loader.load_predictions()
    
    if predictions.empty:
        st.warning("No predictions found. Please run: python models/inference/predict.py")
        return
    
    # Get deal selection
    # Check session state first
    default_deal = st.session_state.get("selected_deal", "")
    
    # Deal selector in sidebar
    with st.sidebar:
        st.header("Select Deal")
        
        deal_options = predictions["opportunity_id"].tolist()
        
        # Find default index
        default_idx = 0
        if default_deal and default_deal in deal_options:
            default_idx = deal_options.index(default_deal) + 1  # +1 for empty option
        
        selected_deal = st.selectbox(
            "Deal ID:",
            options=[""] + deal_options,
            index=default_idx,
            key="detail_deal_select"
        )
        
        if selected_deal:
            st.session_state["selected_deal"] = selected_deal
    
    if not selected_deal:
        st.info("Select a deal from the sidebar to view details.")
        
        # Show recent high-risk deals as suggestions
        st.markdown("### Suggested Deals to Review")
        
        high_risk = predictions[
            predictions["risk_category"].isin(["Critical", "High"])
        ].head(5)
        
        for _, deal in high_risk.iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{deal['opportunity_id']}** - {deal.get('account', 'Unknown')}")
            with col2:
                st.markdown(f"Risk: {deal['risk_score']}")
            with col3:
                if st.button("View", key=f"view_{deal['opportunity_id']}"):
                    st.session_state["selected_deal"] = deal["opportunity_id"]
                    st.rerun()
        
        return
    
    # Get deal data
    deal = loader.get_deal(selected_deal)
    
    if deal is None:
        st.error(f"Deal {selected_deal} not found.")
        return
    
    # Deal header
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"## {deal.get('opportunity_id', 'Unknown')}")
        st.caption(f"{deal.get('account', 'Unknown')} | {deal.get('sales_agent', 'Unknown')} | {deal.get('product', 'Unknown')}")
    
    with col2:
        # Risk badge
        risk_cat = deal.get("risk_category", "Medium")
        risk_score = deal.get("risk_score", 50)
        st.markdown(
            risk_badge_html(risk_cat, risk_score),
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Three-column summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Win Probability")
        win_prob = deal.get("win_probability", 0.5)
        
        # Visual probability bar
        st.progress(win_prob)
        st.markdown(f"<h2 style='text-align: center; color: {'#10B981' if win_prob > 0.6 else '#F59E0B' if win_prob > 0.4 else '#EF4444'};'>{win_prob:.0%}</h2>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Risk Level")
        color = risk_color(risk_cat)
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: {color}20; border-radius: 0.5rem; border: 2px solid {color};">
            <h2 style="color: {color}; margin: 0;">{risk_cat}</h2>
            <p style="color: #94A3B8; margin: 0;">Score: {risk_score}/100</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Est. Close")
        pred_days = deal.get("predicted_close_days", 30)
        close_range = deal.get("predicted_close_range", (15, 45))
        
        if isinstance(close_range, str):
            import json
            close_range = tuple(json.loads(close_range))
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #4F46E5; margin: 0;">{pred_days} days</h2>
            <p style="color: #94A3B8; margin: 0;">Range: {close_range[0]}-{close_range[1]} days</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk drivers section
    st.markdown("### Why This Risk Level")
    
    risk_drivers = deal.get("risk_drivers", [])
    
    if isinstance(risk_drivers, str):
        import json
        try:
            risk_drivers = json.loads(risk_drivers.replace("'", '"'))
        except:
            risk_drivers = []
    
    if risk_drivers:
        for driver in risk_drivers:
            if isinstance(driver, dict):
                icon = get_icon(driver.get("icon", ""))
                driver_name = driver.get("driver", "Unknown")
                detail = driver.get("detail", "")
                impact = driver.get("impact", "")
                
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.markdown(f"**{icon} {driver_name}**")
                with col2:
                    st.markdown(detail)
                with col3:
                    if "+" in str(impact):
                        st.markdown(f"`{impact}`")
                    else:
                        st.markdown(f"`{impact}`")
    else:
        st.info("No specific risk factors identified.")
    
    st.markdown("---")
    
    # Suggested action
    st.markdown("### Suggested Action")
    
    # Generate action based on risk level and drivers
    if risk_cat == "Critical":
        action = "Schedule executive sponsor call immediately"
        confidence = "High"
        reason = "Critical risk level with multiple concerning factors"
    elif risk_cat == "High":
        action = "Review deal with manager and develop recovery plan"
        confidence = "Medium"
        reason = "High risk - needs proactive intervention"
    elif risk_cat == "Medium":
        action = "Monitor closely and check in with champion"
        confidence = "Medium"
        reason = "Moderate risk - stay engaged"
    else:
        action = "Continue normal engagement cadence"
        confidence = "Low"
        reason = "Deal is on track"
    
    st.markdown(f"""
    <div style="background: #1E293B; padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #4F46E5;">
        <h4 style="color: #F8FAFC; margin: 0 0 0.5rem 0;">{action}</h4>
        <p style="color: #94A3B8; margin: 0;">Confidence: {confidence} | Based on: {reason}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Evidence section (collapsible)
    with st.expander("View Evidence & Model Details"):
        st.markdown("#### Data Points Used")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            - **Days in Stage:** {deal.get('days_in_stage', 'N/A')}
            - **Product:** {deal.get('product', 'N/A')}
            - **Deal Value:** {format_currency(deal.get('product_sales_price', 0))}
            """)
        
        with col2:
            st.markdown(f"""
            - **Sales Rep:** {deal.get('sales_agent', 'N/A')}
            - **Account:** {deal.get('account', 'N/A')}
            - **Stage:** {deal.get('deal_stage', 'N/A')}
            """)
        
        st.markdown("#### Model Information")
        st.markdown("""
        - **Model:** LightGBM with Isotonic Calibration
        - **Training AUC:** 0.61
        - **Test AUC:** 0.58
        - **Calibration (ECE):** 0.031
        
        This model is well-calibrated, meaning the predicted probabilities are reliable.
        A 40% win probability truly means ~40% chance of winning.
        """)
        
        st.markdown("#### What We Don't Know")
        st.markdown("""
        - Recent email/call engagement
        - Champion sentiment changes
        - Competitor activity
        - Budget timeline changes
        
        These factors could significantly impact the actual outcome.
        """)
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.page_link("pages/01_Risk_Dashboard.py", label="Back to Dashboard", icon="⬅️")
    
    with col3:
        st.page_link("pages/03_Forecast.py", label="View Forecast", icon="📊")


if __name__ == "__main__":
    main()


