"""
Revenue Intelligence System - Streamlit Dashboard
Phase 1B: UI Implementation

This is the main entry point for the Streamlit application.
Run with: streamlit run app/main.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Revenue Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Revenue Intelligence System - AI-powered deal risk assessment"
    }
)


def load_custom_css():
    """Load custom CSS styling."""
    st.markdown("""
    <style>
    /* Dark theme enhancements */
    .stMetric {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
    }
    
    .stMetric label {
        color: #94A3B8 !important;
        font-size: 0.875rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: 1.75rem;
        font-weight: 700;
    }
    
    /* Risk colors */
    .risk-low { color: #10B981; font-weight: 600; }
    .risk-medium { color: #F59E0B; font-weight: 600; }
    .risk-high { color: #EF4444; font-weight: 600; }
    .risk-critical { color: #DC2626; font-weight: 700; }
    
    /* Card styling */
    .card {
        background: #1E293B;
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.875rem;
    }
    
    /* Header styling */
    h1 {
        color: #F8FAFC;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #E2E8F0;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #CBD5E1;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label {
        color: #94A3B8;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #6366F1 0%, #818CF8 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4F46E5, #818CF8);
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Load custom styling
    load_custom_css()
    
    # Sidebar - Navigation & Info
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%); 
                    padding: 0.75rem 1rem; border-radius: 0.5rem; text-align: center;
                    margin-bottom: 0.5rem;">
            <span style="color: white; font-weight: 700; font-size: 1.25rem;">📈 RevIntel</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### Navigation")
        st.markdown("""
        - **Risk Dashboard** - At-risk deals
        - **Deal Detail** - Drill into specific deals
        - **Forecast** - Revenue projections
        """)
        
        st.markdown("---")
        
        st.markdown("### About")
        st.caption("""
        Revenue Intelligence System  
        Phase 1B - Streamlit UI
        
        Model: LightGBM + Isotonic Calibration  
        AUC: 0.58 | ECE: 0.031
        """)
        
        # Refresh button
        st.markdown("---")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main content
    st.title("Revenue Intelligence")
    st.caption("AI-powered deal risk assessment and forecasting")
    
    st.markdown("---")
    
    # Quick stats
    try:
        from app.services.data_loader import DataLoader
        loader = DataLoader()
        stats = loader.get_summary_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            at_risk = stats.get("at_risk_revenue", 0)
            if at_risk >= 1_000_000:
                at_risk_str = f"${at_risk/1_000_000:.1f}M"
            elif at_risk >= 1_000:
                at_risk_str = f"${at_risk/1_000:.0f}K"
            else:
                at_risk_str = f"${at_risk:.0f}"
            st.metric("At-Risk Revenue", at_risk_str)
        
        with col2:
            high_risk = stats.get("high_risk_count", 0)
            st.metric("High Risk Deals", high_risk)
        
        with col3:
            avg_prob = stats.get("avg_win_probability", 0)
            st.metric("Avg Win Probability", f"{avg_prob:.0%}")
        
        with col4:
            total = stats.get("total_deals", 0)
            st.metric("Total Active Deals", total)
        
        st.markdown("---")
        
        # Risk distribution
        st.markdown("### Risk Distribution")
        
        risk_dist = stats.get("risk_distribution", {})
        if risk_dist:
            cols = st.columns(4)
            categories = ["Critical", "High", "Medium", "Low"]
            colors = {"Critical": "#DC2626", "High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
            
            for i, cat in enumerate(categories):
                count = risk_dist.get(cat, 0)
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #1E293B; border-radius: 0.5rem; border-left: 4px solid {colors[cat]};">
                        <div style="font-size: 2rem; font-weight: 700; color: {colors[cat]};">{count}</div>
                        <div style="color: #94A3B8;">{cat}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigate to pages
        st.info("Use the sidebar navigation or click below to view detailed pages.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.page_link("pages/01_Risk_Dashboard.py", label="Risk Dashboard", icon="🎯")
        
        with col2:
            st.page_link("pages/02_Deal_Detail.py", label="Deal Detail", icon="🔍")
        
        with col3:
            st.page_link("pages/03_Forecast.py", label="Revenue Forecast", icon="📊")
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Make sure predictions have been generated. Run: python models/inference/predict.py")


if __name__ == "__main__":
    main()
