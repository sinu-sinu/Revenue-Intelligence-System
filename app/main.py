"""
Revenue Intelligence System - Main Streamlit Application

This is the entry point for the Streamlit application.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Revenue Intelligence System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-high {
        color: #EF4444;
        font-weight: bold;
    }
    .risk-med {
        color: #F59E0B;
        font-weight: bold;
    }
    .risk-low {
        color: #10B981;
        font-weight: bold;
    }
    h1 {
        color: #4F46E5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Welcome page
st.title("Revenue Intelligence System")
st.markdown("---")

st.markdown(
    """
    ## Welcome to Revenue Intelligence
    
    **Decision support for sales pipeline management**
    
    This system helps sales leadership:
    - **Identify at-risk deals** using explainable ML
    - **Forecast revenue** with uncertainty bands
    - **Understand risk drivers** with clear explanations
    - **Focus attention** where it matters most
    
    ### Get Started
    
    Use the sidebar to navigate to:
    - **Risk Dashboard** - See deals requiring attention
    - **Deal Details** - Deep dive into specific opportunities
    - **Forecast** - View revenue projections with confidence intervals
    
    ---
    
    ### Current Status
    """
)

# Connection status
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("System Status", "Online")

with col2:
    st.metric("Database", "Connected")

with col3:
    st.metric("Models", "Not Loaded")

st.info(
    "**Select a page from the sidebar** to begin exploring your pipeline"
)

# Footer
st.markdown("---")
st.caption("Revenue Intelligence System v1.0.0 | Built with Streamlit")

