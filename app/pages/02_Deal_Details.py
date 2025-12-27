"""
Deal Details - Deep dive into a specific deal.

This file is deprecated. Use 02_Deal_Detail.py instead (without the 's').
Redirecting...
"""

import streamlit as st

st.set_page_config(
    page_title="Deal Detail - Revenue Intelligence",
    page_icon="🔍",
    layout="wide"
)

st.error("⚠️ This page is deprecated. Please use 'Deal Detail' from the sidebar instead.")

st.markdown("""
This file (`02_Deal_Details.py`) was the old version with hardcoded mock data.

The active version is: `02_Deal_Detail.py` (without the 's'), which loads real data from CSV.
""")

st.page_link("pages/02_Deal_Detail.py", label="Go to Deal Detail (Real Data)", icon="🔍")

