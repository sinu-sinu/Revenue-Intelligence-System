"""
Deal Details - Deep dive into a specific deal.

Shows risk drivers, SHAP explanations, and suggested actions.
"""

import streamlit as st

st.title("Deal Details")

# Deal selector
deal_options = ["Acme Corp - $450K", "TechStart Inc - $280K", "GlobalFin - $180K"]
selected_deal = st.selectbox("Select Deal", options=deal_options)

st.markdown("---")

# Deal header
st.subheader(selected_deal.split(" - ")[0])
st.caption("Account: Acme Corp • Owner: Sarah Johnson • Stage: Negotiation")

# Three-column summary
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Win Probability", "67%")
    st.progress(0.67)

with col2:
    st.markdown("**Risk Level**")
    st.markdown('<p class="risk-high">HIGH (85)</p>', unsafe_allow_html=True)

with col3:
    st.metric("Est. Close", "Jan 15-30")
    st.caption("~2 weeks (P50)")

st.markdown("---")

# Risk drivers
st.subheader("Why This Risk Level")

drivers = [
    ("Time Open", "42 days (vs 28 avg)", "+15%"),
    ("Stage Stagnation", "18 days in Negotiation", "+12%"),
    ("Rep Win Rate", "45% historical", "+8%"),
    ("Deal Size", "Top 10% of deals", "+5%"),
]

for icon_label, detail, impact in drivers:
    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.markdown(f"**{icon_label}**")
    with col2:
        st.write(detail)
    with col3:
        st.code(impact)

st.markdown("---")

# Suggested action
st.subheader("Suggested Next Action")

with st.container(border=True):
    st.markdown("**Schedule executive sponsor call**")
    st.caption("Confidence: High • Based on: Stagnation pattern + deal size")
    st.button("Mark as Done", key="action_done")

st.markdown("---")

# Evidence section
with st.expander("Evidence & Data Points"):
    st.markdown(
        """
    **Data points used:**
    - Created: 42 days ago
    - Current stage: Negotiation (18 days)
    - Rep historical win rate: 45%
    - Similar deal avg: 28 days total
    - Deal size percentile: 92nd
    
    **What we don't know:**
    - Last contact date
    - Decision maker engagement level
    - Competitor presence
    
    **Model confidence:** 0.82
    """
    )

st.info("Full explanations will be available after ML pipeline is implemented.")

