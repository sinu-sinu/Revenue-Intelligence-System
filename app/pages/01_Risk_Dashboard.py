"""
Risk Dashboard - Primary view for at-risk deals.

Shows deals requiring attention, sorted by risk × value.
"""

import streamlit as st

st.title("Risk This Week")
st.caption("Deals requiring attention, sorted by risk × value")

st.markdown("---")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="At-Risk Revenue",
        value="$1.2M",
        delta="-$200K",
        delta_color="inverse",
    )

with col2:
    st.metric(label="High Risk Deals", value="7", delta="2", delta_color="inverse")

with col3:
    st.metric(label="Avg Win Probability", value="62%", delta="-5%")

with col4:
    st.metric(
        label="Forecast vs Target", value="87%", delta="-13%", delta_color="inverse"
    )

st.markdown("---")

# Filters in sidebar
with st.sidebar:
    st.header("Filters")

    team_filter = st.multiselect(
        "Sales Team",
        options=["Enterprise", "Mid-Market", "SMB"],
        default=None,
    )

    product_filter = st.multiselect(
        "Product", options=["GTM Suite", "Product X", "Product Y"], default=None
    )

    min_value = st.slider(
        "Min Deal Value", min_value=0, max_value=500000, value=10000, step=10000
    )

    stage_filter = st.multiselect(
        "Stage",
        options=[
            "Prospecting",
            "Qualification",
            "Needs Analysis",
            "Value Proposition",
            "Negotiation",
        ],
        default=None,
    )

    st.divider()
    sort_by = st.selectbox(
        "Sort By",
        options=["Risk × Value", "Risk Score", "Deal Value", "Days Open"],
        index=0,
    )

# Main content
st.subheader("At-Risk Deals")

# Placeholder for actual data
st.info("Data loading will be implemented in Phase 1A after ML pipeline is ready.")

# Demo table structure
st.markdown(
    """
| Deal Name | Stage | Amount | Risk Score | Win Prob | Key Driver | Days Open |
|-----------|-------|---------|-----------|----------|------------|-----------|
| Acme Corp | Negotiation | $450K | HIGH (85) | 34% | Stalled | 42 |
| TechStart Inc | Value Prop | $280K | MED (52) | 58% | Slow Rep | 35 |
| GlobalFin | Needs Analysis | $180K | LOW (28) | 78% | — | 18 |
"""
)

st.caption("This is demo data. Connect to database to see real deals.")

