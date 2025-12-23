"""
Forecast View - Revenue forecasting with uncertainty bands.

Shows P10/P50/P90 projections for upcoming periods.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Revenue Forecast")
st.caption("Projected revenue with confidence intervals")

st.markdown("---")

# Time range selector
col1, col2 = st.columns([3, 1])

with col1:
    time_range = st.radio(
        "Forecast Horizon",
        options=["4 Weeks", "8 Weeks", "12 Weeks", "Quarter"],
        horizontal=True,
        index=2,
    )

with col2:
    st.metric("Target", "$2.5M")

st.markdown("---")

# Demo forecast chart
st.subheader("Revenue Projection")

# Create sample data
weeks = pd.date_range(start="2024-01-01", periods=12, freq="W")
p10 = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]
p50 = [150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480]
p90 = [200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]

fig = go.Figure()

# Add confidence bands
fig.add_trace(
    go.Scatter(
        x=weeks,
        y=p90,
        fill=None,
        mode="lines",
        line=dict(color="rgba(79, 70, 229, 0.1)"),
        name="P90 (Optimistic)",
        showlegend=True,
    )
)

fig.add_trace(
    go.Scatter(
        x=weeks,
        y=p10,
        fill="tonexty",
        mode="lines",
        line=dict(color="rgba(79, 70, 229, 0.1)"),
        name="P10 (Conservative)",
        showlegend=True,
    )
)

# Add P50 line
fig.add_trace(
    go.Scatter(
        x=weeks,
        y=p50,
        mode="lines+markers",
        line=dict(color="#4F46E5", width=3),
        name="Expected (P50)",
        showlegend=True,
    )
)

# Add target line
fig.add_hline(
    y=500, line_dash="dash", line_color="orange", annotation_text="Quarterly Target"
)

fig.update_layout(
    height=500,
    xaxis_title="Week",
    yaxis_title="Revenue ($K)",
    hovermode="x unified",
    template="plotly_dark",
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Summary table
st.subheader("Weekly Breakdown")

summary_data = {
    "Week": [f"Week {i+1}" for i in range(4)],
    "Conservative (P10)": ["$120K", "$140K", "$160K", "$180K"],
    "Expected (P50)": ["$180K", "$210K", "$240K", "$270K"],
    "Optimistic (P90)": ["$240K", "$280K", "$320K", "$360K"],
    "# Deals": [4, 3, 5, 3],
}

st.dataframe(summary_data, use_container_width=True, hide_index=True)

st.info(
    "Real forecast with Monte Carlo simulation will be implemented in Phase 1A."
)

# Insights
st.markdown("---")
st.subheader("Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Forecast vs Target**")
    st.metric("Gap to Target", "-$50K", delta="-$50K")
    st.caption("At P50, we're $50K short of quarterly target")

with col2:
    st.markdown("**Risk Assessment**")
    st.markdown("**Medium Risk**")
    st.caption("Need 2-3 additional deals in pipeline to hit target")

