# Phase 1B: Streamlit UI Implementation

> **Duration**: ~3 days  
> **Goal**: Beautiful, functional decision surfaces — not just data dumps

---

## Design Philosophy

Even in Streamlit, aim for:
- **Clarity over density**: White space is your friend
- **Decision-oriented**: Every element answers "what should I do?"
- **Confidence indicators**: Always show uncertainty
- **Progressive disclosure**: Summary first, details on demand

---

## Checklist

### 1B.1 App Structure & Theming
- [ ] Create `app/main.py` as entrypoint:
  ```python
  import streamlit as st
  
  st.set_page_config(
      page_title="Revenue Intelligence",
      page_icon="📊",
      layout="wide",
      initial_sidebar_state="expanded"
  )
  ```
- [ ] Create custom theme in `.streamlit/config.toml`:
  ```toml
  [theme]
  primaryColor = "#4F46E5"  # Indigo
  backgroundColor = "#0F172A"  # Slate 900
  secondaryBackgroundColor = "#1E293B"  # Slate 800
  textColor = "#F8FAFC"  # Slate 50
  font = "sans serif"
  ```
- [ ] Create reusable components in `app/components/`:
  - `metrics_card.py` — styled metric display
  - `risk_badge.py` — color-coded risk indicator
  - `probability_gauge.py` — visual probability display
  - `data_table.py` — styled dataframe wrapper

### 1B.2 Page: Risk Dashboard (Primary Screen)
- [ ] Create `app/pages/01_🎯_Risk_Dashboard.py`
- [ ] Header section:
  ```python
  st.title("Risk This Week")
  st.caption("Deals requiring attention, sorted by risk × value")
  
  # Key metrics row
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("At-Risk Revenue", "$1.2M", delta="-$200K")
  col2.metric("High Risk Deals", "7", delta="2")
  col3.metric("Avg Win Probability", "62%")
  col4.metric("Forecast vs Target", "87%", delta="-13%")
  ```
- [ ] Filters in sidebar:
  ```python
  with st.sidebar:
      st.header("Filters")
      teams = st.multiselect("Sales Team", options=all_teams)
      products = st.multiselect("Product", options=all_products)
      min_value = st.slider("Min Deal Value", 0, 500000, 10000)
      stages = st.multiselect("Stage", options=all_stages)
  ```
- [ ] Main table with:
  - Deal name (clickable → drill-down)
  - Account name
  - Stage (visual progress indicator)
  - Amount (formatted currency)
  - Risk Score (color-coded badge: 🔴🟡🟢)
  - Win Probability (mini gauge)
  - Key Risk Driver (1-line summary)
  - Days Open
- [ ] Sorting controls (risk×value default)
- [ ] Quick action buttons: "Mark Reviewed", "Add Note"

### 1B.3 Page: Deal Drill-Down
- [ ] Create `app/pages/02_🔍_Deal_Detail.py`
- [ ] URL parameter handling for deal selection
- [ ] Deal header:
  ```python
  st.title(deal.name)
  st.caption(f"{deal.account} • {deal.owner} • {deal.stage}")
  ```
- [ ] Three-column summary:
  ```
  ┌─────────────┬─────────────┬─────────────┐
  │ Win Prob    │ Risk Level  │ Est. Close  │
  │    67%      │    HIGH     │  Jan 15-30  │
  │  ▓▓▓▓▓░░░   │     🔴      │  ~2 weeks   │
  └─────────────┴─────────────┴─────────────┘
  ```
- [ ] Risk drivers section:
  ```python
  st.subheader("Why This Risk Level")
  
  drivers = [
      ("⏱️ Time Open", "42 days (vs 28 avg)", "+15%"),
      ("📉 Stage Stagnation", "18 days in Negotiation", "+12%"),
      ("👤 Rep Win Rate", "45% historical", "+8%"),
  ]
  
  for icon_label, detail, impact in drivers:
      with st.container():
          col1, col2, col3 = st.columns([2, 3, 1])
          col1.write(f"**{icon_label}**")
          col2.write(detail)
          col3.write(f"`{impact}`")
  ```
- [ ] Suggested next action:
  ```python
  st.subheader("Suggested Action")
  with st.container(border=True):
      st.write("**Schedule executive sponsor call**")
      st.caption("Confidence: High • Based on: Stagnation pattern + deal size")
  ```
- [ ] Evidence section (collapsible):
  - Data points used
  - What we don't know (explicit unknowns!)
  - Model confidence interval
- [ ] Deal timeline visualization (Plotly)

### 1B.4 Page: Forecast View
- [ ] Create `app/pages/03_📈_Forecast.py`
- [ ] Time range selector (4w, 8w, 12w, Quarter)
- [ ] Main forecast chart (Plotly):
  ```python
  import plotly.graph_objects as go
  
  fig = go.Figure()
  
  # Confidence bands
  fig.add_trace(go.Scatter(
      x=dates, y=p90,
      fill=None, mode='lines',
      line=dict(color='rgba(79, 70, 229, 0.1)'),
      name='P90 (Optimistic)'
  ))
  fig.add_trace(go.Scatter(
      x=dates, y=p10,
      fill='tonexty', mode='lines',
      line=dict(color='rgba(79, 70, 229, 0.1)'),
      name='P10 (Conservative)'
  ))
  # P50 line
  fig.add_trace(go.Scatter(
      x=dates, y=p50,
      mode='lines+markers',
      line=dict(color='#4F46E5', width=3),
      name='Expected (P50)'
  ))
  # Target line
  fig.add_hline(y=target, line_dash="dash", 
                annotation_text="Target")
  ```
- [ ] Summary table by period:
  ```
  | Week     | Conservative | Expected | Optimistic | # Deals |
  |----------|--------------|----------|------------|---------|
  | Dec 23   | $120K        | $180K    | $250K      | 4       |
  | Dec 30   | $80K         | $150K    | $220K      | 3       |
  ```
- [ ] Drill-down: click period to see contributing deals

### 1B.5 Components & Visualizations
- [ ] Risk score badge component:
  ```python
  def risk_badge(score: int) -> str:
      if score >= 70:
          return f'<span class="risk-high">🔴 {score}</span>'
      elif score >= 40:
          return f'<span class="risk-med">🟡 {score}</span>'
      else:
          return f'<span class="risk-low">🟢 {score}</span>'
  ```
- [ ] Win probability gauge (mini bar)
- [ ] Stage progress indicator
- [ ] Trend sparklines for deals
- [ ] Custom CSS for polish:
  ```python
  st.markdown("""
  <style>
  .stMetric { background: #1E293B; padding: 1rem; border-radius: 0.5rem; }
  .risk-high { color: #EF4444; font-weight: bold; }
  .risk-med { color: #F59E0B; font-weight: bold; }
  .risk-low { color: #10B981; font-weight: bold; }
  </style>
  """, unsafe_allow_html=True)
  ```

### 1B.6 Data Loading & Caching
- [ ] Implement efficient data loading:
  ```python
  @st.cache_data(ttl=300)  # 5 min cache
  def load_deals(filters: dict) -> pd.DataFrame:
      return deal_service.get_filtered_deals(**filters)
  
  @st.cache_resource
  def load_model():
      return ModelLoader.load_latest("win_probability")
  ```
- [ ] Add loading states
- [ ] Handle errors gracefully (not silent failures!)

### 1B.7 Session State & Interactivity
- [ ] Track selected deal in session state
- [ ] Persist filter selections
- [ ] Add "Refresh Data" button with timestamp
- [ ] Implement deal comparison mode (select 2-3 deals)

---

## Acceptance Criteria

✅ All three pages functional and styled  
✅ Filters work correctly and persist  
✅ Clicking deal in table navigates to drill-down  
✅ Forecast shows clear uncertainty bands  
✅ No UI crashes on edge cases (empty data, etc.)  
✅ Loads in < 2 seconds with cached data  

---

## Visual Reference

**Risk Dashboard Layout:**
```
┌────────────────────────────────────────────────────────────────┐
│  📊 Revenue Intelligence                         [Refresh] ⟳   │
├──────────┬─────────────────────────────────────────────────────┤
│          │  Risk This Week                                     │
│ Filters  │  ─────────────────────────────────────────────────  │
│          │  ┌────────┬────────┬────────┬────────┐              │
│ □ Team   │  │ At-Risk│ High   │ Avg    │ Fcst   │              │
│ □ Product│  │ $1.2M  │ 7 deals│ 62%    │ 87%    │              │
│ □ Stage  │  └────────┴────────┴────────┴────────┘              │
│          │                                                      │
│ Value:   │  ┌──────────────────────────────────────────────┐   │
│ ●────○   │  │ Deal         │ Risk │ Win% │ Amount │ Driver │   │
│          │  ├──────────────┼──────┼──────┼────────┼────────┤   │
│          │  │ Acme Corp    │ 🔴85 │ 34%  │ $450K  │ Stalled│   │
│          │  │ TechStart    │ 🟡52 │ 58%  │ $280K  │ Slow   │   │
│          │  │ GlobalFin    │ 🟢28 │ 78%  │ $180K  │ —      │   │
│          │  └──────────────┴──────┴──────┴────────┴────────┘   │
└──────────┴─────────────────────────────────────────────────────┘
```

