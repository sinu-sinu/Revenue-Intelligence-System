PRD — AI Sales Ops Assistant - Revenue Intelligence System
Phase 1: Streamlit-first, API-backed
 Role framing: Staff AI Engineer

1. Product overview
One-line
 An internal AI tool that helps sales leadership and reps prioritize pipeline work, understand forecast risk, and intervene earlier using explainable ML.
Core principle
Decision support over automation.
The system does not change CRM records or contact customers. It surfaces where attention is most valuable and explains why.

2. Target users & jobs-to-be-done
Primary users
CEO / CRO


VP Sales / Sales Director


Sales Ops


User goals
Identify fragile revenue early


Improve forecast credibility


Standardize deal risk assessment


Focus leadership attention where it matters



3. Phase 1 scope (Streamlit-only)
What Phase 1 includes
Streamlit UI


Python backend logic


Trained ML models loaded in-process


Postgres for persistence


Optional LLM for explanations


What Phase 1 explicitly avoids
SPA complexity


Frontend state management


Cloud vendor lock-in


Auto-actions (emails, CRM writes)



4. Core features (Phase 1)
4.1 Weekly “Risk This Week” view
Primary screen
Shows:
Top N open deals sorted by risk × value


Columns:


Deal name


Stage


Amount


Risk score


Win probability


Key risk driver (short text)


Purpose:
“If you only look at one screen this week, look at this.”

4.2 Deal drill-down
Clicking a deal shows:
Summary:


Win probability


Expected close window


Risk level (low/med/high)


Top drivers:


Time open vs peers


Team/product win rate


Stage stagnation


Suggested next action:


Rule + model-backed


Confidence indicator


Evidence section:


Data points used


Explicit “unknowns”


LLM (if used) only summarizes computed facts.

4.3 Forecast view (with uncertainty)
Displays:
Expected revenue (P50)


Conservative (P10) and optimistic (P90) bands


Aggregated from deal-level probabilities


Grouped by week/month


Key requirement:
Forecasts must show uncertainty clearly.

4.4 Lightweight filters
By sales team


By product


By deal size


By stage


All server-side, no heavy UI logic.

5. Data & modeling (Phase 1)
Source data
Public CRM Sales Opportunities dataset (MavenTech-style) or others thats better


Loaded into Postgres


Treated as CRM snapshot data


Models
Win probability model


Logistic Regression (baseline)


LightGBM/XGBoost (primary)


Calibrated probabilities


Time-to-close proxy


Regression or survival-lite approach


Used for forecast timing, not exact dates


Risk score


Composite:


Win probability


Deal age vs peers


Deal size


Normalized to 0–100 for UI clarity


Explainability
Local feature attribution (SHAP or coefficients)


Precomputed for fast UI rendering



6. Phase 1 architecture (Streamlit-first)
┌──────────────────────┐
│      Streamlit UI    │
│  (Decision surfaces) │
└─────────┬────────────┘
          │
┌─────────▼────────────┐
│   Python App Layer   │
│  - Feature logic     │
│  - Scoring           │
│  - Forecasting       │
└─────────┬────────────┘
          │
┌─────────▼────────────┐
│     Postgres DB      │
│  Deals, Scores,     │
│  Forecast snapshots │
└──────────────────────┘

Optional:
LLM API for explanation summaries


Vector store only if needed (not default)



7. Deployment (Phase 1)
Environment
Docker + Docker Compose


Single host (local or small VM)


Services
app → Streamlit + Python logic


db → Postgres


trainer → offline model training container


Deployment goals
One-command startup


Reproducible environment


Low cost


Easy demo



8. Observability & robustness (Phase 1)
Even in Phase 1, you show staff-level hygiene:
Input validation on load


Model version tagging


Prediction timestamping


Simple drift checks:


feature distribution changes


Clear error states in UI (not silent failures)



9. Phase 2 (optional): Vue refactor
Trigger conditions
More users


Need for role-based views


Need for finer UI control


Desire to demonstrate frontend separation


What changes
Streamlit becomes thin or removed


FastAPI layer extracted


Vue frontend consumes API


What stays the same
Models


Database


Business logic


Evaluation


Explanations


This is a refactor, not a rewrite.

10. How this is positioned in your portfolio
Explicit framing
“I intentionally started with a Streamlit interface to minimize UI overhead and focus on decision logic and model quality.
 The backend is structured to support a future Vue frontend without architectural changes.”
This reads as judgment, not compromise.

11. Phase 1 success criteria
A CEO can understand:


Where revenue is at risk


Why it’s at risk


What should be done next


Demo can be completed in under 2 minutes


Codebase is readable and deployable



