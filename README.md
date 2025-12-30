# Revenue Intelligence System

> Production-ready ML system for sales pipeline risk analysis and revenue forecasting

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Vue 3](https://img.shields.io/badge/Vue-3.5+-brightgreen.svg)](https://vuejs.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5+-orange.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A decision-support system that helps sales leadership prioritize pipeline work, understand forecast risk, and intervene earlier using **explainable machine learning**. Built to demonstrate production ML engineering skills.

---

## What This System Does

Sales teams struggle with thousands of open deals. Which ones need attention? Which will close this quarter? This system answers those questions with:

1. **Risk Scoring** - Identifies deals most likely to slip or lose, ranked by potential revenue impact
2. **Win Probability** - Calibrated predictions showing actual likelihood of closing
3. **Revenue Forecasting** - Weekly projections with P10/P50/P90 confidence intervals
4. **Model Explanations** - SHAP-based insights showing *why* a deal is at risk

**Design Philosophy:** Decision support over automation. The system surfaces where attention is valuable and explains why, without modifying CRM records or contacting customers directly.

---

## Key Results

| Metric | Value | Description |
|--------|-------|-------------|
| **Model AUC** | 0.58 | Win probability discrimination (limited by dataset signal) |
| **Calibration** | Isotonic | Ensures 70% predictions win 70% of the time |
| **API Response** | <50ms | Precomputed predictions for instant UI |
| **Forecast Accuracy** | Monte Carlo | 1000 simulations per forecast window |

*Note: The relatively low AUC reflects limited predictive signal in the demo dataset (MavenTech CRM), not model quality. Real CRM data typically yields AUC 0.70-0.85.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Vue 3)                          │
│   TypeScript • TailwindCSS • Vue Query • Chart.js                │
│   Port: 3000                                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │ REST API
┌───────────────────────────────▼─────────────────────────────────┐
│                        Backend (FastAPI)                         │
│   /api/deals    - List, filter, detail with risk drivers        │
│   /api/deals/{id}/explanation - On-demand SHAP explanations     │
│   /api/forecast - Monte Carlo revenue projections               │
│   /api/health   - Container health check                        │
│   Port: 8000                                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                     ML & Business Logic                          │
│  ┌──────────────┬──────────────┬──────────────────────────────┐ │
│  │ Risk Scorer  │ Win Prob     │ Revenue Forecast             │ │
│  │ Composite    │ LightGBM     │ Monte Carlo                  │ │
│  │ formula      │ + Isotonic   │ 1000 simulations             │ │
│  └──────────────┴──────────────┴──────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ SHAP Explainer - TreeExplainer for feature contributions   │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                        Data Layer                                │
│  CSV Files (dataset/) → Feature Engineering → Predictions       │
│  Precomputed scores in data/predictions/latest_predictions.csv  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Training**: CSV files → Feature engineering → LightGBM training → Model artifacts
2. **Inference**: Raw deals → Feature enrichment → Win probability + Risk score → CSV cache
3. **Serving**: API loads cached predictions → Frontend displays with filtering/sorting
4. **Explanations**: On-demand SHAP calculation when user requests deal explanation

---

## Technology Stack

### Backend
- **Python 3.12** - Modern Python with type hints
- **FastAPI** - High-performance async API framework
- **Pydantic** - Data validation and serialization
- **LightGBM 4.5** - Gradient boosting for win probability
- **SHAP** - Model explainability with TreeExplainer
- **scikit-learn** - Preprocessing, calibration, metrics

### Frontend
- **Vue 3** - Composition API with TypeScript
- **Vite** - Fast build tooling
- **TailwindCSS** - Utility-first styling (dark theme)
- **Vue Query (TanStack)** - Server state management
- **Chart.js** - Revenue forecast visualization
- **Heroicons** - UI iconography

### Infrastructure
- **Docker** - Containerized deployment
- **Docker Compose** - Multi-service orchestration
- **Nginx** - Frontend static serving with API proxy

---

## Features

### Risk Dashboard
- Sortable deal table with risk scores
- Multi-select filters (account, rep, product, risk level)
- Risk score range slider
- Summary metrics (total deals, at-risk revenue, average win probability)

### Deal Detail
- Full deal information with risk drivers
- Recommended action based on risk category
- **Model Explanation** - On-demand SHAP analysis showing:
  - Features increasing win probability (green)
  - Features decreasing win probability (red)
  - Human-readable explanations for each factor

### Revenue Forecast
- Weekly revenue projections with confidence intervals
- P10/P50/P90 bands from Monte Carlo simulation
- Interactive Chart.js visualization
- Configurable forecast horizon (4-12 weeks)

---

## Quick Start

### Prerequisites
- Python 3.12+
- Node.js 18+
- Docker (optional)

### Option 1: Docker (Recommended)

```bash
cd docker
docker-compose up

# Access:
# - Frontend: http://localhost:3000
# - API Docs: http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[api]"

# Generate predictions (required first time)
python models/inference/predict.py

# Start API
uvicorn api.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev

# Access: http://localhost:5173 (API at http://localhost:8000)
```

---

## Project Structure

```
revenue-intelligence/
├── api/                    # FastAPI backend
│   ├── main.py             # Application entry
│   ├── routes/             # API endpoints
│   │   ├── deals.py        # /deals endpoints
│   │   ├── forecast.py     # /forecast endpoint
│   │   └── health.py       # /health endpoint
│   ├── schemas/            # Pydantic models
│   └── services/           # SHAP explainer service
├── app/
│   └── services/           # Shared data services
│       ├── data_loader.py  # Prediction loading
│       └── risk_calculator.py
├── core/                   # Business logic
│   ├── data/               # Feature engineering
│   │   ├── preprocessor.py
│   │   └── features.py
│   ├── scoring/            # Risk scoring
│   ├── forecasting/        # Monte Carlo forecasts
│   └── explanations/       # SHAP explainer
├── models/                 # ML pipeline
│   ├── training/           # Model training
│   │   ├── win_probability.py
│   │   ├── time_to_close.py
│   │   └── train_pipeline.py
│   ├── evaluation/         # Metrics
│   ├── inference/          # Prediction generation
│   └── artifacts/          # Saved models
├── frontend/               # Vue 3 SPA
│   ├── src/
│   │   ├── components/     # Reusable components
│   │   ├── views/          # Page views
│   │   ├── composables/    # Vue composables
│   │   └── api/            # API client
│   ├── Dockerfile
│   └── nginx.conf
├── dataset/                # Training data (CSV)
├── data/predictions/       # Cached predictions
├── docker/                 # Docker configuration
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   └── Dockerfile.trainer
└── pyproject.toml          # Python dependencies
```

---

## Dataset

The system includes the MavenTech CRM demo dataset:

| File | Records | Description |
|------|---------|-------------|
| `sales_pipeline.csv` | ~8,800 | Opportunities with stages and outcomes |
| `accounts.csv` | 85 | Customer companies |
| `sales_teams.csv` | 35 | Sales representatives |
| `products.csv` | 7 | Product catalog |

**Production Adaptation:** The data loader is designed for easy integration with:
- Salesforce CSV exports
- HubSpot exports
- Custom CRM APIs (implement `DataLoader` interface)

---

## ML Models

### Win Probability Model
```
Algorithm: LightGBM (gradient boosting)
Calibration: Isotonic regression
Output: P(won) ∈ [0, 1]
Features: 15 (temporal, rep performance, deal characteristics)
Training: Time-series split to prevent leakage
```

**Key Hyperparameters:**
- `num_leaves`: 16 (prevents overfitting)
- `max_depth`: 5
- `min_data_in_leaf`: 50
- `learning_rate`: 0.05
- L1/L2 regularization enabled

### Risk Score
```
Type: Composite formula (not ML)
Range: 0-100 (higher = more risk)
Components:
  - Base: (1 - win_probability) × 50
  - Stagnation: +30 if days_in_stage > 2× average
  - Value: +10-20 based on deal size percentile
```

### Revenue Forecast
```
Method: Monte Carlo simulation (1000 runs)
Per-deal: Bernoulli(win_prob) × deal_value × P(close_in_window)
Aggregation: Weekly totals with P10/P50/P90 percentiles
```

---

## API Reference

### Deals

```
GET /api/deals
  Query: account[], sales_agent[], product[], risk_category[]
         min_risk_score, max_risk_score, sort_by, sort_order, limit
  Returns: { deals: DealSummary[], total: int, limit: int }

GET /api/deals/{id}
  Returns: DealDetail (includes risk_drivers, predicted_close_days)

GET /api/deals/{id}/explanation
  Returns: DealExplanation (SHAP-based feature contributions)

GET /api/deals/summary
  Returns: { total_deals, at_risk_revenue, high_risk_count, avg_win_probability }

GET /api/deals/filters
  Returns: Available filter options from data
```

### Forecast

```
POST /api/forecast
  Body: { horizon_weeks?: int, period?: "week"|"month" }
  Returns: { periods: ForecastPeriod[], summary: ForecastSummary }
```

Full OpenAPI documentation available at `/docs` when running the API.

---

## Why These Choices

| Choice | Reasoning |
|--------|-----------|
| **LightGBM over XGBoost** | Faster training, native categorical handling, efficient on tabular data |
| **Isotonic calibration** | Better for unbalanced datasets than Platt scaling |
| **SHAP TreeExplainer** | Exact values for tree models, no approximation needed |
| **Precomputed predictions** | Sub-50ms API response, predictions regenerated in batch |
| **Vue over React** | Composition API maps well to this data-driven dashboard pattern |
| **CSV over database** | Portability for demo; production would use Postgres/Snowflake |
| **Monte Carlo forecast** | Captures full uncertainty distribution, not just point estimates |

---

## Future Extensibility

1. **Real CRM Integration** - Replace CSV loader with Salesforce/HubSpot API
2. **Database Backend** - PostgreSQL for predictions, Redis for caching
3. **Real-time Scoring** - Kafka/Pub-Sub for streaming updates
4. **A/B Testing** - Track intervention effectiveness
5. **Additional Models** - Churn prediction, upsell probability
6. **Alerting** - Slack/email notifications for high-risk deals

---

## Development

### Code Quality
```bash
black .           # Format
isort .           # Sort imports
flake8 .          # Lint
mypy .            # Type check
pytest            # Test
```

### Rebuild Models
```bash
# Full pipeline
python models/training/train_pipeline.py

# Just predictions
python models/inference/predict.py

# Docker training
docker-compose --profile training up trainer
```

---

## Project Demonstrates

This project showcases production ML engineering practices:

- **End-to-end ML pipeline** - Training, evaluation, inference, serving
- **Model explainability** - SHAP integration for business stakeholders
- **Calibrated predictions** - Probabilities that mean what they say
- **Clean architecture** - Separation of concerns, testable components
- **Modern stack** - FastAPI, Vue 3, TypeScript, Docker
- **Production patterns** - Health checks, error handling, caching
- **Thoughtful design** - Trade-offs documented, not over-engineered

---

## Contact

**Author:** sinu
**Email:** sinu28.sinu@gmail.com

---

## License

MIT License - See LICENSE file for details.

---

*Built as a demonstration of ML Engineer competencies: ML engineering, explainability, production systems, and pragmatic architecture decisions.*
