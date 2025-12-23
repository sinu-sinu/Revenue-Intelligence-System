# Revenue Intelligence System

> AI-powered decision support for sales pipeline management

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An internal AI tool that helps sales leadership prioritize pipeline work, understand forecast risk, and intervene earlier using **explainable ML**.

## 🎯 Core Principle

**Decision support over automation** — This system surfaces where attention is most valuable and explains why, without changing CRM records or contacting customers.

---

## ✨ Features

- **🎯 Risk Dashboard** - At-risk deals sorted by risk × value
- **📊 Win Probability Model** - Calibrated predictions with SHAP explanations
- **📈 Revenue Forecasting** - P10/P50/P90 projections with Monte Carlo simulation
- **🔍 Deal Drill-Down** - Explainable risk drivers and suggested actions
- **⚡ Real-Time Scoring** - Precomputed scores for fast UI rendering

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│        Streamlit UI (Phase 1)           │
│     Decision Surfaces & Dashboards      │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         Core Business Logic             │
│  ┌──────────┬──────────┬──────────┐     │
│  │ Scoring  │ Forecast │ Explain  │     │
│  └──────────┴──────────┴──────────┘     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│          Data Layer                     │
│  ┌──────────┬──────────┬──────────┐     │
│  │Postgres │ Models   │ Features │     │
│  └──────────┴──────────┴──────────┘     │
└─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd revenue-intelligence

# Start all services
cd docker
docker-compose up

# Access the app
open http://localhost:8501
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your database URL

# Start database (if using Docker)
cd docker && docker-compose up db -d

# Run migrations and seed data
python database/seeds/seed_demo_data.py

# Start Streamlit
streamlit run app/main.py
```

---

## 📁 Project Structure

```
revenue-intelligence/
├── app/                    # Streamlit UI
│   ├── main.py            # Entry point
│   ├── config.py          # Configuration
│   └── pages/             # Multi-page app
├── core/                   # Business logic
│   ├── data/              # Data access & features
│   ├── scoring/           # Risk & win probability
│   ├── forecasting/       # Revenue forecasting
│   └── explanations/      # SHAP explainability
├── models/                 # ML models
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation metrics
│   └── artifacts/         # Saved models
├── database/              # Database schema & seeds
├── docker/                # Docker configuration
├── tests/                 # Test suite
├── notebooks/             # EDA & experiments
└── plan/                  # Project planning docs
```

---

## 📊 Data

The system works with standard CRM opportunity data:

**Core Entities:**
- Deals/Opportunities
- Accounts
- Sales Representatives
- Products

**Demo Data:** Included seed script creates realistic sample data for development.

**Production Data:** Design supports loading from:
- Salesforce
- HubSpot
- CSV exports
- Custom CRM systems

---

## 🤖 ML Models

### Win Probability Model
- **Algorithm:** LightGBM with calibration
- **Output:** P(won) ∈ [0, 1]
- **Features:** Time-based, rep performance, deal characteristics
- **Explainability:** SHAP values for each prediction

### Risk Score
- **Type:** Composite formula
- **Range:** 0-100 (higher = more risk)
- **Components:** Win probability, deal size, velocity, stagnation

### Time-to-Close Model
- **Algorithm:** Quantile regression
- **Output:** P10/P50/P90 distributions
- **Use:** Forecast timing, not point estimates

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run integration tests (requires database)
pytest -m integration
```

---

## 📈 Roadmap

### ✅ Phase 0: Foundation (Complete)
- Project structure
- Docker environment
- Database schema
- Basic Streamlit UI

### 🚧 Phase 1A: ML Pipeline (In Progress)
- Feature engineering
- Model training & calibration
- SHAP explanations
- Risk scoring

### 📋 Phase 1B: UI Enhancement
- Connect UI to models
- Real-time scoring
- Interactive visualizations

### 📋 Phase 1C: Polish
- Testing
- Documentation
- Demo preparation

### 🔮 Phase 2: Vue Refactor (Optional)
- FastAPI extraction
- Vue 3 frontend
- Role-based access

---

## 🛠️ Development

### Code Style

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Type checking
mypy .
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

## 📝 Documentation

- [Project Roadmap](plan/00_ROADMAP.md)
- [Data Specification](plan/06_DATA_SPEC.md)
- [Model Specification](plan/07_MODEL_SPEC.md)
- [Phase Plans](plan/)

---

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- Inspired by modern RevOps best practices
- Built with Streamlit, LightGBM, and PostgreSQL
- SHAP for model explainability

---

## 📧 Contact

**Author:** [Your Name]  
**Portfolio:** [Your Portfolio URL]  
**LinkedIn:** [Your LinkedIn]

---

*Built as a demonstration of Staff AI Engineer competencies: ML engineering, explainability, production patterns, and thoughtful architecture.*

