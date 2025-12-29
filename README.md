# Revenue Intelligence System

> AI-powered decision support for sales pipeline management

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
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
│  │  CSV     │ Models   │ Features │     │
│  │  Files   │ Artifacts│ Engine   │     │
│  └──────────┴──────────┴──────────┘     │
└─────────────────────────────────────────┘
```

**Data Flow:**
1. CSV files (dataset/) → Feature engineering
2. Generate predictions → Save to `data/predictions/latest_predictions.csv`
3. Streamlit UI → Load predictions from CSV (cached)
4. Display dashboards, risk scores, forecasts

**Note:** For historical datasets (e.g., 2016-2017 demo data), the system automatically adjusts date calculations to simulate realistic scenarios.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+ (or 3.10+)
- (Optional) Docker & Docker Compose

### Option 1: Local Development (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd revenue-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate predictions (required before running app)
python models/inference/predict.py

# Start Streamlit app
streamlit run app/main.py

# Access the app at http://localhost:8501
```

### Option 2: Docker

```bash
# Navigate to docker directory
cd docker

# Start the app
docker-compose up app

# Run training (optional)
docker-compose --profile training up trainer

# Access the app at http://localhost:8501
```

---

## 📁 Project Structure

```
revenue-intelligence/
├── app/                   # Streamlit UI
│   ├── main.py            # Entry point
│   ├── config.py          # Configuration
│   ├── pages/             # Multi-page app
│   ├── components/        # Reusable UI components
│   └── services/          # Data loading services
├── core/                  # Business logic
│   ├── data/              # Feature engineering
│   ├── scoring/           # Risk & win probability
│   ├── forecasting/       # Revenue forecasting
│   └── explanations/      # SHAP explainability
├── models/                # ML models
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation metrics
│   ├── inference/         # Prediction generation
│   └── artifacts/         # Saved models
├── data/                  # Generated data
│   └── predictions/       # Precomputed predictions
├── dataset/               # Raw training data (CSV)
├── docker/                # Docker configuration
├── experiments/           # MLflow tracking
├── tests/                 # Test suite
├── notebooks/             # EDA & experiments
└── plan/                  # Project planning docs
```

---

## 📊 Data

The system uses **CSV-based data storage** for simplicity and portability.

**Training Data** (`dataset/`):
- `sales_pipeline.csv` - Deal/opportunity data
- `accounts.csv` - Customer accounts
- `sales_teams.csv` - Sales representatives
- `products.csv` - Product catalog

**Generated Data** (`data/predictions/`):
- `latest_predictions.csv` - Precomputed predictions with risk scores
- `predictions_metadata.json` - Metadata about predictions

**Demo Data:** Included MavenTech CRM dataset with ~8,800 opportunities from 2016-2017. The system automatically handles historical dates for realistic predictions (see [Technical Docs](docs/TECHNICAL_DOCS.md#historical-dataset-handling)).

**Production Ready:** Design supports loading from:
- Salesforce CSV exports
- HubSpot exports
- Custom CRM systems
- Easy to adapt data loader for APIs

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
- **Algorithm:** Exponential distribution with age-based adjustments
- **Output:** Days until close (7-120 day range)
- **Use:** Weekly revenue forecasting with uncertainty bands

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## 📈 Roadmap

### ✅ Phase 0: Foundation (Complete)
- Project structure
- Docker environment
- CSV-based data storage
- Basic Streamlit UI

### ✅ Phase 1A: ML Pipeline (Complete)
- Feature engineering
- Model training & calibration
- Risk scoring
- Prediction generation

### ✅ Phase 1B: UI Enhancement (Complete)
- Connect UI to models
- Precomputed scoring
- Interactive visualizations
- Risk Dashboard
- Deal Detail pages
- Revenue Forecast

### 🚧 Phase 1C: Polish (In Progress)
- Testing
- Documentation ✅
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

- **[User Guide](docs/USER_GUIDE.md)** - End-user documentation for the Streamlit app
- **[Technical Docs](docs/TECHNICAL_DOCS.md)** - Developer guide and API reference
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
- Built with Streamlit, LightGBM, and CSV-based data storage
- SHAP for model explainability

---

## 📧 Contact

**Author:** [Your Name]  
**Portfolio:** [Your Portfolio URL]  
**LinkedIn:** [Your LinkedIn]

---

*Built as a demonstration of Staff AI Engineer competencies: ML engineering, explainability, production patterns, and thoughtful architecture.*

