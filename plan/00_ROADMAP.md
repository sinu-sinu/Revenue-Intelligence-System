# Revenue Intelligence System — Project Roadmap

> **Portfolio Focus**: This project demonstrates Staff AI Engineer competencies through thoughtful architecture, explainable ML, and production-grade practices — not feature bloat.

---

## 🎯 Project Vision

Build an internal AI tool that helps sales leadership prioritize pipeline work, understand forecast risk, and intervene earlier using **explainable ML**.

**Core Principle**: Decision support over automation.

---

## 📅 Phases Overview

| Phase | Focus | Duration | Status |
|-------|-------|----------|--------|
| **Phase 0** | Foundation & Data | ~2 days | 🔲 Not Started |
| **Phase 1A** | Core ML Pipeline | ~3 days | 🔲 Not Started |
| **Phase 1B** | Streamlit UI | ~3 days | 🔲 Not Started |
| **Phase 1C** | Polish & Demo | ~2 days | 🔲 Not Started |
| **Phase 2** | Vue Refactor (Optional) | ~5 days | 🔲 Future |

---

## 🏗️ Architecture At-a-Glance

```
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                       │
│  ┌─────────────────┐              ┌─────────────────────┐   │
│  │   Streamlit UI  │   Phase 2 →  │   Vue 3 Frontend    │   │
│  │ (Phase 1)       │              │   + FastAPI         │   │
│  └────────┬────────┘              └──────────┬──────────┘   │
└───────────┼──────────────────────────────────┼──────────────┘
            │                                  │
┌───────────▼──────────────────────────────────▼──────────────┐
│                      SERVICE LAYER                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ Scoring     │  │ Forecasting  │  │ Explanation       │   │
│  │ Engine      │  │ Engine       │  │ Generator         │   │
│  └─────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────┬───────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                       DATA LAYER                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ PostgreSQL  │  │ ML Models    │  │ Feature Store     │   │
│  │ (Deals, etc)│  │ (Pickle/ONNX)│  │ (Precomputed)     │   │
│  └─────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure (Target)

```
revenue-intelligence/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Streamlit entrypoint
│   ├── pages/                  # Streamlit multi-page
│   │   ├── 01_risk_dashboard.py
│   │   ├── 02_deal_drilldown.py
│   │   └── 03_forecast.py
│   ├── components/             # Reusable UI components
│   └── config.py
├── core/
│   ├── __init__.py
│   ├── scoring/                # Risk & Win probability
│   ├── forecasting/            # Revenue forecasting
│   ├── explanations/           # SHAP / feature attribution
│   └── data/                   # Data access layer
├── models/
│   ├── training/               # Training scripts
│   ├── artifacts/              # Saved models
│   └── evaluation/             # Model eval metrics
├── database/
│   ├── migrations/
│   ├── seeds/
│   └── schema.sql
├── tests/
├── notebooks/                  # EDA & experimentation
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── plan/                       # This folder
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🎭 Portfolio Differentiators

What makes this project stand out:

1. **Explainable ML** — Not just predictions, but *why* with SHAP values
2. **Calibrated Probabilities** — Proper uncertainty quantification
3. **Production Patterns** — Model versioning, drift detection, validation
4. **Clean Architecture** — Service layer that survives UI refactors
5. **Thoughtful UX** — Decision surfaces, not data dumps

---

## 📋 Task Files

- `01_PHASE_0_FOUNDATION.md` — Environment, data, database setup
- `02_PHASE_1A_ML_PIPELINE.md` — Model training, scoring, explainability
- `03_PHASE_1B_STREAMLIT_UI.md` — UI implementation
- `04_PHASE_1C_POLISH.md` — Demo prep, documentation, refinement
- `05_PHASE_2_VUE_REFACTOR.md` — Optional frontend separation
- `06_DATA_SPEC.md` — Data dictionary and feature engineering
- `07_MODEL_SPEC.md` — Model specifications and evaluation criteria

