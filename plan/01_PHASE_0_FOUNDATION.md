# Phase 0: Foundation & Data Setup

> **Duration**: ~2 days  
> **Goal**: Solid foundation — environment, database, and clean data ready for ML

---

## Checklist

### 0.1 Project Initialization
- [ ] Initialize Git repository with proper `.gitignore`
- [ ] Create `pyproject.toml` with project metadata
- [ ] Set up `requirements.txt` with pinned versions:
  ```
  streamlit>=1.28.0
  pandas>=2.0.0
  numpy>=1.24.0
  scikit-learn>=1.3.0
  lightgbm>=4.0.0
  shap>=0.42.0
  sqlalchemy>=2.0.0
  psycopg2-binary>=2.9.0
  plotly>=5.17.0
  python-dotenv>=1.0.0
  ```
- [ ] Create virtual environment and install dependencies
- [ ] Set up pre-commit hooks (black, isort, flake8)

### 0.2 Docker Environment
- [ ] Create `docker/Dockerfile` for app container
- [ ] Create `docker/docker-compose.yml`:
  ```yaml
  services:
    app:
      build: .
      ports:
        - "8501:8501"
      depends_on:
        - db
      environment:
        - DATABASE_URL=postgresql://...
    
    db:
      image: postgres:15-alpine
      volumes:
        - postgres_data:/var/lib/postgresql/data
      environment:
        - POSTGRES_DB=revenue_intel
        - POSTGRES_USER=app
        - POSTGRES_PASSWORD=dev_password
  ```
- [ ] Add `trainer` service for offline model training
- [ ] Test one-command startup: `docker-compose up`

### 0.3 Database Schema
- [ ] Design normalized schema (see `06_DATA_SPEC.md`)
- [ ] Create `database/schema.sql`:
  ```sql
  -- Core tables
  CREATE TABLE deals (...);
  CREATE TABLE accounts (...);
  CREATE TABLE sales_reps (...);
  CREATE TABLE products (...);
  
  -- ML tables
  CREATE TABLE deal_scores (...);
  CREATE TABLE score_explanations (...);
  CREATE TABLE forecast_snapshots (...);
  CREATE TABLE model_versions (...);
  ```
- [ ] Create migration runner script
- [ ] Add seed data script for demo

### 0.4 Data Acquisition & Cleaning
- [ ] Download MavenTech CRM dataset (or alternative)
  - Source: [Maven Analytics](https://mavenanalytics.io/data-playground)
  - Alternative: Kaggle B2B sales datasets
- [ ] Create data loading notebook (`notebooks/01_data_exploration.ipynb`)
- [ ] Document data quality issues found
- [ ] Create cleaning pipeline:
  - Handle missing values
  - Standardize date formats
  - Normalize categorical fields
  - Remove obvious duplicates
- [ ] Load cleaned data into Postgres
- [ ] Validate data integrity with SQL checks

### 0.5 Configuration Management
- [ ] Create `app/config.py`:
  ```python
  from pydantic_settings import BaseSettings
  
  class Settings(BaseSettings):
      database_url: str
      model_path: str = "models/artifacts"
      log_level: str = "INFO"
      
      class Config:
          env_file = ".env"
  ```
- [ ] Create `.env.example` template
- [ ] Document all configuration options

---

## Acceptance Criteria

✅ `docker-compose up` starts all services  
✅ Database contains cleaned CRM data  
✅ Can query deals from Python: `SELECT * FROM deals LIMIT 10`  
✅ All dependencies install without conflicts  
✅ Project structure matches roadmap  

---

## Notes for Portfolio

- Document your data cleaning decisions in a notebook
- Keep the Docker setup simple but professional
- Show you understand 12-factor app principles (config via env vars)

