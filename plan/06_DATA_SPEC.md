# Data Specification

> Feature dictionary, schema design, and data quality requirements

---

## 1. Source Data

### Primary Dataset Option: MavenTech CRM
- **Source**: Maven Analytics Data Playground
- **Size**: ~8,800 opportunities, ~85 sales agents, ~30 accounts
- **Time range**: Typically 2 years of closed deals
- **Fits because**: B2B sales pipeline with stages, amounts, outcomes

### Alternative Datasets
| Dataset | Pros | Cons |
|---------|------|------|
| Kaggle B2B Sales | More records | Less realistic stages |
| Synthetically generated | Fully controlled | May seem artificial |
| IBM HR Analytics | Well-known | Not sales-specific |

**Recommendation**: Start with MavenTech, supplement with synthetic data for edge cases.

---

## 2. Database Schema

### Core Tables

```sql
-- Accounts (customers)
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    industry VARCHAR(100),
    employee_count INTEGER,
    annual_revenue DECIMAL(15, 2),
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sales representatives
CREATE TABLE sales_reps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(200),
    team VARCHAR(100),
    manager_id UUID REFERENCES sales_reps(id),
    hire_date DATE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Products/offerings
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    base_price DECIMAL(12, 2),
    is_active BOOLEAN DEFAULT TRUE
);

-- Deals (opportunities)
CREATE TABLE deals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(300) NOT NULL,
    account_id UUID REFERENCES accounts(id) NOT NULL,
    owner_id UUID REFERENCES sales_reps(id) NOT NULL,
    product_id UUID REFERENCES products(id),
    
    -- Amounts
    amount DECIMAL(15, 2) NOT NULL,
    
    -- Stages
    stage VARCHAR(50) NOT NULL,
    stage_entered_at TIMESTAMP,
    
    -- Dates
    created_at TIMESTAMP DEFAULT NOW(),
    expected_close_date DATE,
    actual_close_date DATE,
    
    -- Outcome
    is_won BOOLEAN,  -- NULL = still open
    close_reason VARCHAR(200),
    
    -- Metadata
    source VARCHAR(100),  -- How lead was acquired
    deal_type VARCHAR(50),  -- New, Expansion, Renewal
    
    -- Constraints
    CONSTRAINT valid_outcome CHECK (
        (is_won IS NULL AND actual_close_date IS NULL) OR
        (is_won IS NOT NULL AND actual_close_date IS NOT NULL)
    )
);

-- Deal stage history (for velocity analysis)
CREATE TABLE deal_stage_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id UUID REFERENCES deals(id) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    entered_at TIMESTAMP NOT NULL,
    exited_at TIMESTAMP,
    duration_days INTEGER GENERATED ALWAYS AS (
        EXTRACT(DAY FROM (COALESCE(exited_at, NOW()) - entered_at))
    ) STORED
);

CREATE INDEX idx_stage_history_deal ON deal_stage_history(deal_id);
```

### ML Tables

```sql
-- Model versioning
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL,  -- 'win_probability', 'time_to_close'
    version VARCHAR(20) NOT NULL,
    trained_at TIMESTAMP NOT NULL,
    training_data_cutoff DATE,
    metrics JSONB NOT NULL,
    -- Example: {"auc": 0.85, "brier_score": 0.12, "n_training": 5000}
    feature_list JSONB NOT NULL,
    artifact_path VARCHAR(500),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(model_type, version)
);

-- Deal scores (precomputed)
CREATE TABLE deal_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id UUID REFERENCES deals(id) NOT NULL,
    model_version_id UUID REFERENCES model_versions(id) NOT NULL,
    
    -- Scores
    win_probability DECIMAL(5, 4) NOT NULL,  -- 0.0000 to 1.0000
    risk_score INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    
    -- Time estimates (days from scoring date)
    est_close_p10 INTEGER,  -- Conservative
    est_close_p50 INTEGER,  -- Expected
    est_close_p90 INTEGER,  -- Optimistic
    
    scored_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(deal_id, model_version_id)
);

CREATE INDEX idx_scores_deal ON deal_scores(deal_id);
CREATE INDEX idx_scores_scored_at ON deal_scores(scored_at);

-- SHAP explanations (precomputed)
CREATE TABLE score_explanations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_score_id UUID REFERENCES deal_scores(id) NOT NULL,
    
    -- Feature contributions
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15, 4),  -- Actual value
    shap_value DECIMAL(10, 6) NOT NULL,  -- Contribution to prediction
    human_readable TEXT,  -- "Deal open 42 days (14 above average)"
    
    UNIQUE(deal_score_id, feature_name)
);

CREATE INDEX idx_explanations_score ON score_explanations(deal_score_id);

-- Forecast snapshots (for historical tracking)
CREATE TABLE forecast_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date DATE NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Revenue estimates
    revenue_p10 DECIMAL(15, 2) NOT NULL,
    revenue_p50 DECIMAL(15, 2) NOT NULL,
    revenue_p90 DECIMAL(15, 2) NOT NULL,
    
    -- Filters applied
    team VARCHAR(100),
    product_id UUID REFERENCES products(id),
    
    -- Metadata
    num_deals INTEGER NOT NULL,
    model_version_id UUID REFERENCES model_versions(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_forecast_date ON forecast_snapshots(snapshot_date);
```

---

## 3. Feature Dictionary

### Features for Win Probability Model

| Feature | Type | Description | Computation |
|---------|------|-------------|-------------|
| `days_open` | int | Days since deal created | `NOW() - created_at` |
| `days_in_stage` | int | Days in current stage | `NOW() - stage_entered_at` |
| `stage_numeric` | int | Stage as ordinal (1-7) | Mapping from stage name |
| `amount_log` | float | Log of deal amount | `log(amount + 1)` |
| `amount_percentile` | float | Percentile vs all deals | Rolling window calculation |
| `rep_win_rate` | float | Rep's historical win rate | Last 12 months closed deals |
| `rep_avg_deal_size` | float | Rep's average deal size | Last 12 months won deals |
| `product_win_rate` | float | Product category win rate | Last 12 months |
| `account_prev_deals` | int | Number of prior deals with account | All-time count |
| `account_prev_wins` | int | Number of won deals with account | All-time count |
| `stage_velocity` | float | Days in stage vs peers | Compared to won deals |
| `deal_velocity` | float | Overall speed vs peers | Compared to won deals |
| `num_stage_changes` | int | Times stage has changed | From stage history |
| `has_expected_close` | bool | Expected close date set | Boolean flag |
| `days_to_expected` | int | Days until expected close | `expected_close_date - NOW()` |
| `is_overdue` | bool | Past expected close date | Boolean flag |
| `deal_type_encoded` | int | New/Expansion/Renewal | One-hot or ordinal |
| `source_encoded` | int | Lead source category | One-hot or target encode |

### Features for Time-to-Close Model

Same features plus:

| Feature | Type | Description |
|---------|------|-------------|
| `stage_avg_duration` | float | Average days deals spend in current stage |
| `remaining_stages` | int | Number of stages until Closed Won |
| `similar_deal_avg_close` | float | Avg close time for similar amount/product |

### Feature Engineering Code

```python
class FeatureEngineer:
    """Compute features for ML models"""
    
    def __init__(self, db_session):
        self.db = db_session
        self._cache = {}
    
    def compute_features(self, deal: Deal) -> dict:
        """Generate all features for a single deal"""
        
        return {
            # Time features
            "days_open": (datetime.utcnow() - deal.created_at).days,
            "days_in_stage": (datetime.utcnow() - deal.stage_entered_at).days,
            "stage_numeric": self._stage_to_numeric(deal.stage),
            
            # Amount features
            "amount_log": np.log1p(deal.amount),
            "amount_percentile": self._get_amount_percentile(deal.amount),
            
            # Rep features
            "rep_win_rate": self._get_rep_win_rate(deal.owner_id),
            "rep_avg_deal_size": self._get_rep_avg_deal(deal.owner_id),
            
            # Product features
            "product_win_rate": self._get_product_win_rate(deal.product_id),
            
            # Account features
            "account_prev_deals": self._get_account_deal_count(deal.account_id),
            "account_prev_wins": self._get_account_win_count(deal.account_id),
            
            # Velocity features
            "stage_velocity": self._compute_stage_velocity(deal),
            "deal_velocity": self._compute_deal_velocity(deal),
            
            # Other
            "num_stage_changes": self._get_stage_change_count(deal.id),
            "has_expected_close": deal.expected_close_date is not None,
            "days_to_expected": self._days_to_expected(deal),
            "is_overdue": self._is_overdue(deal),
        }
    
    @staticmethod
    def _stage_to_numeric(stage: str) -> int:
        stages = {
            "Prospecting": 1,
            "Qualification": 2,
            "Needs Analysis": 3,
            "Value Proposition": 4,
            "Negotiation": 5,
            "Closed Won": 6,
            "Closed Lost": 0,
        }
        return stages.get(stage, -1)
```

---

## 4. Data Quality Requirements

### Validation Rules

```python
class DataValidator:
    """Validate incoming CRM data"""
    
    def validate_deal(self, deal: dict) -> list[str]:
        errors = []
        
        # Required fields
        if not deal.get("name"):
            errors.append("Deal name is required")
        if not deal.get("amount") or deal["amount"] <= 0:
            errors.append("Amount must be positive")
        if not deal.get("stage"):
            errors.append("Stage is required")
        
        # Logical checks
        if deal.get("is_won") is True and deal.get("actual_close_date") is None:
            errors.append("Won deals must have close date")
        
        if deal.get("expected_close_date"):
            if deal["expected_close_date"] < deal.get("created_at", datetime.min):
                errors.append("Expected close cannot be before creation")
        
        # Range checks
        if deal.get("amount", 0) > 10_000_000:
            errors.append("Amount suspiciously high - verify")
        
        return errors
```

### Data Freshness Requirements

| Data | Max Staleness | Refresh Frequency |
|------|---------------|-------------------|
| Deals | 1 day | Daily batch or near-real-time |
| Scores | 4 hours | Every 4 hours |
| Forecasts | 1 day | Daily |
| Rep stats | 1 week | Weekly |

---

## 5. Sample Data for Development

### Seed Script

```python
def seed_demo_data():
    """Create realistic demo dataset"""
    
    # Create accounts
    accounts = [
        {"name": "Acme Corporation", "industry": "Technology", "region": "West"},
        {"name": "GlobalTech Inc", "industry": "Software", "region": "East"},
        {"name": "Innovate Labs", "industry": "Healthcare", "region": "Central"},
        # ... more accounts
    ]
    
    # Create reps
    reps = [
        {"name": "Sarah Johnson", "team": "Enterprise", "hire_date": "2021-03-15"},
        {"name": "Michael Chen", "team": "Mid-Market", "hire_date": "2022-01-10"},
        # ... more reps
    ]
    
    # Create deals with realistic distribution
    # - 60% in early stages
    # - 25% in middle stages
    # - 15% in late stages
    # - Mix of risk levels
```

### Demo Scenarios

1. **High-value deal at risk**: $450K deal, 45 days open, stalled in Negotiation
2. **Healthy deal**: $180K deal, 20 days open, moving well through stages
3. **Sleeper risk**: $280K deal looks fine but rep has low win rate for this product

