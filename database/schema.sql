-- Revenue Intelligence System Database Schema
-- Version: 1.0.0
-- Description: Core tables for CRM data and ML pipeline

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- CORE BUSINESS TABLES
-- ============================================================================

-- Accounts (customers/prospects)
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    industry VARCHAR(100),
    employee_count INTEGER,
    annual_revenue DECIMAL(15, 2),
    region VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_accounts_name ON accounts(name);
CREATE INDEX idx_accounts_industry ON accounts(industry);
CREATE INDEX idx_accounts_region ON accounts(region);

-- Sales representatives
CREATE TABLE sales_reps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(200) UNIQUE,
    team VARCHAR(100),
    manager_id UUID REFERENCES sales_reps(id),
    hire_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_sales_reps_team ON sales_reps(team);
CREATE INDEX idx_sales_reps_manager ON sales_reps(manager_id);
CREATE INDEX idx_sales_reps_active ON sales_reps(is_active);

-- Products/offerings
CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    base_price DECIMAL(12, 2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_active ON products(is_active);

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
    stage_entered_at TIMESTAMP DEFAULT NOW(),
    
    -- Dates
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    expected_close_date DATE,
    actual_close_date DATE,
    
    -- Outcome
    is_won BOOLEAN,  -- NULL = still open, TRUE = won, FALSE = lost
    close_reason VARCHAR(200),
    
    -- Metadata
    source VARCHAR(100),  -- How lead was acquired
    deal_type VARCHAR(50) DEFAULT 'New',  -- New, Expansion, Renewal
    
    -- Constraints
    CONSTRAINT valid_outcome CHECK (
        (is_won IS NULL AND actual_close_date IS NULL) OR
        (is_won IS NOT NULL AND actual_close_date IS NOT NULL)
    ),
    CONSTRAINT positive_amount CHECK (amount > 0)
);

CREATE INDEX idx_deals_account ON deals(account_id);
CREATE INDEX idx_deals_owner ON deals(owner_id);
CREATE INDEX idx_deals_product ON deals(product_id);
CREATE INDEX idx_deals_stage ON deals(stage);
CREATE INDEX idx_deals_created_at ON deals(created_at);
CREATE INDEX idx_deals_is_won ON deals(is_won);
CREATE INDEX idx_deals_open ON deals(is_won) WHERE is_won IS NULL;

-- Deal stage history (for velocity analysis)
CREATE TABLE deal_stage_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id UUID REFERENCES deals(id) ON DELETE CASCADE NOT NULL,
    stage VARCHAR(50) NOT NULL,
    entered_at TIMESTAMP NOT NULL,
    exited_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_stage_history_deal ON deal_stage_history(deal_id);
CREATE INDEX idx_stage_history_entered ON deal_stage_history(entered_at);

-- ============================================================================
-- ML TABLES
-- ============================================================================

-- Model versioning
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL,  -- 'win_probability', 'time_to_close', etc.
    version VARCHAR(20) NOT NULL,
    trained_at TIMESTAMP NOT NULL,
    training_data_cutoff DATE,
    metrics JSONB NOT NULL,  -- {"auc": 0.85, "brier_score": 0.12, "n_training": 5000}
    feature_list JSONB NOT NULL,  -- ["days_open", "amount_log", ...]
    artifact_path VARCHAR(500),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(model_type, version)
);

CREATE INDEX idx_model_versions_type ON model_versions(model_type);
CREATE INDEX idx_model_versions_active ON model_versions(is_active);

-- Deal scores (precomputed)
CREATE TABLE deal_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_id UUID REFERENCES deals(id) ON DELETE CASCADE NOT NULL,
    model_version_id UUID REFERENCES model_versions(id) NOT NULL,
    
    -- Scores
    win_probability DECIMAL(5, 4) NOT NULL CHECK (win_probability BETWEEN 0 AND 1),
    risk_score INTEGER NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    
    -- Time estimates (days from scoring date)
    est_close_p10 INTEGER,  -- Conservative (10th percentile)
    est_close_p50 INTEGER,  -- Expected (median)
    est_close_p90 INTEGER,  -- Optimistic (90th percentile)
    
    -- Metadata
    scored_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(deal_id, model_version_id)
);

CREATE INDEX idx_scores_deal ON deal_scores(deal_id);
CREATE INDEX idx_scores_model ON deal_scores(model_version_id);
CREATE INDEX idx_scores_scored_at ON deal_scores(scored_at);
CREATE INDEX idx_scores_risk ON deal_scores(risk_score DESC);

-- SHAP explanations (precomputed)
CREATE TABLE score_explanations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deal_score_id UUID REFERENCES deal_scores(id) ON DELETE CASCADE NOT NULL,
    
    -- Feature contributions
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15, 4),  -- Actual value of the feature
    shap_value DECIMAL(10, 6) NOT NULL,  -- Contribution to prediction
    human_readable TEXT,  -- "Deal open 42 days (14 days above average)"
    
    created_at TIMESTAMP DEFAULT NOW(),
    
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
    revenue_p10 DECIMAL(15, 2) NOT NULL,  -- Conservative
    revenue_p50 DECIMAL(15, 2) NOT NULL,  -- Expected
    revenue_p90 DECIMAL(15, 2) NOT NULL,  -- Optimistic
    
    -- Filters applied
    team VARCHAR(100),
    product_id UUID REFERENCES products(id),
    
    -- Metadata
    num_deals INTEGER NOT NULL,
    model_version_id UUID REFERENCES model_versions(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_forecast_date ON forecast_snapshots(snapshot_date);
CREATE INDEX idx_forecast_period ON forecast_snapshots(period_start, period_end);

-- ============================================================================
-- VIEWS FOR CONVENIENCE
-- ============================================================================

-- Open deals with latest scores
CREATE VIEW open_deals_with_scores AS
SELECT 
    d.id,
    d.name,
    d.amount,
    d.stage,
    d.created_at,
    d.stage_entered_at,
    d.expected_close_date,
    a.name AS account_name,
    sr.name AS owner_name,
    sr.team,
    p.name AS product_name,
    p.category AS product_category,
    ds.win_probability,
    ds.risk_score,
    ds.est_close_p50,
    ds.scored_at
FROM deals d
LEFT JOIN accounts a ON d.account_id = a.id
LEFT JOIN sales_reps sr ON d.owner_id = sr.id
LEFT JOIN products p ON d.product_id = p.id
LEFT JOIN LATERAL (
    SELECT * FROM deal_scores
    WHERE deal_id = d.id
    ORDER BY scored_at DESC
    LIMIT 1
) ds ON true
WHERE d.is_won IS NULL;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for deals table
CREATE TRIGGER update_deals_updated_at 
    BEFORE UPDATE ON deals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for accounts table
CREATE TRIGGER update_accounts_updated_at 
    BEFORE UPDATE ON accounts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- INITIAL DATA / REFERENCE DATA
-- ============================================================================

-- Standard sales stages (can be customized)
CREATE TABLE IF NOT EXISTS sales_stages (
    id SERIAL PRIMARY KEY,
    stage_name VARCHAR(50) NOT NULL UNIQUE,
    stage_order INTEGER NOT NULL UNIQUE,
    is_closed BOOLEAN DEFAULT FALSE
);

INSERT INTO sales_stages (stage_name, stage_order, is_closed) VALUES
    ('Prospecting', 1, FALSE),
    ('Qualification', 2, FALSE),
    ('Needs Analysis', 3, FALSE),
    ('Value Proposition', 4, FALSE),
    ('Negotiation', 5, FALSE),
    ('Closed Won', 6, TRUE),
    ('Closed Lost', 7, TRUE);

COMMENT ON TABLE accounts IS 'Customer and prospect accounts';
COMMENT ON TABLE sales_reps IS 'Sales team members';
COMMENT ON TABLE products IS 'Products and service offerings';
COMMENT ON TABLE deals IS 'Sales opportunities (open and closed)';
COMMENT ON TABLE deal_stage_history IS 'Historical record of stage transitions';
COMMENT ON TABLE model_versions IS 'ML model versioning and metadata';
COMMENT ON TABLE deal_scores IS 'Precomputed risk scores and win probabilities';
COMMENT ON TABLE score_explanations IS 'SHAP explanations for model predictions';
COMMENT ON TABLE forecast_snapshots IS 'Historical forecast snapshots for comparison';

