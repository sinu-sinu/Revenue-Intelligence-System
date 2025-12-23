# Phase 1A: Core ML Pipeline

> **Duration**: ~3 days  
> **Goal**: Trained, calibrated models with explainability — the intellectual core of the project

---

## Checklist

### 1A.1 Feature Engineering
- [ ] Create `core/data/features.py`:
  ```python
  class FeatureEngineer:
      def compute_deal_features(self, deal: Deal) -> dict:
          """
          Features for win probability model
          """
          return {
              "days_open": ...,
              "days_in_current_stage": ...,
              "deal_size_percentile": ...,
              "rep_historical_win_rate": ...,
              "product_category_win_rate": ...,
              "stage_numeric": ...,
              "num_stage_changes": ...,
              "velocity_vs_peers": ...,
          }
  ```
- [ ] Implement time-based features (critical for sales)
- [ ] Implement peer comparison features
- [ ] Create feature validation (ranges, nulls)
- [ ] Document each feature in `06_DATA_SPEC.md`

### 1A.2 Win Probability Model
- [ ] Create `models/training/win_probability.py`
- [ ] Implement baseline Logistic Regression:
  - Interpretable coefficients
  - Good for comparison
- [ ] Implement primary LightGBM model:
  ```python
  params = {
      "objective": "binary",
      "metric": "auc",
      "num_leaves": 31,
      "learning_rate": 0.05,
      "feature_fraction": 0.8,
  }
  ```
- [ ] **Calibrate probabilities** (critical!):
  ```python
  from sklearn.calibration import CalibratedClassifierCV
  calibrated_model = CalibratedClassifierCV(model, method='isotonic')
  ```
- [ ] Validate calibration with reliability diagrams
- [ ] Save model with version metadata:
  ```python
  artifact = {
      "model": model,
      "version": "1.0.0",
      "trained_at": datetime.utcnow(),
      "features": feature_list,
      "metrics": {"auc": 0.85, "brier": 0.12}
  }
  ```

### 1A.3 Time-to-Close Model
- [ ] Create `models/training/time_to_close.py`
- [ ] Approach options:
  - **Simple**: Regression on log(days_to_close)
  - **Better**: Survival analysis with `lifelines`
- [ ] Output: distribution of close dates, not point estimate
- [ ] Used for P10/P50/P90 forecast bands

### 1A.4 Risk Score Computation
- [ ] Create `core/scoring/risk_scorer.py`:
  ```python
  class RiskScorer:
      def compute_risk(self, deal: Deal, win_prob: float) -> RiskScore:
          """
          Composite risk score 0-100
          
          Components:
          - Win probability (inverted)
          - Deal value (higher = more risky if stuck)
          - Time vs peers (slower = more risky)
          - Stage stagnation
          """
          # Weighted combination, normalized
          raw_risk = (
              0.40 * (1 - win_prob) +
              0.25 * self._time_risk(deal) +
              0.20 * self._value_risk(deal) +
              0.15 * self._stagnation_risk(deal)
          )
          return int(raw_risk * 100)
  ```
- [ ] Document risk formula clearly
- [ ] Make weights configurable

### 1A.5 Explainability Layer
- [ ] Create `core/explanations/shap_explainer.py`:
  ```python
  import shap
  
  class DealExplainer:
      def __init__(self, model, background_data):
          self.explainer = shap.TreeExplainer(model)
      
      def explain_deal(self, features: pd.DataFrame) -> Explanation:
          shap_values = self.explainer.shap_values(features)
          return Explanation(
              feature_contributions=dict(zip(
                  features.columns, 
                  shap_values[0]
              )),
              base_value=self.explainer.expected_value
          )
  ```
- [ ] Precompute SHAP for all open deals (batch)
- [ ] Store explanations in database
- [ ] Create human-readable explanation templates:
  ```python
  TEMPLATES = {
      "days_open": "Deal has been open {value} days ({comparison} vs avg)",
      "rep_win_rate": "Rep's historical win rate is {value}%",
  }
  ```

### 1A.6 Forecasting Engine
- [ ] Create `core/forecasting/revenue_forecast.py`:
  ```python
  class RevenueForecast:
      def generate_forecast(
          self, 
          deals: list[Deal], 
          horizon_weeks: int = 12
      ) -> ForecastResult:
          """
          Monte Carlo simulation for revenue forecast
          
          For each deal:
          1. Sample from win probability (Bernoulli)
          2. Sample close date from time model
          3. Aggregate to weekly/monthly buckets
          4. Repeat N times for distribution
          """
          # Returns P10, P50, P90 bands
  ```
- [ ] Implement Monte Carlo simulation (1000+ runs)
- [ ] Group by configurable time periods
- [ ] Cache forecast snapshots for historical comparison

### 1A.7 Model Evaluation & Validation
- [ ] Create `models/evaluation/evaluate.py`
- [ ] Implement metrics:
  - AUC-ROC (discrimination)
  - Brier Score (calibration)
  - Expected Calibration Error
  - Precision/Recall at thresholds
- [ ] Create evaluation notebook with visualizations
- [ ] Implement simple drift detection:
  ```python
  def check_feature_drift(current_data, training_data):
      """Compare distributions, flag if KS > threshold"""
  ```

---

## Acceptance Criteria

✅ Win probability model trained with AUC > 0.75  
✅ Probabilities are calibrated (reliability diagram looks good)  
✅ SHAP explanations generated for all open deals  
✅ Risk scores computed for all deals (0-100 scale)  
✅ Forecast produces sensible P10/P50/P90 bands  
✅ All models versioned with metadata  

---

## Portfolio Emphasis

This phase is **the intellectual core**. Interviewers will ask about:

1. **Why calibration matters**: "A 70% probability should win 70% of the time"
2. **SHAP vs LIME**: "SHAP has theoretical guarantees from game theory"
3. **Risk formula design**: "Weights came from domain knowledge + backtesting"
4. **Uncertainty in forecasts**: "Point estimates are dishonest — bands are honest"

Document your thinking in notebooks and code comments.

