# Model Specification

> Technical specifications for ML models, evaluation criteria, and deployment

---

## 1. Model Overview

| Model | Purpose | Output | Primary Algorithm |
|-------|---------|--------|-------------------|
| **Win Probability** | Predict deal close likelihood | P(won) ∈ [0, 1] | LightGBM (calibrated) |
| **Time-to-Close** | Estimate closing timeline | Distribution of days | LightGBM Regressor |
| **Risk Score** | Composite risk ranking | Score ∈ [0, 100] | Formula-based |

---

## 2. Win Probability Model

### 2.1 Problem Formulation

- **Task**: Binary classification
- **Target**: `is_won` (1 = Closed Won, 0 = Closed Lost)
- **Training data**: Historical closed deals only
- **Inference**: Score all open deals

### 2.2 Algorithm Selection

| Algorithm | Pros | Cons | Use |
|-----------|------|------|-----|
| Logistic Regression | Interpretable, fast | Less accurate | Baseline |
| LightGBM | Accurate, handles categoricals | Less interpretable | Primary |
| XGBoost | Battle-tested | Slower than LightGBM | Alternative |
| Neural Network | Flexible | Overkill, less interpretable | Skip |

**Decision**: LightGBM as primary, Logistic Regression as interpretable baseline.

### 2.3 Training Pipeline

```python
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

class WinProbabilityTrainer:
    """Train win probability model with proper validation"""
    
    def __init__(self):
        self.base_model = LGBMClassifier(
            objective="binary",
            metric="auc",
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            n_estimators=500,
            early_stopping_rounds=50,
            verbose=-1,
        )
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_split: float = 0.2
    ) -> CalibratedClassifierCV:
        """
        Train and calibrate the model
        
        Uses time-based split to prevent data leakage
        """
        # Time-based train/val split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train base model
        self.base_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method="isotonic",  # or "sigmoid"
            cv="prefit"
        )
        calibrated_model.fit(X_val, y_val)
        
        return calibrated_model
    
    def evaluate(
        self, 
        model, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> dict:
        """Comprehensive evaluation metrics"""
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        return {
            "auc_roc": roc_auc_score(y_test, y_prob),
            "auc_pr": average_precision_score(y_test, y_prob),
            "brier_score": brier_score_loss(y_test, y_prob),
            "log_loss": log_loss(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "expected_calibration_error": self._compute_ece(y_test, y_prob),
        }
    
    def _compute_ece(self, y_true, y_prob, n_bins=10):
        """Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(y_prob[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece
```

### 2.4 Calibration

**Why calibration matters**: A model might have good AUC but poor calibration. We need probabilities to be *reliable* — a 70% prediction should win ~70% of the time.

```python
def plot_calibration(y_true, y_prob, n_bins=10):
    """Generate reliability diagram"""
    
    from sklearn.calibration import calibration_curve
    
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    
    # Model calibration
    ax.plot(mean_predicted, fraction_positive, 's-', label='Model')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend()
    
    return fig
```

### 2.5 Hyperparameter Tuning

```python
from optuna import create_study

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
    }
    
    model = LGBMClassifier(**params, n_estimators=500)
    
    # Cross-validation
    cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    
    return scores.mean()

study = create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## 3. Time-to-Close Model

### 3.1 Approach Options

| Approach | Description | Complexity |
|----------|-------------|------------|
| **Simple Regression** | Predict log(days_to_close) | Low |
| **Quantile Regression** | Predict P10, P50, P90 directly | Medium |
| **Survival Analysis** | Handle censoring properly | High |

**Recommendation**: Quantile regression for good balance of accuracy and simplicity.

### 3.2 Quantile Regression

```python
from lightgbm import LGBMRegressor

class TimeToClosePredictor:
    """Predict closing time distribution"""
    
    def __init__(self):
        self.models = {}
        for quantile in [0.1, 0.5, 0.9]:
            self.models[quantile] = LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                num_leaves=31,
                learning_rate=0.05,
            )
    
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train models for each quantile"""
        for quantile, model in self.models.items():
            model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> dict:
        """Return P10, P50, P90 predictions"""
        return {
            "p10": self.models[0.1].predict(X),
            "p50": self.models[0.5].predict(X),
            "p90": self.models[0.9].predict(X),
        }
```

### 3.3 Handling Open Deals

Training data only includes closed deals. For inference on open deals:

1. Use time-since-creation as reference
2. Predict *remaining* days to close
3. Add to current date for expected close window

---

## 4. Risk Score

### 4.1 Formula

Risk score is a **composite metric**, not a learned model:

```python
class RiskScorer:
    """
    Compute composite risk score (0-100)
    
    Components:
    - Win probability (inverted): lower prob = higher risk
    - Value exposure: larger deals = more risk if lost
    - Time risk: slower than peers = higher risk  
    - Stagnation: stuck in stage = higher risk
    """
    
    WEIGHTS = {
        "win_prob": 0.40,
        "value": 0.20,
        "time": 0.25,
        "stagnation": 0.15,
    }
    
    def compute(
        self,
        deal: Deal,
        win_prob: float,
        peer_stats: PeerStats
    ) -> int:
        
        # Win probability component (inverted)
        prob_risk = 1 - win_prob
        
        # Value component (normalized by max deal)
        value_risk = min(deal.amount / peer_stats.max_deal_value, 1.0)
        
        # Time component (vs peer average)
        days_open = (datetime.utcnow() - deal.created_at).days
        time_ratio = days_open / max(peer_stats.avg_days_to_close, 1)
        time_risk = min(time_ratio, 2.0) / 2.0  # Cap at 2x average
        
        # Stagnation component
        days_in_stage = (datetime.utcnow() - deal.stage_entered_at).days
        stage_avg = peer_stats.avg_days_per_stage.get(deal.stage, 14)
        stag_ratio = days_in_stage / max(stage_avg, 1)
        stagnation_risk = min(stag_ratio, 2.0) / 2.0
        
        # Weighted combination
        raw_risk = (
            self.WEIGHTS["win_prob"] * prob_risk +
            self.WEIGHTS["value"] * value_risk +
            self.WEIGHTS["time"] * time_risk +
            self.WEIGHTS["stagnation"] * stagnation_risk
        )
        
        # Scale to 0-100
        return int(round(raw_risk * 100))
```

### 4.2 Weight Justification

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Win Probability | 40% | Most direct indicator of outcome |
| Time Risk | 25% | Deals slowing down is a key warning sign |
| Value Risk | 20% | Larger deals warrant more attention |
| Stagnation | 15% | Stage-specific slowdown indicator |

Weights should be validated against historical outcomes and tuned based on business feedback.

---

## 5. Model Evaluation Criteria

### 5.1 Minimum Thresholds for Production

| Metric | Threshold | Priority |
|--------|-----------|----------|
| AUC-ROC | > 0.75 | Must-have |
| Brier Score | < 0.20 | Must-have |
| ECE (Calibration) | < 0.05 | Must-have |
| AUC-PR | > 0.60 | Nice-to-have |

### 5.2 Evaluation Notebook Outline

```markdown
# Model Evaluation Report

## 1. Dataset Statistics
- Training set: N deals
- Test set: M deals  
- Win rate: X%
- Time period: YYYY-MM to YYYY-MM

## 2. Performance Metrics
[Table of all metrics]

## 3. Calibration Analysis
[Reliability diagram]
[ECE by probability bucket]

## 4. Feature Importance
[SHAP summary plot]
[Top 10 features]

## 5. Error Analysis
[False positive examples]
[False negative examples]
[Edge cases]

## 6. Comparison to Baseline
[vs Logistic Regression]
[vs Simple heuristics]

## 7. Recommendations
[Model selected]
[Areas for improvement]
```

---

## 6. Model Deployment

### 6.1 Artifact Structure

```python
@dataclass
class ModelArtifact:
    """Complete model artifact for deployment"""
    
    model: Any  # Trained model object
    version: str  # Semantic version
    model_type: str  # "win_probability" or "time_to_close"
    trained_at: datetime
    training_data_cutoff: date
    feature_names: list[str]
    feature_dtypes: dict[str, str]
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    
    def save(self, path: str):
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> "ModelArtifact":
        return joblib.load(path)
```

### 6.2 Model Loading

```python
class ModelLoader:
    """Load and manage model versions"""
    
    def __init__(self, artifact_dir: str = "models/artifacts"):
        self.artifact_dir = Path(artifact_dir)
    
    def load_latest(self, model_type: str) -> ModelArtifact:
        """Load most recent active model"""
        
        pattern = f"{model_type}_*.pkl"
        files = sorted(self.artifact_dir.glob(pattern), reverse=True)
        
        if not files:
            raise ModelNotFoundError(f"No model found for {model_type}")
        
        return ModelArtifact.load(files[0])
    
    def load_version(self, model_type: str, version: str) -> ModelArtifact:
        """Load specific model version"""
        
        path = self.artifact_dir / f"{model_type}_{version}.pkl"
        
        if not path.exists():
            raise ModelNotFoundError(f"Version {version} not found")
        
        return ModelArtifact.load(path)
```

### 6.3 Drift Detection

```python
class DriftDetector:
    """Simple drift detection for feature distributions"""
    
    def __init__(self, reference_stats: dict):
        self.reference = reference_stats
    
    def check_drift(self, current_data: pd.DataFrame) -> list[DriftWarning]:
        """Compare current data to training distribution"""
        
        warnings = []
        
        for col in current_data.columns:
            if col not in self.reference:
                continue
            
            ref = self.reference[col]
            curr_mean = current_data[col].mean()
            curr_std = current_data[col].std()
            
            # Check mean shift
            z_score = abs(curr_mean - ref["mean"]) / ref["std"]
            if z_score > 2:
                warnings.append(DriftWarning(
                    feature=col,
                    type="mean_shift",
                    severity="high" if z_score > 3 else "medium",
                    message=f"{col} mean shifted by {z_score:.1f} std deviations"
                ))
            
            # Check variance change
            var_ratio = curr_std / ref["std"]
            if var_ratio > 2 or var_ratio < 0.5:
                warnings.append(DriftWarning(
                    feature=col,
                    type="variance_change",
                    severity="medium",
                    message=f"{col} variance changed by {var_ratio:.1f}x"
                ))
        
        return warnings
```

---

## 7. Explainability

### 7.1 SHAP Integration

```python
import shap

class ModelExplainer:
    """Generate SHAP explanations for predictions"""
    
    def __init__(self, model, background_data: pd.DataFrame):
        """
        Args:
            model: Trained tree model (LightGBM)
            background_data: Sample of training data for baseline
        """
        self.explainer = shap.TreeExplainer(model)
        self.background = background_data
    
    def explain(self, X: pd.DataFrame) -> list[dict]:
        """Generate explanations for each row"""
        
        shap_values = self.explainer.shap_values(X)
        
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        explanations = []
        for i in range(len(X)):
            contributions = dict(zip(X.columns, shap_values[i]))
            
            # Sort by absolute contribution
            sorted_contrib = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            explanations.append({
                "base_value": self.explainer.expected_value,
                "contributions": sorted_contrib,
                "prediction": self.explainer.expected_value + sum(shap_values[i])
            })
        
        return explanations
    
    def top_drivers(self, X: pd.DataFrame, n: int = 5) -> list[list[tuple]]:
        """Get top N drivers for each prediction"""
        
        explanations = self.explain(X)
        return [exp["contributions"][:n] for exp in explanations]
```

### 7.2 Human-Readable Explanations

```python
EXPLANATION_TEMPLATES = {
    "days_open": {
        "positive": "Deal has been open {value} days ({diff:+d} vs average), suggesting momentum loss",
        "negative": "Deal is moving faster than average ({diff:+d} days ahead)",
    },
    "rep_win_rate": {
        "positive": "Rep's historical win rate ({value:.0%}) is above average",
        "negative": "Rep's historical win rate ({value:.0%}) is below average",
    },
    "amount_percentile": {
        "positive": "Deal is larger than {value:.0%} of deals, attracting more scrutiny",
        "negative": "Deal size is typical, reducing complexity risk",
    },
    "stage_velocity": {
        "positive": "Moving through {stage} faster than typical deals",
        "negative": "Spending longer in {stage} than comparable won deals",
    },
}

def generate_human_explanation(
    feature: str, 
    value: float, 
    shap_value: float,
    context: dict
) -> str:
    """Convert SHAP value to human-readable text"""
    
    templates = EXPLANATION_TEMPLATES.get(feature, {})
    direction = "positive" if shap_value > 0 else "negative"
    
    template = templates.get(direction, f"{feature}: {value:.2f}")
    
    return template.format(value=value, **context)
```

