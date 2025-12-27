"""
SHAP Explainability Layer
Phase 1A: Core ML Pipeline

This module provides model explanations using SHAP values.
Key responsibilities:
- Explain individual deal predictions
- Generate human-readable explanation text
- Support batch explanation for performance
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logging.warning("shap not installed. Explanations unavailable.")

logger = logging.getLogger(__name__)


# Human-readable templates for feature explanations
EXPLANATION_TEMPLATES = {
    "days_in_engaging": {
        "positive": "Deal has been open {value:.0f} days, longer than typical ({avg:.0f} day avg), reducing win likelihood",
        "negative": "Deal moving quickly at {value:.0f} days (vs {avg:.0f} day avg), increasing win likelihood",
    },
    "rep_historical_win_rate": {
        "positive": "Rep's historical win rate of {value:.0%} is above average, boosting prediction",
        "negative": "Rep's historical win rate of {value:.0%} is below average, lowering prediction",
    },
    "product_historical_win_rate": {
        "positive": "This product has historically won {value:.0%} of deals, above average",
        "negative": "This product has historically won {value:.0%} of deals, below average",
    },
    "product_sales_price": {
        "positive": "Higher price point (${value:,.0f}) associated with higher win rate",
        "negative": "Higher price point (${value:,.0f}) associated with lower win rate",
    },
    "account_revenue": {
        "positive": "Larger accounts (${value:.0f}M revenue) tend to close more often",
        "negative": "Account size (${value:.0f}M revenue) suggests lower win rate",
    },
    "account_employees": {
        "positive": "Larger organizations ({value:.0f} employees) tend to convert",
        "negative": "Organization size ({value:.0f} employees) less favorable",
    },
    "account_age": {
        "positive": "Established company ({value:.0f} years) has positive signal",
        "negative": "Company age ({value:.0f} years) has negative signal",
    },
    "rep_deal_count": {
        "positive": "Rep's experience ({value:.0f} prior deals) is positive",
        "negative": "Rep's limited experience ({value:.0f} prior deals) is a factor",
    },
    "sales_agent": {
        "positive": "This rep has historically strong performance",
        "negative": "This rep's historical performance is below average",
    },
    "product": {
        "positive": "This product line performs well",
        "negative": "This product line has lower typical win rates",
    },
    "account_sector": {
        "positive": "The {value} sector has favorable win rates",
        "negative": "The {value} sector has lower typical win rates",
    },
    "product_series": {
        "positive": "The {value} series performs above average",
        "negative": "The {value} series performs below average",
    },
    "manager": {
        "positive": "Deals under this manager tend to close",
        "negative": "Deals under this manager close less often",
    },
    "regional_office": {
        "positive": "The {value} region has strong win rates",
        "negative": "The {value} region has lower win rates",
    },
}

# Average values for comparison (from Phase 1A analysis)
FEATURE_AVERAGES = {
    "days_in_engaging": 48,
    "rep_historical_win_rate": 0.63,
    "product_historical_win_rate": 0.63,
    "product_sales_price": 1491,
    "account_revenue": 2000,
    "account_employees": 5000,
    "account_age": 25,
    "rep_deal_count": 100,
}


@dataclass
class Explanation:
    """
    Structured explanation for a single deal prediction.
    """
    opportunity_id: str
    win_probability: float
    base_value: float
    feature_contributions: Dict[str, float]
    feature_values: Dict[str, Any]
    top_positive: List[Tuple[str, float, str]]  # (feature, contribution, text)
    top_negative: List[Tuple[str, float, str]]  # (feature, contribution, text)
    summary_text: str


class DealExplainer:
    """
    SHAP-based explainer for win probability predictions.
    
    Uses TreeExplainer for LightGBM models (exact and fast).
    Generates both numerical contributions and human-readable text.
    """
    
    def __init__(
        self,
        model: Any,
        background_data: pd.DataFrame,
        feature_names: List[str],
        max_background_samples: int = 100
    ):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (LightGBM Booster)
            background_data: Representative data for SHAP background
            feature_names: List of feature names
            max_background_samples: Max samples for background (performance)
        """
        if not HAS_SHAP:
            raise ImportError("shap package required for explanations")
        
        self.model = model
        self.feature_names = feature_names
        
        # Sample background data for performance
        if len(background_data) > max_background_samples:
            background_data = background_data.sample(
                max_background_samples, 
                random_state=42
            )
        
        self.background_data = background_data
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(model)
        self.expected_value = self.explainer.expected_value
        
        logger.info(f"Explainer initialized. Base value: {self.expected_value:.4f}")
    
    def _get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            SHAP values array
        """
        shap_values = self.explainer.shap_values(X[self.feature_names])
        return shap_values
    
    def _generate_explanation_text(
        self,
        feature: str,
        shap_value: float,
        feature_value: Any
    ) -> str:
        """
        Generate human-readable explanation for a feature.
        
        Args:
            feature: Feature name
            shap_value: SHAP contribution value
            feature_value: Actual feature value
            
        Returns:
            Explanation text
        """
        direction = "positive" if shap_value > 0 else "negative"
        
        if feature in EXPLANATION_TEMPLATES:
            template = EXPLANATION_TEMPLATES[feature][direction]
            avg = FEATURE_AVERAGES.get(feature, 0)
            
            try:
                return template.format(value=feature_value, avg=avg)
            except (ValueError, KeyError):
                return f"{feature}: {'increases' if shap_value > 0 else 'decreases'} win probability"
        
        # Default template
        return f"{feature}: {'increases' if shap_value > 0 else 'decreases'} win probability by {abs(shap_value):.3f}"
    
    def explain_deal(
        self,
        features: pd.DataFrame,
        opportunity_id: str = "unknown",
        top_k: int = 5
    ) -> Explanation:
        """
        Explain prediction for a single deal.
        
        Args:
            features: Single-row DataFrame with features
            opportunity_id: Deal identifier
            top_k: Number of top features to highlight
            
        Returns:
            Explanation object
        """
        # Get SHAP values
        shap_values = self._get_shap_values(features)[0]
        
        # Build feature contributions dict
        contributions = dict(zip(self.feature_names, shap_values))
        
        # Get feature values
        feature_values = features.iloc[0].to_dict()
        
        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top positive and negative
        top_positive = []
        top_negative = []
        
        for feature, contrib in sorted_features:
            value = feature_values.get(feature, 0)
            text = self._generate_explanation_text(feature, contrib, value)
            
            if contrib > 0:
                top_positive.append((feature, contrib, text))
            else:
                top_negative.append((feature, contrib, text))
        
        top_positive = top_positive[:top_k]
        top_negative = top_negative[:top_k]
        
        # Compute predicted probability
        # SHAP values sum to prediction difference from base
        pred_logit = self.expected_value + sum(shap_values)
        win_probability = 1 / (1 + np.exp(-pred_logit))  # Sigmoid for binary
        
        # Generate summary
        summary_parts = []
        if top_positive:
            summary_parts.append(f"Key drivers: {top_positive[0][2]}")
        if top_negative:
            summary_parts.append(f"Risk factor: {top_negative[0][2]}")
        summary_text = " ".join(summary_parts)
        
        return Explanation(
            opportunity_id=opportunity_id,
            win_probability=float(win_probability),
            base_value=float(self.expected_value),
            feature_contributions=contributions,
            feature_values=feature_values,
            top_positive=top_positive,
            top_negative=top_negative,
            summary_text=summary_text
        )
    
    def explain_batch(
        self,
        features: pd.DataFrame,
        opportunity_ids: Optional[List[str]] = None,
        top_k: int = 3
    ) -> List[Explanation]:
        """
        Explain predictions for multiple deals.
        
        More efficient than calling explain_deal repeatedly.
        
        Args:
            features: DataFrame with multiple rows
            opportunity_ids: Optional list of deal IDs
            top_k: Number of top features per deal
            
        Returns:
            List of Explanation objects
        """
        if opportunity_ids is None:
            opportunity_ids = [f"deal_{i}" for i in range(len(features))]
        
        logger.info(f"Generating explanations for {len(features)} deals...")
        
        # Batch SHAP computation
        shap_values = self._get_shap_values(features)
        
        explanations = []
        for i in range(len(features)):
            row_features = features.iloc[[i]]
            row_shap = shap_values[i]
            
            # Build explanation
            contributions = dict(zip(self.feature_names, row_shap))
            feature_values = features.iloc[i].to_dict()
            
            sorted_features = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            top_positive = []
            top_negative = []
            
            for feature, contrib in sorted_features:
                value = feature_values.get(feature, 0)
                text = self._generate_explanation_text(feature, contrib, value)
                
                if contrib > 0:
                    top_positive.append((feature, contrib, text))
                else:
                    top_negative.append((feature, contrib, text))
            
            pred_logit = self.expected_value + sum(row_shap)
            win_probability = 1 / (1 + np.exp(-pred_logit))
            
            explanation = Explanation(
                opportunity_id=opportunity_ids[i],
                win_probability=float(win_probability),
                base_value=float(self.expected_value),
                feature_contributions=contributions,
                feature_values=feature_values,
                top_positive=top_positive[:top_k],
                top_negative=top_negative[:top_k],
                summary_text=""  # Skip for batch performance
            )
            explanations.append(explanation)
        
        logger.info(f"Generated {len(explanations)} explanations")
        
        return explanations
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get global feature importance from SHAP.
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with feature importance
        """
        shap_values = self._get_shap_values(X)
        
        importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        
        return importance


def create_explainer_from_trainer(
    trainer: Any,
    train_data: pd.DataFrame
) -> DealExplainer:
    """
    Create explainer from a trained WinProbabilityTrainer.
    
    Args:
        trainer: WinProbabilityTrainer instance
        train_data: Training data for background
        
    Returns:
        DealExplainer instance
    """
    feature_cols = trainer.all_features
    
    # Need to encode training data same way
    encoded_data = trainer._encode_features(train_data[feature_cols], fit=False)
    
    return DealExplainer(
        model=trainer.lgbm_model,
        background_data=encoded_data[feature_cols],
        feature_names=feature_cols
    )


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    if not HAS_SHAP:
        print("SHAP not installed. Run: pip install shap")
    else:
        # This would normally use the trained model
        print("SHAP explainer module loaded successfully")
        print("See win_probability.py for integration example")

