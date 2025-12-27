"""
Risk Score Computation
Phase 1A: Core ML Pipeline

This module computes composite risk scores for deals.
Risk indicates likelihood of deal failure or delay.

Key components:
- Win probability (inverted)
- Time in stage (slower = riskier)
- Deal value proxy (larger deals = more impact if lost)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskWeights:
    """
    Configurable weights for risk score components.
    
    Total should sum to 1.0 for normalized 0-100 output.
    """
    win_probability: float = 0.50  # Primary driver
    time_risk: float = 0.30        # Stage stagnation
    value_risk: float = 0.20       # Deal size impact
    
    def validate(self) -> bool:
        """Validate weights sum to 1.0."""
        total = self.win_probability + self.time_risk + self.value_risk
        if not np.isclose(total, 1.0):
            logger.warning(f"Risk weights sum to {total}, not 1.0")
            return False
        return True


@dataclass  
class RiskThresholds:
    """
    Thresholds for risk categorization.
    
    Deals are categorized as Low/Medium/High/Critical based on score.
    """
    low_max: int = 25
    medium_max: int = 50
    high_max: int = 75
    # Above high_max is Critical


class RiskScorer:
    """
    Computes composite risk scores for deals.
    
    Risk Score Formula:
    
    risk = w1 * (1 - win_probability) +
           w2 * time_risk_normalized +
           w3 * value_risk_normalized
    
    Where:
    - win_probability: From ML model (0-1)
    - time_risk: Based on days in stage vs typical
    - value_risk: Based on deal value tier
    
    Output: Integer 0-100 (higher = more risky)
    """
    
    # Default reference values (from Phase 1A data analysis)
    DEFAULT_MEDIAN_DAYS = 45  # Median days to close
    DEFAULT_P75_DAYS = 85     # 75th percentile days
    DEFAULT_P90_DAYS = 110    # 90th percentile days
    
    # Product price tiers for value risk
    PRICE_TIERS = {
        "low": (0, 100),
        "medium": (100, 2000),
        "high": (2000, 10000),
        "premium": (10000, float("inf"))
    }
    
    def __init__(
        self,
        weights: Optional[RiskWeights] = None,
        thresholds: Optional[RiskThresholds] = None,
        median_days: float = DEFAULT_MEDIAN_DAYS,
        p75_days: float = DEFAULT_P75_DAYS,
        p90_days: float = DEFAULT_P90_DAYS
    ):
        """
        Initialize risk scorer.
        
        Args:
            weights: Custom risk component weights
            thresholds: Custom risk category thresholds
            median_days: Reference median days to close
            p75_days: Reference 75th percentile days
            p90_days: Reference 90th percentile days
        """
        self.weights = weights or RiskWeights()
        self.thresholds = thresholds or RiskThresholds()
        self.median_days = median_days
        self.p75_days = p75_days
        self.p90_days = p90_days
        
        self.weights.validate()
    
    def _compute_probability_risk(self, win_prob: float) -> float:
        """
        Compute risk from win probability.
        
        Simply inverts probability: low win prob = high risk.
        
        Args:
            win_prob: Win probability (0-1)
            
        Returns:
            Risk score (0-1)
        """
        return 1.0 - np.clip(win_prob, 0, 1)
    
    def _compute_time_risk(self, days_in_stage: int) -> float:
        """
        Compute risk from time in stage.
        
        Uses a smooth curve that increases as deal exceeds typical timelines:
        - At median days: ~0.2 risk
        - At P75 days: ~0.5 risk
        - At P90 days: ~0.8 risk
        - Beyond P90: approaches 1.0
        
        Args:
            days_in_stage: Days deal has been in current stage
            
        Returns:
            Risk score (0-1)
        """
        if days_in_stage <= 0:
            return 0.0
        
        # Normalize using log scale for smooth curve
        if days_in_stage <= self.median_days:
            # Below median: low risk, linear scale
            return 0.2 * (days_in_stage / self.median_days)
        elif days_in_stage <= self.p75_days:
            # Between median and P75: moderate risk
            progress = (days_in_stage - self.median_days) / (self.p75_days - self.median_days)
            return 0.2 + 0.3 * progress
        elif days_in_stage <= self.p90_days:
            # Between P75 and P90: high risk
            progress = (days_in_stage - self.p75_days) / (self.p90_days - self.p75_days)
            return 0.5 + 0.3 * progress
        else:
            # Beyond P90: very high risk, asymptotic to 1.0
            excess = days_in_stage - self.p90_days
            return 0.8 + 0.2 * (1 - np.exp(-excess / 30))
    
    def _compute_value_risk(
        self,
        product_sales_price: float,
        max_price: float = 30000
    ) -> float:
        """
        Compute risk from deal value.
        
        Higher value deals have more business impact if lost,
        so they carry higher risk weight.
        
        Args:
            product_sales_price: Product list price
            max_price: Maximum price for normalization
            
        Returns:
            Risk score (0-1)
        """
        # Normalize price to 0-1 range with log scaling
        if product_sales_price <= 0:
            return 0.0
        
        normalized = np.log1p(product_sales_price) / np.log1p(max_price)
        return np.clip(normalized, 0, 1)
    
    def compute_risk(
        self,
        win_prob: float,
        days_in_stage: int,
        product_sales_price: float
    ) -> int:
        """
        Compute composite risk score for a deal.
        
        Args:
            win_prob: Win probability from ML model (0-1)
            days_in_stage: Days in current stage
            product_sales_price: Product list price
            
        Returns:
            Risk score 0-100 (integer)
        """
        # Compute components
        prob_risk = self._compute_probability_risk(win_prob)
        time_risk = self._compute_time_risk(days_in_stage)
        value_risk = self._compute_value_risk(product_sales_price)
        
        # Weighted combination
        raw_risk = (
            self.weights.win_probability * prob_risk +
            self.weights.time_risk * time_risk +
            self.weights.value_risk * value_risk
        )
        
        # Convert to 0-100 integer scale
        return int(np.clip(raw_risk * 100, 0, 100))
    
    def compute_risk_batch(
        self,
        df: pd.DataFrame,
        win_prob_col: str = "win_probability",
        days_col: str = "days_in_engaging",
        price_col: str = "product_sales_price"
    ) -> pd.Series:
        """
        Compute risk scores for multiple deals.
        
        Args:
            df: DataFrame with required columns
            win_prob_col: Column name for win probability
            days_col: Column name for days in stage
            price_col: Column name for product price
            
        Returns:
            Series of risk scores
        """
        return df.apply(
            lambda row: self.compute_risk(
                win_prob=row.get(win_prob_col, 0.5),
                days_in_stage=int(row.get(days_col, 0)),
                product_sales_price=float(row.get(price_col, 0))
            ),
            axis=1
        )
    
    def categorize_risk(self, risk_score: int) -> str:
        """
        Categorize risk score into named levels.
        
        Args:
            risk_score: Risk score 0-100
            
        Returns:
            Category: "Low", "Medium", "High", or "Critical"
        """
        if risk_score <= self.thresholds.low_max:
            return "Low"
        elif risk_score <= self.thresholds.medium_max:
            return "Medium"
        elif risk_score <= self.thresholds.high_max:
            return "High"
        else:
            return "Critical"
    
    def get_risk_breakdown(
        self,
        win_prob: float,
        days_in_stage: int,
        product_sales_price: float
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of risk components.
        
        Useful for explanations and debugging.
        
        Args:
            win_prob: Win probability
            days_in_stage: Days in stage
            product_sales_price: Product price
            
        Returns:
            Dictionary with component scores
        """
        prob_risk = self._compute_probability_risk(win_prob)
        time_risk = self._compute_time_risk(days_in_stage)
        value_risk = self._compute_value_risk(product_sales_price)
        
        return {
            "probability_risk": prob_risk,
            "probability_contribution": self.weights.win_probability * prob_risk,
            "time_risk": time_risk,
            "time_contribution": self.weights.time_risk * time_risk,
            "value_risk": value_risk,
            "value_contribution": self.weights.value_risk * value_risk,
            "total_risk": self.compute_risk(win_prob, days_in_stage, product_sales_price),
        }
    
    def get_risk_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get summary of risk distribution across deals.
        
        Args:
            df: DataFrame with 'risk_score' column
            
        Returns:
            Dictionary with counts per category
        """
        if "risk_score" not in df.columns:
            raise ValueError("DataFrame must have 'risk_score' column")
        
        categories = df["risk_score"].apply(self.categorize_risk)
        return categories.value_counts().to_dict()


def compute_portfolio_risk(
    df: pd.DataFrame,
    scorer: Optional[RiskScorer] = None
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Compute risk scores for entire portfolio.
    
    Args:
        df: DataFrame with required columns
        scorer: Optional custom scorer
        
    Returns:
        Tuple of (DataFrame with risk_score, summary dict)
    """
    if scorer is None:
        scorer = RiskScorer()
    
    df = df.copy()
    df["risk_score"] = scorer.compute_risk_batch(df)
    df["risk_category"] = df["risk_score"].apply(scorer.categorize_risk)
    
    summary = scorer.get_risk_summary(df)
    
    logger.info(f"Portfolio risk summary: {summary}")
    
    return df, summary


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Example deals
    deals = pd.DataFrame([
        {"deal_id": 1, "win_probability": 0.85, "days_in_engaging": 10, "product_sales_price": 500},
        {"deal_id": 2, "win_probability": 0.60, "days_in_engaging": 50, "product_sales_price": 5000},
        {"deal_id": 3, "win_probability": 0.30, "days_in_engaging": 100, "product_sales_price": 25000},
        {"deal_id": 4, "win_probability": 0.15, "days_in_engaging": 120, "product_sales_price": 1000},
    ])
    
    # Compute risk
    scorer = RiskScorer()
    deals_with_risk, summary = compute_portfolio_risk(deals, scorer)
    
    print("\n=== Deal Risk Scores ===")
    print(deals_with_risk[["deal_id", "win_probability", "days_in_engaging", 
                           "product_sales_price", "risk_score", "risk_category"]])
    
    print("\n=== Portfolio Summary ===")
    print(summary)
    
    # Detailed breakdown for one deal
    print("\n=== Risk Breakdown (Deal 3) ===")
    breakdown = scorer.get_risk_breakdown(
        win_prob=0.30,
        days_in_stage=100,
        product_sales_price=25000
    )
    for key, value in breakdown.items():
        print(f"  {key}: {value:.3f}")

