"""
Risk calculation service for the Streamlit app.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class RiskCalculator:
    """
    Calculate and aggregate risk metrics.
    """
    
    def __init__(self):
        """Initialize the calculator."""
        pass
    
    def calculate_portfolio_risk(
        self,
        predictions: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate overall portfolio risk metrics.
        
        Args:
            predictions: Predictions DataFrame
            
        Returns:
            Portfolio risk metrics
        """
        if predictions.empty:
            return {
                "weighted_risk": 0,
                "risk_concentration": 0,
                "avg_win_prob": 0,
            }
        
        # Weighted average risk by deal value
        if "product_sales_price" in predictions.columns:
            weights = predictions["product_sales_price"].fillna(1)
        else:
            weights = pd.Series(1, index=predictions.index)
        
        weighted_risk = (predictions["risk_score"] * weights).sum() / weights.sum()
        
        # Risk concentration (% of value in high risk deals)
        high_risk = predictions[predictions["risk_category"].isin(["High", "Critical"])]
        if weights.sum() > 0:
            risk_concentration = (
                weights[high_risk.index].sum() / weights.sum() * 100
            )
        else:
            risk_concentration = 0
        
        return {
            "weighted_risk": float(weighted_risk),
            "risk_concentration": float(risk_concentration),
            "avg_win_prob": float(predictions["win_probability"].mean()),
        }
    
    def rank_by_risk_value(
        self,
        predictions: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Rank deals by risk x value product.
        
        High risk + high value = highest priority.
        
        Args:
            predictions: Predictions DataFrame
            top_n: Number of deals to return
            
        Returns:
            Top deals by risk priority
        """
        df = predictions.copy()
        
        # Get deal value
        if "product_sales_price" in df.columns:
            value = df["product_sales_price"].fillna(1000)
        else:
            value = 1000
        
        # Risk x Value priority
        df["priority_score"] = df["risk_score"] * np.log1p(value)
        
        # Sort and return top N
        return df.nlargest(top_n, "priority_score")
    
    def segment_by_action(
        self,
        predictions: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Segment deals by recommended action.
        
        Args:
            predictions: Predictions DataFrame
            
        Returns:
            Dictionary of segment DataFrames
        """
        segments = {}
        
        # Urgent attention needed (Critical risk or very low win prob)
        urgent_mask = (
            (predictions["risk_category"] == "Critical") |
            (predictions["win_probability"] < 0.3)
        )
        segments["urgent"] = predictions[urgent_mask]
        
        # Need monitoring (High risk)
        monitor_mask = (
            (predictions["risk_category"] == "High") &
            ~urgent_mask
        )
        segments["monitor"] = predictions[monitor_mask]
        
        # On track (Medium or Low risk, good win prob)
        ontrack_mask = (
            predictions["risk_category"].isin(["Low", "Medium"]) &
            (predictions["win_probability"] >= 0.5)
        )
        segments["on_track"] = predictions[ontrack_mask]
        
        # Other
        other_mask = ~(urgent_mask | monitor_mask | ontrack_mask)
        segments["other"] = predictions[other_mask]
        
        return segments
    
    def get_risk_trend(
        self,
        predictions: pd.DataFrame,
        previous_predictions: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate risk trend vs previous period.
        
        Args:
            predictions: Current predictions
            previous_predictions: Previous period predictions
            
        Returns:
            Trend metrics
        """
        current_avg = predictions["risk_score"].mean() if not predictions.empty else 0
        
        if previous_predictions is None or previous_predictions.empty:
            return {
                "current_avg_risk": current_avg,
                "previous_avg_risk": 0,
                "risk_change": 0,
                "trend": "stable",
            }
        
        previous_avg = previous_predictions["risk_score"].mean()
        change = current_avg - previous_avg
        
        if change > 5:
            trend = "increasing"
        elif change < -5:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "current_avg_risk": current_avg,
            "previous_avg_risk": previous_avg,
            "risk_change": change,
            "trend": trend,
        }


