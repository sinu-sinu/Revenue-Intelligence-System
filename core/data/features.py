"""
Feature Engineering for Revenue Intelligence System
Core ML Pipeline

This module handles all feature computation for the ML models.
Key responsibilities:
- Compute features available at prediction time
- Implement temporal cutoffs for historical aggregations
- Prevent data leakage
- Provide feature documentation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Feature documentation for explainability
FEATURE_DEFINITIONS = {
    # Categorical features
    "sales_agent": "Sales representative assigned to the deal",
    "product": "Product being sold (7 SKUs)",
    "account": "Customer company name (or 'Unknown')",
    "account_sector": "Industry sector of the account",
    "account_location": "Headquarters location of account",
    "product_series": "Product line (GTX, GTK, MG)",
    "manager": "Sales manager overseeing the rep",
    "regional_office": "Regional office (East, West, Central)",
    
    # Numeric features
    "product_sales_price": "List price of the product ($)",
    "account_revenue": "Annual revenue of account (millions $)",
    "account_employees": "Number of employees at account",
    "account_age": "Years since account was established",
    "days_in_engaging": "Days since deal entered Engaging stage",
    
    # Historical aggregation features
    "rep_historical_win_rate": "Rep's win rate on deals closed before this deal's engage_date",
    "product_historical_win_rate": "Product's win rate on deals closed before this deal's engage_date",
    "rep_deal_count": "Number of deals rep has closed before this deal's engage_date",
}

# ============================================================================
# TIERED FEATURE SETS
# Start with minimal features to establish baseline, then add incrementally
# ============================================================================

# Tier 1: Most robust features (start here for baseline)
# Low cardinality, no cold-start issues
MINIMAL_FEATURES = {
    "categorical": ["product_series", "regional_office"],
    "numeric": ["product_sales_price", "days_in_engaging"],
}

# Tier 2: Add account and rep context
# Medium complexity, some cold-start for historical features
STANDARD_FEATURES = {
    "categorical": ["product_series", "regional_office", "account_sector", "manager"],
    "numeric": [
        "product_sales_price", 
        "days_in_engaging",
        "account_revenue",
        "account_employees",
        "rep_historical_win_rate",
    ],
}

# Tier 3: Full feature set (current)
# Highest complexity, potential overfitting risk
FULL_FEATURES = {
    "categorical": [
        "sales_agent",
        "product", 
        "account_sector",
        "product_series",
        "manager",
        "regional_office",
    ],
    "numeric": [
        "product_sales_price",
        "account_revenue",
        "account_employees",
        "account_age",
        "days_in_engaging",
        "rep_historical_win_rate",
        "product_historical_win_rate",
        "rep_deal_count",
    ],
}

# Default feature lists (for backwards compatibility)
CATEGORICAL_FEATURES = FULL_FEATURES["categorical"]
NUMERIC_FEATURES = FULL_FEATURES["numeric"]


def get_feature_tier(tier: str = "standard") -> dict:
    """
    Get feature set for a specific tier.
    
    Args:
        tier: 'minimal', 'standard', or 'full'
        
    Returns:
        Dictionary with 'categorical' and 'numeric' feature lists
    """
    tiers = {
        "minimal": MINIMAL_FEATURES,
        "standard": STANDARD_FEATURES,
        "full": FULL_FEATURES,
    }
    
    if tier not in tiers:
        raise ValueError(f"Unknown tier: {tier}. Use 'minimal', 'standard', or 'full'")
    
    return tiers[tier]


# Features to exclude (leakage risk)
EXCLUDED_FEATURES = [
    "opportunity_id",  # Unique identifier
    "close_date",      # Only known after close
    "close_value",     # Only known after close
    "deal_stage",      # Target variable
    "target",          # Target variable
    "account",         # High cardinality, use sector instead
    "account_location", # High cardinality
    "account_year_established",  # Use account_age instead
]


class FeatureEngineer:
    """
    Feature engineering for win probability and time-to-close models.
    
    Key design decisions:
    - All features must be available at prediction time (when deal is in Engaging)
    - Historical aggregations use strict temporal cutoffs to prevent leakage
    - Missing values handled with explicit strategies
    """
    
    # Minimum sample size for reliable historical win rate
    MIN_HISTORICAL_SAMPLES = 5
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize feature engineer.
        
        Args:
            reference_date: Date to use for computing temporal features.
                           If None, uses each deal's engage_date.
        """
        self.reference_date = reference_date
        self._historical_cache: Dict[str, pd.DataFrame] = {}
    
    def compute_days_in_engaging(
        self, 
        df: pd.DataFrame,
        prediction_date: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Compute days since deal entered Engaging stage.
        
        For training: Use close_date as "prediction date" (retrospective)
        For inference: Use current date
        
        Args:
            df: DataFrame with engage_date column
            prediction_date: Optional fixed date for all deals
            
        Returns:
            Series of days in engaging
        """
        if prediction_date is not None:
            # Fixed prediction date for all deals
            days = (prediction_date - df["engage_date"]).dt.days
        elif "close_date" in df.columns and df["close_date"].notna().any():
            # For training: use close_date as "when prediction would have been made"
            # This simulates predicting at various points during the deal lifecycle
            # We use a random point between engage and close for realism
            # Simplified: use engage_date + (close_date - engage_date) / 2
            midpoint = df["engage_date"] + (df["close_date"] - df["engage_date"]) / 2
            days = (midpoint - df["engage_date"]).dt.days
        else:
            # For open deals: use current date
            now = pd.Timestamp.now()
            days = (now - df["engage_date"]).dt.days
        
        return days.clip(lower=0).fillna(0).astype(int)
    
    def compute_historical_win_rates(
        self,
        df: pd.DataFrame,
        historical_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute historical win rates with temporal cutoff.
        
        CRITICAL: Only use deals closed BEFORE each deal's engage_date
        to prevent temporal leakage.
        
        Args:
            df: DataFrame of deals to compute features for
            historical_data: Full historical data for aggregation
            
        Returns:
            DataFrame with historical win rate columns added
        """
        df = df.copy()
        
        # Ensure we have required columns
        required_cols = ["engage_date", "sales_agent", "product"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Prepare historical data (closed deals only)
        hist = historical_data[
            (historical_data["deal_stage"].isin(["Won", "Lost"])) &
            (historical_data["close_date"].notna())
        ].copy()
        
        # Initialize columns
        df["rep_historical_win_rate"] = np.nan
        df["product_historical_win_rate"] = np.nan
        df["rep_deal_count"] = 0
        
        # Compute for each deal (vectorized approach would be complex due to temporal constraints)
        # For efficiency, we pre-compute cumulative stats
        
        # Sort historical data by close_date
        hist = hist.sort_values("close_date")
        
        # Create lookup for each unique engage_date
        unique_dates = df["engage_date"].dropna().unique()
        
        for engage_date in unique_dates:
            # Get deals that closed before this engage_date
            prior_deals = hist[hist["close_date"] < engage_date]
            
            if len(prior_deals) == 0:
                continue
            
            # Compute rep win rates
            rep_stats = prior_deals.groupby("sales_agent").agg(
                wins=("target", "sum"),
                total=("target", "count")
            )
            rep_stats["win_rate"] = rep_stats["wins"] / rep_stats["total"]
            rep_stats.loc[rep_stats["total"] < self.MIN_HISTORICAL_SAMPLES, "win_rate"] = np.nan
            
            # Compute product win rates  
            product_stats = prior_deals.groupby("product").agg(
                wins=("target", "sum"),
                total=("target", "count")
            )
            product_stats["win_rate"] = product_stats["wins"] / product_stats["total"]
            product_stats.loc[product_stats["total"] < self.MIN_HISTORICAL_SAMPLES, "win_rate"] = np.nan
            
            # Apply to deals with this engage_date
            mask = df["engage_date"] == engage_date
            
            for idx in df[mask].index:
                agent = df.loc[idx, "sales_agent"]
                product = df.loc[idx, "product"]
                
                if agent in rep_stats.index:
                    df.loc[idx, "rep_historical_win_rate"] = rep_stats.loc[agent, "win_rate"]
                    df.loc[idx, "rep_deal_count"] = rep_stats.loc[agent, "total"]
                
                if product in product_stats.index:
                    df.loc[idx, "product_historical_win_rate"] = product_stats.loc[product, "win_rate"]
        
        # Fill NaN with global average for cold-start
        global_win_rate = hist["target"].mean() if len(hist) > 0 else 0.5
        df["rep_historical_win_rate"] = df["rep_historical_win_rate"].fillna(global_win_rate)
        df["product_historical_win_rate"] = df["product_historical_win_rate"].fillna(global_win_rate)
        
        logger.info(f"Computed historical win rates for {len(df)} deals")
        
        return df
    
    def compute_all_features(
        self,
        df: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute all features for the win probability model.
        
        Args:
            df: DataFrame of deals to featurize
            historical_data: Optional separate historical data for aggregations.
                           If None, uses df itself.
                           
        Returns:
            DataFrame with all features computed
        """
        df = df.copy()
        
        logger.info(f"Computing features for {len(df)} deals...")
        
        # Compute temporal features
        df["days_in_engaging"] = self.compute_days_in_engaging(df)
        
        # Compute historical aggregations
        hist_data = historical_data if historical_data is not None else df
        df = self.compute_historical_win_rates(df, hist_data)
        
        # Handle missing numeric values
        df = self._handle_missing_values(df)
        
        logger.info("Feature computation complete")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Strategy:
        - Numeric: Fill with median (computed on training data ideally)
        - Categorical: Already handled as "Unknown" in preprocessing
        """
        df = df.copy()
        
        # Numeric features: fill with median
        numeric_fill = {
            "account_revenue": df["account_revenue"].median(),
            "account_employees": df["account_employees"].median(),
            "account_age": df["account_age"].median(),
        }
        
        for col, fill_val in numeric_fill.items():
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    df[col] = df[col].fillna(fill_val)
                    logger.debug(f"Filled {null_count} nulls in {col} with {fill_val:.2f}")
        
        return df
    
    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        include_categorical: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract feature matrix for model training.
        
        Args:
            df: DataFrame with all features computed
            include_categorical: Whether to include categorical features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        feature_cols = NUMERIC_FEATURES.copy()
        
        if include_categorical:
            feature_cols = CATEGORICAL_FEATURES + feature_cols
        
        # Filter to columns that exist
        available_cols = [c for c in feature_cols if c in df.columns]
        missing_cols = set(feature_cols) - set(available_cols)
        
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
        
        return df[available_cols], available_cols
    
    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract target variable.
        
        Args:
            df: DataFrame with target column
            
        Returns:
            Series of target values
        """
        if "target" not in df.columns:
            raise ValueError("Target column not found. Run preprocessing first.")
        
        return df["target"]


def get_feature_documentation() -> Dict[str, str]:
    """
    Get feature documentation for explainability.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return FEATURE_DEFINITIONS.copy()


def get_feature_lists() -> Dict[str, List[str]]:
    """
    Get feature lists by type.
    
    Returns:
        Dictionary with 'categorical', 'numeric', and 'excluded' keys
    """
    return {
        "categorical": CATEGORICAL_FEATURES.copy(),
        "numeric": NUMERIC_FEATURES.copy(),
        "excluded": EXCLUDED_FEATURES.copy(),
    }


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    from preprocessor import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor("dataset")
    full_data = preprocessor.preprocess(for_training=True)
    
    # Compute features
    engineer = FeatureEngineer()
    featured_data = engineer.compute_all_features(full_data, full_data)
    
    # Get feature matrix
    X, feature_names = engineer.get_feature_matrix(featured_data)
    y = engineer.get_target(featured_data)
    
    print("\n=== Feature Matrix ===")
    print(f"Shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"\nTarget distribution:")
    print(y.value_counts())

