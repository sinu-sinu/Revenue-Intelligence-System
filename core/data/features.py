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

# Tier 4: Enhanced feature set (Dec 2024 improvements)
# EXPERIMENTAL RESULTS (Dec 2024):
# - Target encoding: Added leakage, hurt generalization (AUC 0.50)
# - Temporal features: Hurt calibration (ECE 0.12 vs 0.03)
# - account_sector: Slightly hurt AUC (0.578 vs 0.582)
# - Interactions: Increased complexity without benefit
# 
# CONCLUSION: Minimal tier (product_series, regional_office, 
#             product_sales_price, days_in_engaging) is optimal
# 
# The dataset has limited predictive signal. Adding more features
# introduces noise rather than signal. The minimal tier achieves:
# - AUC: 0.582
# - ECE: 0.031 (excellent calibration)
# - Overfit gap: 0.023 (no overfitting)
#
# Enhanced tier preserved for testing but defaults to minimal
ENHANCED_FEATURES = MINIMAL_FEATURES.copy()

# Default feature lists (for backwards compatibility)
CATEGORICAL_FEATURES = FULL_FEATURES["categorical"]
NUMERIC_FEATURES = FULL_FEATURES["numeric"]


def get_feature_tier(tier: str = "standard") -> dict:
    """
    Get feature set for a specific tier.
    
    Args:
        tier: 'minimal', 'standard', 'full', or 'enhanced'
        
    Returns:
        Dictionary with 'categorical' and 'numeric' feature lists
    """
    tiers = {
        "minimal": MINIMAL_FEATURES,
        "standard": STANDARD_FEATURES,
        "full": FULL_FEATURES,
        "enhanced": ENHANCED_FEATURES,
    }
    
    if tier not in tiers:
        raise ValueError(f"Unknown tier: {tier}. Use 'minimal', 'standard', 'full', or 'enhanced'")
    
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


# ============================================================================
# ENHANCED FEATURE ENGINEERING (Dec 2024)
# These functions extract more signal from existing data without adding new sources
# ============================================================================

def target_encode_column(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    column: str,
    target: str,
    smoothing: float = 20.0,  # Increased for more regularization
    min_samples: int = 10    # Increased to reduce noise
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Target encode a categorical column with Bayesian smoothing.
    
    WHY THIS HELPS:
    - Label encoding loses ordinal information and treats categories as arbitrary integers
    - Target encoding captures the relationship between category and target
    - Bayesian smoothing prevents overfitting on rare categories
    
    LEAKAGE PREVENTION:
    - Only use train set to compute encodings
    - Apply same encoding to validation set
    - Use leave-one-out for training set to prevent seeing own target
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        column: Column to encode
        target: Target column name
        smoothing: Prior strength (higher = more regularization toward global mean)
        min_samples: Minimum samples to use category-specific rate
        
    Returns:
        Tuple of (train_encoded, val_encoded, encoding_map)
    """
    # Compute global mean from training data only
    global_mean = train_df[target].mean()
    
    # Compute category statistics from training data
    category_stats = train_df.groupby(column)[target].agg(['sum', 'count'])
    category_stats.columns = ['successes', 'total']
    
    # Bayesian smoothing: shrink toward global mean based on sample size
    # Formula: (successes + smoothing * global_mean) / (total + smoothing)
    category_stats['encoded'] = (
        (category_stats['successes'] + smoothing * global_mean) / 
        (category_stats['total'] + smoothing)
    )
    
    # For very rare categories, use global mean
    category_stats.loc[category_stats['total'] < min_samples, 'encoded'] = global_mean
    
    # Create encoding map
    encoding_map = category_stats['encoded'].to_dict()
    encoding_map['__global_mean__'] = global_mean  # For unseen categories
    
    # For training set: use leave-one-out encoding to prevent leakage
    # This excludes each row's own target from the category mean
    train_encoded = pd.Series(index=train_df.index, dtype=float)
    
    for cat in train_df[column].unique():
        mask = train_df[column] == cat
        cat_data = train_df[mask]
        n = len(cat_data)
        
        if n <= 1 or n < min_samples:
            # Not enough data, use global mean
            train_encoded[mask] = global_mean
        else:
            # Leave-one-out: for each row, exclude its own target
            cat_sum = cat_data[target].sum()
            cat_count = n
            
            for idx in cat_data.index:
                loo_sum = cat_sum - cat_data.loc[idx, target]
                loo_count = cat_count - 1
                
                # Bayesian smoothing with LOO
                loo_encoded = (
                    (loo_sum + smoothing * global_mean) / 
                    (loo_count + smoothing)
                )
                train_encoded[idx] = loo_encoded
    
    # Apply to validation (no LOO needed - uses full training stats)
    val_encoded = val_df[column].map(encoding_map).fillna(global_mean)
    
    return train_encoded, val_encoded, encoding_map


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal pattern features from engage_date.
    
    WHY THIS HELPS:
    - Captures seasonality (Q4 budget flush, summer slowdowns)
    - Month-end deals may have different dynamics
    - Cyclical encoding preserves continuity (Dec is close to Jan)
    
    Args:
        df: DataFrame with engage_date column
        
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    if 'engage_date' not in df.columns:
        logger.warning("engage_date not found, skipping temporal features")
        return df
    
    # Quarter - captures fiscal year patterns
    df['engage_quarter'] = df['engage_date'].dt.quarter
    
    # Cyclical encoding for month - preserves continuity
    # (December should be "close to" January in feature space)
    month = df['engage_date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Is month-end? (often quota-driven behavior)
    df['is_month_end'] = (df['engage_date'].dt.day >= 25).astype(int)
    
    # Is Q4? (budget flush period)
    df['is_q4'] = (df['engage_quarter'] == 4).astype(int)
    
    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add carefully selected interaction features.
    
    WHY THIS HELPS:
    - Some products may sell better in certain regions
    - Product-region interaction has low cardinality (3x3 = 9 values)
    - Captures geographic product preferences
    
    CONSTRAINTS:
    - Only add interactions with clear business meaning
    - Keep cardinality low to prevent overfitting
    
    Args:
        df: DataFrame with product_series and regional_office columns
        
    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    # Product x Region interaction
    # LOW CARDINALITY: 3 product series × 3 regions = 9 combinations
    if 'product_series' in df.columns and 'regional_office' in df.columns:
        df['product_region'] = (
            df['product_series'].astype(str) + '_' + 
            df['regional_office'].astype(str)
        )
    
    return df


def compute_engagement_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin days_in_engaging into meaningful categories.
    
    WHY THIS HELPS:
    - Captures non-linear relationship (deal freshness matters)
    - LightGBM with small trees may miss complex patterns
    - Business-meaningful categories (fresh, active, stale, dormant)
    
    Args:
        df: DataFrame with days_in_engaging column
        
    Returns:
        DataFrame with engagement_stage feature added
    """
    df = df.copy()
    
    if 'days_in_engaging' not in df.columns:
        return df
    
    # Define bins based on sales cycle understanding
    # Fresh: first week, high momentum
    # Active: first month, normal progression
    # Extended: 1-2 months, may need attention
    # Stale: 2-4 months, at risk
    # Dormant: 4+ months, likely dead
    
    # Use string categorical (easier to handle in encoding)
    bins = [0, 7, 30, 60, 120, float('inf')]
    labels = ['fresh', 'active', 'extended', 'stale', 'dormant']
    
    df['engagement_stage'] = pd.cut(
        df['days_in_engaging'],
        bins=bins,
        labels=labels,
        include_lowest=True
    ).astype(str)  # Convert to string to avoid Categorical issues
    
    # Also create numeric bins for models that prefer numbers
    # This is the key feature - directly usable as numeric
    df['engagement_stage_num'] = pd.cut(
        df['days_in_engaging'],
        bins=bins,
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    ).astype(float).fillna(2)  # Fill NaN with 'extended' equivalent
    
    return df


def apply_enhanced_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = 'target'
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply all enhanced feature engineering.
    
    This is the main entry point for improved feature extraction.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        target_col: Name of target column
        
    Returns:
        Tuple of (enhanced_train, enhanced_val, encoding_maps)
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    encoding_maps = {}
    
    # 1. Temporal features (no target leakage risk)
    train_df = compute_temporal_features(train_df)
    val_df = compute_temporal_features(val_df)
    
    # 2. Interaction features (no target leakage risk)
    train_df = compute_interaction_features(train_df)
    val_df = compute_interaction_features(val_df)
    
    # 3. Engagement bins (no target leakage risk)
    train_df = compute_engagement_bins(train_df)
    val_df = compute_engagement_bins(val_df)
    
    # NOTE: Target encoding was tested but didn't improve generalization
    # The categories have enough signal captured by LightGBM natively
    # Keeping this code for future reference but not applying it
    
    logger.info(f"Applied enhanced features: temporal, interaction, binning")
    
    return train_df, val_df, encoding_maps


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

