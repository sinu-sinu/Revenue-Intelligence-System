"""
Data Preprocessor for Revenue Intelligence System
Core ML Pipeline

This module handles all data loading, cleaning, and preparation
for the ML pipeline. Key responsibilities:
- Load and merge datasets
- Fix known data quality issues (GTXPro mapping, NULL accounts)
- Create train/test splits with temporal integrity
- Prepare features for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for sales pipeline data.
    
    Handles:
    - Loading raw CSV files
    - Fixing data quality issues
    - Joining reference tables
    - Creating temporal train/test splits
    """
    
    # Known data quality fixes
    PRODUCT_NAME_MAPPING = {
        "GTXPro": "GTX Pro"
    }
    
    # Train/test split cutoff (based on Phase 1A analysis)
    TRAIN_TEST_CUTOFF = pd.Timestamp("2017-07-01")
    
    # Reference year for computing account age
    REFERENCE_YEAR = 2017
    
    def __init__(self, data_dir: str = "dataset"):
        """
        Initialize preprocessor with data directory path.
        
        Args:
            data_dir: Path to directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self._validate_data_dir()
        
        # Cached dataframes
        self._sales_pipeline: Optional[pd.DataFrame] = None
        self._accounts: Optional[pd.DataFrame] = None
        self._products: Optional[pd.DataFrame] = None
        self._sales_teams: Optional[pd.DataFrame] = None
        self._merged_data: Optional[pd.DataFrame] = None
    
    def _validate_data_dir(self) -> None:
        """Validate that required files exist."""
        required_files = [
            "sales_pipeline.csv",
            "accounts.csv", 
            "products.csv",
            "sales_teams.csv"
        ]
        for file in required_files:
            if not (self.data_dir / file).exists():
                raise FileNotFoundError(f"Required file not found: {self.data_dir / file}")
    
    def load_raw_data(self) -> dict:
        """
        Load all raw CSV files.
        
        Returns:
            Dictionary with dataframe names as keys
        """
        logger.info("Loading raw data files...")
        
        self._sales_pipeline = pd.read_csv(self.data_dir / "sales_pipeline.csv")
        self._accounts = pd.read_csv(self.data_dir / "accounts.csv")
        self._products = pd.read_csv(self.data_dir / "products.csv")
        self._sales_teams = pd.read_csv(self.data_dir / "sales_teams.csv")
        
        logger.info(f"Loaded sales_pipeline: {len(self._sales_pipeline)} rows")
        logger.info(f"Loaded accounts: {len(self._accounts)} rows")
        logger.info(f"Loaded products: {len(self._products)} rows")
        logger.info(f"Loaded sales_teams: {len(self._sales_teams)} rows")
        
        return {
            "sales_pipeline": self._sales_pipeline,
            "accounts": self._accounts,
            "products": self._products,
            "sales_teams": self._sales_teams
        }
    
    def fix_product_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix known product name inconsistencies.
        
        Issue: 'GTXPro' in pipeline doesn't match 'GTX Pro' in products table.
        Resolution: Map GTXPro -> GTX Pro
        
        Args:
            df: DataFrame with 'product' column
            
        Returns:
            DataFrame with fixed product names
        """
        df = df.copy()
        original_values = df["product"].value_counts().to_dict()
        
        df["product"] = df["product"].replace(self.PRODUCT_NAME_MAPPING)
        
        fixed_count = sum(
            original_values.get(old, 0) 
            for old in self.PRODUCT_NAME_MAPPING.keys()
        )
        if fixed_count > 0:
            logger.info(f"Fixed {fixed_count} product name mappings")
        
        return df
    
    def handle_null_accounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle NULL account values.
        
        Issue: 16% of deals have NULL account
        Resolution: Replace with "Unknown" category
        
        Args:
            df: DataFrame with 'account' column
            
        Returns:
            DataFrame with NULL accounts replaced
        """
        df = df.copy()
        null_count = df["account"].isna().sum()
        
        df["account"] = df["account"].fillna("Unknown")
        
        if null_count > 0:
            logger.info(f"Replaced {null_count} NULL accounts with 'Unknown'")
        
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date columns to datetime.
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with parsed dates
        """
        df = df.copy()
        
        df["engage_date"] = pd.to_datetime(df["engage_date"], errors="coerce")
        df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce")
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable.
        
        Target: 1 = Won, 0 = Lost
        Only applicable to closed deals (Won/Lost stage)
        
        Args:
            df: DataFrame with 'deal_stage' column
            
        Returns:
            DataFrame with 'target' column added
        """
        df = df.copy()
        
        # Create target only for closed deals
        df["target"] = np.where(
            df["deal_stage"] == "Won", 1,
            np.where(df["deal_stage"] == "Lost", 0, np.nan)
        )
        
        return df
    
    def join_reference_tables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Join reference tables to enrich features.
        
        Joins:
        - products: series, sales_price
        - accounts: sector, revenue, employees, year_established, office_location
        - sales_teams: manager, regional_office
        
        Args:
            df: Sales pipeline DataFrame
            
        Returns:
            Enriched DataFrame with joined columns
        """
        if self._products is None or self._accounts is None or self._sales_teams is None:
            self.load_raw_data()
        
        df = df.copy()
        
        # Join products
        df = df.merge(
            self._products[["product", "series", "sales_price"]].rename(
                columns={"series": "product_series", "sales_price": "product_sales_price"}
            ),
            on="product",
            how="left"
        )
        
        # Join accounts (handle "Unknown" category)
        accounts_with_unknown = pd.concat([
            self._accounts,
            pd.DataFrame([{
                "account": "Unknown",
                "sector": "Unknown",
                "revenue": np.nan,
                "employees": np.nan,
                "year_established": np.nan,
                "office_location": "Unknown",
                "subsidiary_of": np.nan
            }])
        ], ignore_index=True)
        
        df = df.merge(
            accounts_with_unknown[[
                "account", "sector", "revenue", "employees", 
                "year_established", "office_location"
            ]].rename(columns={
                "sector": "account_sector",
                "revenue": "account_revenue",
                "employees": "account_employees",
                "year_established": "account_year_established",
                "office_location": "account_location"
            }),
            on="account",
            how="left"
        )
        
        # Compute account age
        df["account_age"] = self.REFERENCE_YEAR - df["account_year_established"]
        
        # Join sales teams
        df = df.merge(
            self._sales_teams[["sales_agent", "manager", "regional_office"]],
            on="sales_agent",
            how="left"
        )
        
        logger.info("Joined reference tables: products, accounts, sales_teams")
        
        return df
    
    def get_closed_deals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to closed deals only (for training).
        
        Args:
            df: Full pipeline DataFrame
            
        Returns:
            DataFrame with only Won/Lost deals
        """
        closed = df[df["deal_stage"].isin(["Won", "Lost"])].copy()
        logger.info(f"Filtered to {len(closed)} closed deals (Won/Lost)")
        return closed
    
    def get_open_deals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to open deals only (for prediction).
        
        Args:
            df: Full pipeline DataFrame
            
        Returns:
            DataFrame with only Engaging/Prospecting deals
        """
        open_deals = df[df["deal_stage"].isin(["Engaging", "Prospecting"])].copy()
        logger.info(f"Filtered to {len(open_deals)} open deals")
        return open_deals
    
    def create_time_split(
        self, 
        df: pd.DataFrame,
        cutoff_date: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/test split.
        
        Split based on engage_date to prevent temporal leakage.
        Default cutoff: 2017-07-01
        
        Args:
            df: DataFrame with engage_date column
            cutoff_date: Optional custom cutoff date
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if cutoff_date is None:
            cutoff_date = self.TRAIN_TEST_CUTOFF
        
        # Filter to deals with valid engage_date
        df_valid = df[df["engage_date"].notna()].copy()
        
        train = df_valid[df_valid["engage_date"] < cutoff_date]
        test = df_valid[df_valid["engage_date"] >= cutoff_date]
        
        logger.info(f"Time-based split at {cutoff_date.date()}:")
        logger.info(f"  Train: {len(train)} deals, win rate: {train['target'].mean():.1%}")
        logger.info(f"  Test: {len(test)} deals, win rate: {test['target'].mean():.1%}")
        
        return train, test
    
    def preprocess(self, for_training: bool = True) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.
        
        Args:
            for_training: If True, filter to closed deals only
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        self.load_raw_data()
        
        # Apply fixes
        df = self._sales_pipeline.copy()
        df = self.fix_product_names(df)
        df = self.handle_null_accounts(df)
        df = self.parse_dates(df)
        df = self.create_target(df)
        df = self.join_reference_tables(df)
        
        # Filter if for training
        if for_training:
            df = self.get_closed_deals(df)
        
        self._merged_data = df
        logger.info(f"Preprocessing complete: {len(df)} rows")
        
        return df
    
    def get_train_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get preprocessed train and test datasets.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if self._merged_data is None:
            self.preprocess(for_training=True)
        
        return self.create_time_split(self._merged_data)


def load_and_preprocess(data_dir: str = "dataset") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and preprocess data.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Tuple of (train_df, test_df)
    """
    preprocessor = DataPreprocessor(data_dir)
    return preprocessor.get_train_test_data()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = DataPreprocessor("dataset")
    train, test = preprocessor.get_train_test_data()
    
    print("\n=== Train Data ===")
    print(train.head())
    print(f"\nShape: {train.shape}")
    print(f"Columns: {list(train.columns)}")
    
    print("\n=== Test Data ===")
    print(test.head())
    print(f"\nShape: {test.shape}")

