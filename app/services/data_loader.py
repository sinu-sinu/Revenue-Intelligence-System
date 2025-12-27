"""
Data loading service for the Streamlit app.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and cache data for the Streamlit app.
    """
    
    def __init__(
        self,
        predictions_path: str = "data/predictions/latest_predictions.csv",
        dataset_dir: str = "dataset"
    ):
        """
        Initialize the data loader.
        
        Args:
            predictions_path: Path to precomputed predictions
            dataset_dir: Path to raw dataset files
        """
        self.predictions_path = Path(predictions_path)
        self.dataset_dir = Path(dataset_dir)
        
        self._predictions_cache = None
        self._metadata_cache = None
    
    def load_predictions(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load precomputed predictions.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Predictions DataFrame
        """
        if self._predictions_cache is not None and not force_reload:
            return self._predictions_cache
        
        if not self.predictions_path.exists():
            logger.warning(f"Predictions not found at {self.predictions_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.predictions_path)
        
        # Parse JSON columns
        if "risk_drivers" in df.columns:
            df["risk_drivers"] = df["risk_drivers"].apply(
                lambda x: json.loads(str(x).replace("'", '"')) if pd.notna(x) else []
            )
        
        if "predicted_close_range" in df.columns:
            df["predicted_close_range"] = df["predicted_close_range"].apply(
                lambda x: tuple(json.loads(str(x))) if pd.notna(x) else (0, 0)
            )
        
        self._predictions_cache = df
        logger.info(f"Loaded {len(df)} predictions")
        
        return df
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load predictions metadata.
        
        Returns:
            Metadata dictionary
        """
        if self._metadata_cache is not None:
            return self._metadata_cache
        
        meta_path = self.predictions_path.parent / "predictions_metadata.json"
        
        if not meta_path.exists():
            return {
                "generated_at": "Unknown",
                "n_predictions": 0,
                "risk_distribution": {},
                "avg_win_probability": 0.0,
            }
        
        with open(meta_path, "r") as f:
            self._metadata_cache = json.load(f)
        
        return self._metadata_cache
    
    def get_filters(self) -> Dict[str, List[str]]:
        """
        Get available filter options from predictions data.
        
        Returns:
            Dictionary of filter options
        """
        df = self.load_predictions()
        
        if df.empty:
            return {
                "accounts": [],
                "sales_agents": [],
                "products": [],
                "risk_categories": [],
            }
        
        return {
            "accounts": sorted(df["account"].dropna().unique().tolist()),
            "sales_agents": sorted(df["sales_agent"].dropna().unique().tolist()),
            "products": sorted(df["product"].dropna().unique().tolist()),
            "risk_categories": ["Critical", "High", "Medium", "Low"],
        }
    
    def filter_predictions(
        self,
        accounts: Optional[List[str]] = None,
        sales_agents: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        risk_categories: Optional[List[str]] = None,
        min_risk_score: int = 0,
        max_risk_score: int = 100,
    ) -> pd.DataFrame:
        """
        Filter predictions based on criteria.
        
        Args:
            accounts: Filter by account names
            sales_agents: Filter by sales agent names
            products: Filter by product names
            risk_categories: Filter by risk category
            min_risk_score: Minimum risk score
            max_risk_score: Maximum risk score
            
        Returns:
            Filtered DataFrame
        """
        df = self.load_predictions()
        
        if df.empty:
            return df
        
        mask = pd.Series(True, index=df.index)
        
        if accounts:
            mask &= df["account"].isin(accounts)
        
        if sales_agents:
            mask &= df["sales_agent"].isin(sales_agents)
        
        if products:
            mask &= df["product"].isin(products)
        
        if risk_categories:
            mask &= df["risk_category"].isin(risk_categories)
        
        mask &= (df["risk_score"] >= min_risk_score) & (df["risk_score"] <= max_risk_score)
        
        return df[mask]
    
    def get_deal(self, opportunity_id: str) -> Optional[pd.Series]:
        """
        Get a single deal by ID.
        
        Args:
            opportunity_id: Deal ID
            
        Returns:
            Deal data as Series, or None if not found
        """
        df = self.load_predictions()
        
        matches = df[df["opportunity_id"] == opportunity_id]
        
        if matches.empty:
            return None
        
        return matches.iloc[0]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the dashboard.
        
        Returns:
            Summary statistics dictionary
        """
        df = self.load_predictions()
        
        if df.empty:
            return {
                "total_deals": 0,
                "at_risk_revenue": 0,
                "high_risk_count": 0,
                "avg_win_probability": 0,
            }
        
        # At-risk = deals with risk_score > 50
        at_risk = df[df["risk_score"] > 50]
        
        # Calculate revenue (using product_sales_price as proxy)
        if "product_sales_price" in at_risk.columns:
            at_risk_revenue = at_risk["product_sales_price"].sum()
        else:
            at_risk_revenue = len(at_risk) * 1000  # Default estimate
        
        return {
            "total_deals": len(df),
            "at_risk_revenue": at_risk_revenue,
            "high_risk_count": len(df[df["risk_category"].isin(["High", "Critical"])]),
            "avg_win_probability": df["win_probability"].mean(),
            "risk_distribution": df["risk_category"].value_counts().to_dict(),
        }
    
    def load_sales_teams(self) -> pd.DataFrame:
        """
        Load sales teams reference data.
        
        Returns:
            Sales teams DataFrame
        """
        path = self.dataset_dir / "sales_teams.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    
    def load_products(self) -> pd.DataFrame:
        """
        Load products reference data.
        
        Returns:
            Products DataFrame
        """
        path = self.dataset_dir / "products.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)


