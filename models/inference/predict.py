"""
Deal Prediction Pipeline
Phase 1B: Streamlit UI

This module provides prediction capabilities for the Streamlit app.
It loads trained models and generates predictions for deals.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DealPrediction:
    """
    Prediction result for a single deal.
    """
    opportunity_id: str
    account: str
    sales_agent: str
    product: str
    deal_stage: str
    
    # Predictions
    win_probability: float
    risk_score: int
    risk_category: str
    predicted_close_days: int
    predicted_close_range: Tuple[int, int]  # (min, max)
    
    # Context
    days_in_stage: int
    product_sales_price: float
    
    # Explanations
    risk_drivers: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["predicted_close_range"] = list(d["predicted_close_range"])
        return d


class DealPredictor:
    """
    Prediction pipeline for deal win probability and risk.
    
    Loads trained models and generates predictions with explanations.
    Designed for the Streamlit UI to use precomputed predictions.
    """
    
    def __init__(
        self,
        models_dir: str = "models/artifacts",
        dataset_dir: str = "dataset"
    ):
        """
        Initialize predictor with model paths.
        
        Args:
            models_dir: Directory containing trained model artifacts
            dataset_dir: Directory containing dataset files
        """
        self.models_dir = Path(models_dir)
        self.dataset_dir = Path(dataset_dir)
        
        self.lgbm_model = None
        self.calibrator = None
        self.time_model = None
        self.label_encoders = None
        self.feature_cols = None
        
        self._models_loaded = False
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if models loaded successfully
        """
        try:
            # Load LightGBM model
            lgbm_path = self.models_dir / "lgbm_model.pkl"
            if lgbm_path.exists():
                with open(lgbm_path, "rb") as f:
                    self.lgbm_model = pickle.load(f)
                logger.info("Loaded LightGBM model")
            
            # Load calibrator
            cal_path = self.models_dir / "calibrator.pkl"
            if cal_path.exists():
                with open(cal_path, "rb") as f:
                    self.calibrator = pickle.load(f)
                logger.info("Loaded calibrator")
            
            # Load time-to-close model
            time_path = self.models_dir / "time_model.pkl"
            if time_path.exists():
                with open(time_path, "rb") as f:
                    self.time_model = pickle.load(f)
                logger.info("Loaded time-to-close model")
            
            # Load label encoders
            enc_path = self.models_dir / "label_encoders.pkl"
            if enc_path.exists():
                with open(enc_path, "rb") as f:
                    self.label_encoders = pickle.load(f)
                logger.info("Loaded label encoders")
            
            # Load feature config
            config_path = self.models_dir / "feature_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self.feature_cols = config.get("features", [])
                logger.info(f"Loaded feature config: {len(self.feature_cols)} features")
            
            self._models_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def _load_reference_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load reference tables for feature enrichment.
        
        Returns:
            Tuple of (accounts, products, sales_teams, pipeline)
        """
        accounts = pd.read_csv(self.dataset_dir / "accounts.csv")
        products = pd.read_csv(self.dataset_dir / "products.csv")
        sales_teams = pd.read_csv(self.dataset_dir / "sales_teams.csv")
        pipeline = pd.read_csv(self.dataset_dir / "sales_pipeline.csv")
        
        return accounts, products, sales_teams, pipeline
    
    def _enrich_deals(self, deals: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich deals with reference data.
        
        Args:
            deals: Raw deal DataFrame
            
        Returns:
            Enriched DataFrame
        """
        accounts, products, sales_teams, _ = self._load_reference_data()
        
        # Fix product name mismatch
        deals = deals.copy()
        deals["product_clean"] = deals["product"].replace("GTXPro", "GTX Pro")
        
        # Join products (note: actual column names are 'series' and 'sales_price')
        deals = deals.merge(
            products[["product", "series", "sales_price"]].rename(
                columns={"product": "product_clean", "series": "product_series", "sales_price": "product_sales_price"}
            ),
            on="product_clean",
            how="left"
        )
        
        # Join accounts
        deals = deals.merge(
            accounts[["account", "sector", "revenue", "employees", "year_established"]].rename(
                columns={"sector": "account_sector", "revenue": "account_revenue"}
            ),
            on="account",
            how="left"
        )
        
        # Join sales teams
        deals = deals.merge(
            sales_teams[["sales_agent", "manager", "regional_office"]],
            on="sales_agent",
            how="left"
        )
        
        # Fill missing account info
        deals["account"] = deals["account"].fillna("Unknown")
        deals["account_sector"] = deals["account_sector"].fillna("Unknown")
        deals["account_revenue"] = deals["account_revenue"].fillna(0)
        
        # Parse dates and compute temporal features
        if "engage_date" in deals.columns:
            deals["engage_date"] = pd.to_datetime(deals["engage_date"])
            deals["engage_month"] = deals["engage_date"].dt.month
            deals["engage_quarter"] = deals["engage_date"].dt.quarter
            deals["engage_day_of_week"] = deals["engage_date"].dt.dayofweek
            
            # Days in stage - use dataset's latest date as "today" for realistic calculations
            # The dataset is from 2016-2017, so using current date would give 3000+ days
            # Instead, simulate as if "today" is the end of the dataset period
            dataset_end_date = deals["engage_date"].max() + pd.Timedelta(days=30)
            deals["days_in_engaging"] = (dataset_end_date - deals["engage_date"]).dt.days
        
        return deals
    
    def _compute_risk_score(
        self,
        win_prob: float,
        days_in_stage: int,
        product_price: float
    ) -> Tuple[int, str]:
        """
        Compute composite risk score.
        
        Args:
            win_prob: Win probability (0-1)
            days_in_stage: Days in current stage
            product_price: Product price
            
        Returns:
            Tuple of (risk_score 0-100, risk_category)
        """
        # Weight components
        prob_risk = (1 - win_prob) * 50  # 0-50
        
        # Time risk (normalized to 0-30)
        if days_in_stage < 30:
            time_risk = 0
        elif days_in_stage < 60:
            time_risk = 10
        elif days_in_stage < 90:
            time_risk = 20
        else:
            time_risk = 30
        
        # Value risk (log scale, 0-20)
        value_risk = min(20, np.log1p(product_price) / np.log1p(30000) * 20)
        
        risk_score = int(prob_risk + time_risk + value_risk)
        
        # Categorize
        if risk_score <= 25:
            category = "Low"
        elif risk_score <= 50:
            category = "Medium"
        elif risk_score <= 75:
            category = "High"
        else:
            category = "Critical"
        
        return risk_score, category
    
    def _generate_risk_drivers(
        self,
        win_prob: float,
        days_in_stage: int,
        product_price: float,
        rep_win_rate: float = 0.63
    ) -> List[Dict[str, Any]]:
        """
        Generate human-readable risk drivers.
        
        Args:
            win_prob: Win probability
            days_in_stage: Days in stage
            product_price: Product price
            rep_win_rate: Historical rep win rate
            
        Returns:
            List of risk driver dictionaries
        """
        drivers = []
        
        # Low win probability
        if win_prob < 0.4:
            drivers.append({
                "driver": "Low Win Probability",
                "detail": f"{win_prob:.0%} predicted win rate",
                "impact": f"+{int((0.5 - win_prob) * 50)}%",
                "icon": "chart_decreasing"
            })
        
        # Stage stagnation
        if days_in_stage > 60:
            avg_days = 48
            drivers.append({
                "driver": "Stage Stagnation",
                "detail": f"{days_in_stage} days in stage (avg: {avg_days})",
                "impact": f"+{min(30, days_in_stage - 30)}%",
                "icon": "hourglass"
            })
        
        # High value at risk
        if product_price > 5000:
            drivers.append({
                "driver": "High Value at Risk",
                "detail": f"${product_price:,.0f} deal value",
                "impact": "+10%",
                "icon": "dollar"
            })
        
        # Rep performance (if below average)
        if rep_win_rate < 0.5:
            drivers.append({
                "driver": "Rep Performance",
                "detail": f"{rep_win_rate:.0%} historical win rate",
                "impact": f"+{int((0.63 - rep_win_rate) * 30)}%",
                "icon": "person"
            })
        
        # If no drivers, deal is healthy
        if not drivers:
            drivers.append({
                "driver": "Healthy Deal",
                "detail": "No significant risk factors",
                "impact": "0%",
                "icon": "check"
            })
        
        return drivers
    
    def predict_batch(
        self,
        deals: pd.DataFrame = None,
        stage_filter: str = "Engaging"
    ) -> pd.DataFrame:
        """
        Generate predictions for all deals.
        
        Args:
            deals: Optional DataFrame of deals (loads from disk if None)
            stage_filter: Only predict for deals in this stage
            
        Returns:
            DataFrame with predictions
        """
        # Load deals if not provided
        if deals is None:
            deals = pd.read_csv(self.dataset_dir / "sales_pipeline.csv")
        
        # Filter to open deals in specified stage
        if stage_filter:
            open_deals = deals[deals["deal_stage"] == stage_filter].copy()
        else:
            open_deals = deals.copy()
        
        logger.info(f"Predicting for {len(open_deals)} deals")
        
        # Enrich with reference data
        enriched = self._enrich_deals(open_deals)
        
        # For demo/precomputed predictions, use a simple heuristic
        # In production, this would use the trained models
        predictions = []
        
        for idx, row in enriched.iterrows():
            # Base win probability (simple heuristic for demo)
            base_prob = 0.63  # Overall win rate
            
            # Adjust by days in stage
            days = row.get("days_in_engaging", 30)
            if days < 30:
                prob_adj = 0.1
            elif days < 60:
                prob_adj = 0
            elif days < 90:
                prob_adj = -0.1
            else:
                prob_adj = -0.2
            
            # Adjust by product
            price = row.get("product_sales_price", 1000)
            if price > 5000:
                prob_adj -= 0.05
            
            win_prob = np.clip(base_prob + prob_adj + np.random.normal(0, 0.05), 0.1, 0.9)
            
            # Risk score
            risk_score, risk_cat = self._compute_risk_score(win_prob, days, price)
            
            # Time to close prediction
            # Use a more realistic distribution: shorter for newer deals, longer for older ones
            # This ensures some deals close soon (for forecast visibility)
            base_close = 14 + np.random.exponential(21)  # Mean of ~35 days
            # Older deals slightly longer to close
            age_factor = min(days / 365, 1) * 20  # Up to 20 extra days for oldest deals
            pred_days = int(base_close + age_factor + np.random.normal(0, 7))
            pred_days = max(7, min(pred_days, 120))  # Clamp to 7-120 days
            
            # Risk drivers
            drivers = self._generate_risk_drivers(win_prob, days, price)
            
            pred = DealPrediction(
                opportunity_id=row.get("opportunity_id", f"OPP_{idx}"),
                account=row.get("account", "Unknown"),
                sales_agent=row.get("sales_agent", "Unknown"),
                product=row.get("product", "Unknown"),
                deal_stage=row.get("deal_stage", stage_filter),
                win_probability=round(win_prob, 3),
                risk_score=risk_score,
                risk_category=risk_cat,
                predicted_close_days=pred_days,
                predicted_close_range=(max(7, pred_days - 15), pred_days + 15),
                days_in_stage=days,
                product_sales_price=price if pd.notna(price) else 0,
                risk_drivers=drivers
            )
            predictions.append(pred)
        
        # Convert to DataFrame
        pred_df = pd.DataFrame([p.to_dict() for p in predictions])
        
        logger.info(f"Generated {len(pred_df)} predictions")
        
        return pred_df
    
    def save_predictions(
        self,
        predictions: pd.DataFrame,
        output_path: str = "data/predictions/latest_predictions.csv"
    ) -> str:
        """
        Save predictions to disk.
        
        Args:
            predictions: Predictions DataFrame
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        predictions.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        # Also save metadata
        meta_path = output_path.parent / "predictions_metadata.json"
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "n_predictions": len(predictions),
            "risk_distribution": predictions["risk_category"].value_counts().to_dict(),
            "avg_win_probability": float(predictions["win_probability"].mean()),
        }
        
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(output_path)
    
    def load_predictions(
        self,
        path: str = "data/predictions/latest_predictions.csv"
    ) -> pd.DataFrame:
        """
        Load precomputed predictions from disk.
        
        Args:
            path: Path to predictions file
            
        Returns:
            Predictions DataFrame
        """
        pred_df = pd.read_csv(path)
        
        # Parse list columns
        if "risk_drivers" in pred_df.columns:
            pred_df["risk_drivers"] = pred_df["risk_drivers"].apply(
                lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else []
            )
        
        if "predicted_close_range" in pred_df.columns:
            pred_df["predicted_close_range"] = pred_df["predicted_close_range"].apply(
                lambda x: tuple(json.loads(x)) if isinstance(x, str) else (0, 0)
            )
        
        return pred_df


def generate_predictions(output_path: str = "data/predictions/latest_predictions.csv") -> str:
    """
    Generate and save predictions for all open deals.
    
    Args:
        output_path: Where to save predictions
        
    Returns:
        Path to saved file
    """
    predictor = DealPredictor()
    predictions = predictor.predict_batch()
    return predictor.save_predictions(predictions, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate predictions
    output = generate_predictions()
    print(f"\nPredictions saved to: {output}")
    
    # Show sample
    predictor = DealPredictor()
    preds = predictor.load_predictions(output)
    print(f"\nSample predictions:")
    print(preds[["opportunity_id", "account", "win_probability", "risk_score", "risk_category"]].head(10))

