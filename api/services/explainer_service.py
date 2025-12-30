"""
SHAP Explainer Service for Deal Predictions.

This service provides on-demand SHAP explanations for deal win probability predictions.
It lazily initializes the explainer on first request for better startup performance.
"""

import sys
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """A single feature's contribution to the prediction."""
    feature: str
    value: Any
    contribution: float
    explanation: str


@dataclass
class DealExplanationResult:
    """Complete SHAP explanation for a deal."""
    opportunity_id: str
    win_probability: float
    base_value: float
    top_positive: List[FeatureContribution]
    top_negative: List[FeatureContribution]
    summary_text: str


class ExplainerService:
    """
    Service for generating SHAP-based explanations for deal predictions.

    Uses lazy initialization to avoid loading SHAP and models until needed.
    Provides on-demand explanations for individual deals.
    """

    def __init__(
        self,
        models_dir: str = "models/artifacts",
        dataset_dir: str = "dataset",
        max_background_samples: int = 100
    ):
        """
        Initialize the explainer service.

        Args:
            models_dir: Directory containing trained model artifacts
            dataset_dir: Directory containing dataset files
            max_background_samples: Max samples for SHAP background data
        """
        self.models_dir = Path(models_dir)
        self.dataset_dir = Path(dataset_dir)
        self.max_background_samples = max_background_samples

        self._explainer = None
        self._model = None
        self._feature_cols = None
        self._label_encoders = None
        self._initialized = False
        self._init_error: Optional[str] = None

    def _initialize(self) -> bool:
        """
        Lazy initialization of SHAP explainer.

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return self._init_error is None

        try:
            # Import SHAP
            try:
                import shap
            except ImportError:
                self._init_error = "SHAP package not installed. Install with: pip install shap"
                logger.error(self._init_error)
                self._initialized = True
                return False

            # Load LightGBM model
            model_path = self.models_dir / "lgbm_model.pkl"
            if not model_path.exists():
                self._init_error = f"Model not found at {model_path}. Please train the model first."
                logger.error(self._init_error)
                self._initialized = True
                return False

            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            logger.info("Loaded LightGBM model for SHAP")

            # Load label encoders
            encoders_path = self.models_dir / "label_encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, "rb") as f:
                    self._label_encoders = pickle.load(f)
                logger.info("Loaded label encoders")

            # Load feature config
            import json
            config_path = self.models_dir / "feature_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    self._feature_cols = config.get("features", [])
                logger.info(f"Loaded {len(self._feature_cols)} feature columns")
            else:
                self._init_error = "Feature config not found"
                self._initialized = True
                return False

            # Load or create background data for SHAP
            background_path = self.models_dir / "shap_background.parquet"
            if background_path.exists():
                background_data = pd.read_parquet(background_path)
                logger.info(f"Loaded SHAP background data: {len(background_data)} samples")
            else:
                # Create background data from predictions
                logger.info("Creating SHAP background data from dataset...")
                background_data = self._create_background_data()
                if background_data is None:
                    self._init_error = "Could not create background data for SHAP"
                    self._initialized = True
                    return False

            # Initialize SHAP TreeExplainer
            logger.info("Initializing SHAP TreeExplainer...")
            self._explainer = shap.TreeExplainer(self._model)
            self._background_data = background_data

            logger.info(f"SHAP Explainer initialized. Base value: {self._explainer.expected_value:.4f}")

            self._initialized = True
            return True

        except Exception as e:
            self._init_error = f"Failed to initialize SHAP explainer: {str(e)}"
            logger.error(self._init_error, exc_info=True)
            self._initialized = True
            return False

    def _create_background_data(self) -> Optional[pd.DataFrame]:
        """
        Create background data for SHAP from the dataset.

        Returns:
            DataFrame with encoded features for SHAP background
        """
        try:
            # Load sales pipeline
            pipeline_path = self.dataset_dir / "sales_pipeline.csv"
            if not pipeline_path.exists():
                return None

            pipeline = pd.read_csv(pipeline_path)

            # Filter to engaging stage (what we predict on)
            engaging = pipeline[pipeline["deal_stage"] == "Engaging"].copy()

            # Sample if too large
            if len(engaging) > self.max_background_samples:
                engaging = engaging.sample(self.max_background_samples, random_state=42)

            # Enrich with reference data
            enriched = self._enrich_deals(engaging)

            # Encode features
            encoded = self._encode_features(enriched)

            # Save for future use
            background_path = self.models_dir / "shap_background.parquet"
            encoded[self._feature_cols].to_parquet(background_path)
            logger.info(f"Saved SHAP background data to {background_path}")

            return encoded[self._feature_cols]

        except Exception as e:
            logger.error(f"Failed to create background data: {e}")
            return None

    def _enrich_deals(self, deals: pd.DataFrame) -> pd.DataFrame:
        """Enrich deals with reference data."""
        deals = deals.copy()

        # Load reference tables
        accounts = pd.read_csv(self.dataset_dir / "accounts.csv")
        products = pd.read_csv(self.dataset_dir / "products.csv")
        sales_teams = pd.read_csv(self.dataset_dir / "sales_teams.csv")

        # Fix product name mismatch
        deals["product_clean"] = deals["product"].replace("GTXPro", "GTX Pro")

        # Join products
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

        # Fill missing values
        deals["account"] = deals["account"].fillna("Unknown")
        deals["account_sector"] = deals["account_sector"].fillna("Unknown")
        deals["account_revenue"] = deals["account_revenue"].fillna(0)

        # Parse dates and compute temporal features
        if "engage_date" in deals.columns:
            deals["engage_date"] = pd.to_datetime(deals["engage_date"])
            deals["engage_month"] = deals["engage_date"].dt.month
            deals["engage_quarter"] = deals["engage_date"].dt.quarter
            deals["engage_day_of_week"] = deals["engage_date"].dt.dayofweek

            dataset_end_date = deals["engage_date"].max() + pd.Timedelta(days=30)
            deals["days_in_engaging"] = (dataset_end_date - deals["engage_date"]).dt.days

        return deals

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using saved label encoders."""
        df = df.copy()

        if self._label_encoders is None:
            return df

        for col, encoder in self._label_encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[col] = df[col].fillna("Unknown")
                known_classes = set(encoder.classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else "Unknown")

                # Add "Unknown" to encoder if not present
                if "Unknown" not in known_classes:
                    # Encode as -1 for unknown
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in known_classes else -1
                    )
                else:
                    df[col] = encoder.transform(df[col])

        return df

    def is_available(self) -> bool:
        """Check if the explainer service is available."""
        if not self._initialized:
            self._initialize()
        return self._init_error is None

    def get_error(self) -> Optional[str]:
        """Get initialization error if any."""
        if not self._initialized:
            self._initialize()
        return self._init_error

    def explain_deal(
        self,
        deal_data: Dict[str, Any],
        opportunity_id: str,
        top_k: int = 5
    ) -> Optional[DealExplanationResult]:
        """
        Generate SHAP explanation for a single deal.

        Args:
            deal_data: Dictionary with deal features
            opportunity_id: Deal identifier
            top_k: Number of top features to include

        Returns:
            DealExplanationResult or None if explainer unavailable
        """
        if not self._initialize():
            return None

        try:
            # Convert to DataFrame
            df = pd.DataFrame([deal_data])

            # Enrich and encode
            enriched = self._enrich_deals(df)
            encoded = self._encode_features(enriched)

            # Ensure all feature columns exist
            for col in self._feature_cols:
                if col not in encoded.columns:
                    encoded[col] = 0

            # Get features for SHAP
            X = encoded[self._feature_cols]

            # Compute SHAP values
            shap_values = self._explainer.shap_values(X)[0]

            # Build contributions
            contributions = dict(zip(self._feature_cols, shap_values))
            feature_values = enriched.iloc[0].to_dict()

            # Sort by absolute contribution
            sorted_features = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Separate positive and negative
            top_positive = []
            top_negative = []

            for feature, contrib in sorted_features:
                value = feature_values.get(feature, 0)
                text = self._generate_explanation_text(feature, contrib, value)

                fc = FeatureContribution(
                    feature=self._format_feature_name(feature),
                    value=value,
                    contribution=round(contrib, 4),
                    explanation=text
                )

                if contrib > 0:
                    top_positive.append(fc)
                else:
                    top_negative.append(fc)

            top_positive = top_positive[:top_k]
            top_negative = top_negative[:top_k]

            # Compute predicted probability
            pred_logit = self._explainer.expected_value + sum(shap_values)
            win_probability = 1 / (1 + np.exp(-pred_logit))

            # Generate summary
            summary_parts = []
            if top_positive:
                summary_parts.append(f"Key driver: {top_positive[0].explanation}")
            if top_negative:
                summary_parts.append(f"Risk factor: {top_negative[0].explanation}")
            summary_text = " ".join(summary_parts)

            return DealExplanationResult(
                opportunity_id=opportunity_id,
                win_probability=round(float(win_probability), 4),
                base_value=round(float(self._explainer.expected_value), 4),
                top_positive=top_positive,
                top_negative=top_negative,
                summary_text=summary_text
            )

        except Exception as e:
            logger.error(f"Failed to explain deal {opportunity_id}: {e}", exc_info=True)
            return None

    def _format_feature_name(self, feature: str) -> str:
        """Convert feature name to human-readable format."""
        name_map = {
            "days_in_engaging": "Days in Stage",
            "rep_historical_win_rate": "Rep Win Rate",
            "product_historical_win_rate": "Product Win Rate",
            "product_sales_price": "Deal Value",
            "account_revenue": "Account Revenue",
            "account_employees": "Account Size",
            "account_age": "Company Age",
            "rep_deal_count": "Rep Experience",
            "sales_agent": "Sales Rep",
            "product": "Product",
            "account_sector": "Industry Sector",
            "product_series": "Product Series",
            "manager": "Sales Manager",
            "regional_office": "Region",
            "engage_month": "Month",
            "engage_quarter": "Quarter",
            "engage_day_of_week": "Day of Week",
        }
        return name_map.get(feature, feature.replace("_", " ").title())

    def _generate_explanation_text(
        self,
        feature: str,
        shap_value: float,
        feature_value: Any
    ) -> str:
        """Generate human-readable explanation for a feature contribution."""
        direction = "increases" if shap_value > 0 else "decreases"

        # Feature-specific templates
        templates = {
            "days_in_engaging": {
                "positive": f"Deal open {feature_value:.0f} days, longer than typical, reducing win likelihood",
                "negative": f"Deal moving quickly at {feature_value:.0f} days, increasing win likelihood",
            },
            "rep_historical_win_rate": {
                "positive": f"Rep's {feature_value:.0%} win rate is above average",
                "negative": f"Rep's {feature_value:.0%} win rate is below average",
            },
            "product_sales_price": {
                "positive": f"Higher price point (${feature_value:,.0f}) has positive signal",
                "negative": f"Price point (${feature_value:,.0f}) has negative signal",
            },
            "account_revenue": {
                "positive": f"Larger account (${feature_value/1e6:.1f}M revenue) tends to close",
                "negative": f"Account size (${feature_value/1e6:.1f}M revenue) is a risk factor",
            },
        }

        if feature in templates:
            key = "positive" if shap_value > 0 else "negative"
            try:
                return templates[feature][key]
            except (KeyError, ValueError, TypeError):
                pass

        # Default template
        return f"{self._format_feature_name(feature)} {direction} win probability"


# Singleton instance
_explainer_service: Optional[ExplainerService] = None


def get_explainer_service() -> ExplainerService:
    """Get or create the singleton explainer service."""
    global _explainer_service
    if _explainer_service is None:
        _explainer_service = ExplainerService()
    return _explainer_service
