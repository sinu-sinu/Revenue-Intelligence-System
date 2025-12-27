"""
Time-to-Close Model Training
Phase 1A: Core ML Pipeline

This module handles training of the time-to-close prediction model.
Key responsibilities:
- Predict distribution of close dates (not point estimates)
- Handle right-censoring for open deals
- Provide P10/P50/P90 estimates for forecasting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json
import logging

from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

# Optional: survival analysis
try:
    from lifelines import WeibullAFTFitter, KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    logging.warning("lifelines not installed. Survival analysis unavailable.")

logger = logging.getLogger(__name__)


class TimeToCloseTrainer:
    """
    Trainer for time-to-close prediction model.
    
    Two approaches implemented:
    1. Simple: Log-transformed regression (LightGBM)
    2. Advanced: Survival analysis with Weibull AFT (if lifelines available)
    
    Output is a distribution, not a point estimate, enabling P10/P50/P90 bands.
    """
    
    # LightGBM hyperparameters for regression
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    
    def __init__(
        self,
        categorical_features: List[str],
        numeric_features: List[str],
        artifacts_dir: str = "models/artifacts"
    ):
        """
        Initialize trainer.
        
        Args:
            categorical_features: List of categorical feature names
            numeric_features: List of numeric feature names
            artifacts_dir: Directory to save model artifacts
        """
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.all_features = categorical_features + numeric_features
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Encoders
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        
        # Models
        self.lgbm_model: Optional[lgb.Booster] = None
        self.survival_model: Optional[Any] = None
        
        # Residual distribution for uncertainty
        self.residual_std: float = 0.0
        self.log_target_mean: float = 0.0
        self.log_target_std: float = 0.0
        
        # Metadata
        self.training_metadata: Dict[str, Any] = {}
    
    def _compute_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute days_to_close target.
        
        Args:
            df: DataFrame with engage_date and close_date
            
        Returns:
            Series of days to close
        """
        days = (df["close_date"] - df["engage_date"]).dt.days
        return days.clip(lower=1)  # Minimum 1 day
    
    def _encode_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features and scale numeric features.
        
        Args:
            df: DataFrame with features
            fit: Whether to fit encoders
            
        Returns:
            Encoded DataFrame
        """
        df = df.copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if col not in df.columns:
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = df[col].fillna("Unknown").astype(str)
                self.label_encoders[col].fit(df[col])
            
            df[col] = df[col].fillna("Unknown").astype(str)
            known_classes = set(self.label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else "Unknown")
            
            if fit and "Unknown" not in known_classes:
                all_values = list(known_classes) + ["Unknown"]
                self.label_encoders[col].fit(all_values)
            
            df[col] = self.label_encoders[col].transform(df[col])
        
        # Scale numeric features
        numeric_cols = [c for c in self.numeric_features if c in df.columns]
        
        if fit:
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols].fillna(0))
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols].fillna(0))
        
        return df
    
    def train_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 500
    ) -> lgb.Booster:
        """
        Train log-transformed regression model.
        
        We predict log(days_to_close) for better distribution properties.
        
        Args:
            X_train: Training features
            y_train: Training target (days to close)
            X_val: Validation features
            y_val: Validation target
            num_boost_round: Maximum boosting rounds
            
        Returns:
            Trained LightGBM Booster
        """
        logger.info("Training time-to-close regression model...")
        
        # Log-transform target
        log_y_train = np.log1p(y_train)
        self.log_target_mean = log_y_train.mean()
        self.log_target_std = log_y_train.std()
        
        # Encode features
        X_train_encoded = self._encode_features(X_train, fit=True)
        feature_cols = [c for c in self.all_features if c in X_train_encoded.columns]
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train_encoded[feature_cols],
            label=log_y_train,
            feature_name=feature_cols
        )
        
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            log_y_val = np.log1p(y_val)
            X_val_encoded = self._encode_features(X_val, fit=False)
            val_data = lgb.Dataset(
                X_val_encoded[feature_cols],
                label=log_y_val,
                reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
        
        # Train
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=50)
        ]
        
        self.lgbm_model = lgb.train(
            self.LGBM_PARAMS,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Compute residual std for uncertainty estimation
        train_pred = self.lgbm_model.predict(X_train_encoded[feature_cols])
        residuals = log_y_train - train_pred
        self.residual_std = residuals.std()
        
        logger.info(f"Residual std (log scale): {self.residual_std:.4f}")
        logger.info(f"Target stats: mean={self.log_target_mean:.2f}, std={self.log_target_std:.2f}")
        
        return self.lgbm_model
    
    def train_survival(
        self,
        df: pd.DataFrame,
        duration_col: str = "days_to_close",
        event_col: str = "event"
    ) -> Any:
        """
        Train survival analysis model (Weibull AFT).
        
        Survival models naturally handle:
        - Right-censoring (open deals)
        - Non-negative predictions
        - Uncertainty quantification
        
        Args:
            df: DataFrame with duration and event columns
            duration_col: Column with time to event
            event_col: Column indicating if event occurred (1) or censored (0)
            
        Returns:
            Fitted Weibull AFT model
        """
        if not HAS_LIFELINES:
            logger.warning("lifelines not available. Skipping survival model.")
            return None
        
        logger.info("Training Weibull AFT survival model...")
        
        # Prepare data
        feature_cols = [c for c in self.all_features if c in df.columns]
        survival_df = df[feature_cols + [duration_col, event_col]].copy()
        
        # Fit Weibull AFT
        self.survival_model = WeibullAFTFitter()
        self.survival_model.fit(
            survival_df,
            duration_col=duration_col,
            event_col=event_col
        )
        
        logger.info(f"Weibull AFT fitted. Median survival time: {self.survival_model.median_survival_time_:.1f} days")
        
        return self.survival_model
    
    def train_full_pipeline(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        include_survival: bool = True
    ) -> Dict[str, Any]:
        """
        Run full training pipeline.
        
        Args:
            train_df: Training data with features and dates
            val_df: Validation data
            include_survival: Whether to train survival model
            
        Returns:
            Training metadata
        """
        logger.info("=" * 60)
        logger.info("Starting time-to-close training pipeline")
        logger.info("=" * 60)
        
        # Compute target
        y_train = self._compute_target(train_df)
        y_val = self._compute_target(val_df) if val_df is not None else None
        
        # Log target statistics
        logger.info(f"Target statistics:")
        logger.info(f"  Mean: {y_train.mean():.1f} days")
        logger.info(f"  Median: {y_train.median():.1f} days")
        logger.info(f"  Std: {y_train.std():.1f} days")
        logger.info(f"  Range: {y_train.min():.0f} - {y_train.max():.0f} days")
        
        # Train regression model
        feature_cols = [c for c in self.all_features if c in train_df.columns]
        self.train_regression(
            X_train=train_df[feature_cols],
            y_train=y_train,
            X_val=val_df[feature_cols] if val_df is not None else None,
            y_val=y_val
        )
        
        # Train survival model (optional)
        if include_survival and HAS_LIFELINES:
            train_surv = train_df.copy()
            train_surv["days_to_close"] = y_train
            train_surv["event"] = 1  # All training deals are closed
            self.train_survival(train_surv)
        
        # Store metadata
        self.training_metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "train_samples": len(train_df),
            "val_samples": len(val_df) if val_df is not None else 0,
            "features": self.all_features,
            "target_mean": float(y_train.mean()),
            "target_median": float(y_train.median()),
            "target_std": float(y_train.std()),
            "residual_std": float(self.residual_std),
            "has_survival_model": self.survival_model is not None,
        }
        
        logger.info("Training pipeline complete")
        
        return self.training_metadata
    
    def predict_distribution(
        self,
        X: pd.DataFrame,
        percentiles: List[float] = [10, 50, 90]
    ) -> pd.DataFrame:
        """
        Predict distribution of close times.
        
        Returns P10, P50, P90 (or custom percentiles) for each deal.
        
        Args:
            X: Features DataFrame
            percentiles: List of percentiles to compute
            
        Returns:
            DataFrame with percentile columns
        """
        if self.lgbm_model is None:
            raise ValueError("Model not trained")
        
        # Encode features
        X_encoded = self._encode_features(X, fit=False)
        feature_cols = [c for c in self.all_features if c in X_encoded.columns]
        
        # Predict log mean
        log_pred_mean = self.lgbm_model.predict(X_encoded[feature_cols])
        
        # Generate percentiles using normal approximation on log scale
        results = {}
        for p in percentiles:
            z_score = np.percentile(np.random.randn(10000), p)
            log_pred_p = log_pred_mean + z_score * self.residual_std
            results[f"days_p{p}"] = np.clip(np.expm1(log_pred_p), 1, None)
        
        # Point estimate (median)
        results["days_predicted"] = np.clip(np.expm1(log_pred_mean), 1, None)
        
        return pd.DataFrame(results)
    
    def predict_close_dates(
        self,
        X: pd.DataFrame,
        base_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Predict close date distributions.
        
        Args:
            X: Features DataFrame with engage_date
            base_date: Base date for predictions (default: now)
            
        Returns:
            DataFrame with predicted close dates
        """
        if base_date is None:
            base_date = datetime.now()
        
        # Get days distribution
        days_df = self.predict_distribution(X)
        
        # Convert to dates
        result = pd.DataFrame()
        result["close_date_p10"] = base_date + pd.to_timedelta(days_df["days_p10"], unit="D")
        result["close_date_p50"] = base_date + pd.to_timedelta(days_df["days_p50"], unit="D")
        result["close_date_p90"] = base_date + pd.to_timedelta(days_df["days_p90"], unit="D")
        result["days_predicted"] = days_df["days_predicted"]
        
        return result
    
    def save(self, version: str = "1.0.0") -> Path:
        """
        Save model artifacts.
        
        Args:
            version: Model version string
            
        Returns:
            Path to saved artifact
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_name = f"time_to_close_v{version}_{timestamp}"
        artifact_path = self.artifacts_dir / f"{artifact_name}.pkl"
        
        artifact = {
            "version": version,
            "trained_at": self.training_metadata.get("trained_at"),
            "lgbm_model": self.lgbm_model,
            "survival_model": self.survival_model,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "residual_std": self.residual_std,
            "log_target_mean": self.log_target_mean,
            "log_target_std": self.log_target_std,
            "features": self.all_features,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "metadata": self.training_metadata,
        }
        
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)
        
        # Save metadata JSON
        metadata_path = self.artifacts_dir / f"{artifact_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {artifact_path}")
        
        return artifact_path
    
    @classmethod
    def load(cls, artifact_path: str) -> "TimeToCloseTrainer":
        """
        Load model from artifact.
        
        Args:
            artifact_path: Path to saved artifact
            
        Returns:
            Loaded trainer instance
        """
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)
        
        trainer = cls(
            categorical_features=artifact["categorical_features"],
            numeric_features=artifact["numeric_features"]
        )
        
        trainer.lgbm_model = artifact["lgbm_model"]
        trainer.survival_model = artifact["survival_model"]
        trainer.label_encoders = artifact["label_encoders"]
        trainer.scaler = artifact["scaler"]
        trainer.residual_std = artifact["residual_std"]
        trainer.log_target_mean = artifact["log_target_mean"]
        trainer.log_target_std = artifact["log_target_std"]
        trainer.training_metadata = artifact["metadata"]
        
        return trainer


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    from core.data.preprocessor import DataPreprocessor
    from core.data.features import FeatureEngineer, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    
    # Load and preprocess (Won deals only for time-to-close)
    preprocessor = DataPreprocessor("dataset")
    full_data = preprocessor.preprocess(for_training=True)
    
    # Filter to Won deals only (they have actual close times)
    won_deals = full_data[full_data["deal_stage"] == "Won"].copy()
    
    # Feature engineering
    engineer = FeatureEngineer()
    featured_data = engineer.compute_all_features(won_deals, full_data)
    
    # Split
    train_df, test_df = preprocessor.create_time_split(featured_data)
    
    # Train
    trainer = TimeToCloseTrainer(
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=NUMERIC_FEATURES
    )
    trainer.train_full_pipeline(train_df, test_df)
    
    # Predict
    predictions = trainer.predict_distribution(test_df)
    
    print("\n=== Predictions ===")
    print(predictions.head(10))
    
    # Save
    trainer.save()

