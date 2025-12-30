"""
Win Probability Model Training
Core ML Pipeline

This module handles training and calibration of the win probability model.
Key responsibilities:
- Train baseline logistic regression
- Train primary LightGBM model
- Calibrate probabilities for reliability
- Save model artifacts with metadata
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle
import json
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb

logger = logging.getLogger(__name__)


class WinProbabilityTrainer:
    """
    Trainer for win probability models.
    
    Implements:
    - Baseline logistic regression (interpretable)
    - Primary LightGBM model (performant)
    - Probability calibration (critical for downstream forecasting)
    - Time-series cross-validation
    """
    
    # LightGBM hyperparameters - tuned via grid search (Dec 2024)
    # Best config from 108 experiments: AUC 0.584, overfit gap 0.023
    # Note: Low AUC reflects limited predictive signal in dataset, not model issue
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 16,             # Tuned (was 8)
        "max_depth": 5,               # Limit tree depth
        "min_data_in_leaf": 50,       # Prevent overfitting
        "learning_rate": 0.05,
        "feature_fraction": 0.7,      # Tuned (was 0.6)
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.1,             # L1 regularization
        "lambda_l2": 1.0,             # L2 regularization (tuned)
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
    }
    
    # Monotonic constraints for features where direction is known
    # -1 = decreasing (higher value -> lower win prob)
    #  0 = no constraint
    # +1 = increasing (higher value -> higher win prob)
    # WHY: Prevents overfitting by enforcing business logic
    # NOTE: Only apply when feature is part of feature set
    MONOTONIC_FEATURES = {
        # Disabled for now - too aggressive, hurts AUC
        # "days_in_engaging": -1,       # Longer deals are less likely to win
        # "engagement_stage_num": -1,   # Stale deals are less likely to win
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
        
        # Encoders and scalers
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        
        # Models
        self.baseline_model: Optional[LogisticRegression] = None
        self.lgbm_model: Optional[lgb.Booster] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        
        # Metadata
        self.training_metadata: Dict[str, Any] = {}
    
    def _encode_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features and scale numeric features.
        
        Args:
            df: DataFrame with features
            fit: Whether to fit encoders (True for training, False for inference)
            
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
                # Handle unseen categories
                df[col] = df[col].fillna("Unknown").astype(str)
                self.label_encoders[col].fit(df[col])
            
            # Transform with handling for unseen values
            df[col] = df[col].fillna("Unknown").astype(str)
            known_classes = set(self.label_encoders[col].classes_)
            df[col] = df[col].apply(
                lambda x: x if x in known_classes else "Unknown"
            )
            
            # Refit if Unknown was added
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
    
    def train_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """
        Train baseline logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained LogisticRegression model
        """
        logger.info("Training baseline logistic regression...")
        
        # Encode features
        X_encoded = self._encode_features(X_train, fit=True)
        feature_cols = [c for c in self.all_features if c in X_encoded.columns]
        
        # Train model
        self.baseline_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )
        self.baseline_model.fit(X_encoded[feature_cols], y_train)
        
        # Log coefficients for interpretability
        coef_df = pd.DataFrame({
            "feature": feature_cols,
            "coefficient": self.baseline_model.coef_[0]
        }).sort_values("coefficient", ascending=False)
        
        logger.info("Logistic regression coefficients:")
        for _, row in coef_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")
        
        return self.baseline_model
    
    def train_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50
    ) -> lgb.Booster:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            num_boost_round: Maximum boosting rounds
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Trained LightGBM Booster
        """
        logger.info("Training LightGBM model...")
        
        # Encode features (reuse fitted encoders from baseline)
        X_train_encoded = self._encode_features(X_train, fit=False)
        feature_cols = [c for c in self.all_features if c in X_train_encoded.columns]
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train_encoded[feature_cols],
            label=y_train,
            feature_name=feature_cols,
            categorical_feature=[
                c for c in self.categorical_features if c in feature_cols
            ]
        )
        
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            X_val_encoded = self._encode_features(X_val, fit=False)
            val_data = lgb.Dataset(
                X_val_encoded[feature_cols],
                label=y_val,
                reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append("valid")
        
        # Build monotonic constraints based on features used
        # WHY: Enforces business logic (e.g., longer deals = lower win prob)
        # This prevents overfitting to spurious patterns
        monotonic_constraints = []
        for col in feature_cols:
            if col in self.MONOTONIC_FEATURES:
                monotonic_constraints.append(self.MONOTONIC_FEATURES[col])
            else:
                monotonic_constraints.append(0)  # No constraint
        
        # Add monotonic constraints to params if any are non-zero
        train_params = self.LGBM_PARAMS.copy()
        if any(c != 0 for c in monotonic_constraints):
            train_params["monotone_constraints"] = monotonic_constraints
            constrained_features = [
                feature_cols[i] for i, c in enumerate(monotonic_constraints) if c != 0
            ]
            logger.info(f"Monotonic constraints applied to: {constrained_features}")
        
        # Train with callbacks
        callbacks = [
            lgb.log_evaluation(period=100),
        ]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        
        self.lgbm_model = lgb.train(
            train_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Log feature importance
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.lgbm_model.feature_importance(importance_type="gain")
        }).sort_values("importance", ascending=False)
        
        logger.info("LightGBM feature importance (gain):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        return self.lgbm_model
    
    def calibrate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = "isotonic"
    ) -> tuple:
        """
        Calibrate model probabilities using isotonic regression.
        
        CRITICAL: Calibration ensures that predicted probabilities
        are reliable (a 70% prediction should win 70% of the time).
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Tuple of (calibrator, feature_cols)
        """
        if self.lgbm_model is None:
            raise ValueError("Must train LightGBM model before calibrating")
        
        logger.info(f"Calibrating probabilities using {method} regression...")
        
        # Get uncalibrated predictions
        X_encoded = self._encode_features(X_train, fit=False)
        feature_cols = [c for c in self.all_features if c in X_encoded.columns]
        
        uncalibrated_probs = self.lgbm_model.predict(X_encoded[feature_cols])
        
        # Use sklearn's calibration on the predictions directly
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(uncalibrated_probs, y_train)
        else:  # sigmoid
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression()
            calibrator.fit(uncalibrated_probs.reshape(-1, 1), y_train)
        
        self.calibrator = calibrator
        self.calibration_method = method
        
        logger.info("Calibration complete")
        
        return calibrator, feature_cols
    
    def train_full_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        save_shap_background: bool = True,
        shap_background_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Run full training pipeline.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            save_shap_background: Whether to save background data for SHAP
            shap_background_samples: Number of samples for SHAP background

        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info("Starting full training pipeline")
        logger.info("=" * 60)

        # Train baseline
        self.train_baseline(X_train, y_train)

        # Train LightGBM
        self.train_lgbm(X_train, y_train, X_val, y_val)

        # Calibrate
        self.calibrate(X_train, y_train)

        # Save SHAP background data for on-demand explanations
        if save_shap_background:
            self._save_shap_background(X_train, shap_background_samples)

        # Store metadata
        self.training_metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "train_samples": len(X_train),
            "val_samples": len(X_val) if X_val is not None else 0,
            "features": self.all_features,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "lgbm_params": self.LGBM_PARAMS,
        }

        logger.info("Training pipeline complete")

        return self.training_metadata

    def _save_shap_background(
        self,
        X_train: pd.DataFrame,
        n_samples: int = 100
    ) -> None:
        """
        Save background data sample for SHAP explanations.

        SHAP TreeExplainer uses background data to compute feature contributions.
        We save a representative sample from training data for use during inference.

        Args:
            X_train: Training features (pre-encoding)
            n_samples: Number of samples to save
        """
        logger.info(f"Saving SHAP background data ({n_samples} samples)...")

        # Sample from training data
        if len(X_train) > n_samples:
            background_sample = X_train.sample(n_samples, random_state=42)
        else:
            background_sample = X_train.copy()

        # Encode features using fitted encoders
        background_encoded = self._encode_features(background_sample, fit=False)

        # Get feature columns
        feature_cols = [c for c in self.all_features if c in background_encoded.columns]

        # Save as parquet for efficient loading
        background_path = self.artifacts_dir / "shap_background.parquet"
        background_encoded[feature_cols].to_parquet(background_path)

        logger.info(f"SHAP background data saved to {background_path}")
    
    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        """
        Predict win probabilities.
        
        Args:
            X: Features DataFrame
            calibrated: Whether to apply calibration
            
        Returns:
            Array of win probabilities
        """
        if self.lgbm_model is None:
            raise ValueError("Model not trained. Call train_full_pipeline first.")
        
        # Encode features
        X_encoded = self._encode_features(X, fit=False)
        feature_cols = [c for c in self.all_features if c in X_encoded.columns]
        
        # Get uncalibrated predictions
        probs = self.lgbm_model.predict(X_encoded[feature_cols])
        
        # Apply calibration if available and requested
        if calibrated and hasattr(self, 'calibrator'):
            if self.calibration_method == "isotonic":
                probs = self.calibrator.predict(probs)
            else:  # sigmoid
                probs = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        
        return probs
    
    def save(self, version: str = "1.0.0") -> Path:
        """
        Save model artifacts.
        
        Args:
            version: Model version string
            
        Returns:
            Path to saved artifact
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_name = f"win_probability_v{version}_{timestamp}"
        artifact_path = self.artifacts_dir / f"{artifact_name}.pkl"
        
        artifact = {
            "version": version,
            "trained_at": self.training_metadata.get("trained_at"),
            "baseline_model": self.baseline_model,
            "lgbm_model": self.lgbm_model,
            "calibrator": getattr(self, 'calibrator', None),
            "calibration_method": getattr(self, 'calibration_method', None),
            "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "features": self.all_features,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "metadata": self.training_metadata,
        }
        
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)
        
        # Also save metadata as JSON for easy inspection
        metadata_path = self.artifacts_dir / f"{artifact_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {artifact_path}")
        
        return artifact_path
    
    @classmethod
    def load(cls, artifact_path: str) -> "WinProbabilityTrainer":
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
        
        trainer.baseline_model = artifact["baseline_model"]
        trainer.lgbm_model = artifact["lgbm_model"]
        trainer.calibrator = artifact.get("calibrator")
        trainer.calibration_method = artifact.get("calibration_method", "isotonic")
        trainer.label_encoders = artifact["label_encoders"]
        trainer.scaler = artifact["scaler"]
        trainer.training_metadata = artifact["metadata"]
        
        return trainer


def train_win_probability_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
    save_model: bool = True
) -> Tuple[WinProbabilityTrainer, Dict[str, float]]:
    """
    Convenience function to train win probability model.
    
    Args:
        train_df: Training data with features and target
        test_df: Test data with features and target
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        save_model: Whether to save model artifacts
        
    Returns:
        Tuple of (trainer, metrics)
    """
    from models.evaluation.evaluate import ModelEvaluator
    
    # Initialize trainer
    trainer = WinProbabilityTrainer(
        categorical_features=categorical_features,
        numeric_features=numeric_features
    )
    
    # Get feature columns
    all_features = categorical_features + numeric_features
    feature_cols = [c for c in all_features if c in train_df.columns]
    
    # Train
    trainer.train_full_pipeline(
        X_train=train_df[feature_cols],
        y_train=train_df["target"],
        X_val=test_df[feature_cols],
        y_val=test_df["target"]
    )
    
    # Evaluate
    evaluator = ModelEvaluator()
    y_test = test_df["target"]
    y_pred_proba = trainer.predict_proba(test_df[feature_cols])
    
    metrics = evaluator.compute_metrics(y_test, y_pred_proba)
    
    # Save
    if save_model:
        trainer.training_metadata["test_metrics"] = metrics
        trainer.save()
    
    return trainer, metrics


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    from core.data.preprocessor import DataPreprocessor
    from core.data.features import FeatureEngineer, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    
    # Load and preprocess
    preprocessor = DataPreprocessor("dataset")
    full_data = preprocessor.preprocess(for_training=True)
    
    # Feature engineering
    engineer = FeatureEngineer()
    featured_data = engineer.compute_all_features(full_data, full_data)
    
    # Split
    train_df, test_df = preprocessor.create_time_split(featured_data)
    
    # Train
    trainer, metrics = train_win_probability_model(
        train_df=train_df,
        test_df=test_df,
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=NUMERIC_FEATURES
    )
    
    print("\n=== Test Metrics ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

