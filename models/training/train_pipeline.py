"""
Phase 1A Model Training Pipeline
Revenue Intelligence System

This script runs the complete Phase 1A ML pipeline:
1. Data preprocessing
2. Feature engineering
3. Win probability model training
4. Time-to-close model training
5. Model evaluation
6. Artifact saving
7. Experiment tracking (JSON + optional MLflow)

Usage:
    python -m models.training.train_pipeline
    python -m models.training.train_pipeline --feature-tier minimal
    python -m models.training.train_pipeline --experiment tuning_v1
    
    Or with custom data path:
    python -m models.training.train_pipeline --data-dir /path/to/data
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

from core.data.preprocessor import DataPreprocessor
from core.data.features import (
    FeatureEngineer, 
    CATEGORICAL_FEATURES, 
    NUMERIC_FEATURES,
    get_feature_documentation,
    get_feature_tier,
)
from models.training.win_probability import WinProbabilityTrainer
from models.training.time_to_close import TimeToCloseTrainer
from models.evaluation.evaluate import ModelEvaluator, print_evaluation_report
from models.tracking.experiment_tracker import ExperimentTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


def train_pipeline(
    data_dir: str = "dataset",
    save_models: bool = True,
    experiment_name: str = "phase1a",
    feature_tier: str = "standard",
) -> dict:
    """
    Run complete Phase 1A training pipeline.
    
    Args:
        data_dir: Path to dataset directory
        save_models: Whether to save trained models
        experiment_name: Name for experiment tracking
        feature_tier: Feature set tier ('minimal', 'standard', 'full')
        
    Returns:
        Dictionary with training results
    """
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": data_dir,
        "feature_tier": feature_tier,
    }
    
    # Get feature set for this tier
    tier_features = get_feature_tier(feature_tier)
    tier_categorical = tier_features["categorical"]
    tier_numeric = tier_features["numeric"]
    
    logger.info(f"Using feature tier: {feature_tier}")
    logger.info(f"  Categorical: {tier_categorical}")
    logger.info(f"  Numeric: {tier_numeric}")
    
    # =========================================================================
    # Step 1: Data Preprocessing
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 70)
    
    preprocessor = DataPreprocessor(data_dir)
    full_data = preprocessor.preprocess(for_training=True)
    
    results["total_deals"] = len(full_data)
    results["won_deals"] = (full_data["deal_stage"] == "Won").sum()
    results["lost_deals"] = (full_data["deal_stage"] == "Lost").sum()
    
    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    engineer = FeatureEngineer()
    featured_data = engineer.compute_all_features(full_data, full_data)
    
    # Get available feature columns based on tier
    available_categorical = [c for c in tier_categorical if c in featured_data.columns]
    available_numeric = [c for c in tier_numeric if c in featured_data.columns]
    all_features = available_categorical + available_numeric
    
    logger.info(f"Categorical features: {available_categorical}")
    logger.info(f"Numeric features: {available_numeric}")
    
    results["features_used"] = all_features
    results["n_features"] = len(all_features)
    
    # =========================================================================
    # Step 3: Train/Test Split
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 3: TIME-BASED TRAIN/TEST SPLIT")
    logger.info("=" * 70)
    
    train_df, test_df = preprocessor.create_time_split(featured_data)
    
    results["train_samples"] = len(train_df)
    results["test_samples"] = len(test_df)
    results["train_win_rate"] = train_df["target"].mean()
    results["test_win_rate"] = test_df["target"].mean()
    
    # Log parameters to tracker
    tracker.log_params({
        "data_dir": data_dir,
        "feature_tier": feature_tier,
        "n_features": len(all_features),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_win_rate": round(train_df["target"].mean(), 4),
        "test_win_rate": round(test_df["target"].mean(), 4),
        # LightGBM hyperparams will be logged from the trainer
    })
    
    # =========================================================================
    # Step 4: Win Probability Model
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 4: WIN PROBABILITY MODEL")
    logger.info("=" * 70)
    
    win_trainer = WinProbabilityTrainer(
        categorical_features=available_categorical,
        numeric_features=available_numeric
    )
    
    # Log LightGBM hyperparameters
    tracker.log_params({
        "lgbm_" + k: v for k, v in win_trainer.LGBM_PARAMS.items()
        if k not in ["verbose", "n_jobs", "random_state"]  # Skip non-essential params
    })
    
    win_trainer.train_full_pipeline(
        X_train=train_df[all_features],
        y_train=train_df["target"],
        X_val=test_df[all_features],
        y_val=test_df["target"]
    )
    
    # Evaluate win probability model
    evaluator = ModelEvaluator()
    y_test = test_df["target"].values
    y_pred_proba = win_trainer.predict_proba(test_df[all_features])
    
    win_result = evaluator.full_evaluation(y_test, y_pred_proba)
    print_evaluation_report(win_result)
    
    results["win_prob_metrics"] = {
        "auc_roc": win_result.auc_roc,
        "brier_score": win_result.brier_score,
        "ece": win_result.expected_calibration_error,
        "precision": win_result.precision,
        "recall": win_result.recall,
        "f1": win_result.f1,
    }
    
    # Check acceptance criteria
    passes, checks = evaluator.check_passes_threshold(results["win_prob_metrics"])
    results["win_prob_passes_criteria"] = passes
    
    if save_models:
        win_artifact_path = win_trainer.save()
        results["win_prob_artifact"] = str(win_artifact_path)
    
    # =========================================================================
    # Step 5: Time-to-Close Model
    # =========================================================================
    logger.info("=" * 70)
    logger.info("STEP 5: TIME-TO-CLOSE MODEL")
    logger.info("=" * 70)
    
    # Filter to Won deals only (they have actual close times)
    train_won = train_df[train_df["deal_stage"] == "Won"].copy()
    test_won = test_df[test_df["deal_stage"] == "Won"].copy()
    
    ttc_trainer = TimeToCloseTrainer(
        categorical_features=available_categorical,
        numeric_features=available_numeric
    )
    
    ttc_trainer.train_full_pipeline(
        train_df=train_won,
        val_df=test_won,
        include_survival=False  # Skip survival model for now
    )
    
    # Evaluate time-to-close predictions
    ttc_predictions = ttc_trainer.predict_distribution(test_won[all_features])
    actual_days = (test_won["close_date"] - test_won["engage_date"]).dt.days
    
    # Compute RMSE and MAE
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(actual_days, ttc_predictions["days_predicted"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_days, ttc_predictions["days_predicted"])
    
    logger.info(f"Time-to-close RMSE: {rmse:.1f} days")
    logger.info(f"Time-to-close MAE: {mae:.1f} days")
    
    results["ttc_metrics"] = {
        "rmse_days": rmse,
        "mae_days": mae,
        "target_mean": actual_days.mean(),
        "predicted_mean": ttc_predictions["days_predicted"].mean(),
    }
    
    if save_models:
        ttc_artifact_path = ttc_trainer.save()
        results["ttc_artifact"] = str(ttc_artifact_path)
    
    # =========================================================================
    # Step 6: Summary
    # =========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    logger.info(f"Win Probability AUC: {results['win_prob_metrics']['auc_roc']:.4f}")
    logger.info(f"Win Probability Brier: {results['win_prob_metrics']['brier_score']:.4f}")
    logger.info(f"Time-to-Close RMSE: {results['ttc_metrics']['rmse_days']:.1f} days")
    logger.info(f"Passes criteria: {'YES' if results['win_prob_passes_criteria'] else 'NO'}")
    
    # Log metrics to tracker
    tracker.log_metrics({
        # Win probability metrics
        "auc_roc": results["win_prob_metrics"]["auc_roc"],
        "brier_score": results["win_prob_metrics"]["brier_score"],
        "ece": results["win_prob_metrics"]["ece"],
        "precision": results["win_prob_metrics"]["precision"],
        "recall": results["win_prob_metrics"]["recall"],
        "f1": results["win_prob_metrics"]["f1"],
        # Time-to-close metrics
        "ttc_rmse": results["ttc_metrics"]["rmse_days"],
        "ttc_mae": results["ttc_metrics"]["mae_days"],
        # Pass/fail
        "passes_criteria": 1.0 if results["win_prob_passes_criteria"] else 0.0,
    })
    
    # Log model artifacts if saved
    if save_models and "win_prob_artifact" in results:
        tracker.log_artifact(results["win_prob_artifact"])
    if save_models and "ttc_artifact" in results:
        tracker.log_artifact(results["ttc_artifact"])
    
    # End tracking run
    tracker.end_run("completed")
    results["run_id"] = tracker.run_id
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1A ML models")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save model artifacts"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="phase1a",
        help="Experiment name for tracking (default: phase1a)"
    )
    parser.add_argument(
        "--feature-tier",
        type=str,
        default="standard",
        choices=["minimal", "standard", "full"],
        help="Feature set tier: minimal, standard, or full (default: standard)"
    )
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run training
    results = train_pipeline(
        data_dir=args.data_dir,
        save_models=not args.no_save,
        experiment_name=args.experiment,
        feature_tier=args.feature_tier,
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 1A TRAINING RESULTS")
    print("=" * 70)
    print(f"Total deals processed: {results['total_deals']}")
    print(f"Train samples: {results['train_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"\nWin Probability Model:")
    print(f"  AUC-ROC: {results['win_prob_metrics']['auc_roc']:.4f}")
    print(f"  Brier Score: {results['win_prob_metrics']['brier_score']:.4f}")
    print(f"  ECE: {results['win_prob_metrics']['ece']:.4f}")
    print(f"\nTime-to-Close Model:")
    print(f"  RMSE: {results['ttc_metrics']['rmse_days']:.1f} days")
    print(f"  MAE: {results['ttc_metrics']['mae_days']:.1f} days")
    print(f"\nPasses Phase 1A Criteria: {'YES' if results['win_prob_passes_criteria'] else 'NO'}")
    print(f"\n[TRACKING]")
    print(f"  Run ID: {results.get('run_id', 'N/A')}")
    print(f"  Feature Tier: {results.get('feature_tier', 'N/A')}")
    print(f"  Compare runs: python -m models.tracking.compare_runs")
    
    return 0 if results['win_prob_passes_criteria'] else 1


if __name__ == "__main__":
    sys.exit(main())

