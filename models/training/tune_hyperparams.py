"""
Hyperparameter Tuning for Win Probability Model
Revenue Intelligence System

This script performs grid search over key hyperparameters to find
the best configuration for the win probability model.

Usage:
    python -m models.training.tune_hyperparams
    python -m models.training.tune_hyperparams --quick  # Reduced grid for testing
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
from typing import Dict, List, Any
import numpy as np

from core.data.preprocessor import DataPreprocessor
from core.data.features import FeatureEngineer, get_feature_tier
from models.training.win_probability import WinProbabilityTrainer
from models.evaluation.evaluate import ModelEvaluator
from models.tracking.experiment_tracker import ExperimentTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# Hyperparameter search grid
PARAM_GRID = {
    "num_leaves": [4, 8, 16],
    "min_data_in_leaf": [20, 50, 100],
    "lambda_l2": [0.1, 1.0, 10.0],
    "feature_fraction": [0.5, 0.7],
}

# Reduced grid for quick testing
QUICK_PARAM_GRID = {
    "num_leaves": [4, 8],
    "min_data_in_leaf": [50],
    "lambda_l2": [1.0],
    "feature_fraction": [0.6],
}

# Feature tiers to test
FEATURE_TIERS = ["minimal", "standard"]


def generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of hyperparameters."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


def run_single_experiment(
    train_df,
    test_df,
    categorical_features: List[str],
    numeric_features: List[str],
    params: Dict[str, Any],
    feature_tier: str,
    experiment_name: str,
) -> Dict[str, Any]:
    """Run a single training experiment with specified parameters."""
    
    all_features = categorical_features + numeric_features
    
    # Initialize tracker
    tracker = ExperimentTracker(experiment_name)
    
    # Log parameters
    tracker.log_params({
        "feature_tier": feature_tier,
        "n_features": len(all_features),
        **{f"lgbm_{k}": v for k, v in params.items()},
    })
    tracker.set_tag("tuning_run", "true")
    
    # Create trainer with custom params
    trainer = WinProbabilityTrainer(
        categorical_features=categorical_features,
        numeric_features=numeric_features
    )
    
    # Override default params
    for key, value in params.items():
        trainer.LGBM_PARAMS[key] = value
    
    try:
        # Train
        trainer.train_full_pipeline(
            X_train=train_df[all_features],
            y_train=train_df["target"],
            X_val=test_df[all_features],
            y_val=test_df["target"]
        )
        
        # Evaluate
        evaluator = ModelEvaluator()
        y_test = test_df["target"].values
        y_pred_proba = trainer.predict_proba(test_df[all_features])
        
        result = evaluator.full_evaluation(y_test, y_pred_proba)
        
        metrics = {
            "auc_roc": result.auc_roc,
            "brier_score": result.brier_score,
            "ece": result.expected_calibration_error,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        }
        
        # Also compute train AUC for overfitting detection
        y_train = train_df["target"].values
        y_train_pred = trainer.predict_proba(train_df[all_features])
        train_auc = evaluator.compute_auc(y_train, y_train_pred)
        metrics["train_auc"] = train_auc
        metrics["overfit_gap"] = train_auc - result.auc_roc
        
        tracker.log_metrics(metrics)
        tracker.end_run("completed")
        
        return {
            "run_id": tracker.run_id,
            "params": params,
            "feature_tier": feature_tier,
            "metrics": metrics,
            "status": "success",
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        tracker.end_run("failed")
        return {
            "run_id": tracker.run_id,
            "params": params,
            "feature_tier": feature_tier,
            "metrics": {},
            "status": "failed",
            "error": str(e),
        }


def run_tuning(
    data_dir: str = "dataset",
    quick: bool = False,
    experiment_name: str = "tuning",
) -> List[Dict[str, Any]]:
    """Run full hyperparameter tuning."""
    
    param_grid = QUICK_PARAM_GRID if quick else PARAM_GRID
    param_combos = generate_param_combinations(param_grid)
    
    total_experiments = len(param_combos) * len(FEATURE_TIERS)
    logger.info(f"Starting hyperparameter tuning")
    logger.info(f"  Parameter combinations: {len(param_combos)}")
    logger.info(f"  Feature tiers: {FEATURE_TIERS}")
    logger.info(f"  Total experiments: {total_experiments}")
    
    # Prepare data once
    logger.info("Preparing data...")
    preprocessor = DataPreprocessor(data_dir)
    full_data = preprocessor.preprocess(for_training=True)
    
    engineer = FeatureEngineer()
    featured_data = engineer.compute_all_features(full_data, full_data)
    
    train_df, test_df = preprocessor.create_time_split(featured_data)
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Run experiments
    all_results = []
    
    for tier in FEATURE_TIERS:
        tier_features = get_feature_tier(tier)
        categorical = [c for c in tier_features["categorical"] if c in featured_data.columns]
        numeric = [n for n in tier_features["numeric"] if n in featured_data.columns]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Feature Tier: {tier} ({len(categorical)} cat, {len(numeric)} num)")
        logger.info(f"{'='*60}")
        
        for i, params in enumerate(param_combos):
            logger.info(f"  Experiment {i+1}/{len(param_combos)}: {params}")
            
            result = run_single_experiment(
                train_df=train_df,
                test_df=test_df,
                categorical_features=categorical,
                numeric_features=numeric,
                params=params,
                feature_tier=tier,
                experiment_name=experiment_name,
            )
            
            all_results.append(result)
            
            if result["status"] == "success":
                auc = result["metrics"]["auc_roc"]
                gap = result["metrics"]["overfit_gap"]
                logger.info(f"    AUC: {auc:.4f}, Overfit Gap: {gap:.4f}")
            else:
                logger.info(f"    FAILED: {result.get('error', 'Unknown')}")
    
    return all_results


def print_tuning_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary of tuning results."""
    
    successful = [r for r in results if r["status"] == "success"]
    
    if not successful:
        print("\nNo successful experiments!")
        return
    
    # Sort by AUC
    successful.sort(key=lambda x: x["metrics"]["auc_roc"], reverse=True)
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 80)
    
    print(f"\nTotal experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    
    print("\n" + "-" * 80)
    print("TOP 5 CONFIGURATIONS (by AUC)")
    print("-" * 80)
    print(f"{'Rank':<5} {'Tier':<10} {'AUC':<8} {'Gap':<8} {'Brier':<8} {'Params'}")
    print("-" * 80)
    
    for i, r in enumerate(successful[:5]):
        tier = r["feature_tier"]
        auc = r["metrics"]["auc_roc"]
        gap = r["metrics"]["overfit_gap"]
        brier = r["metrics"]["brier_score"]
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        
        print(f"{i+1:<5} {tier:<10} {auc:<8.4f} {gap:<8.4f} {brier:<8.4f} {params_str}")
    
    # Best overall
    best = successful[0]
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(f"  Feature Tier: {best['feature_tier']}")
    print(f"  AUC-ROC: {best['metrics']['auc_roc']:.4f}")
    print(f"  Overfit Gap: {best['metrics']['overfit_gap']:.4f}")
    print(f"  Brier Score: {best['metrics']['brier_score']:.4f}")
    print(f"  Parameters:")
    for k, v in best["params"].items():
        print(f"    {k}: {v}")
    print(f"\n  Run ID: {best['run_id']}")
    
    # Find best with low overfitting
    low_overfit = [r for r in successful if r["metrics"]["overfit_gap"] < 0.10]
    if low_overfit:
        low_overfit.sort(key=lambda x: x["metrics"]["auc_roc"], reverse=True)
        best_stable = low_overfit[0]
        
        if best_stable != best:
            print("\n" + "-" * 80)
            print("BEST STABLE CONFIG (overfit gap < 0.10)")
            print("-" * 80)
            print(f"  Feature Tier: {best_stable['feature_tier']}")
            print(f"  AUC-ROC: {best_stable['metrics']['auc_roc']:.4f}")
            print(f"  Overfit Gap: {best_stable['metrics']['overfit_gap']:.4f}")
            print(f"  Parameters:")
            for k, v in best_stable["params"].items():
                print(f"    {k}: {v}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Update win_probability.py with best params")
    print("  2. Run: python train.py --feature-tier <best_tier>")
    print("  3. Compare runs: python -m models.tracking.compare_runs")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for win probability model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced parameter grid for quick testing"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="tuning",
        help="Experiment name for tracking"
    )
    
    args = parser.parse_args()
    
    # Run tuning
    results = run_tuning(
        data_dir=args.data_dir,
        quick=args.quick,
        experiment_name=args.experiment,
    )
    
    # Print summary
    print_tuning_summary(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

