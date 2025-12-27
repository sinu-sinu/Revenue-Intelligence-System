"""
Model Evaluation Module
Phase 1A: Core ML Pipeline

This module provides comprehensive evaluation metrics for ML models.
Key responsibilities:
- Compute discrimination metrics (AUC-ROC)
- Compute calibration metrics (Brier, ECE)
- Generate reliability diagrams
- Detect feature/prediction drift
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Complete evaluation results for a model.
    """
    # Discrimination
    auc_roc: float
    
    # Calibration
    brier_score: float
    expected_calibration_error: float
    
    # Classification (at 0.5 threshold)
    precision: float
    recall: float
    f1: float
    
    # Additional
    confusion_matrix: np.ndarray
    n_samples: int
    positive_rate: float
    
    # Calibration curve data
    calibration_bins: List[float]
    calibration_true_prob: List[float]
    calibration_pred_prob: List[float]
    calibration_counts: List[int]


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    
    Implements metrics critical for:
    - Discrimination: Can the model rank deals correctly?
    - Calibration: Are predicted probabilities reliable?
    """
    
    def __init__(self, n_calibration_bins: int = 10):
        """
        Initialize evaluator.
        
        Args:
            n_calibration_bins: Number of bins for calibration analysis
        """
        self.n_calibration_bins = n_calibration_bins
    
    def compute_auc(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Compute AUC-ROC score.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            AUC-ROC score
        """
        return roc_auc_score(np.asarray(y_true), np.asarray(y_pred_proba))
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            # Discrimination
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            
            # Calibration
            "brier_score": brier_score_loss(y_true, y_pred_proba),
            "ece": self._expected_calibration_error(y_true, y_pred_proba),
            
            # Classification
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            
            # Summary stats
            "n_samples": len(y_true),
            "positive_rate": y_true.mean(),
            "predicted_positive_rate": y_pred.mean(),
        }
        
        return metrics
    
    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE measures the average gap between predicted probability
        and actual frequency in probability bins.
        
        Lower is better. < 0.05 is well-calibrated.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            ECE value
        """
        bin_edges = np.linspace(0, 1, self.n_calibration_bins + 1)
        ece = 0.0
        
        for i in range(self.n_calibration_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_confidence = y_pred_proba[mask].mean()
                bin_accuracy = y_true[mask].mean()
                bin_weight = mask.sum() / len(y_true)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def get_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for reliability diagram.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Tuple of (bin_centers, true_probs, pred_probs, counts)
        """
        bin_edges = np.linspace(0, 1, self.n_calibration_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        true_probs = []
        pred_probs = []
        counts = []
        
        for i in range(self.n_calibration_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            count = mask.sum()
            counts.append(count)
            
            if count > 0:
                true_probs.append(y_true[mask].mean())
                pred_probs.append(y_pred_proba[mask].mean())
            else:
                true_probs.append(np.nan)
                pred_probs.append(np.nan)
        
        return np.array(bin_centers), np.array(true_probs), np.array(pred_probs), np.array(counts)
    
    def full_evaluation(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> EvaluationResult:
        """
        Run full evaluation and return structured result.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            EvaluationResult object
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        bin_centers, true_probs, pred_probs, counts = self.get_calibration_curve(
            y_true, y_pred_proba
        )
        
        return EvaluationResult(
            auc_roc=roc_auc_score(y_true, y_pred_proba),
            brier_score=brier_score_loss(y_true, y_pred_proba),
            expected_calibration_error=self._expected_calibration_error(y_true, y_pred_proba),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            n_samples=len(y_true),
            positive_rate=y_true.mean(),
            calibration_bins=bin_centers.tolist(),
            calibration_true_prob=true_probs.tolist(),
            calibration_pred_prob=pred_probs.tolist(),
            calibration_counts=counts.tolist()
        )
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            y_true: True labels
            predictions: Dict mapping model name to predictions
            
        Returns:
            DataFrame with metrics for each model
        """
        results = []
        
        for name, y_pred_proba in predictions.items():
            metrics = self.compute_metrics(y_true, y_pred_proba)
            metrics["model"] = name
            results.append(metrics)
        
        return pd.DataFrame(results).set_index("model")
    
    def check_passes_threshold(
        self,
        metrics: Dict[str, float],
        auc_threshold: float = 0.75,
        brier_threshold: float = 0.20,
        ece_threshold: float = 0.10
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if model passes Phase 1A acceptance criteria.
        
        Args:
            metrics: Dictionary of metrics
            auc_threshold: Minimum AUC-ROC
            brier_threshold: Maximum Brier score
            ece_threshold: Maximum ECE
            
        Returns:
            Tuple of (passes_all, individual_checks)
        """
        checks = {
            "auc_roc >= 0.75": metrics["auc_roc"] >= auc_threshold,
            "brier <= 0.20": metrics["brier_score"] <= brier_threshold,
            "ece <= 0.10": metrics["ece"] <= ece_threshold,
        }
        
        passes_all = all(checks.values())
        
        logger.info("Acceptance criteria check:")
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {check}: {status}")
        
        return passes_all, checks


class DriftDetector:
    """
    Detect feature and prediction drift.
    
    Uses Kolmogorov-Smirnov test to compare distributions.
    """
    
    def __init__(self, ks_threshold: float = 0.1):
        """
        Initialize detector.
        
        Args:
            ks_threshold: KS statistic threshold for flagging drift
        """
        self.ks_threshold = ks_threshold
    
    def check_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """
        Check for feature distribution drift.
        
        Args:
            reference_data: Training data (reference distribution)
            current_data: New data to check
            features: List of feature columns to check
            
        Returns:
            DataFrame with drift statistics
        """
        results = []
        
        for feature in features:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            ref_values = reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Only for numeric features
            if pd.api.types.is_numeric_dtype(ref_values):
                ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
                
                results.append({
                    "feature": feature,
                    "ks_statistic": ks_stat,
                    "p_value": p_value,
                    "drift_detected": ks_stat > self.ks_threshold,
                    "ref_mean": ref_values.mean(),
                    "cur_mean": cur_values.mean(),
                    "mean_shift": (cur_values.mean() - ref_values.mean()) / (ref_values.std() + 1e-10)
                })
        
        return pd.DataFrame(results)
    
    def check_prediction_drift(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray
    ) -> Dict[str, float]:
        """
        Check for prediction distribution drift.
        
        Args:
            reference_preds: Reference predictions
            current_preds: Current predictions
            
        Returns:
            Drift statistics
        """
        ks_stat, p_value = stats.ks_2samp(reference_preds, current_preds)
        
        return {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": ks_stat > self.ks_threshold,
            "ref_mean": reference_preds.mean(),
            "cur_mean": current_preds.mean(),
            "ref_std": reference_preds.std(),
            "cur_std": current_preds.std(),
        }


def print_evaluation_report(result: EvaluationResult) -> None:
    """
    Print formatted evaluation report.
    
    Args:
        result: EvaluationResult object
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    
    print("\n[DISCRIMINATION METRICS]")
    print(f"  AUC-ROC: {result.auc_roc:.4f}")
    print(f"  {'PASS' if result.auc_roc >= 0.75 else 'FAIL'} (threshold: 0.75)")
    
    print("\n[CALIBRATION METRICS]")
    print(f"  Brier Score: {result.brier_score:.4f}")
    print(f"  {'PASS' if result.brier_score <= 0.20 else 'FAIL'} (threshold: <=0.20)")
    print(f"  Expected Calibration Error: {result.expected_calibration_error:.4f}")
    print(f"  {'PASS' if result.expected_calibration_error <= 0.10 else 'FAIL'} (threshold: <=0.10)")
    
    print("\n[CLASSIFICATION METRICS] (threshold=0.5)")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall: {result.recall:.4f}")
    print(f"  F1 Score: {result.f1:.4f}")
    
    print("\n[CONFUSION MATRIX]")
    tn, fp, fn, tp = result.confusion_matrix.ravel()
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    
    print("\n[SAMPLE INFO]")
    print(f"  Total Samples: {result.n_samples}")
    print(f"  Positive Rate: {result.positive_rate:.2%}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulated ground truth and predictions
    y_true = np.random.binomial(1, 0.6, n_samples)
    
    # Good model predictions (correlated with truth)
    y_pred_good = np.clip(
        y_true * 0.4 + np.random.normal(0.3, 0.2, n_samples),
        0, 1
    )
    
    # Poor model predictions (less correlated)
    y_pred_poor = np.random.uniform(0.3, 0.7, n_samples)
    
    # Evaluate
    evaluator = ModelEvaluator()
    
    print("\n=== Good Model ===")
    result_good = evaluator.full_evaluation(y_true, y_pred_good)
    print_evaluation_report(result_good)
    
    print("\n=== Poor Model ===")
    result_poor = evaluator.full_evaluation(y_true, y_pred_poor)
    print_evaluation_report(result_poor)
    
    # Compare models
    print("\n=== Model Comparison ===")
    comparison = evaluator.compare_models(y_true, {
        "good_model": y_pred_good,
        "poor_model": y_pred_poor
    })
    print(comparison[["auc_roc", "brier_score", "ece", "f1"]])

