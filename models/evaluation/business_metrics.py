"""
Business Metrics for Model Evaluation
Revenue Intelligence System

These metrics evaluate model utility for business decisions, not just statistical performance.
A 0.58 AUC model can still drive significant value if it provides:
- Good ranking (prioritize the right deals)
- Calibrated probabilities (trustworthy predictions)
- Lift over baseline (better than random)
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy import stats
import logging

logger = logging.getLogger(__name__)


def compute_lift_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute lift curve showing how much better the model is vs random.
    
    Lift = (% of positives in top k%) / (overall positive rate)
    
    A lift of 1.5 at 10% means: the top 10% of predictions contain
    50% more positives than a random 10% sample.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of decile bins
        
    Returns:
        Dictionary with percentiles, lift values, and cumulative gains
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Sort by predicted probability (descending)
    sorted_idx = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_idx]
    
    n = len(y_true)
    overall_positive_rate = y_true.mean()
    
    percentiles = []
    lifts = []
    cumulative_gains = []
    
    for i in range(1, n_bins + 1):
        pct = i / n_bins
        top_k = int(n * pct)
        
        # Positive rate in top k%
        top_k_positive_rate = y_true_sorted[:top_k].mean()
        
        # Lift = local positive rate / global positive rate
        lift = top_k_positive_rate / overall_positive_rate if overall_positive_rate > 0 else 1.0
        
        # Cumulative gain = % of all positives captured
        cumulative_gain = y_true_sorted[:top_k].sum() / y_true.sum() if y_true.sum() > 0 else 0.0
        
        percentiles.append(pct)
        lifts.append(lift)
        cumulative_gains.append(cumulative_gain)
    
    return {
        "percentiles": np.array(percentiles),
        "lift": np.array(lifts),
        "cumulative_gain": np.array(cumulative_gains),
        "baseline_gain": np.array(percentiles),  # Random model's cumulative gain
    }


def compute_top_k_precision(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: int = 10,
    high_risk_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute precision for top-k predictions.
    
    For risk identification: what % of deals we flag as "high risk" 
    (low win probability) actually fail?
    
    Args:
        y_true: True binary labels (1 = won, 0 = lost)
        y_pred_proba: Predicted win probabilities
        k: Number of top predictions to evaluate
        high_risk_threshold: Probability below which a deal is "high risk"
        
    Returns:
        Dictionary with precision metrics
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    n = len(y_true)
    k = min(k, n)
    
    # For "at-risk" deals, we want LOW probabilities to be correct
    # Sort ascending (lowest win prob first = highest risk)
    sorted_idx = np.argsort(y_pred_proba)
    
    # Top-k highest risk deals (lowest win probability)
    top_k_indices = sorted_idx[:k]
    top_k_true = y_true[top_k_indices]
    top_k_pred = y_pred_proba[top_k_indices]
    
    # Precision for risk: what % actually lost?
    # (y_true = 0 means the deal was lost)
    top_k_risk_precision = (top_k_true == 0).mean()
    
    # Also compute for deals below threshold
    high_risk_mask = y_pred_proba < high_risk_threshold
    if high_risk_mask.sum() > 0:
        threshold_precision = (y_true[high_risk_mask] == 0).mean()
    else:
        threshold_precision = 0.0
    
    # For comparison: bottom-k (highest win prob)
    bottom_k_indices = sorted_idx[-k:]
    bottom_k_true = y_true[bottom_k_indices]
    bottom_k_win_precision = (bottom_k_true == 1).mean()
    
    return {
        "top_k": k,
        "top_k_risk_precision": top_k_risk_precision,  # % of top-k low prob that actually lost
        "top_k_win_precision": bottom_k_win_precision,  # % of top-k high prob that actually won
        "threshold": high_risk_threshold,
        "threshold_risk_precision": threshold_precision,
        "n_high_risk": int(high_risk_mask.sum()),
    }


def compute_rank_correlation(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute rank correlation between predictions and outcomes.
    
    Spearman correlation measures how well the ranking is preserved.
    A model with good ranking ability will have high Spearman correlation.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary with correlation metrics
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Spearman rank correlation
    spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred_proba)
    
    # Kendall's tau (another rank correlation)
    kendall_corr, kendall_p = stats.kendalltau(y_true, y_pred_proba)
    
    return {
        "spearman_correlation": spearman_corr,
        "spearman_pvalue": spearman_p,
        "kendall_tau": kendall_corr,
        "kendall_pvalue": kendall_p,
    }


def compute_expected_revenue_impact(
    deals_df: pd.DataFrame,
    predictions: np.ndarray,
    value_column: str = "close_value",
    target_column: str = "target",
    action_threshold: float = 0.4,
    action_lift: float = 0.10
) -> Dict[str, float]:
    """
    Estimate expected revenue impact of using the model.
    
    Assumptions:
    - Deals below action_threshold get intervention
    - Intervention improves win rate by action_lift (e.g., 10%)
    - Compare to no-model baseline (random intervention)
    
    Args:
        deals_df: DataFrame with deal values
        predictions: Win probability predictions
        value_column: Column with deal values
        target_column: Column with win/loss labels
        action_threshold: Probability below which we intervene
        action_lift: Assumed improvement from intervention
        
    Returns:
        Dictionary with revenue impact estimates
    """
    df = deals_df.copy()
    df["win_prob"] = predictions
    
    # Deals we would intervene on
    intervention_mask = df["win_prob"] < action_threshold
    n_interventions = intervention_mask.sum()
    
    if value_column not in df.columns:
        # Use a default value if not available
        df[value_column] = 100000
    
    # Expected value with model-guided intervention
    intervention_deals = df[intervention_mask]
    non_intervention_deals = df[~intervention_mask]
    
    # Expected revenue from intervention deals (boosted win rate)
    if len(intervention_deals) > 0:
        boosted_win_rate = intervention_deals["win_prob"] + action_lift
        boosted_win_rate = boosted_win_rate.clip(0, 1)
        intervention_expected_revenue = (
            intervention_deals[value_column] * boosted_win_rate
        ).sum()
    else:
        intervention_expected_revenue = 0
    
    # Expected revenue from non-intervention deals (natural win rate)
    if len(non_intervention_deals) > 0:
        non_intervention_expected_revenue = (
            non_intervention_deals[value_column] * non_intervention_deals["win_prob"]
        ).sum()
    else:
        non_intervention_expected_revenue = 0
    
    total_expected_with_model = intervention_expected_revenue + non_intervention_expected_revenue
    
    # Baseline: random intervention (same number of deals)
    if n_interventions > 0 and len(df) > 0:
        # Random selection would have average win rate
        avg_win_prob = df["win_prob"].mean()
        random_intervention_revenue = (
            df[value_column].head(n_interventions) * (avg_win_prob + action_lift)
        ).sum()
        random_non_intervention_revenue = (
            df[value_column].tail(len(df) - n_interventions) * avg_win_prob
        ).sum()
        total_expected_random = random_intervention_revenue + random_non_intervention_revenue
    else:
        total_expected_random = (df[value_column] * df["win_prob"]).sum()
    
    # Lift from using model
    revenue_lift = total_expected_with_model - total_expected_random
    
    return {
        "n_interventions": int(n_interventions),
        "intervention_threshold": action_threshold,
        "assumed_lift": action_lift,
        "expected_revenue_with_model": total_expected_with_model,
        "expected_revenue_random": total_expected_random,
        "revenue_lift": revenue_lift,
        "lift_percentage": (revenue_lift / total_expected_random * 100) if total_expected_random > 0 else 0,
    }


def print_business_metrics_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    deals_df: pd.DataFrame = None
) -> None:
    """
    Print a comprehensive business metrics report.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        deals_df: Optional DataFrame with deal values
    """
    print("\n" + "=" * 60)
    print("BUSINESS METRICS REPORT")
    print("=" * 60)
    
    # Lift curve
    lift = compute_lift_curve(y_true, y_pred_proba)
    print("\n[LIFT ANALYSIS]")
    print(f"  Lift @ 10%: {lift['lift'][0]:.2f}x (top 10% are {(lift['lift'][0]-1)*100:.0f}% more likely to win)")
    print(f"  Lift @ 20%: {lift['lift'][1]:.2f}x")
    print(f"  Lift @ 50%: {lift['lift'][4]:.2f}x")
    print(f"\n  Cumulative Gain @ 20%: {lift['cumulative_gain'][1]*100:.1f}% of wins captured")
    print(f"  Cumulative Gain @ 50%: {lift['cumulative_gain'][4]*100:.1f}% of wins captured")
    
    # Top-k precision
    for k in [10, 25, 50]:
        precision = compute_top_k_precision(y_true, y_pred_proba, k=k)
        print(f"\n[TOP-{k} ANALYSIS]")
        print(f"  Risk Precision: {precision['top_k_risk_precision']*100:.1f}% of top-{k} low-prob deals actually lost")
        print(f"  Win Precision: {precision['top_k_win_precision']*100:.1f}% of top-{k} high-prob deals actually won")
    
    # Rank correlation
    rank = compute_rank_correlation(y_true, y_pred_proba)
    print(f"\n[RANKING QUALITY]")
    print(f"  Spearman Correlation: {rank['spearman_correlation']:.3f} (p={rank['spearman_pvalue']:.4f})")
    print(f"  Kendall's Tau: {rank['kendall_tau']:.3f}")
    
    # Revenue impact (if deals_df provided)
    if deals_df is not None:
        impact = compute_expected_revenue_impact(deals_df, y_pred_proba)
        print(f"\n[EXPECTED REVENUE IMPACT]")
        print(f"  Interventions: {impact['n_interventions']} deals (below {impact['intervention_threshold']} win prob)")
        print(f"  Revenue with model: ${impact['expected_revenue_with_model']:,.0f}")
        print(f"  Revenue random: ${impact['expected_revenue_random']:,.0f}")
        print(f"  Model lift: ${impact['revenue_lift']:,.0f} ({impact['lift_percentage']:.1f}%)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n = 1000
    
    # Create correlated predictions (simulating a real model)
    y_true = np.random.binomial(1, 0.6, n)
    noise = np.random.normal(0, 0.3, n)
    y_pred_proba = np.clip(y_true * 0.3 + 0.5 + noise, 0.1, 0.9)
    
    print_business_metrics_report(y_true, y_pred_proba)


