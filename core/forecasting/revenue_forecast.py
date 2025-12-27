"""
Revenue Forecasting Engine
Phase 1A: Core ML Pipeline

This module generates probabilistic revenue forecasts using Monte Carlo simulation.
Key responsibilities:
- Combine win probability and time-to-close predictions
- Generate P10/P50/P90 forecast bands
- Aggregate by time periods (weekly/monthly)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """
    Result of revenue forecast.
    """
    # Time period aggregations
    periods: List[str]  # Period labels
    p10: List[float]    # 10th percentile (pessimistic)
    p50: List[float]    # 50th percentile (expected)
    p90: List[float]    # 90th percentile (optimistic)
    
    # Deal-level details
    deal_forecasts: pd.DataFrame  # Per-deal forecasts
    
    # Metadata
    simulation_runs: int
    forecast_date: datetime
    horizon_days: int


class RevenueForecast:
    """
    Monte Carlo revenue forecasting engine.
    
    For each deal:
    1. Sample win/loss from Bernoulli(win_probability)
    2. If win, sample close date from time-to-close distribution
    3. Use product_sales_price as value estimate
    4. Aggregate to time buckets
    5. Repeat N times for distribution
    
    Note: Uses product_sales_price since actual close_value
    is unknown for open deals (would be leakage).
    """
    
    def __init__(
        self,
        win_prob_model: Optional[object] = None,
        time_to_close_model: Optional[object] = None,
        n_simulations: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize forecast engine.
        
        Args:
            win_prob_model: Trained win probability model
            time_to_close_model: Trained time-to-close model
            n_simulations: Number of Monte Carlo simulations
            random_state: Random seed for reproducibility
        """
        self.win_prob_model = win_prob_model
        self.time_to_close_model = time_to_close_model
        self.n_simulations = n_simulations
        self.random_state = random_state
        
        np.random.seed(random_state)
    
    def _sample_close_dates(
        self,
        base_date: datetime,
        days_p10: np.ndarray,
        days_p50: np.ndarray,
        days_p90: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample close dates from distribution defined by percentiles.
        
        Uses triangular approximation based on P10/P50/P90.
        
        Args:
            base_date: Starting date for calculation
            days_p10: 10th percentile days to close
            days_p50: 50th percentile days to close  
            days_p90: 90th percentile days to close
            n_samples: Number of samples per deal
            
        Returns:
            Array of sampled days (n_deals x n_samples)
        """
        n_deals = len(days_p10)
        
        # Sample uniformly in [0,1] and map to triangular
        # Approximate: sample from normal with mean=p50, std=(p90-p10)/3.3
        # (3.3 corresponds to ~80% interval for normal)
        
        std = (days_p90 - days_p10) / 3.3
        std = np.maximum(std, 1)  # Minimum 1 day std
        
        samples = np.random.normal(
            loc=days_p50.reshape(-1, 1),
            scale=std.reshape(-1, 1),
            size=(n_deals, n_samples)
        )
        
        # Clip to reasonable range
        samples = np.clip(samples, days_p10.reshape(-1, 1), days_p90.reshape(-1, 1) * 1.5)
        
        return samples.astype(int)
    
    def _aggregate_by_period(
        self,
        close_dates: np.ndarray,
        values: np.ndarray,
        wins: np.ndarray,
        forecast_date: datetime,
        period: str = "week"
    ) -> pd.DataFrame:
        """
        Aggregate simulated revenues by time period.
        
        Args:
            close_dates: Simulated close dates (n_deals x n_sims)
            values: Deal values (n_deals,)
            wins: Win indicators (n_deals x n_sims)
            forecast_date: Base date for period calculation
            period: 'week' or 'month'
            
        Returns:
            DataFrame with period-level aggregations
        """
        n_deals, n_sims = close_dates.shape
        
        # Convert days to period indices
        if period == "week":
            period_idx = close_dates // 7
            max_periods = 13  # 13 weeks = ~quarter
            period_labels = [
                (forecast_date + timedelta(weeks=i)).strftime("%Y-W%W")
                for i in range(max_periods)
            ]
        else:  # month
            period_idx = close_dates // 30
            max_periods = 6  # 6 months
            period_labels = [
                (forecast_date + timedelta(days=30*i)).strftime("%Y-%m")
                for i in range(max_periods)
            ]
        
        # Calculate revenue per period per simulation
        period_revenues = np.zeros((max_periods, n_sims))
        
        for sim in range(n_sims):
            for deal in range(n_deals):
                if wins[deal, sim]:
                    p_idx = period_idx[deal, sim]
                    if 0 <= p_idx < max_periods:
                        period_revenues[p_idx, sim] += values[deal]
        
        # Cumulative revenue
        cumulative = np.cumsum(period_revenues, axis=0)
        
        # Calculate percentiles
        results = []
        for i, label in enumerate(period_labels):
            results.append({
                "period": label,
                "period_p10": np.percentile(period_revenues[i], 10),
                "period_p50": np.percentile(period_revenues[i], 50),
                "period_p90": np.percentile(period_revenues[i], 90),
                "cumulative_p10": np.percentile(cumulative[i], 10),
                "cumulative_p50": np.percentile(cumulative[i], 50),
                "cumulative_p90": np.percentile(cumulative[i], 90),
            })
        
        return pd.DataFrame(results)
    
    def generate_forecast(
        self,
        deals: pd.DataFrame,
        horizon_weeks: int = 12,
        period: str = "week"
    ) -> ForecastResult:
        """
        Generate probabilistic revenue forecast.
        
        Args:
            deals: DataFrame with columns:
                - opportunity_id
                - win_probability
                - days_p10, days_p50, days_p90 (or will estimate)
                - product_sales_price
            horizon_weeks: Forecast horizon
            period: Aggregation period ('week' or 'month')
            
        Returns:
            ForecastResult with percentile bands
        """
        logger.info(f"Generating forecast for {len(deals)} deals, {horizon_weeks} week horizon")
        
        forecast_date = datetime.now()
        n_deals = len(deals)
        
        # Get win probabilities
        win_probs = deals["win_probability"].values
        
        # Get time-to-close distributions
        if "days_p50" in deals.columns:
            days_p10 = deals["days_p10"].values
            days_p50 = deals["days_p50"].values
            days_p90 = deals["days_p90"].values
        else:
            # Default distribution if not provided
            logger.warning("No time-to-close distribution provided, using defaults")
            days_p10 = np.full(n_deals, 8)
            days_p50 = np.full(n_deals, 45)
            days_p90 = np.full(n_deals, 85)
        
        # Get deal values
        values = deals["product_sales_price"].values
        
        # Monte Carlo simulation
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations...")
        
        # Sample wins (Bernoulli for each deal, each simulation)
        wins = np.random.random((n_deals, self.n_simulations)) < win_probs.reshape(-1, 1)
        
        # Sample close dates
        close_dates = self._sample_close_dates(
            forecast_date,
            days_p10,
            days_p50,
            days_p90,
            self.n_simulations
        )
        
        # Aggregate by period
        period_df = self._aggregate_by_period(
            close_dates,
            values,
            wins,
            forecast_date,
            period
        )
        
        # Create deal-level summary
        deal_forecasts = deals[["opportunity_id", "win_probability", "product_sales_price"]].copy()
        deal_forecasts["expected_value"] = win_probs * values
        deal_forecasts["expected_close_days"] = days_p50
        
        result = ForecastResult(
            periods=period_df["period"].tolist(),
            p10=period_df["cumulative_p10"].tolist(),
            p50=period_df["cumulative_p50"].tolist(),
            p90=period_df["cumulative_p90"].tolist(),
            deal_forecasts=deal_forecasts,
            simulation_runs=self.n_simulations,
            forecast_date=forecast_date,
            horizon_days=horizon_weeks * 7
        )
        
        # Log summary
        total_pipeline = values.sum()
        expected_revenue = (win_probs * values).sum()
        logger.info(f"Total pipeline value: ${total_pipeline:,.0f}")
        logger.info(f"Expected revenue: ${expected_revenue:,.0f}")
        logger.info(f"P50 forecast: ${result.p50[-1]:,.0f}")
        logger.info(f"Forecast range: ${result.p10[-1]:,.0f} - ${result.p90[-1]:,.0f}")
        
        return result
    
    def forecast_with_models(
        self,
        deals: pd.DataFrame,
        feature_cols: List[str],
        horizon_weeks: int = 12
    ) -> ForecastResult:
        """
        Generate forecast using trained models.
        
        Args:
            deals: DataFrame with features
            feature_cols: List of feature columns
            horizon_weeks: Forecast horizon
            
        Returns:
            ForecastResult
        """
        if self.win_prob_model is None:
            raise ValueError("Win probability model not provided")
        
        # Get win probabilities from model
        deals = deals.copy()
        deals["win_probability"] = self.win_prob_model.predict_proba(deals[feature_cols])
        
        # Get time-to-close distributions
        if self.time_to_close_model is not None:
            time_preds = self.time_to_close_model.predict_distribution(deals[feature_cols])
            deals["days_p10"] = time_preds["days_p10"]
            deals["days_p50"] = time_preds["days_p50"]
            deals["days_p90"] = time_preds["days_p90"]
        
        return self.generate_forecast(deals, horizon_weeks)
    
    def get_forecast_summary(self, result: ForecastResult) -> Dict:
        """
        Get summary statistics from forecast.
        
        Args:
            result: ForecastResult object
            
        Returns:
            Summary dictionary
        """
        deal_df = result.deal_forecasts
        
        return {
            "forecast_date": result.forecast_date.isoformat(),
            "horizon_days": result.horizon_days,
            "total_deals": len(deal_df),
            "total_pipeline": float(deal_df["product_sales_price"].sum()),
            "expected_revenue": float(deal_df["expected_value"].sum()),
            "p10_forecast": result.p10[-1] if result.p10 else 0,
            "p50_forecast": result.p50[-1] if result.p50 else 0,
            "p90_forecast": result.p90[-1] if result.p90 else 0,
            "simulation_runs": result.simulation_runs,
        }


def create_sample_forecast(n_deals: int = 50) -> ForecastResult:
    """
    Create sample forecast for demo/testing.
    
    Args:
        n_deals: Number of deals to simulate
        
    Returns:
        ForecastResult
    """
    np.random.seed(42)
    
    # Generate sample deals
    deals = pd.DataFrame({
        "opportunity_id": [f"deal_{i}" for i in range(n_deals)],
        "win_probability": np.random.beta(3, 2, n_deals),  # Skewed toward winning
        "product_sales_price": np.random.choice([550, 1096, 3393, 4821, 5482, 26768], n_deals),
        "days_p10": np.random.randint(5, 15, n_deals),
        "days_p50": np.random.randint(30, 60, n_deals),
        "days_p90": np.random.randint(70, 100, n_deals),
    })
    
    forecaster = RevenueForecast(n_simulations=1000)
    return forecaster.generate_forecast(deals)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample forecast
    result = create_sample_forecast(100)
    
    print("\n=== Forecast Summary ===")
    summary = RevenueForecast().get_forecast_summary(result)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: ${value:,.0f}" if "revenue" in key or "forecast" in key or "pipeline" in key else f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== Weekly Forecast ===")
    for i, period in enumerate(result.periods[:8]):
        print(f"{period}: P10=${result.p10[i]:,.0f}, P50=${result.p50[i]:,.0f}, P90=${result.p90[i]:,.0f}")
    
    print("\n=== Top 5 Deals by Expected Value ===")
    top_deals = result.deal_forecasts.nlargest(5, "expected_value")
    print(top_deals[["opportunity_id", "win_probability", "product_sales_price", "expected_value"]])

