"""
Configuration management for the application.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Paths
    model_path: Path = Path("models/artifacts")
    data_path: Path = Path("data")
    predictions_path: Path = Path("data/predictions/latest_predictions.csv")
    dataset_path: Path = Path("dataset")
    log_path: Path = Path("logs")

    # Logging
    log_level: str = "INFO"

    # Application
    app_name: str = "Revenue Intelligence System"
    app_version: str = "1.0.0"

    # ML Model Settings
    default_forecast_weeks: int = 12
    min_training_samples: int = 100
    shap_sample_size: int = 100

    # UI Settings
    max_deals_display: int = 50
    refresh_interval_seconds: int = 300  # 5 minutes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create global settings instance
settings = Settings()

# Ensure directories exist
settings.model_path.mkdir(parents=True, exist_ok=True)
settings.data_path.mkdir(parents=True, exist_ok=True)
settings.log_path.mkdir(parents=True, exist_ok=True)


