"""
Experiment Tracker for Revenue Intelligence System

Dual tracking approach:
- Always logs to JSON (no dependencies, quick access)
- Optionally logs to MLflow if installed (rich UI, artifact management)
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.info("MLflow not installed. Using JSON-only tracking.")


class ExperimentTracker:
    """
    Dual experiment tracker for ML experiments.
    
    Features:
    - Always logs to JSON file (experiments/runs.json)
    - Optionally logs to MLflow if installed
    - Provides run comparison utilities
    """
    
    def __init__(
        self,
        experiment_name: str = "default",
        tracking_dir: str = "experiments",
        use_mlflow: bool = True
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_dir: Directory for JSON logs
            use_mlflow: Whether to use MLflow (if available)
        """
        self.experiment_name = experiment_name
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_path = self.tracking_dir / "runs.json"
        self.use_mlflow = use_mlflow and HAS_MLFLOW
        
        # Current run state
        self.run_id = self._generate_run_id()
        self.run_data: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "status": "running",
            "params": {},
            "metrics": {},
            "artifacts": [],
            "tags": {},
        }
        
        # Initialize MLflow if available
        if self.use_mlflow:
            self._init_mlflow()
        
        logger.info(f"Started experiment run: {self.run_id}")
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{short_uuid}"
    
    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        try:
            # Set tracking URI to local directory
            # Use relative path with forward slashes (works on Windows & Unix)
            mlflow_dir = self.tracking_dir / "mlruns"
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            
            # Use as_posix() to convert to forward slashes for cross-platform compatibility
            tracking_uri = mlflow_dir.as_posix()
            mlflow.set_tracking_uri(tracking_uri)
            
            # Set or create experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Start run
            mlflow.start_run(run_name=self.run_id)
            
            logger.info(f"MLflow tracking enabled: {tracking_uri}")
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}. Falling back to JSON only.")
            self.use_mlflow = False
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for this run.
        
        Args:
            params: Dictionary of parameter names and values
        """
        # Convert non-serializable types
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                clean_params[key] = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            else:
                clean_params[key] = value
        
        self.run_data["params"].update(clean_params)
        
        # Log to MLflow
        if self.use_mlflow:
            try:
                for key, value in clean_params.items():
                    mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"MLflow param logging failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics for this run.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        self.run_data["metrics"].update(metrics)
        
        # Log to MLflow
        if self.use_mlflow:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"MLflow metric logging failed: {e}")
    
    def log_artifact(self, file_path: str, artifact_path: Optional[str] = None) -> None:
        """
        Log an artifact (file) for this run.
        
        Args:
            file_path: Path to the file to log
            artifact_path: Optional subdirectory in artifact store
        """
        self.run_data["artifacts"].append(str(file_path))
        
        # Log to MLflow
        if self.use_mlflow:
            try:
                mlflow.log_artifact(file_path, artifact_path)
            except Exception as e:
                logger.warning(f"MLflow artifact logging failed: {e}")
    
    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag for this run.
        
        Args:
            key: Tag name
            value: Tag value
        """
        self.run_data["tags"][key] = value
        
        if self.use_mlflow:
            try:
                mlflow.set_tag(key, value)
            except Exception as e:
                logger.warning(f"MLflow tag setting failed: {e}")
    
    def end_run(self, status: str = "completed") -> None:
        """
        End the current run and save to JSON.
        
        Args:
            status: Run status ('completed', 'failed', 'cancelled')
        """
        self.run_data["end_time"] = datetime.now().isoformat()
        self.run_data["status"] = status
        
        # Calculate duration
        start = datetime.fromisoformat(self.run_data["start_time"])
        end = datetime.fromisoformat(self.run_data["end_time"])
        self.run_data["duration_seconds"] = (end - start).total_seconds()
        
        # Save to JSON
        self._save_to_json()
        
        # End MLflow run
        if self.use_mlflow:
            try:
                mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")
            except Exception as e:
                logger.warning(f"MLflow run end failed: {e}")
        
        logger.info(f"Run {self.run_id} {status} in {self.run_data['duration_seconds']:.1f}s")
    
    def _save_to_json(self) -> None:
        """Save current run to JSON file."""
        # Load existing runs
        runs = self._load_runs()
        
        # Add or update current run
        runs[self.run_id] = self.run_data
        
        # Save back
        with open(self.json_path, "w") as f:
            json.dump(runs, f, indent=2, default=str)
    
    def _load_runs(self) -> Dict[str, Any]:
        """Load existing runs from JSON."""
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    @classmethod
    def load_all_runs(cls, tracking_dir: str = "experiments") -> List[Dict[str, Any]]:
        """
        Load all runs from JSON file.
        
        Args:
            tracking_dir: Directory containing runs.json
            
        Returns:
            List of run dictionaries sorted by start time
        """
        json_path = Path(tracking_dir) / "runs.json"
        
        if not json_path.exists():
            return []
        
        try:
            with open(json_path, "r") as f:
                runs = json.load(f)
            
            # Convert to list and sort by start time
            run_list = list(runs.values())
            run_list.sort(key=lambda x: x.get("start_time", ""), reverse=True)
            
            return run_list
        except (json.JSONDecodeError, IOError):
            return []
    
    @classmethod
    def get_best_run(
        cls,
        tracking_dir: str = "experiments",
        metric: str = "auc_roc",
        higher_is_better: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            tracking_dir: Directory containing runs.json
            metric: Metric to optimize
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Best run dictionary or None
        """
        runs = cls.load_all_runs(tracking_dir)
        
        if not runs:
            return None
        
        # Filter runs with the metric
        valid_runs = [r for r in runs if metric in r.get("metrics", {})]
        
        if not valid_runs:
            return None
        
        # Sort by metric
        valid_runs.sort(
            key=lambda x: x["metrics"][metric],
            reverse=higher_is_better
        )
        
        return valid_runs[0]


def get_mlflow_ui_command() -> str:
    """Get command to start MLflow UI."""
    return "mlflow ui --backend-store-uri experiments/mlruns"


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    tracker = ExperimentTracker("demo_experiment")
    
    tracker.log_params({
        "num_leaves": 8,
        "learning_rate": 0.05,
        "features": ["a", "b", "c"],
    })
    
    tracker.log_metrics({
        "auc_roc": 0.75,
        "brier_score": 0.18,
    })
    
    tracker.set_tag("model_type", "lightgbm")
    tracker.end_run()
    
    # Show all runs
    print("\nAll runs:")
    for run in ExperimentTracker.load_all_runs():
        print(f"  {run['run_id']}: AUC={run['metrics'].get('auc_roc', 'N/A')}")

