"""
Optional MLflow client for experiment tracking.
This module provides a singleton MLflow client that can be used throughout the app
without requiring initialization everywhere.
"""

import mlflow
import os
from typing import Optional, Any, Dict
from contextlib import contextmanager
from functools import wraps
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.settings import settings


class MLflowClient:
    """
    Singleton MLflow client for optional experiment tracking.
    
    Usage:
        from app.mlflow_client import mlflow_client
        
        # Optional tracking - won't fail if MLflow is unavailable
        mlflow_client.log_param("model", "text-embedding-3-small")
        mlflow_client.log_metric("accuracy", 0.95)
        
        # Context manager for runs
        with mlflow_client.start_run("embedding_experiment"):
            mlflow_client.log_param("strategy", "normalized")
            mlflow_client.log_metric("precision", 0.92)
    """
    
    def __init__(self):
        self._enabled = False
        self._current_experiment = None
        self._initialize()
    
    def _initialize(self):
        """Initialize MLflow client if available."""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            
            # Set up S3/MinIO environment for artifacts
            protocol = "https" if settings.minio_secure else "http"
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"{protocol}://{settings.minio_endpoint}"
            os.environ["AWS_ACCESS_KEY_ID"] = settings.minio_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = settings.minio_secret_key
            
            # Also set for boto3/S3 compatibility
            os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  # MinIO doesn't use regions but boto3 requires it
            
            # Debug print to verify credentials
            print(f"  → MinIO endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
            print(f"  → MinIO access key: {settings.minio_access_key}")
            
            # Create MinIO bucket if needed
            self._ensure_bucket_exists()
            
            # Test connection
            mlflow.get_tracking_uri()
            self._enabled = True
            
            # Set default experiment if specified
            if settings.mlflow_experiment:
                self.set_experiment(settings.mlflow_experiment)
            
            print("✅ MLflow client initialized successfully")
            
        except Exception as e:
            print(f"⚠️  MLflow unavailable (optional): {e}")
            self._enabled = False
    
    def _ensure_bucket_exists(self):
        """Ensure the MinIO bucket exists for MLflow artifacts."""
        try:
            # Create S3 client for MinIO
            protocol = "https" if settings.minio_secure else "http"
            s3_client = boto3.client(
                's3',
                endpoint_url=f"{protocol}://{settings.minio_endpoint}",
                aws_access_key_id=settings.minio_access_key,
                aws_secret_access_key=settings.minio_secret_key,
                region_name='us-east-1'  # MinIO doesn't use regions but boto3 requires it
            )
            
            bucket_name = settings.minio_bucket_name
            
            # Check if bucket exists
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                print(f"  ✓ MinIO bucket '{bucket_name}' exists")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    # Bucket doesn't exist, create it
                    print(f"  → Creating MinIO bucket '{bucket_name}'...")
                    s3_client.create_bucket(Bucket=bucket_name)
                    print(f"  ✓ MinIO bucket '{bucket_name}' created")
                else:
                    raise
                    
        except (NoCredentialsError, ClientError) as e:
            print(f"  ⚠️  Could not ensure MinIO bucket exists: {e}")
            # Don't fail initialization - MLflow can still work without artifacts
        except Exception as e:
            print(f"  ⚠️  MinIO bucket check failed: {e}")
    
    def _safe_call(self, func, *args, **kwargs):
        """Safely call MLflow function, returning None if disabled."""
        if not self._enabled:
            return None
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️  MLflow operation failed (continuing anyway): {e}")
            return None
    
    def create_experiment(self, name: str, artifact_location: Optional[str] = None) -> Optional[str]:
        """Create or get existing experiment."""
        if not self._enabled:
            return None
            
        try:
            # Try to get existing experiment first
            experiment = mlflow.get_experiment_by_name(name)
            if experiment:
                self._current_experiment = experiment.experiment_id
                return experiment.experiment_id
        except:
            pass
        
        # Create new experiment
        artifact_location = artifact_location or f"s3://{settings.minio_bucket_name}/mlflow-artifacts/experiments"
        experiment_id = self._safe_call(mlflow.create_experiment, name, artifact_location)
        if experiment_id:
            self._current_experiment = experiment_id
        return experiment_id
    
    def set_experiment(self, name: str) -> bool:
        """Set the active experiment."""
        if not self._enabled:
            return False
        
        try:
            mlflow.set_experiment(name)
            experiment = mlflow.get_experiment_by_name(name)
            if experiment:
                self._current_experiment = experiment.experiment_id
                return True
        except Exception as e:
            print(f"⚠️  Could not set experiment '{name}': {e}")
        return False
    
    @contextmanager
    def start_run(self, experiment_name: Optional[str] = None, run_name: Optional[str] = None):
        """Context manager for MLflow runs."""
        if not self._enabled:
            yield None
            return
        
        # Set experiment - use provided name, or default from settings, or MLflow's default
        if experiment_name:
            self.set_experiment(experiment_name)
        elif settings.mlflow_experiment and not self._current_experiment:
            self.set_experiment(settings.mlflow_experiment)
        
        try:
            with mlflow.start_run(run_name=run_name):
                yield mlflow.active_run()
        except Exception as e:
            print(f"⚠️  MLflow run failed: {e}")
            yield None
    
    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter."""
        self._safe_call(mlflow.log_param, key, value)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self._safe_call(mlflow.log_params, params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric."""
        self._safe_call(mlflow.log_metric, key, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics."""
        self._safe_call(mlflow.log_metrics, metrics, step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact file."""
        self._safe_call(mlflow.log_artifact, local_path, artifact_path)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifact directory."""
        self._safe_call(mlflow.log_artifacts, local_dir, artifact_path)
    
    def log_figure(self, figure, artifact_file: str) -> None:
        """Log a matplotlib/plotly figure."""
        if not self._enabled:
            return
        
        try:
            import matplotlib.pyplot as plt
            if hasattr(figure, 'savefig'):  # matplotlib figure
                figure.savefig(artifact_file)
                self.log_artifact(artifact_file)
                os.remove(artifact_file)
            elif hasattr(figure, 'write_html'):  # plotly figure
                figure.write_html(artifact_file)
                self.log_artifact(artifact_file)
                os.remove(artifact_file)
        except Exception as e:
            print(f"⚠️  Could not log figure: {e}")
    
    def add_tags(self, tags: Dict[str, str]) -> None:
        """Add tags to current run."""
        if not self._enabled:
            return
        for key, value in tags.items():
            self._safe_call(mlflow.set_tag, key, value)
    
    @property
    def enabled(self) -> bool:
        """Check if MLflow tracking is enabled."""
        return self._enabled
    
    @property
    def tracking_uri(self) -> Optional[str]:
        """Get current tracking URI."""
        return settings.mlflow_tracking_uri if self._enabled else None


# Singleton instance
mlflow_client = MLflowClient()


def track_experiment(experiment_name: str, run_name: Optional[str] = None):
    """
    Decorator for automatic experiment tracking.
    
    Usage:
        @track_experiment("embedding_experiments", "normalize_test")
        def test_embedding_normalization():
            mlflow_client.log_param("strategy", "l2_normalized")
            # ... your code ...
            mlflow_client.log_metric("accuracy", 0.95)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with mlflow_client.start_run(experiment_name, run_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
