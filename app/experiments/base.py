"""
Base experiment class with MLflow integration.
"""
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC, abstractmethod

from app.mlflow_client import mlflow_client


class BaseExperiment(ABC):
    """Base class for all experiments with MLflow tracking."""
    
    def __init__(
        self, 
        experiment_name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ):
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or {}
        self.start_time = None
        self.metrics = {}
        self.params = {}
        self.artifacts_dir = Path("/tmp") / f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def setup(self):
        """Setup experiment, create MLflow experiment."""
        print(f"\n🔬 Setting up experiment: {self.experiment_name}")
        print(f"📝 Description: {self.description}")
        
        # Create artifacts directory
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set MLflow experiment
        mlflow_client.set_experiment(self.experiment_name)
        
        # Add default tags
        self.tags.update({
            "experiment_type": self.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        })
        
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Main experiment logic - override in subclasses."""
        pass
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the full experiment lifecycle."""
        try:
            # Setup
            await self.setup()
            
            # Start MLflow run
            with mlflow_client.start_run(run_name=f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log tags
                mlflow_client.add_tags(self.tags)
                
                # Log parameters
                if self.params:
                    mlflow_client.log_params(self.params)
                
                # Record start time
                self.start_time = time.time()
                
                # Run the experiment
                print(f"\n🚀 Running experiment...")
                results = await self.run()
                
                # Calculate duration
                duration = time.time() - self.start_time
                mlflow_client.log_metric("experiment_duration_seconds", duration)
                
                # Log any collected metrics
                if self.metrics:
                    mlflow_client.log_metrics(self.metrics)
                
                # Log artifacts
                await self.log_artifacts()
                
                print(f"\n✅ Experiment completed in {duration:.2f} seconds")
                
                return {
                    "status": "success",
                    "duration": duration,
                    "results": results,
                    "metrics": self.metrics,
                    "params": self.params
                }
                
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            mlflow_client.log_metric("experiment_failed", 1)
            mlflow_client.add_tags({"error": str(e)})
            
            return {
                "status": "failed",
                "error": str(e)
            }
        finally:
            await self.cleanup()
    
    async def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow and store locally."""
        for key, value in metrics.items():
            if step is not None:
                mlflow_client.log_metric(key, value, step)
            else:
                mlflow_client.log_metric(key, value)
            # Store locally for summary
            self.metrics[key] = value
    
    async def log_artifacts(self):
        """Save and log artifacts to MLflow."""
        # Save metrics summary
        metrics_file = self.artifacts_dir / "metrics_summary.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "experiment": self.experiment_name,
                "metrics": self.metrics,
                "params": self.params,
                "duration": time.time() - self.start_time if self.start_time else None
            }, f, indent=2)
        
        # Log all artifacts in directory
        if self.artifacts_dir.exists() and any(self.artifacts_dir.iterdir()):
            mlflow_client.log_artifacts(str(self.artifacts_dir))
    
    async def save_artifact(self, data: Any, filename: str, format: str = "json"):
        """Save data as an artifact."""
        filepath = self.artifacts_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "text":
            with open(filepath, 'w') as f:
                f.write(str(data))
        elif format == "csv":
            # Assume data is a pandas DataFrame
            data.to_csv(filepath, index=False)
        
        return filepath
    
    async def cleanup(self):
        """Cleanup after experiment."""
        # Remove temporary artifacts if needed
        # We'll keep them for now for debugging
        print(f"📁 Artifacts saved to: {self.artifacts_dir}")
    
    def add_param(self, key: str, value: Any):
        """Add a parameter to track."""
        self.params[key] = value
        mlflow_client.log_param(key, value)
    
    def add_metric(self, key: str, value: float):
        """Add a metric to track."""
        self.metrics[key] = value
        mlflow_client.log_metric(key, value)