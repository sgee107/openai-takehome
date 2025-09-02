#!/usr/bin/env python3
"""
Test script for the optional MLflow client.
This demonstrates how to use MLflow tracking throughout your app without
requiring MLflow to be available everywhere.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import our optional MLflow client
from app.mlflow_client import mlflow_client, track_experiment


def test_basic_logging():
    """Test basic MLflow logging functionality."""
    print("🧪 Testing basic MLflow logging...")
    
    # Create/set experiment
    mlflow_client.create_experiment("client_test_basic")
    
    # Start a run manually
    with mlflow_client.start_run("client_test_basic", "manual_run"):
        # Log parameters
        mlflow_client.log_param("test_type", "basic_logging")
        mlflow_client.log_param("framework", "sklearn")
        
        # Log multiple params at once
        mlflow_client.log_params({
            "n_samples": 100,
            "n_features": 20,
            "random_state": 42
        })
        
        # Log metrics
        mlflow_client.log_metric("dummy_accuracy", 0.85)
        mlflow_client.log_metric("dummy_precision", 0.82)
        
        # Log multiple metrics
        mlflow_client.log_metrics({
            "recall": 0.88,
            "f1_score": 0.85
        })
        
        # Add tags
        mlflow_client.add_tags({
            "model_type": "logistic_regression",
            "data_type": "synthetic"
        })
        
        print("✅ Basic logging completed")


@track_experiment("client_test_decorator", "decorated_run")
def test_decorator_logging():
    """Test MLflow logging using the decorator."""
    print("🎯 Testing decorator-based logging...")
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Log experiment parameters
    mlflow_client.log_params({
        "model": "LogisticRegression",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1]
    })
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate and log metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    mlflow_client.log_metrics({
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "accuracy_diff": train_score - test_score
    })
    
    # Create and log a simple plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.6, cmap='viridis')
    plt.title(f'Test Data (Accuracy: {test_score:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    
    # Log the figure
    mlflow_client.log_figure(plt.gcf(), "test_data_scatter.png")
    plt.close()
    
    print(f"✅ Decorator logging completed (Test Accuracy: {test_score:.3f})")
    
    return test_score


def test_embedding_simulation():
    """Simulate an embedding experiment."""
    print("🔗 Testing embedding experiment simulation...")
    
    with mlflow_client.start_run("embedding_experiments", "normalization_test"):
        # Simulate embedding experiment parameters
        mlflow_client.log_params({
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": 1536,
            "preprocessing_strategy": "simple_concat",
            "normalization": "l2_normalized",
            "n_products": 300
        })
        
        # Simulate some metrics you might track
        mlflow_client.log_metrics({
            "precision_at_5": 0.84,
            "precision_at_10": 0.78,
            "recall_at_10": 0.65,
            "mrr": 0.72,
            "query_latency_p95": 45.2,
            "index_build_time": 12.5
        })
        
        # Log tags for filtering
        mlflow_client.add_tags({
            "strategy": "normalization_comparison",
            "data_source": "amazon_fashion",
            "experiment_type": "preprocessing"
        })
        
        # Simulate UMAP visualization
        from sklearn.decomposition import PCA
        X = np.random.randn(300, 1536)  # Simulate 300 embeddings
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=30)
        plt.title('Embedding Space Visualization (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        
        mlflow_client.log_figure(plt.gcf(), "embedding_space_pca.png")
        plt.close()
        
        print("✅ Embedding simulation completed")


def main():
    """Run all tests."""
    print("🚀 Testing MLflow Client")
    print("=" * 50)
    
    if not mlflow_client.enabled:
        print("⚠️  MLflow not available - tests will run but won't log anything")
        print("💡 Start MLflow server with: uv run mlflow-server")
    else:
        print(f"✅ MLflow enabled - tracking URI: {mlflow_client.tracking_uri}")
    
    print()
    
    # Run tests
    test_basic_logging()
    print()
    
    accuracy = test_decorator_logging()
    print()
    
    test_embedding_simulation()
    print()
    
    print("🎉 All tests completed!")
    if mlflow_client.enabled:
        print(f"🌐 View results at: {mlflow_client.tracking_uri}")
    
    return accuracy


if __name__ == "__main__":
    main()