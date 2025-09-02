#!/usr/bin/env python3
"""
MLflow server startup script.
Starts MLflow tracking server with PostgreSQL backend and MinIO artifact storage.
"""

import subprocess
import sys
import os


def start_mlflow_server():
    """Start MLflow server with proper configuration."""
    # Set environment variables for S3/MinIO artifact storage
    env = os.environ.copy()
    env.update({
        "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin", 
        "AWS_SECRET_ACCESS_KEY": "minioadmin",
    })
    
    # MLflow server command
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "postgresql://postgres:postgres@localhost:5432/chatdb",
        "--default-artifact-root", "s3://mlflow-artifacts/",
        "--host", "127.0.0.1",
        "--port", "5001"
    ]
    
    print("🚀 Starting MLflow server...")
    print(f"📊 UI will be available at: http://127.0.0.1:5001")
    print(f"🗄️  Backend: PostgreSQL")
    print(f"📦 Artifacts: MinIO S3 (s3://mlflow-artifacts/)")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 MLflow server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start MLflow server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_mlflow_server()