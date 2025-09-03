#!/usr/bin/env python3
"""
Script to retrieve MLflow artifacts from MinIO.
"""
import os
from app.dependencies import get_minio_client
from app.settings import settings
import mlflow

def list_mlflow_artifacts():
    """List all artifacts in the MLflow bucket."""
    client = get_minio_client()
    bucket_name = settings.minio_bucket_name
    
    print(f"🪣 Checking bucket: {bucket_name}")
    
    # Check if bucket exists
    if not client.bucket_exists(bucket_name):
        print(f"❌ Bucket '{bucket_name}' does not exist")
        return
    
    print(f"✅ Bucket '{bucket_name}' exists")
    
    # List objects in the bucket
    objects = client.list_objects(bucket_name, recursive=True)
    
    artifact_count = 0
    png_files = []
    
    print("\n📁 MLflow artifacts in MinIO:")
    for obj in objects:
        artifact_count += 1
        print(f"  - {obj.object_name} ({obj.size} bytes)")
        
        # Track PNG files
        if obj.object_name.endswith('.png'):
            png_files.append(obj.object_name)
    
    if artifact_count == 0:
        print("  (No artifacts found)")
    else:
        print(f"\n📊 Total artifacts: {artifact_count}")
        
    # Download PNG files
    if png_files:
        print(f"\n🖼️  Found {len(png_files)} visualization files:")
        download_dir = "retrieved_artifacts"
        os.makedirs(download_dir, exist_ok=True)
        
        for png_file in png_files:
            local_path = os.path.join(download_dir, os.path.basename(png_file))
            try:
                client.fget_object(bucket_name, png_file, local_path)
                print(f"  ✅ Downloaded: {png_file} -> {local_path}")
            except Exception as e:
                print(f"  ❌ Failed to download {png_file}: {e}")
    
    # Also check MLflow for recent runs
    print("\n🔍 Checking recent MLflow experiments:")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    experiments = mlflow.search_experiments()
    for exp in experiments:
        if "embedding_strategy" in exp.name:
            print(f"\n📋 Experiment: {exp.name} (ID: {exp.experiment_id})")
            
            # Get recent runs
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=5,
                order_by=["start_time DESC"]
            )
            
            if not runs.empty:
                print(f"  Latest runs:")
                for idx, run in runs.iterrows():
                    run_id = run['run_id']
                    start_time = run['start_time']
                    print(f"    - Run {run_id} at {start_time}")
                    
                    # Try to get artifact URI
                    run_obj = mlflow.get_run(run_id)
                    artifact_uri = run_obj.info.artifact_uri
                    print(f"      Artifact URI: {artifact_uri}")

if __name__ == "__main__":
    list_mlflow_artifacts()
