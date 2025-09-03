#!/bin/bash
# Script to run experiments with MinIO credentials

# Set MinIO credentials (overriding any AWS credentials)
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Run the experiment
python -m app.experiments.run_experiments --experiment strategy --num-products 50
