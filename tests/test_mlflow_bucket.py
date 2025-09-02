"""
Test MLflow bucket creation and artifact logging functionality.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
import boto3
from botocore.exceptions import ClientError

from app.mlflow_client import MLflowClient
from app.settings import settings


class TestMLflowBucketCreation:
    """Test suite for MLflow bucket creation and management."""
    
    def test_bucket_creation_on_init(self, monkeypatch):
        """Test that MLflow client creates bucket during initialization."""
        # Mock boto3 client
        mock_s3 = MagicMock()
        mock_boto3_client = MagicMock(return_value=mock_s3)
        monkeypatch.setattr('app.mlflow_client.boto3.client', mock_boto3_client)
        
        # Mock head_bucket to raise 404 (bucket doesn't exist)
        mock_s3.head_bucket.side_effect = ClientError(
            {'Error': {'Code': '404'}}, 
            'HeadBucket'
        )
        
        # Mock mlflow functions
        monkeypatch.setattr('mlflow.set_tracking_uri', MagicMock())
        monkeypatch.setattr('mlflow.get_tracking_uri', MagicMock())
        
        # Create a new MLflow client instance
        client = MLflowClient()
        
        # Verify S3 client was created with correct parameters
        mock_boto3_client.assert_called_with(
            's3',
            endpoint_url=f"http://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            region_name='us-east-1'
        )
        
        # Verify bucket creation was attempted
        mock_s3.create_bucket.assert_called_once_with(Bucket=settings.minio_bucket_name)
    
    def test_bucket_exists_no_creation(self, monkeypatch):
        """Test that MLflow client doesn't create bucket if it already exists."""
        # Mock boto3 client
        mock_s3 = MagicMock()
        mock_boto3_client = MagicMock(return_value=mock_s3)
        monkeypatch.setattr('app.mlflow_client.boto3.client', mock_boto3_client)
        
        # Mock head_bucket to succeed (bucket exists)
        mock_s3.head_bucket.return_value = True
        
        # Mock mlflow functions
        monkeypatch.setattr('mlflow.set_tracking_uri', MagicMock())
        monkeypatch.setattr('mlflow.get_tracking_uri', MagicMock())
        
        # Create a new MLflow client instance
        client = MLflowClient()
        
        # Verify bucket existence was checked
        mock_s3.head_bucket.assert_called_once_with(Bucket=settings.minio_bucket_name)
        
        # Verify bucket creation was NOT attempted
        mock_s3.create_bucket.assert_not_called()
    
    def test_bucket_creation_failure_continues(self, monkeypatch):
        """Test that MLflow client continues even if bucket creation fails."""
        # Mock boto3 client
        mock_s3 = MagicMock()
        mock_boto3_client = MagicMock(return_value=mock_s3)
        monkeypatch.setattr('app.mlflow_client.boto3.client', mock_boto3_client)
        
        # Mock head_bucket to raise a different error
        mock_s3.head_bucket.side_effect = ClientError(
            {'Error': {'Code': 'AccessDenied'}}, 
            'HeadBucket'
        )
        
        # Mock mlflow functions
        monkeypatch.setattr('mlflow.set_tracking_uri', MagicMock())
        monkeypatch.setattr('mlflow.get_tracking_uri', MagicMock())
        
        # Create a new MLflow client instance - should not raise
        client = MLflowClient()
        
        # Verify initialization completed (client is created)
        assert client is not None


class TestMLflowBucketIntegration:
    """Integration tests for MLflow bucket functionality."""
    
    def test_artifact_logging_with_bucket(self):
        """Test actual artifact logging to MinIO bucket."""
        # Note: This test requires MLflow server and MinIO to be running
        pytest.skip("Integration test - requires MLflow server and MinIO running")
        
        from app.mlflow_client import mlflow_client
        
        if not mlflow_client.enabled:
            pytest.skip("MLflow is not enabled")
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test artifact content for bucket verification")
            test_file = f.name
        
        try:
            # Create a test run and log artifact
            with mlflow_client.start_run("test_bucket_integration", "artifact_test"):
                mlflow_client.log_param("test_type", "bucket_integration")
                mlflow_client.log_metric("test_score", 1.0)
                mlflow_client.log_artifact(test_file)
        finally:
            # Clean up
            if os.path.exists(test_file):
                os.remove(test_file)
