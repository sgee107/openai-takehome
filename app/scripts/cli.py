"""CLI entry point for the takehome application."""

import click
from minio import Minio
from minio.error import S3Error

from app.settings import settings


def ensure_minio_bucket(bucket_name: str) -> bool:
    """Ensure a MinIO bucket exists; create if missing.

    Returns True if bucket exists or was created, False otherwise.
    """
    client = Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
        return True
    except S3Error as e:
        click.echo(f"MinIO error ensuring bucket '{bucket_name}': {e}")
    except Exception as e:
        click.echo(f"Unexpected error ensuring bucket '{bucket_name}': {e}")
    return False


@click.group()
def data_loader():
    """Data loader CLI for the takehome application."""
    pass


@data_loader.command()
@click.option("--bucket", default=None, help="Bucket to ensure (default: app + mlflow)")
def ensure_buckets(bucket: str | None = None):
    """Ensure MinIO buckets exist (app and MLflow)."""
    targets = [bucket] if bucket else [settings.minio_bucket_name, "mlflow-artifacts"]
    for b in targets:
        if ensure_minio_bucket(b):
            click.echo(f"✓ Bucket ensured: {b}")
        else:
            click.echo(f"✗ Failed to ensure bucket: {b}")


@data_loader.command()
def load():
    """Load data into the database."""
    # Ensure needed buckets before load
    ensure_minio_bucket(settings.minio_bucket_name)
    ensure_minio_bucket("mlflow-artifacts")
    click.echo("Loading data...")
    # TODO: Implement data loading
    raise NotImplementedError("Data loading not yet implemented")


@data_loader.command()
def validate():
    """Validate loaded data."""
    click.echo("Validating data...")
    # TODO: Implement validation
    raise NotImplementedError("Validation not yet implemented")


if __name__ == '__main__':
    data_loader()
