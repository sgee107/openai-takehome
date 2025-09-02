from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database configuration
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="chatdb", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="postgres", description="Database password")
    
    # OpenAI configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-5-mini", description="OpenAI model to use")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    openai_embedding_dimension: int = Field(default=1536, description="Embedding vector dimension")
    
    # CORS configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )
    
    # Application settings
    debug: bool = Field(default=False, description="Debug mode")

    # MLflow configuration
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLflow tracking server URI"
    )
    mlflow_experiment: str | None = Field(
        default=None,
        description="Default MLflow experiment name (optional)"
    )

    # MinIO configuration
    minio_endpoint: str = Field(
        default="localhost:9000",
        description="MinIO endpoint host:port"
    )
    minio_access_key: str = Field(
        default="minioadmin",
        description="MinIO access key"
    )
    minio_secret_key: str = Field(
        default="minioadmin",
        description="MinIO secret key"
    )
    minio_secure: bool = Field(
        default=False,
        description="Use HTTPS for MinIO"
    )
    minio_bucket_name: str = Field(
        default="mlflow-artifacts",
        description="Default MinIO bucket for MLflow artifacts"
    )
    
    @computed_field
    @property
    def database_url(self) -> str:
        """Construct database URL from individual components."""
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()
