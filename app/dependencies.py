from typing import Annotated
from fastapi import Depends
from openai import AsyncOpenAI
from minio import Minio

from app.settings import settings


def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_minio_client() -> Minio:
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure
    )


# OpenAIClient = Annotated[AsyncOpenAI, Depends(get_openai_client)]
# MinioClient = Annotated[Minio, Depends(get_minio_client)]