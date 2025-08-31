FROM python:3.12-slim as base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

FROM base as dependencies

COPY pyproject.toml .
COPY README.md .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .

FROM base as dev

COPY --from=dependencies /usr/local /usr/local

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

FROM base as prod

COPY --from=dependencies /usr/local /usr/local

COPY app app/
COPY main.py .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]