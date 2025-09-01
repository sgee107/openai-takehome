# Takehome Application

## Quickstart Guide

Get up and running with the application in just a few steps.

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) and Docker Compose

### Development Setup

#### 1. Start Infrastructure Services

Start PostgreSQL, MinIO, and MLflow services:

```bash
docker compose -f docker-compose-dev.yml up -d
```

This will start:
- **PostgreSQL** (port 5432) - Database with pgvector extension
- **MinIO** (ports 9000/9001) - S3-compatible object storage
- **MLflow** (port 5001) - ML experiment tracking

#### 2. Install Dependencies

```bash
uv sync
```

#### 3. Run the Backend

Start the FastAPI application:

```bash
uv run fastapi dev app/app.py
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Services Access

- **MLflow UI**: http://localhost:5001
- **MinIO Console**: http://localhost:9001 (admin/minioadmin)
- **API Documentation**: http://localhost:8000/docs

### Environment Configuration

Copy `.env.example` to `.env` and adjust settings as needed:

```bash
cp .env.example .env
```

### Testing

Run the test suite:

```bash
uv run pytest
```

### Stopping Services

```bash
docker compose -f docker-compose-dev.yml down
```