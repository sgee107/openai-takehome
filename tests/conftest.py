"""Pytest configuration and fixtures for testing embeddings and outputs."""

import sys
from pathlib import Path
import pytest
from typing import Generator
import os
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import AsyncSessionLocal, async_engine, init_db
from app.db.models import Base
from app.settings import settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_session():
    """Create a database session for testing."""
    # Initialize database
    await init_db()
    
    async with AsyncSessionLocal() as session:
        yield session


@pytest.fixture
def sample_queries():
    """Sample queries for testing different embedding scenarios."""
    return [
        "protein powder",
        "organic vegetables",
        "dairy-free milk alternatives",
        "high protein snacks",
        "gluten free bread",
        "vitamin supplements",
        "energy drinks",
        "chocolate bars",
        "fresh fruits",
        "cooking oils"
    ]


@pytest.fixture
def embedding_configs():
    """Different embedding configuration scenarios to test."""
    return {
        "small": {
            "model": "text-embedding-3-small", 
            "dimension": 512
        }
    }


@pytest.fixture
def output_formatter():
    """Helper to format outputs for review."""
    def format_results(query: str, results: list, config: dict = None):
        output = f"\n{'='*60}\n"
        output += f"Query: {query}\n"
        if config:
            output += f"Config: {config}\n"
        output += f"{'='*60}\n"
        
        for idx, result in enumerate(results, 1):
            output += f"\n{idx}. {result.get('name', 'Unknown')}\n"
            output += f"   Category: {result.get('category', 'N/A')}\n"
            output += f"   Score: {result.get('score', 0):.4f}\n"
            output += f"   Description: {result.get('description', 'N/A')[:100]}...\n"
        
        return output
    
    return format_results
