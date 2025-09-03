"""
Test mock API endpoints for frontend development.
These tests verify the mock endpoints work correctly without database dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from app.types.frontend import ChatRequest, ChatResponse, HealthResponse, ProductResult, ProductImage


class TestSettings(BaseModel):
    """Test configuration that disables database dependencies"""
    database_url: str = "sqlite:///:memory:"
    openai_api_key: str = "test-key-mock"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    mlflow_tracking_uri: str = "file:///tmp/test_mlruns"


@pytest.fixture(scope="module")
def test_settings():
    """Module-level test settings fixture"""
    return TestSettings()


@pytest.fixture
def mock_data_service():
    """Mock the data service to avoid file dependencies"""
    mock_service = MagicMock()
    
    # Create sample product data
    sample_product = ProductResult(
        parent_asin="B08BHN9PK5",
        title="YUEDGE 5 Pairs Men's Moisture Control Cushioned Dry Fit Casual Athletic Crew Socks",
        main_category="AMAZON FASHION",
        store="GiveGift",
        images=[
            ProductImage(
                thumb="https://m.media-amazon.com/images/I/41+cCfaVOFS._AC_SR38,50_.jpg",
                large="https://m.media-amazon.com/images/I/41+cCfaVOFS._AC_.jpg",
                hi_res="https://m.media-amazon.com/images/I/81XlFXImFrS._AC_UL1500_.jpg",
                variant="MAIN"
            )
        ],
        price=29.81,
        average_rating=4.6,
        rating_number=16,
        features=["Moisture Control", "Cushioned"],
        description=["Comfortable athletic socks"],
        details={"Package Dimensions": "10.31 x 8.5 x 1.73 inches"},
        categories=[["Clothing", "Men", "Socks"]],
        videos=[],
        bought_together=None,
        similarity_score=0.95,
        rank=1
    )
    
    # Mock service methods
    mock_service.total_products = 300
    mock_service.get_all_products.return_value = [sample_product] * 20
    mock_service.search_products.return_value = [sample_product] * 5
    mock_service.get_product_by_asin.return_value = sample_product
    mock_service.get_random_products.return_value = [sample_product] * 10
    mock_service.get_sample_images.return_value = [
        "https://m.media-amazon.com/images/I/41+cCfaVOFS._AC_SR38,50_.jpg",
        "https://m.media-amazon.com/images/I/41jBdP7etRS._AC_SR38,50_.jpg"
    ]
    
    return mock_service


@pytest.fixture
def client(mock_data_service, test_settings):
    """Create test client with mocked dependencies"""
    with patch('app.services.mock_data.mock_data_service', mock_data_service), \
         patch('app.settings.settings', test_settings), \
         patch('app.db.database.init_db', return_value=None), \
         patch('app.db.database.get_async_session', return_value=MagicMock()):
        # Import after patching to ensure mocks are used
        from app.app import app
        return TestClient(app)


def test_mock_chat_endpoint(client, mock_data_service):
    """Test the mock chat endpoint"""
    response = client.post("/api/mock/chat", json={
        "query": "blue shirts",
        "limit": 10,
        "strategy": "mock"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert "results" in data
    assert "query" in data
    assert "strategy" in data
    assert "total" in data
    assert data["query"] == "blue shirts"
    assert data["strategy"] == "mock"
    assert len(data["results"]) > 0
    
    # Check product structure
    product = data["results"][0]
    assert "parent_asin" in product
    assert "title" in product
    assert "similarity_score" in product
    assert "rank" in product
    assert "images" in product


def test_mock_health_endpoint(client, mock_data_service):
    """Test the mock health endpoint"""
    response = client.get("/api/mock/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["total_products"] == 300
    assert "sample_product_titles" in data


def test_get_product_by_asin(client, mock_data_service):
    """Test getting a specific product"""
    asin = "B08BHN9PK5"
    response = client.get(f"/api/mock/products/{asin}")
    
    assert response.status_code == 200
    product = response.json()
    
    assert product["parent_asin"] == asin
    assert "title" in product
    assert "similarity_score" in product


def test_get_random_products(client, mock_data_service):
    """Test getting random products"""
    response = client.get("/api/mock/random?count=5")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "results" in data
    assert len(data["results"]) == 10  # Based on mock return value
    assert data["query"] == "random"


def test_get_sample_images(client, mock_data_service):
    """Test getting sample images for background"""
    response = client.get("/api/mock/images/sample?count=10")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "image_urls" in data
    assert "total" in data
    assert len(data["image_urls"]) == 2  # Based on mock return value


def test_chat_with_empty_query(client, mock_data_service):
    """Test chat endpoint with empty query"""
    response = client.post("/api/mock/chat", json={
        "query": "",
        "limit": 5
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == ""
    assert len(data["results"]) > 0


def test_chat_request_validation():
    """Test ChatRequest model validation"""
    # Valid request
    request = ChatRequest(query="test query", limit=10)
    assert request.query == "test query"
    assert request.limit == 10
    assert request.strategy == "mock"  # Default value
    
    # Minimal request
    request = ChatRequest(query="test")
    assert request.query == "test"
    assert request.limit == 20  # Default value


def test_product_result_model():
    """Test ProductResult model structure"""
    product = ProductResult(
        parent_asin="TEST123",
        title="Test Product",
        main_category="Fashion",
        similarity_score=0.85,
        rank=1
    )
    
    assert product.parent_asin == "TEST123"
    assert product.title == "Test Product"
    assert product.similarity_score == 0.85
    assert product.rank == 1
    assert product.images == []  # Default empty list
    assert product.features == []  # Default empty list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])