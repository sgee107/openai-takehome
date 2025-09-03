"""
Mock API endpoints for frontend development.
These endpoints simulate the real search functionality with sample data.
"""

import asyncio
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.types.frontend import ChatRequest, ChatResponse, HealthResponse
from app.services.mock_data import mock_data_service


router = APIRouter(prefix="/api/mock", tags=["mock"])


@router.post("/chat", response_model=ChatResponse)
async def mock_chat(request: ChatRequest):
    """
    Mock chat/search endpoint that returns ranked fashion products.
    Simulates embedding-based semantic search with simple text matching.
    """
    start_time = time.time()
    
    try:
        # Add realistic delay to show skeleton loading
        await asyncio.sleep(0.5)
        
        # Perform mock search
        if request.query.strip():
            results = mock_data_service.search_products(request.query, request.limit)
        else:
            # Return top products for empty query
            results = mock_data_service.get_all_products()[:request.limit]
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return ChatResponse(
            results=results,
            query=request.query,
            strategy=request.strategy or "mock",
            total=len(results),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock search failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def mock_health():
    """Health check endpoint with sample data info"""
    try:
        all_products = mock_data_service.get_all_products()
        sample_titles = [p.title for p in all_products[:5]]
        
        return HealthResponse(
            status="healthy",
            version="1.0.0-mock",
            total_products=mock_data_service.total_products,
            sample_product_titles=sample_titles
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/products/{asin}")
async def get_product(asin: str):
    """Get a specific product by ASIN"""
    product = mock_data_service.get_product_by_asin(asin)
    
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {asin} not found")
    
    return product


@router.get("/random")
async def get_random_products(count: int = 20):
    """Get random products for testing"""
    if count > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 products allowed")
    
    products = mock_data_service.get_random_products(count)
    
    return {
        "results": products,
        "total": len(products),
        "query": "random",
        "strategy": "random"
    }


@router.get("/images/sample")
async def get_sample_images(count: int = 50):
    """Get sample product image URLs for background generation"""
    if count > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 images allowed")
    
    image_urls = mock_data_service.get_sample_images(count)
    
    return {
        "image_urls": image_urls,
        "total": len(image_urls)
    }