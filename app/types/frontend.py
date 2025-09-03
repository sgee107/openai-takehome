"""
TypeScript-compatible type definitions for frontend interface.
These will be used to generate the actual TypeScript interfaces.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel


class ProductImage(BaseModel):
    """Product image with different resolutions"""
    thumb: Optional[str] = None
    large: Optional[str] = None
    hi_res: Optional[str] = None
    variant: Optional[str] = None


class ProductResult(BaseModel):
    """Product result for search responses"""
    parent_asin: str
    title: str
    main_category: str
    store: Optional[str] = None
    images: List[ProductImage] = []
    price: Optional[float] = None
    average_rating: Optional[float] = None
    rating_number: Optional[int] = None
    features: List[str] = []
    description: List[str] = []
    details: Dict[str, Any] = {}
    categories: List[List[str]] = []
    videos: List[Any] = []
    bought_together: Optional[Any] = None
    
    # Mock fields for frontend
    similarity_score: float
    rank: int


class ChatRequest(BaseModel):
    """Request to chat/search endpoint"""
    query: str
    limit: Optional[int] = 20
    strategy: Optional[str] = "mock"


class ChatResponse(BaseModel):
    """Response from chat/search endpoint"""
    results: List[ProductResult]
    query: str
    strategy: str
    total: int
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    total_products: int
    sample_product_titles: List[str]