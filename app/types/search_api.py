"""TypeScript-compatible types for the Chat Search API vNext."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from enum import IntEnum


class QueryComplexity(IntEnum):
    """Query complexity levels."""
    DIRECT = 1      # SKU/brand+model, quoted terms
    FILTERED = 2    # category + constraints  
    AMBIGUOUS = 3   # broad terms, conflicting signals


class SearchRequest(BaseModel):
    """Request schema for /chat/search endpoint."""
    query: str = Field(..., description="User search query")
    topK: Optional[int] = Field(200, description="Number of candidates to retrieve")
    lambda_blend: Optional[float] = Field(0.85, description="Semantic weight (0-1)")
    debug: Optional[bool] = Field(False, description="Include debug information")


class ParsedQuery(BaseModel):
    """Structured query parsing results."""
    category: Optional[str] = Field(None, description="Product category")
    brand: Optional[str] = Field(None, description="Brand name")
    price_min: Optional[float] = Field(None, description="Minimum price")
    price_max: Optional[float] = Field(None, description="Maximum price")
    
    def get_price_filters(self, product_model):
        """Get database filter conditions for price constraints."""
        conditions = []
        if self.price_min is not None:
            conditions.append(product_model.price >= self.price_min)
        if self.price_max is not None:
            conditions.append(product_model.price <= self.price_max)
        return conditions
    
    @property
    def has_price_filter(self) -> bool:
        """Check if any price constraints are specified."""
        return self.price_min is not None or self.price_max is not None


class AgentDecision(BaseModel):
    """Agent classification and reasoning."""
    complexity: QueryComplexity = Field(..., description="Query complexity level")
    parsed: Optional[ParsedQuery] = Field(None, description="Parsed query structure")
    reason: Optional[str] = Field(None, description="Classification reasoning")


class ProductMatch(BaseModel):
    """Matching scores and explanation for a product result."""
    final: float = Field(..., description="Final blended score (0-1)")
    semantic: float = Field(..., description="Normalized semantic similarity (0-1)")
    rating: float = Field(..., description="Normalized Bayesian rating (0-1)")
    lambda_used: float = Field(..., description="Lambda value used in blend")
    explanation: Optional[str] = Field(None, description="Score explanation")


class ProductResult(BaseModel):
    """Product result with match information."""
    id: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    image: Optional[str] = Field(None, description="Primary product image URL")
    url: Optional[str] = Field(None, description="Product detail page URL")
    price: Optional[float] = Field(None, description="Product price")
    rating: Optional[float] = Field(None, description="Average rating (0-5)")
    ratingCount: Optional[int] = Field(None, description="Number of ratings")
    match: ProductMatch = Field(..., description="Matching scores")


class FacetOption(BaseModel):
    """Individual facet option with count."""
    value: str = Field(..., description="Facet option value")
    count: int = Field(..., description="Number of products with this value")


class FacetGroup(BaseModel):
    """Group of related facet options."""
    name: str = Field(..., description="Facet group name")
    options: List[FacetOption] = Field(..., description="Available facet options")


class FollowupPrompt(BaseModel):
    """Follow-up question for ambiguous queries."""
    text: str = Field(..., description="Follow-up question text")
    rationale: Optional[str] = Field(None, description="Why this question is helpful")


class UIHints(BaseModel):
    """UI rendering hints for the frontend."""
    layout: Literal["grid", "list"] = Field("grid", description="Recommended layout")
    showRating: bool = Field(True, description="Whether to show rating information")
    showFacets: bool = Field(True, description="Whether to show facets")
    emptyStateCopy: Optional[str] = Field(None, description="Empty state message")


class DebugTrace(BaseModel):
    """Debug information for performance analysis."""
    timings: Optional[Dict[str, float]] = Field(None, description="Timing breakdown")
    plan: Optional[str] = Field(None, description="Execution plan summary")
    rawScores: Optional[List[Dict[str, Any]]] = Field(None, description="Raw candidate scores")


class ChatSearchResponse(BaseModel):
    """Complete response schema for /chat/search endpoint."""
    agent: AgentDecision = Field(..., description="Agent classification results")
    ui: UIHints = Field(..., description="UI rendering hints")
    results: List[ProductResult] = Field(..., description="Ranked product results")
    facets: Optional[List[FacetGroup]] = Field(None, description="Available facets")
    followups: Optional[List[FollowupPrompt]] = Field(None, description="Follow-up questions")
    debug: Optional[DebugTrace] = Field(None, description="Debug information")
