"""Search API router for the Chat Search vNext implementation."""

import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.types.search_api import SearchRequest, ChatSearchResponse
from app.agents.fashion_agent import FashionAgent
from app.dependencies import get_openai_client
from app.db.database import get_async_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["search"])


@router.post("/search")
async def search_products(
    request: SearchRequest,
    client: AsyncOpenAI = Depends(get_openai_client),
    session: AsyncSession = Depends(get_async_session)
) -> ChatSearchResponse:
    """
    Product search endpoint using the agent-based architecture.
    
    This endpoint uses the FashionAgent with ComprehensiveSearchTool to execute:
    1. Intent classification and query parsing
    2. Semantic similarity search with filters
    3. Bayesian rating adjustment and linear blending
    4. Facet generation
    5. Structured response formatting
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing search request: query='{request.query}', topK={request.topK}, λ={request.lambda_blend}")
        
        # Initialize fashion agent
        agent = FashionAgent(client)
        
        # Execute comprehensive search through agent
        response = await agent.search_products(
            query=request.query,
            session=session,
            topk=request.topK,
            lambda_blend=request.lambda_blend,
            debug=request.debug
        )
        
        logger.info(f"Search completed in {time.time() - start_time:.3f}s: {len(response.results)} results")
        return response
    
    except ValueError as e:
        logger.error(f"Validation error in search request: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Search pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error in search pipeline"
        )


@router.get("/search/health")
async def search_health():
    """Health check endpoint for search service."""
    return {
        "status": "healthy",
        "service": "chat-search-vnext",
        "timestamp": time.time()
    }
