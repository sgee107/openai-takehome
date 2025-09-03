"""Search API router for the Chat Search vNext implementation."""

import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.types.search_api import (
    SearchRequest, ChatSearchResponse, AgentDecision, UIHints, 
    ProductResult, ProductMatch, DebugTrace, QueryComplexity
)
from app.services.intent_classifier import IntentClassifier
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.services.facets import FacetsService
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
    Product search endpoint with intent classification and linear-blend ranking.
    
    This endpoint implements the full vNext search pipeline:
    1. Classify query complexity and parse structured elements
    2. Retrieve top-K candidates using semantic similarity
    3. Re-rank using linear blend of semantic + Bayesian rating scores
    4. Generate facets and follow-up prompts
    5. Return structured response with UI hints
    """
    start_time = time.time()
    debug_trace = DebugTrace(timings={}) if request.debug else None
    
    try:
        logger.info(f"Processing search request: query='{request.query}', topK={request.topK}, λ={request.lambda_blend}")
        
        # Initialize services
        classifier = IntentClassifier(client)
        retriever = RetrievalService(client) 
        ranker = RankingService()
        facets_service = FacetsService(client)
        
        # Step 1: Classify intent and parse query
        classify_start = time.time()
        complexity, parsed = await classifier.classify(request.query)
        if debug_trace:
            debug_trace.timings["classify"] = time.time() - classify_start
        
        classification_reason = f"Classified as complexity {complexity.value}"
        agent_decision = AgentDecision(
            complexity=complexity,
            parsed=parsed,
            reason=classification_reason
        )
        
        # Step 2: Retrieve top-K candidates
        retrieve_start = time.time()
        
        # Extract filters from parsed query
        category_filter = parsed.category if parsed else None
        brand_filter = parsed.brand if parsed else None
        price_min = parsed.price_min if parsed else None
        price_max = parsed.price_max if parsed else None
        
        candidates = await retriever.topk(
            query=request.query,
            session=session,
            k=request.topK,
            category_filter=category_filter,
            brand_filter=brand_filter,
            price_min=price_min,
            price_max=price_max
        )
        if debug_trace:
            debug_trace.timings["retrieve"] = time.time() - retrieve_start
        
        # Step 3: Re-rank with linear blend
        rank_start = time.time()
        ranked_candidates = ranker.rerank(candidates, request.lambda_blend)
        if debug_trace:
            debug_trace.timings["rank"] = time.time() - rank_start
        
        # Step 4: Generate facets and follow-ups
        facets_start = time.time()
        facets = facets_service.generate_facets(ranked_candidates)
        followups = await facets_service.generate_followups(complexity, request.query, classification_reason)
        if debug_trace:
            debug_trace.timings["facets"] = time.time() - facets_start
        
        # Step 5: Format response
        format_start = time.time()
        results = []
        for candidate in ranked_candidates:
            # Get primary image
            images = candidate.metadata.get('images', [])
            primary_image = None
            if images:
                for img in images:
                    if isinstance(img, dict):
                        # Try different image quality levels
                        primary_image = (img.get('hi_res') or 
                                       img.get('large') or 
                                       img.get('thumb'))
                        if primary_image:
                            break
                    elif isinstance(img, str):
                        # Handle direct URL strings
                        primary_image = img
                        break
            
            # Generate explanation
            explanation = ranker.get_ranking_explanation(candidate)
            
            product_result = ProductResult(
                id=candidate.product_id,
                title=candidate.title,
                image=primary_image,
                url=f"/product/{candidate.metadata.get('parent_asin', candidate.product_id)}",
                price=candidate.price,
                rating=candidate.rating,
                ratingCount=candidate.rating_count,
                match=ProductMatch(
                    final=candidate.final_score,
                    semantic=candidate.semantic_norm,
                    rating=candidate.rating_norm,
                    lambda_used=candidate.lambda_used,
                    explanation=explanation
                )
            )
            results.append(product_result)
        
        if debug_trace:
            debug_trace.timings["format"] = time.time() - format_start
        
        # UI hints based on complexity
        ui_hints = UIHints(
            layout="list" if complexity == QueryComplexity.DIRECT else "grid",
            showRating=True,
            showFacets=complexity != QueryComplexity.DIRECT,
            emptyStateCopy="No products found. Try a broader search or different terms." if not results else None
        )
        
        # Debug information
        if debug_trace:
            debug_trace.timings["total"] = time.time() - start_time
            debug_trace.plan = (
                f"Complexity {complexity.value} → "
                f"TopK({len(candidates)}) → "
                f"LinearBlend(λ={request.lambda_blend}) → "
                f"Results({len(results)})"
            )
            if request.debug and ranked_candidates:
                debug_trace.rawScores = [
                    {
                        "product_id": c.product_id,
                        "title": c.title[:50],
                        "similarity": round(c.similarity, 4),
                        "final_score": round(c.final_score, 4),
                        "semantic_norm": round(c.semantic_norm, 4),
                        "rating_norm": round(c.rating_norm, 4)
                    }
                    for c in ranked_candidates[:10]  # Top 10 for debug
                ]
        
        response = ChatSearchResponse(
            agent=agent_decision,
            ui=ui_hints,
            results=results,
            facets=facets if ui_hints.showFacets and facets else None,
            followups=followups if followups else None,
            debug=debug_trace
        )
        
        logger.info(f"Search completed in {time.time() - start_time:.3f}s: {len(results)} results")
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