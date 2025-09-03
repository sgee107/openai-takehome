"""Comprehensive fashion product search tool combining the full search pipeline."""

import json
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from openai import AsyncOpenAI

from app.db.models import Product, ProductEmbedding
from app.types.search_api import (
    QueryComplexity, ParsedQuery, FacetGroup, FacetOption, 
    ProductResult, ProductMatch, AgentDecision, UIHints
)
from app.prompts.v1.intent_classifier import (
    INTENT_CLASSIFICATION_SYSTEM,
    INTENT_CLASSIFICATION_USER_TEMPLATE
)
from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchCandidate:
    """Internal candidate structure for the search pipeline."""
    product_id: str
    title: str
    price: Optional[float]
    rating: Optional[float]
    rating_count: Optional[int]
    category: str
    similarity: float
    metadata: Dict[str, Any]
    # Ranking scores
    final_score: float = 0.0
    semantic_norm: float = 0.0
    rating_norm: float = 0.0
    lambda_used: float = 0.0


class ComprehensiveSearchTool:
    """
    Single comprehensive search tool that handles the complete fashion search pipeline:
    1. Intent classification and query parsing
    2. Semantic similarity search with filters
    3. Bayesian rating adjustment and linear blending
    4. Facet generation
    5. Result formatting
    """
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.classification_model = "gpt-5-mini"
        self.embedding_model = "text-embedding-3-small"
        
        # Load configurable parameters from settings
        self.lambda_blend = settings.ranking_lambda_blend
        self.bayesian_mu = settings.ranking_bayesian_mu
        self.bayesian_w = settings.ranking_bayesian_w
        self.default_topk = settings.search_topk_default
        self.embedding_strategy = settings.startup_embedding_strategy
    
    async def search_products(
        self,
        query: str,
        session: AsyncSession,
        topk: Optional[int] = None,
        lambda_blend: Optional[float] = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Execute comprehensive product search pipeline.
        
        Args:
            query: User search query
            session: Database session
            topk: Number of candidates to retrieve (default from settings)
            lambda_blend: Semantic vs rating blend weight (default from settings)
            debug: Whether to include debug information
            
        Returns:
            Dictionary containing search results and metadata
        """
        start_time = time.time()
        debug_trace = {"timings": {}} if debug else None
        
        # Use defaults from settings if not provided
        topk = topk or self.default_topk
        lambda_blend = lambda_blend if lambda_blend is not None else self.lambda_blend
        
        try:
            logger.info(f"Starting comprehensive search: query='{query}', topK={topk}, λ={lambda_blend}")
            print(f"[ComprehensiveSearchTool] Starting search for: '{query}'")
            
            # Step 1: Intent Classification and Query Parsing
            classify_start = time.time()
            complexity, parsed = await self._classify_intent(query)
            if debug_trace:
                debug_trace["timings"]["classify"] = time.time() - classify_start
            
            agent_decision = AgentDecision(
                complexity=complexity,
                parsed=parsed,
                reason=f"Classified as complexity {complexity.value}"
            )
            
            # Step 2: Semantic Retrieval with Filters
            retrieve_start = time.time()
            candidates = await self._retrieve_candidates(
                query, session, topk, parsed, debug_trace
            )
            if debug_trace:
                debug_trace["timings"]["retrieve"] = time.time() - retrieve_start
            
            print(f"[ComprehensiveSearchTool] Retrieved {len(candidates)} candidates")
            
            # Step 3: Re-ranking with Linear Blend
            rank_start = time.time()
            ranked_candidates = self._rerank_candidates(candidates, lambda_blend)
            if debug_trace:
                debug_trace["timings"]["rank"] = time.time() - rank_start
            
            # Step 4: Generate Facets (no follow-ups)
            facets_start = time.time()
            facets = self._generate_facets(ranked_candidates)
            if debug_trace:
                debug_trace["timings"]["facets"] = time.time() - facets_start
            
            # Step 5: Format Results
            format_start = time.time()
            results = self._format_results(ranked_candidates)
            if debug_trace:
                debug_trace["timings"]["format"] = time.time() - format_start
            
            # Generate UI hints
            ui_hints = UIHints(
                layout="list" if complexity == QueryComplexity.DIRECT else "grid",
                showRating=True,
                showFacets=complexity != QueryComplexity.DIRECT,
                emptyStateCopy="No products found. Try a broader search or different terms." if not results else None
            )
            
            # Finalize debug information
            if debug_trace:
                debug_trace["timings"]["total"] = time.time() - start_time
                debug_trace["plan"] = (
                    f"Complexity {complexity.value} → "
                    f"TopK({len(candidates)}) → "
                    f"LinearBlend(λ={lambda_blend}) → "
                    f"Results({len(results)})"
                )
                if ranked_candidates:
                    debug_trace["rawScores"] = [
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
            
            logger.info(f"Search completed in {time.time() - start_time:.3f}s: {len(results)} results")
            print(f"[ComprehensiveSearchTool] Search completed: {len(results)} results in {time.time() - start_time:.3f}s")
            
            return {
                "agent": agent_decision,
                "ui": ui_hints,
                "results": results,
                "facets": facets if ui_hints.showFacets and facets else None,
                "followups": None,  # Removed as per requirements
                "debug": debug_trace
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive search pipeline: {e}", exc_info=True)
            raise
    
    async def _classify_intent(self, query: str) -> Tuple[QueryComplexity, ParsedQuery]:
        """Classify query complexity and parse structured elements."""
        if not query or not query.strip():
            return QueryComplexity.AMBIGUOUS, ParsedQuery()
        
        query = query.strip()
        user_prompt = INTENT_CLASSIFICATION_USER_TEMPLATE.format(query=query)
        
        try:
            logger.debug(f"Classifying query: {query}")
            
            response = await self.openai_client.chat.completions.create(
                model=self.classification_model,
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.debug(f"LLM classification result: {result}")
            
            # Extract complexity
            complexity = QueryComplexity(result["complexity"])
            
            # Build ParsedQuery from extracted data (only fields used in semantic retrieval)
            parsed_data = result.get("parsed", {})
            parsed = ParsedQuery(**parsed_data)
            
            logger.info(f"Query classified as complexity {complexity.value}: {result.get('reason', '')}")
            return complexity, parsed
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM classification response: {e}")
            return self._fallback_classification(query)
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Tuple[QueryComplexity, ParsedQuery]:
        """Fallback classification when LLM fails."""
        logger.warning(f"Using fallback classification for query: {query}")
        return QueryComplexity.AMBIGUOUS, ParsedQuery()
    
    async def _retrieve_candidates(
        self,
        query: str,
        session: AsyncSession,
        topk: int,
        parsed: ParsedQuery,
        debug_trace: Optional[Dict] = None
    ) -> List[SearchCandidate]:
        """Retrieve top-K candidates using semantic similarity with filters."""
        if not query.strip():
            return []
        
        try:
            # Generate query embedding
            embedding_start = time.time()
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            query_embedding = response.data[0].embedding
            if debug_trace:
                debug_trace["timings"]["embedding"] = time.time() - embedding_start
            
            # Build similarity query
            query_stmt = (
                select(
                    ProductEmbedding,
                    Product,
                    (1 - ProductEmbedding.embedding.cosine_distance(query_embedding)).label('similarity')
                )
                .join(Product, ProductEmbedding.product_id == Product.id)
            )
            
            # Apply embedding strategy filter
            query_stmt = query_stmt.where(ProductEmbedding.strategy == self.embedding_strategy)
            logger.debug(f"Using embedding strategy: {self.embedding_strategy}")
            
            # Apply parsed filters
            # Note: Category filtering disabled since all products have main_category="AMAZON FASHION"
            # if parsed and parsed.category:
            #     query_stmt = query_stmt.where(Product.main_category.ilike(f"%{parsed.category}%"))
            #     logger.debug(f"Applied category filter: {parsed.category}")
            
            if parsed and parsed.brand:
                query_stmt = query_stmt.where(Product.store.ilike(f"%{parsed.brand}%"))
                logger.debug(f"Applied brand filter: {parsed.brand}")
            
            # Apply price filters using the ParsedQuery method
            if parsed and parsed.has_price_filter:
                price_conditions = parsed.get_price_filters(Product)
                for condition in price_conditions:
                    query_stmt = query_stmt.where(condition)
                logger.debug(f"Applied price filters: min={parsed.price_min}, max={parsed.price_max}")
            
            # Order by similarity and limit
            query_stmt = query_stmt.order_by(text('similarity DESC')).limit(topk)
            
            # Execute query
            result = await session.execute(query_stmt)
            rows = result.all()
            
            logger.info(f"Retrieved {len(rows)} candidates from database")
            
            # Convert to search candidates
            candidates = []
            for row in rows:
                embedding_record, product, similarity = row
                
                candidates.append(SearchCandidate(
                    product_id=str(product.id),  # Convert to string for API compatibility
                    title=product.title,
                    price=product.price,
                    rating=product.average_rating,
                    rating_count=product.rating_number,
                    category=product.main_category or "Unknown",
                    similarity=float(similarity),
                    metadata={
                        'parent_asin': product.parent_asin,
                        'store': product.store,
                        'features': product.features or [],
                        'images': [],  # Will be populated if needed
                        'categories': product.categories or [],
                        'description': product.description,
                        'embedding_strategy': embedding_record.strategy
                    }
                ))
            
            # Log similarity score distribution
            if candidates:
                similarities = [c.similarity for c in candidates]
                logger.debug(f"Similarity scores: min={min(similarities):.3f}, max={max(similarities):.3f}, avg={sum(similarities)/len(similarities):.3f}")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in candidate retrieval: {e}")
            raise
    
    def _rerank_candidates(
        self,
        candidates: List[SearchCandidate],
        lambda_blend: float
    ) -> List[SearchCandidate]:
        """Re-rank candidates using linear blend of semantic + rating scores."""
        if not candidates:
            return []
        
        logger.info(f"Re-ranking {len(candidates)} candidates with λ={lambda_blend}")
        
        # Validate lambda parameter
        lambda_blend = max(0.0, min(1.0, lambda_blend))
        
        # Extract similarity scores
        similarities = np.array([c.similarity for c in candidates])
        
        # Normalize similarities per-query
        sim_min = similarities.min()
        sim_max = similarities.max()
        sim_range = sim_max - sim_min
        
        if sim_range < 1e-9:
            # Zero variance case - use rank-based scoring
            logger.warning("Zero variance in similarity scores, using rank-based scoring")
            s_prime = 1.0 - np.arange(len(candidates)) / (len(candidates) + 1)
        else:
            s_prime = (similarities - sim_min) / sim_range
        
        # Compute Bayesian ratings and normalize
        bayes_ratings = np.array([
            self._bayes_rating(c.rating, c.rating_count) 
            for c in candidates
        ])
        r_prime = bayes_ratings / 5.0  # Normalize to [0,1]
        
        # Linear blend
        final_scores = lambda_blend * s_prime + (1 - lambda_blend) * r_prime
        
        # Update candidates with ranking scores
        for i, candidate in enumerate(candidates):
            candidate.final_score = float(final_scores[i])
            candidate.semantic_norm = float(s_prime[i])
            candidate.rating_norm = float(r_prime[i])
            candidate.lambda_used = lambda_blend
        
        # Sort by final score (descending), then by rating_count (descending), then by price (ascending)
        candidates.sort(
            key=lambda x: (
                -x.final_score,           # Higher final score first
                -(x.rating_count or 0),   # More reviews first (confidence)
                x.price or float('inf')   # Lower price first (affordability)
            )
        )
        
        # Log ranking statistics
        final_scores_sorted = [c.final_score for c in candidates]
        logger.debug(f"Final scores: min={min(final_scores_sorted):.3f}, max={max(final_scores_sorted):.3f}")
        
        return candidates
    
    def _bayes_rating(self, rating: Optional[float], count: Optional[int]) -> float:
        """Compute Bayesian-adjusted rating with shrinkage toward prior."""
        if rating is None or count is None or count <= 0:
            return self.bayesian_mu  # Default to prior mean
        
        # Bayesian shrinkage formula
        adjusted_rating = (self.bayesian_w * self.bayesian_mu + rating * count) / (self.bayesian_w + count)
        
        # Ensure result is within valid range
        return max(0.0, min(5.0, adjusted_rating))
    
    def _generate_facets(self, candidates: List[SearchCandidate]) -> List[FacetGroup]:
        """Generate simplified facets: price and rating only (direct metadata fields)."""
        if not candidates:
            return []
        
        logger.debug(f"Generating facets from {len(candidates)} candidates")
        facets = []
        
        try:
            # 1. Price range facets
            prices = [c.price for c in candidates if c.price is not None and c.price > 0]
            if prices and len(prices) > 3:
                price_ranges = self._create_price_ranges(prices)
                price_options = [
                    FacetOption(value=range_name, count=count)
                    for range_name, count in price_ranges.items()
                    if count > 0
                ]
                if price_options:
                    facets.append(FacetGroup(name="Price", options=price_options))
            
            # 2. Rating facets
            ratings = [c.rating for c in candidates if c.rating is not None and c.rating > 0]
            if ratings and len(ratings) > 3:
                rating_ranges = self._create_rating_ranges(ratings)
                rating_options = [
                    FacetOption(value=range_name, count=count)
                    for range_name, count in rating_ranges.items()
                    if count > 0
                ]
                if rating_options:
                    facets.append(FacetGroup(name="Rating", options=rating_options))
            
            logger.debug(f"Generated {len(facets)} facet groups")
            return facets
            
        except Exception as e:
            logger.error(f"Error generating facets: {e}")
            return []
    
    def _create_price_ranges(self, prices: List[float]) -> Dict[str, int]:
        """Create simplified price range buckets."""
        ranges = {
            "Under $50": 0,
            "$50-$100": 0,
            "Over $100": 0
        }
        
        for price in prices:
            if price < 50:
                ranges["Under $50"] += 1
            elif price <= 100:
                ranges["$50-$100"] += 1
            else:
                ranges["Over $100"] += 1
        
        return ranges
    
    def _create_rating_ranges(self, ratings: List[float]) -> Dict[str, int]:
        """Create simplified rating range buckets."""
        ranges = {
            "4+ Stars": 0,
            "3+ Stars": 0,
            "Under 3 Stars": 0
        }
        
        for rating in ratings:
            if rating >= 4.0:
                ranges["4+ Stars"] += 1
            elif rating >= 3.0:
                ranges["3+ Stars"] += 1
            else:
                ranges["Under 3 Stars"] += 1
        
        return ranges
    
    def _format_results(self, candidates: List[SearchCandidate]) -> List[ProductResult]:
        """Format search candidates as ProductResult objects."""
        results = []
        
        for candidate in candidates:
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
                        primary_image = img
                        break
            
            # Generate explanation
            explanation = self._get_ranking_explanation(candidate)
            
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
        
        return results
    
    def _get_ranking_explanation(self, candidate: SearchCandidate) -> str:
        """Generate human-readable explanation for ranking score."""
        semantic_pct = candidate.semantic_norm * candidate.lambda_used * 100
        rating_pct = candidate.rating_norm * (1 - candidate.lambda_used) * 100
        
        return (
            f"Final: {candidate.final_score:.3f} "
            f"(Semantic: {semantic_pct:.1f}% + Rating: {rating_pct:.1f}%)"
        )
