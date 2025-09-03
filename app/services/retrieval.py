"""Retrieval service for semantic similarity search."""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from openai import AsyncOpenAI
from dataclasses import dataclass

from app.db.models import Product, ProductEmbedding
from app.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """Retrieved candidate product with similarity score."""
    product_id: str
    title: str
    price: Optional[float]
    rating: Optional[float]
    rating_count: Optional[int]
    category: str
    similarity: float
    metadata: Dict[str, Any]


class RetrievalService:
    """Service for retrieving top-K candidates using semantic similarity."""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-3-small"
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=query
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for query '{query}': {e}")
            raise
    
    async def topk(
        self, 
        query: str,
        session: AsyncSession,
        k: int = 200,
        category_filter: Optional[str] = None,
        brand_filter: Optional[str] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        embedding_strategy: str = "key_value_with_images"
    ) -> List[Candidate]:
        """
        Retrieve top-K candidates using semantic similarity.
        
        Args:
            query: Search query text
            session: Database session
            k: Number of candidates to retrieve
            category_filter: Optional category filter
            brand_filter: Optional brand filter
            price_min: Optional minimum price filter
            price_max: Optional maximum price filter
            embedding_strategy: Embedding strategy to use (default: key_value_with_images)
            
        Returns:
            List of Candidate objects sorted by similarity descending
        """
        if not query.strip():
            return []
        
        try:
            # Get query embedding
            logger.info(f"Retrieving top-{k} candidates for query: {query}")
            query_embedding = await self.get_query_embedding(query)
            
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
            query_stmt = query_stmt.where(ProductEmbedding.strategy == embedding_strategy)
            logger.debug(f"Using embedding strategy: {embedding_strategy}")
            
            # Apply other filters
            if category_filter:
                # Use ILIKE for case-insensitive partial matching
                query_stmt = query_stmt.where(Product.main_category.ilike(f"%{category_filter}%"))
                logger.debug(f"Applied category filter: {category_filter}")
            
            if brand_filter:
                # Check both store field and details for brand information
                brand_condition = Product.store.ilike(f"%{brand_filter}%")
                query_stmt = query_stmt.where(brand_condition)
                logger.debug(f"Applied brand filter: {brand_filter}")
            
            if price_min is not None:
                query_stmt = query_stmt.where(Product.price >= price_min)
                logger.debug(f"Applied minimum price filter: {price_min}")
            
            if price_max is not None:
                query_stmt = query_stmt.where(Product.price <= price_max)
                logger.debug(f"Applied maximum price filter: {price_max}")
            
            # Order by similarity and limit
            query_stmt = query_stmt.order_by(text('similarity DESC')).limit(k)
            
            # Execute query
            result = await session.execute(query_stmt)
            rows = result.all()
            
            logger.info(f"Retrieved {len(rows)} candidates from database")
            
            # Convert to candidates
            candidates = []
            for row in rows:
                embedding_record, product, similarity = row
                
                # Images will be loaded separately if needed
                # Avoid lazy loading in async context
                images = []
                
                candidates.append(Candidate(
                    product_id=product.id,
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
                        'images': images,
                        'categories': product.categories or [],
                        'description': product.description,
                        'embedding_strategy': embedding_record.strategy
                    }
                ))
            
            # Log similarity score distribution
            if candidates:
                similarities = [c.similarity for c in candidates]
                logger.info(f"Similarity scores: min={min(similarities):.3f}, max={max(similarities):.3f}, avg={sum(similarities)/len(similarities):.3f}")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error in topk retrieval: {e}")
            raise