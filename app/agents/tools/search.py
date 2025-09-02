from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from openai import AsyncOpenAI
from app.db.models import Product, ProductEmbedding
from app.settings import settings


class SemanticSearchTool:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-3-small"
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the search query."""
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    async def search_products(
        self, 
        query: str, 
        session: AsyncSession,
        limit: int = 10,
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for products using cosine similarity on embeddings.
        
        Args:
            query: The search query text
            session: Database session
            limit: Number of results to return
            strategy: Optional embedding strategy to filter by
        
        Returns:
            List of products with similarity scores
        """
        # Get query embedding
        query_embedding = await self.get_query_embedding(query)
        
        # Build the cosine similarity query
        # Using pgvector's <=> operator for cosine distance (1 - cosine_similarity)
        # Convert to similarity score by subtracting from 1
        query_stmt = (
            select(
                ProductEmbedding,
                Product,
                (1 - ProductEmbedding.embedding.cosine_distance(query_embedding)).label('similarity')
            )
            .join(Product, ProductEmbedding.product_id == Product.id)
            .order_by(text('similarity DESC'))
            .limit(limit)
        )
        
        # Add strategy filter if provided
        if strategy:
            query_stmt = query_stmt.where(ProductEmbedding.strategy == strategy)
        
        # Execute query
        result = await session.execute(query_stmt)
        rows = result.all()
        
        # Format results
        products = []
        for row in rows:
            embedding_record, product, similarity = row
            products.append({
                'product_id': product.id,
                'parent_asin': product.parent_asin,
                'title': product.title,
                'price': product.price,
                'average_rating': product.average_rating,
                'rating_number': product.rating_number,
                'main_category': product.main_category,
                'categories': product.categories,
                'features': product.features,
                'description': product.description,
                'store': product.store,
                'similarity_score': float(similarity),
                'embedding_strategy': embedding_record.strategy
            })
        
        return products
    
    async def search_with_filters(
        self,
        query: str,
        session: AsyncSession,
        limit: int = 10,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with additional filters on product attributes.
        
        Args:
            query: The search query text
            session: Database session
            limit: Number of results to return
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_rating: Minimum average rating filter
            category: Main category filter
        
        Returns:
            List of filtered products with similarity scores
        """
        # Get query embedding
        query_embedding = await self.get_query_embedding(query)
        
        # Build query with filters
        query_stmt = (
            select(
                ProductEmbedding,
                Product,
                (1 - ProductEmbedding.embedding.cosine_distance(query_embedding)).label('similarity')
            )
            .join(Product, ProductEmbedding.product_id == Product.id)
        )
        
        # Apply filters
        if min_price is not None:
            query_stmt = query_stmt.where(Product.price >= min_price)
        if max_price is not None:
            query_stmt = query_stmt.where(Product.price <= max_price)
        if min_rating is not None:
            query_stmt = query_stmt.where(Product.average_rating >= min_rating)
        if category:
            query_stmt = query_stmt.where(Product.main_category == category)
        
        # Order by similarity and limit
        query_stmt = query_stmt.order_by(text('similarity DESC')).limit(limit)
        
        # Execute query
        result = await session.execute(query_stmt)
        rows = result.all()
        
        # Format results
        products = []
        for row in rows:
            embedding_record, product, similarity = row
            products.append({
                'product_id': product.id,
                'parent_asin': product.parent_asin,
                'title': product.title,
                'price': product.price,
                'average_rating': product.average_rating,
                'rating_number': product.rating_number,
                'main_category': product.main_category,
                'categories': product.categories,
                'features': product.features,
                'description': product.description,
                'store': product.store,
                'similarity_score': float(similarity),
                'embedding_strategy': embedding_record.strategy
            })
        
        return products