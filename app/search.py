from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from openai import AsyncOpenAI
from app.db.models import Product, ProductEmbedding


async def search_products(query: str, openai_client: AsyncOpenAI, session: AsyncSession, limit: int = 5, strategy: str = "title_only") -> List[Dict[str, Any]]:
    """Basic semantic search for products using specific embedding strategy."""
    
    # Get embedding for query
    response = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Search using cosine similarity with specific strategy
    stmt = (
        select(
            Product,
            ProductEmbedding,
            (1 - ProductEmbedding.embedding.cosine_distance(query_embedding)).label('similarity')
        )
        .join(ProductEmbedding, Product.id == ProductEmbedding.product_id)
        .where(ProductEmbedding.strategy == strategy)
        .order_by(text('similarity DESC'))
        .limit(limit)
    )
    
    result = await session.execute(stmt)
    rows = result.all()
    
    # Return simple product data
    products = []
    for row in rows:
        product, embedding, similarity = row
        products.append({
            'title': product.title,
            'price': product.price,
            'rating': product.average_rating,
            'similarity': float(similarity),
            'strategy': embedding.strategy,
            'embedding_text': embedding.embedding_text[:200] + "..." if len(embedding.embedding_text) > 200 else embedding.embedding_text
        })
    
    return products