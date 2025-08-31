from typing import List, Optional
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from app.models import Review
from app.database import SessionLocal
from openai import OpenAI
from app.settings import settings


class ReviewSearcher:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            dimensions=1536
        )
        return response.data[0].embedding
    
    def search_similar_reviews(
        self, 
        query: str, 
        limit: int = 10,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None
    ) -> List[Review]:
        """Search for reviews similar to the query using vector similarity"""
        
        # Generate embedding for the query
        query_embedding = self.generate_query_embedding(query)
        
        session = SessionLocal()
        try:
            # Build the base query with vector similarity
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # Create the similarity query
            sql_query = text("""
                SELECT *, 
                       embedding <=> :embedding::vector as distance
                FROM reviews
                WHERE embedding IS NOT NULL
                {}
                ORDER BY embedding <=> :embedding::vector
                LIMIT :limit
            """.format(
                "AND rating >= :min_rating AND rating <= :max_rating" 
                if min_rating is not None and max_rating is not None 
                else ""
            ))
            
            # Set parameters
            params = {"embedding": embedding_str, "limit": limit}
            if min_rating is not None and max_rating is not None:
                params["min_rating"] = min_rating
                params["max_rating"] = max_rating
            
            # Execute query
            result = session.execute(sql_query, params)
            
            # Convert results to Review objects
            reviews = []
            for row in result:
                review = Review(
                    id=row.id,
                    asin=row.asin,
                    user_id=row.user_id,
                    rating=row.rating,
                    title=row.title,
                    text=row.text,
                    parent_asin=row.parent_asin,
                    timestamp=row.timestamp,
                    helpful_vote=row.helpful_vote,
                    verified_purchase=row.verified_purchase,
                    created_at=row.created_at,
                    metadata=row.metadata
                )
                reviews.append(review)
            
            return reviews
            
        finally:
            session.close()
    
    def search_by_keyword(
        self, 
        keyword: str, 
        limit: int = 10
    ) -> List[Review]:
        """Search for reviews containing a keyword"""
        session = SessionLocal()
        try:
            query = session.query(Review).filter(
                (Review.title.ilike(f"%{keyword}%")) | 
                (Review.text.ilike(f"%{keyword}%"))
            ).limit(limit)
            
            return query.all()
        finally:
            session.close()


async def search_reviews_endpoint(
    query: str,
    search_type: str = "semantic",
    limit: int = 10,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None
):
    """FastAPI endpoint for searching reviews"""
    searcher = ReviewSearcher()
    
    if search_type == "semantic":
        reviews = searcher.search_similar_reviews(
            query, 
            limit=limit,
            min_rating=min_rating,
            max_rating=max_rating
        )
    else:
        reviews = searcher.search_by_keyword(query, limit=limit)
    
    return [
        {
            "id": str(review.id),
            "asin": review.asin,
            "rating": review.rating,
            "title": review.title,
            "text": review.text,
            "verified_purchase": review.verified_purchase,
            "helpful_vote": review.helpful_vote
        }
        for review in reviews
    ]