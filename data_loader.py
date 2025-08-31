import asyncio
import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
from sqlalchemy.orm import Session
from openai import OpenAI
import time
from datetime import datetime

from app.database import SessionLocal, sync_engine
from app.models import Review, Base
from app.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        batch_size: int = 100,
        max_reviews: Optional[int] = None,
        generate_embeddings: bool = True
    ):
        self.batch_size = batch_size
        self.max_reviews = max_reviews
        self.generate_embeddings = generate_embeddings
        self.openai_client = OpenAI(api_key=settings.openai_api_key) if generate_embeddings else None
        
    def load_dataset(self):
        """Load the Amazon Reviews dataset from Hugging Face"""
        logger.info("Loading Amazon Reviews dataset from Hugging Face...")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            "raw_review_All_Beauty", 
            trust_remote_code=True
        )
        return dataset
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a text using OpenAI API"""
        if not self.generate_embeddings or not text:
            return None
            
        try:
            # Limit text length to avoid token limits
            text = text[:8000] if len(text) > 8000 else text
            
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    def process_review(self, review_data: Dict[str, Any]) -> Review:
        """Process a single review and create a Review object"""
        # Combine title and text for embedding
        review_text = ""
        if review_data.get("title"):
            review_text += review_data["title"] + " "
        if review_data.get("text"):
            review_text += review_data["text"]
        
        # Generate embedding if enabled
        embedding = None
        if self.generate_embeddings and review_text:
            embedding = self.generate_embedding(review_text)
            # Add small delay to respect rate limits
            time.sleep(0.1)
        
        # Create Review object
        review = Review(
            asin=review_data.get("asin"),
            user_id=review_data.get("user_id"),
            rating=float(review_data.get("rating")) if review_data.get("rating") else None,
            title=review_data.get("title"),
            text=review_data.get("text"),
            parent_asin=review_data.get("parent_asin"),
            timestamp=review_data.get("timestamp"),
            helpful_vote=review_data.get("helpful_vote"),
            verified_purchase=review_data.get("verified_purchase"),
            embedding=embedding,
            metadata={
                "images": review_data.get("images"),
                "loaded_at": datetime.utcnow().isoformat()
            }
        )
        
        return review
    
    def load_to_database(self, dataset):
        """Load reviews into the PostgreSQL database"""
        # Create tables if they don't exist
        Base.metadata.create_all(bind=sync_engine)
        
        # Get the dataset split
        reviews_data = dataset["full"]
        total_reviews = len(reviews_data) if not self.max_reviews else min(self.max_reviews, len(reviews_data))
        
        logger.info(f"Loading {total_reviews} reviews into database...")
        
        session = SessionLocal()
        batch = []
        
        try:
            for i in tqdm(range(total_reviews), desc="Processing reviews"):
                review_data = reviews_data[i]
                review = self.process_review(review_data)
                batch.append(review)
                
                # Commit batch when it reaches batch_size
                if len(batch) >= self.batch_size:
                    session.bulk_save_objects(batch)
                    session.commit()
                    logger.info(f"Committed batch of {len(batch)} reviews")
                    batch = []
            
            # Commit remaining reviews
            if batch:
                session.bulk_save_objects(batch)
                session.commit()
                logger.info(f"Committed final batch of {len(batch)} reviews")
                
            logger.info(f"Successfully loaded {total_reviews} reviews into database")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def run(self):
        """Main method to run the data loading process"""
        try:
            # Load dataset
            dataset = self.load_dataset()
            
            # Display dataset info
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Dataset info: {dataset}")
            
            # Show sample review
            if len(dataset["full"]) > 0:
                logger.info(f"Sample review: {dataset['full'][0]}")
            
            # Load to database
            self.load_to_database(dataset)
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise


def main():
    """Main entry point for the data loader script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load Amazon Reviews into PostgreSQL database")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for database inserts")
    parser.add_argument("--max-reviews", type=int, default=None, help="Maximum number of reviews to load")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip generating embeddings")
    
    args = parser.parse_args()
    
    loader = DataLoader(
        batch_size=args.batch_size,
        max_reviews=args.max_reviews,
        generate_embeddings=not args.no_embeddings
    )
    
    loader.run()


if __name__ == "__main__":
    main()