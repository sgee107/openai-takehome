"""
OpenAI embedding generation with batching and rate limiting.

Consolidates the EmbeddingGenerator classes from both scripts and experiments
into a single, unified implementation.
"""
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from app.settings import settings


class EmbeddingGenerator:
    """Generate embeddings using OpenAI's embedding model with batching and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key. If None, uses settings.openai_api_key
        """
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimension = settings.openai_embedding_dimension
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 20,
        rate_limit_delay: float = 0.5
    ) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            rate_limit_delay: Delay in seconds between batches to respect rate limits
            
        Returns:
            List of embeddings (same length as input texts)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tasks = [self.generate_embedding(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)
            
            # Rate limiting - add delay between batches
            if (i + batch_size) < len(texts):
                await asyncio.sleep(rate_limit_delay)
        
        return embeddings
    
    async def generate_embeddings_dict(
        self,
        text_dict: Dict[str, str],
        batch_size: int = 20,
        rate_limit_delay: float = 0.5
    ) -> Dict[str, Optional[List[float]]]:
        """Generate embeddings for a dictionary of named texts.
        
        Args:
            text_dict: Dictionary mapping names to texts
            batch_size: Number of texts to process in each batch
            rate_limit_delay: Delay in seconds between batches
            
        Returns:
            Dictionary mapping names to embeddings
        """
        names = list(text_dict.keys())
        texts = list(text_dict.values())
        
        embeddings = await self.generate_embeddings_batch(
            texts, batch_size, rate_limit_delay
        )
        
        return dict(zip(names, embeddings))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model being used.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model': self.model,
            'dimension': self.dimension,
            'max_input_tokens': 8192,  # Standard limit for text-embedding-3-small
        }
