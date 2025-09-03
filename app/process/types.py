"""
Type definitions and protocols for the data processing pipeline.
"""
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass


class EmbeddingStrategy(Protocol):
    """Protocol for embedding text generation strategies."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Generate embedding text from product data.
        
        Args:
            product: Product data dictionary
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Generated text for embedding
        """
        ...


@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""
    num_products: Optional[int] = None
    batch_size: int = 20
    strategies: Optional[List[str]] = None
    save_to_db: bool = True
    enable_experiments: bool = False
    experiment_name: Optional[str] = None


@dataclass
class ProcessingResults:
    """Results from pipeline execution."""
    products_loaded: int
    embeddings_created: int
    strategies_processed: List[str]
    duration: float
    experiment_run_id: Optional[str] = None
