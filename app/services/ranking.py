"""Ranking service with Bayesian rating adjustment and linear blending."""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, replace

from app.services.retrieval import Candidate

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate(Candidate):
    """Extended candidate with ranking scores."""
    final_score: float
    semantic_norm: float
    rating_norm: float
    lambda_used: float


class RankingService:
    """Service for re-ranking candidates using linear blend of semantic + rating scores."""
    
    def __init__(self, mu: float = 4.0, w: float = 20.0):
        """
        Initialize ranking service with Bayesian rating parameters.
        
        Args:
            mu: Prior rating mean (default 4.0 for 5-star scale)
            w: Prior weight for Bayesian shrinkage (default 20.0)
        """
        self.mu = mu  # Prior rating mean
        self.w = w    # Prior weight for Bayesian shrinkage
    
    def bayes_rating(self, rating: Optional[float], count: Optional[int]) -> float:
        """
        Compute Bayesian-adjusted rating with shrinkage toward prior.
        
        Formula: (w * μ + rating * count) / (w + count)
        
        Args:
            rating: Average rating (0-5 scale)
            count: Number of ratings
            
        Returns:
            Bayesian-adjusted rating
        """
        if rating is None or count is None or count <= 0:
            return self.mu  # Default to prior mean
        
        # Bayesian shrinkage formula
        adjusted_rating = (self.w * self.mu + rating * count) / (self.w + count)
        
        # Ensure result is within valid range
        return max(0.0, min(5.0, adjusted_rating))
    
    def rerank(
        self, 
        candidates: List[Candidate], 
        lambda_blend: float = 0.85
    ) -> List[RankedCandidate]:
        """
        Re-rank candidates using linear blend of semantic + rating scores.
        
        Formula: final_score = λ * s' + (1-λ) * r'
        Where s' and r' are normalized per-query scores
        
        Args:
            candidates: List of retrieved candidates
            lambda_blend: Semantic weight (0-1), higher values favor semantic similarity
            
        Returns:
            List of RankedCandidate objects sorted by final_score descending
        """
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
            self.bayes_rating(c.rating, c.rating_count) 
            for c in candidates
        ])
        r_prime = bayes_ratings / 5.0  # Normalize to [0,1]
        
        # Linear blend
        final_scores = lambda_blend * s_prime + (1 - lambda_blend) * r_prime
        
        # Create ranked candidates
        ranked = []
        for i, candidate in enumerate(candidates):
            ranked.append(RankedCandidate(
                # Copy all fields from original candidate
                product_id=candidate.product_id,
                title=candidate.title,
                price=candidate.price,
                rating=candidate.rating,
                rating_count=candidate.rating_count,
                category=candidate.category,
                similarity=candidate.similarity,
                metadata=candidate.metadata,
                # Add ranking scores
                final_score=float(final_scores[i]),
                semantic_norm=float(s_prime[i]),
                rating_norm=float(r_prime[i]),
                lambda_used=lambda_blend
            ))
        
        # Sort by final score (descending), then by rating_count (descending), then by price (ascending)
        # This provides stable ordering and breaks ties logically
        ranked.sort(
            key=lambda x: (
                -x.final_score,           # Higher final score first
                -(x.rating_count or 0),   # More reviews first (confidence)
                x.price or float('inf')   # Lower price first (affordability)
            )
        )
        
        # Log ranking statistics
        final_scores_sorted = [r.final_score for r in ranked]
        logger.info(f"Final scores: min={min(final_scores_sorted):.3f}, max={max(final_scores_sorted):.3f}")
        logger.debug(f"Top 3 products: {[(r.title[:50], r.final_score) for r in ranked[:3]]}")
        
        return ranked
    
    def get_ranking_explanation(self, candidate: RankedCandidate) -> str:
        """
        Generate human-readable explanation for ranking score.
        
        Args:
            candidate: Ranked candidate
            
        Returns:
            Explanation string
        """
        semantic_pct = candidate.semantic_norm * candidate.lambda_used * 100
        rating_pct = candidate.rating_norm * (1 - candidate.lambda_used) * 100
        
        return (
            f"Final: {candidate.final_score:.3f} "
            f"(Semantic: {semantic_pct:.1f}% + Rating: {rating_pct:.1f}%)"
        )