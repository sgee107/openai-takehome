"""Unit tests for the ranking service with Bayesian adjustment."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from app.services.ranking import RankingService, RankedCandidate
from app.services.retrieval import Candidate


class TestRankingService:
    """Test the Bayesian rating adjustment and linear blend ranking."""

    @pytest.fixture
    def ranker(self):
        """Create a RankingService instance with default parameters."""
        return RankingService(mu=4.0, w=20.0)

    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates for testing."""
        return [
            Candidate(
                product_id="1",
                title="High Similarity, Low Rating Product",
                price=50.0,
                rating=2.5,
                rating_count=5,
                category="Clothing",
                similarity=0.95,
                metadata={"store": "Brand A"}
            ),
            Candidate(
                product_id="2", 
                title="Medium Similarity, High Rating Product",
                price=75.0,
                rating=4.8,
                rating_count=100,
                category="Clothing",
                similarity=0.80,
                metadata={"store": "Brand B"}
            ),
            Candidate(
                product_id="3",
                title="Low Similarity, Medium Rating Product", 
                price=30.0,
                rating=3.5,
                rating_count=25,
                category="Clothing",
                similarity=0.60,
                metadata={"store": "Brand C"}
            )
        ]

    def test_bayesian_rating_adjustment(self, ranker):
        """Test Bayesian rating adjustment formula."""
        # Test with high rating, low count (should shrink toward prior)
        adjusted = ranker.bayes_rating(5.0, 1)
        assert 4.0 < adjusted < 5.0  # Should be between prior (4.0) and rating (5.0)
        assert adjusted == pytest.approx((20 * 4.0 + 5.0 * 1) / (20 + 1), rel=1e-3)

        # Test with high rating, high count (should stay close to original)
        adjusted = ranker.bayes_rating(4.8, 100)
        assert adjusted > 4.6  # Should be reasonably close to original rating
        assert adjusted == pytest.approx((20 * 4.0 + 4.8 * 100) / (20 + 100), rel=1e-3)

        # Test with no rating data (should return prior)
        assert ranker.bayes_rating(None, None) == 4.0
        assert ranker.bayes_rating(4.5, None) == 4.0
        assert ranker.bayes_rating(None, 10) == 4.0
        assert ranker.bayes_rating(4.5, 0) == 4.0

    def test_empty_candidates_list(self, ranker):
        """Test ranking with empty candidates list."""
        result = ranker.rerank([])
        assert result == []

    def test_single_candidate_ranking(self, ranker):
        """Test ranking with a single candidate."""
        candidates = [
            Candidate("1", "Test Product", 50.0, 4.0, 10, "Category", 0.8, {})
        ]
        
        ranked = ranker.rerank(candidates)
        
        assert len(ranked) == 1
        assert isinstance(ranked[0], RankedCandidate)
        assert ranked[0].product_id == "1"
        assert ranked[0].final_score > 0
        assert ranked[0].semantic_norm == 1.0  # Single candidate gets max normalized score
        assert ranked[0].lambda_used == 0.85

    def test_linear_blend_ranking(self, ranker, sample_candidates):
        """Test the linear blend ranking algorithm."""
        ranked = ranker.rerank(sample_candidates, lambda_blend=0.85)
        
        assert len(ranked) == 3
        assert all(isinstance(r, RankedCandidate) for r in ranked)
        
        # Check that final scores are calculated correctly
        for candidate in ranked:
            expected_final = (
                candidate.lambda_used * candidate.semantic_norm + 
                (1 - candidate.lambda_used) * candidate.rating_norm
            )
            assert candidate.final_score == pytest.approx(expected_final, rel=1e-3)

        # Results should be sorted by final score (descending)
        final_scores = [r.final_score for r in ranked]
        assert final_scores == sorted(final_scores, reverse=True)

    def test_lambda_parameter_effects(self, ranker, sample_candidates):
        """Test that lambda parameter affects ranking as expected."""
        # High lambda (favor semantic similarity)
        high_semantic = ranker.rerank(sample_candidates, lambda_blend=0.95)
        
        # Low lambda (favor ratings)
        high_rating = ranker.rerank(sample_candidates, lambda_blend=0.1)
        
        # The high similarity product should rank higher with high lambda
        high_sim_product_id = "1"  # Product with 0.95 similarity but low rating
        
        high_semantic_pos = next(i for i, r in enumerate(high_semantic) if r.product_id == high_sim_product_id)
        high_rating_pos = next(i for i, r in enumerate(high_rating) if r.product_id == high_sim_product_id)
        
        # High similarity product should rank better with high lambda
        assert high_semantic_pos <= high_rating_pos

    def test_lambda_bounds_validation(self, ranker, sample_candidates):
        """Test that lambda parameter is properly bounded."""
        # Test lambda > 1 (should be clamped to 1)
        ranked_high = ranker.rerank(sample_candidates, lambda_blend=1.5)
        assert all(r.lambda_used == 1.0 for r in ranked_high)
        
        # Test lambda < 0 (should be clamped to 0)
        ranked_low = ranker.rerank(sample_candidates, lambda_blend=-0.5)
        assert all(r.lambda_used == 0.0 for r in ranked_low)

    def test_zero_variance_similarity_handling(self, ranker):
        """Test handling of candidates with identical similarity scores."""
        candidates = [
            Candidate("1", "Product 1", 50.0, 4.0, 10, "Category", 0.8, {}),
            Candidate("2", "Product 2", 60.0, 3.5, 20, "Category", 0.8, {}),
            Candidate("3", "Product 3", 40.0, 4.5, 5, "Category", 0.8, {})
        ]
        
        ranked = ranker.rerank(candidates)
        
        # Should use rank-based scoring when similarities are identical
        semantic_scores = [r.semantic_norm for r in ranked]
        
        # Should have different normalized scores despite identical similarities
        assert len(set(semantic_scores)) > 1
        
        # Should still produce valid rankings
        assert all(0 <= score <= 1 for score in semantic_scores)

    def test_tie_breaking_logic(self, ranker):
        """Test tie-breaking when final scores are very close."""
        candidates = [
            Candidate("1", "Product 1", 100.0, 4.0, 50, "Category", 0.8, {}),  # Higher price
            Candidate("2", "Product 2", 50.0, 4.0, 100, "Category", 0.8, {}),  # More reviews
            Candidate("3", "Product 3", 75.0, 4.0, 50, "Category", 0.8, {})   # Middle ground
        ]
        
        ranked = ranker.rerank(candidates, lambda_blend=0.0)  # Pure rating-based
        
        # With identical ratings and similarities, should prefer:
        # 1. More rating count (more confidence)  
        # 2. Lower price (better value)
        
        # Product 2 should rank first (most reviews)
        assert ranked[0].product_id == "2"

    def test_ranking_explanation_generation(self, ranker, sample_candidates):
        """Test the ranking explanation generation."""
        ranked = ranker.rerank(sample_candidates)
        
        explanation = ranker.get_ranking_explanation(ranked[0])
        
        # Should be a non-empty string
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        
        # Should contain key information
        assert "Final:" in explanation
        assert "Semantic:" in explanation
        assert "Rating:" in explanation

    def test_rating_normalization(self, ranker):
        """Test that ratings are properly normalized to [0,1] scale."""
        candidates = [
            Candidate("1", "Product 1", 50.0, 1.0, 10, "Category", 0.8, {}),  # Min rating
            Candidate("2", "Product 2", 50.0, 5.0, 10, "Category", 0.8, {}),  # Max rating
            Candidate("3", "Product 3", 50.0, 3.0, 10, "Category", 0.8, {})   # Mid rating
        ]
        
        ranked = ranker.rerank(candidates)
        
        # All normalized ratings should be in [0,1]
        for candidate in ranked:
            assert 0 <= candidate.rating_norm <= 1
            
        # Bayesian adjustment should be applied correctly
        for candidate in ranked:
            expected_bayes = ranker.bayes_rating(candidate.rating, candidate.rating_count)
            expected_norm = expected_bayes / 5.0
            assert candidate.rating_norm == pytest.approx(expected_norm, rel=1e-3)