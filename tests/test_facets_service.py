"""Unit tests for the facets service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from app.services.facets import FacetsService
from app.services.ranking import RankedCandidate
from app.types.search_api import QueryComplexity, FacetGroup, FollowupPrompt


class TestFacetsService:
    """Test the facets generation and follow-up prompts."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client for testing."""
        mock_client = AsyncMock()
        return mock_client

    @pytest.fixture
    def facets_service(self, mock_openai_client):
        """Create a FacetsService instance with mock client."""
        return FacetsService(mock_openai_client)

    @pytest.fixture
    def sample_ranked_candidates(self):
        """Create sample ranked candidates for facets testing."""
        return [
            RankedCandidate(
                product_id="1", title="Blue Shirt", price=25.0, rating=4.0, rating_count=10,
                category="Shirts", similarity=0.9, metadata={"store": "Brand A"},
                final_score=0.85, semantic_norm=0.9, rating_norm=0.8, lambda_used=0.85
            ),
            RankedCandidate(
                product_id="2", title="Red Shirt", price=75.0, rating=4.5, rating_count=50, 
                category="Shirts", similarity=0.8, metadata={"store": "Brand B"},
                final_score=0.80, semantic_norm=0.8, rating_norm=0.9, lambda_used=0.85
            ),
            RankedCandidate(
                product_id="3", title="Blue Jeans", price=150.0, rating=3.8, rating_count=25,
                category="Jeans", similarity=0.7, metadata={"store": "Brand A"},
                final_score=0.75, semantic_norm=0.7, rating_norm=0.76, lambda_used=0.85
            ),
            RankedCandidate(
                product_id="4", title="Black Jacket", price=200.0, rating=4.8, rating_count=100,
                category="Jackets", similarity=0.6, metadata={"store": "Brand C"},
                final_score=0.70, semantic_norm=0.6, rating_norm=0.96, lambda_used=0.85
            )
        ]

    def test_generate_facets_empty_candidates(self, facets_service):
        """Test facets generation with empty candidates list."""
        facets = facets_service.generate_facets([])
        assert facets == []

    def test_generate_category_facets(self, facets_service, sample_ranked_candidates):
        """Test category facets generation."""
        facets = facets_service.generate_facets(sample_ranked_candidates)
        
        # Find category facet group
        category_facet = next((f for f in facets if f.name == "Category"), None)
        assert category_facet is not None
        
        # Should have options for each category
        category_values = [opt.value for opt in category_facet.options]
        assert "Shirts" in category_values
        assert "Jeans" in category_values  
        assert "Jackets" in category_values
        
        # Should have correct counts
        shirts_option = next(opt for opt in category_facet.options if opt.value == "Shirts")
        assert shirts_option.count == 2  # Two shirt products

    def test_generate_price_facets(self, facets_service):
        """Test price range facets generation with sufficient data."""
        # Create 8 products with prices to meet the >5 threshold
        candidates = [
            RankedCandidate(
                product_id=str(i), title=f"Product {i}", 
                price=20.0 + i * 30.0,  # Prices: 20, 50, 80, 110, 140, 170, 200, 230
                rating=4.0, rating_count=10, category="Category", similarity=0.8, 
                metadata={"store": "Brand"}, final_score=0.8, semantic_norm=0.8, 
                rating_norm=0.8, lambda_used=0.85
            )
            for i in range(8)
        ]
        
        facets = facets_service.generate_facets(candidates)
        
        # Find price facet group
        price_facet = next((f for f in facets if f.name == "Price Range"), None)
        assert price_facet is not None
        
        # Should have price ranges based on sample data
        price_ranges = [opt.value for opt in price_facet.options]
        assert any("$25" in range_name or "Under" in range_name for range_name in price_ranges)
        assert any("$50" in range_name for range_name in price_ranges)

    def test_generate_brand_facets(self, facets_service, sample_ranked_candidates):
        """Test brand facets generation."""
        facets = facets_service.generate_facets(sample_ranked_candidates)
        
        # Find brand facet group
        brand_facet = next((f for f in facets if f.name == "Brand"), None)
        assert brand_facet is not None
        
        # Should have options for each brand
        brand_values = [opt.value for opt in brand_facet.options]
        assert "Brand A" in brand_values
        assert "Brand B" in brand_values
        assert "Brand C" in brand_values
        
        # Brand A should have count of 2
        brand_a_option = next(opt for opt in brand_facet.options if opt.value == "Brand A")
        assert brand_a_option.count == 2

    def test_generate_rating_facets(self, facets_service):
        """Test rating facets generation with sufficient rated products."""
        # Create more products with ratings to trigger rating facets
        candidates = []
        for i in range(12):  # Need >10 for rating facets
            candidates.append(RankedCandidate(
                product_id=str(i), title=f"Product {i}", price=50.0, 
                rating=4.0 + (i % 5) * 0.2, rating_count=10,  # Ratings from 4.0 to 4.8
                category="Category", similarity=0.8, metadata={"store": "Brand"},
                final_score=0.8, semantic_norm=0.8, rating_norm=0.8, lambda_used=0.85
            ))
        
        facets = facets_service.generate_facets(candidates)
        
        # Should have rating facets
        rating_facet = next((f for f in facets if f.name == "Customer Rating"), None)
        assert rating_facet is not None
        
        # Should have "4+ Stars" option since all products have 4+ ratings
        rating_values = [opt.value for opt in rating_facet.options]
        assert "4+ Stars" in rating_values

    def test_price_range_bucketing(self, facets_service):
        """Test price range bucketing logic."""
        prices = [10.0, 30.0, 60.0, 120.0, 300.0]
        ranges = facets_service._create_price_ranges(prices)
        
        assert ranges["Under $25"] == 1    # $10
        assert ranges["$25 - $50"] == 1    # $30
        assert ranges["$50 - $100"] == 1   # $60
        assert ranges["$100 - $200"] == 1  # $120
        assert ranges["Over $200"] == 1    # $300

    def test_rating_range_bucketing(self, facets_service):
        """Test rating range bucketing logic."""
        ratings = [1.5, 2.5, 3.5, 4.2, 4.8]
        ranges = facets_service._create_rating_ranges(ratings)
        
        assert ranges["1+ Stars"] == 1  # 1.5
        assert ranges["2+ Stars"] == 1  # 2.5  
        assert ranges["3+ Stars"] == 1  # 3.5
        assert ranges["4+ Stars"] == 2  # 4.2, 4.8

    @pytest.mark.asyncio
    async def test_generate_followups_direct_query(self, facets_service):
        """Test that direct queries don't generate follow-ups."""
        followups = await facets_service.generate_followups(
            QueryComplexity.DIRECT, 
            "Nike Air Max 270", 
            "Specific product query"
        )
        assert followups == []

    @pytest.mark.asyncio
    async def test_generate_followups_filtered_query(self, facets_service):
        """Test that filtered queries don't generate follow-ups."""
        followups = await facets_service.generate_followups(
            QueryComplexity.FILTERED,
            "blue jeans under $50",
            "Category search with constraints"
        )
        assert followups == []

    @pytest.mark.asyncio 
    async def test_generate_followups_ambiguous_query(self, facets_service, mock_openai_client):
        """Test follow-up generation for ambiguous queries."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "followups": [
                {
                    "text": "Are you looking for men's or women's items?",
                    "rationale": "Query lacks gender specification"
                },
                {
                    "text": "What's your budget range?", 
                    "rationale": "No price preferences indicated"
                }
            ]
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        followups = await facets_service.generate_followups(
            QueryComplexity.AMBIGUOUS,
            "work clothes",
            "Broad search term"
        )
        
        assert len(followups) == 2
        assert followups[0].text == "Are you looking for men's or women's items?"
        assert followups[0].rationale == "Query lacks gender specification"
        assert followups[1].text == "What's your budget range?"
        assert followups[1].rationale == "No price preferences indicated"

    @pytest.mark.asyncio
    async def test_followup_generation_error_handling(self, facets_service, mock_openai_client):
        """Test error handling in follow-up generation."""
        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        followups = await facets_service.generate_followups(
            QueryComplexity.AMBIGUOUS,
            "test query",
            "Test reason"
        )
        
        # Should return fallback follow-ups
        assert len(followups) >= 1
        assert any("more specific" in f.text.lower() for f in followups)

    @pytest.mark.asyncio
    async def test_followup_invalid_json_handling(self, facets_service, mock_openai_client):
        """Test handling of invalid JSON in follow-up generation."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Invalid JSON"
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        followups = await facets_service.generate_followups(
            QueryComplexity.AMBIGUOUS,
            "test query", 
            "Test reason"
        )
        
        # Should return fallback follow-ups
        assert len(followups) >= 1

    def test_facets_skip_insufficient_data(self, facets_service):
        """Test that facets are skipped when there's insufficient data."""
        # Single product - should not generate multi-option facets
        single_candidate = [RankedCandidate(
            product_id="1", title="Product", price=50.0, rating=4.0, rating_count=10,
            category="Category", similarity=0.8, metadata={"store": "Brand"},
            final_score=0.8, semantic_norm=0.8, rating_norm=0.8, lambda_used=0.85
        )]
        
        facets = facets_service.generate_facets(single_candidate)
        
        # Should have minimal or no facets since there's only one option for each
        category_facets = [f for f in facets if f.name == "Category"]
        if category_facets:
            # If category facet exists, it should have only one option
            assert len(category_facets[0].options) == 1

    def test_facets_without_prices(self, facets_service):
        """Test facets generation when products don't have prices."""
        candidates = [
            RankedCandidate(
                product_id="1", title="Product 1", price=None, rating=4.0, rating_count=10,
                category="Category A", similarity=0.8, metadata={"store": "Brand A"},
                final_score=0.8, semantic_norm=0.8, rating_norm=0.8, lambda_used=0.85
            ),
            RankedCandidate(
                product_id="2", title="Product 2", price=None, rating=3.5, rating_count=20,
                category="Category B", similarity=0.7, metadata={"store": "Brand B"},
                final_score=0.7, semantic_norm=0.7, rating_norm=0.7, lambda_used=0.85
            )
        ]
        
        facets = facets_service.generate_facets(candidates)
        
        # Should not have price facets
        price_facets = [f for f in facets if f.name == "Price Range"]
        assert len(price_facets) == 0
        
        # Should still have category and brand facets
        category_facets = [f for f in facets if f.name == "Category"]
        brand_facets = [f for f in facets if f.name == "Brand"]
        assert len(category_facets) == 1
        assert len(brand_facets) == 1