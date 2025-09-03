"""Unit tests for the intent classifier service."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from app.services.intent_classifier import IntentClassifier
from app.types.search_api import QueryComplexity, ParsedQuery


class TestIntentClassifier:
    """Test the intent classification and query parsing logic."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client for testing."""
        mock_client = AsyncMock()
        return mock_client

    @pytest.fixture
    def classifier(self, mock_openai_client):
        """Create an IntentClassifier instance with mock client."""
        return IntentClassifier(mock_openai_client)

    @pytest.mark.asyncio
    async def test_classify_direct_query(self, classifier, mock_openai_client):
        """Test classification of direct/specific queries."""
        # Mock LLM response for direct query
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "complexity": 1,
            "reason": "Specific product with exact brand and model",
            "parsed": {
                "terms": ["Nike", "Air", "Max", "270"],
                "category": "shoes",
                "brand": "Nike",
                "colors": None,
                "price_min": None,
                "price_max": None,
                "size": "10",
                "gender": "men",
                "occasion": None
            }
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test direct query
        complexity, parsed = await classifier.classify("Nike Air Max 270 men's size 10")
        
        assert complexity == QueryComplexity.DIRECT
        assert parsed.terms == ["Nike", "Air", "Max", "270"]
        assert parsed.category == "shoes"
        assert parsed.brand == "Nike"
        assert parsed.filters["size"] == "10"
        assert parsed.filters["gender"] == "men"

    @pytest.mark.asyncio
    async def test_classify_filtered_query(self, classifier, mock_openai_client):
        """Test classification of filtered queries."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "complexity": 2,
            "reason": "Category search with color and price constraints",
            "parsed": {
                "terms": ["blue", "jeans", "under", "$50"],
                "category": "jeans",
                "brand": None,
                "colors": ["blue"],
                "price_min": None,
                "price_max": 50.0,
                "size": None,
                "gender": None,
                "occasion": None
            }
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        complexity, parsed = await classifier.classify("blue jeans under $50")
        
        assert complexity == QueryComplexity.FILTERED
        assert parsed.terms == ["blue", "jeans", "under", "$50"]
        assert parsed.category == "jeans"
        assert parsed.colors == ["blue"]
        assert parsed.price_max == 50.0
        assert parsed.price_min is None

    @pytest.mark.asyncio
    async def test_classify_ambiguous_query(self, classifier, mock_openai_client):
        """Test classification of ambiguous/broad queries."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "complexity": 3,
            "reason": "Broad search term without specific constraints",
            "parsed": {
                "terms": ["work", "clothes"],
                "category": None,
                "brand": None,
                "colors": None,
                "price_min": None,
                "price_max": None,
                "size": None,
                "gender": None,
                "occasion": "work"
            }
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        complexity, parsed = await classifier.classify("work clothes")
        
        assert complexity == QueryComplexity.AMBIGUOUS
        assert parsed.terms == ["work", "clothes"]
        assert parsed.category is None
        assert parsed.filters["occasion"] == "work"

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, classifier, mock_openai_client):
        """Test handling of empty or whitespace-only queries."""
        # Should not call OpenAI for empty queries
        complexity, parsed = await classifier.classify("")
        
        assert complexity == QueryComplexity.AMBIGUOUS
        assert parsed.terms == []
        mock_openai_client.chat.completions.create.assert_not_called()
        
        # Test whitespace-only query
        complexity, parsed = await classifier.classify("   \n\t  ")
        
        assert complexity == QueryComplexity.AMBIGUOUS
        assert parsed.terms == []
        mock_openai_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_error_handling(self, classifier, mock_openai_client):
        """Test handling of LLM API errors."""
        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        complexity, parsed = await classifier.classify("test query")
        
        # Should fallback to ambiguous classification
        assert complexity == QueryComplexity.AMBIGUOUS
        assert parsed.terms == ["test", "query"]

    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, classifier, mock_openai_client):
        """Test handling of invalid JSON from LLM."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is not valid JSON"
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        complexity, parsed = await classifier.classify("test query")
        
        # Should fallback to ambiguous classification
        assert complexity == QueryComplexity.AMBIGUOUS
        assert parsed.terms == ["test", "query"]

    @pytest.mark.asyncio
    async def test_invalid_complexity_handling(self, classifier, mock_openai_client):
        """Test handling of invalid complexity values from LLM."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "complexity": 999,  # Invalid complexity
            "reason": "Test",
            "parsed": {"terms": ["test"]}
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        complexity, parsed = await classifier.classify("test query")
        
        # Should fallback to ambiguous classification
        assert complexity == QueryComplexity.AMBIGUOUS
        assert parsed.terms == ["test", "query"]

    @pytest.mark.asyncio
    async def test_data_cleaning(self, classifier, mock_openai_client):
        """Test that extracted data is properly cleaned."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "complexity": 2,
            "reason": "Test",
            "parsed": {
                "terms": ["", "blue", "", "shirt", ""],  # Empty strings should be removed
                "colors": ["", "Blue", "RED", ""],  # Should be cleaned and lowercased
                "category": "shirts",
                "brand": None,
                "price_min": None,
                "price_max": None
            }
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        complexity, parsed = await classifier.classify("blue shirt")
        
        assert complexity == QueryComplexity.FILTERED
        assert parsed.terms == ["blue", "shirt"]  # Empty strings removed
        assert parsed.colors == ["blue", "red"]  # Cleaned and lowercased
        assert parsed.category == "shirts"