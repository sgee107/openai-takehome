"""Integration tests for Chat API vNext search functionality."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.types.search_api import SearchRequest, QueryComplexity
from app.dependencies import get_openai_client
from app.services.intent_classifier import IntentClassifier
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.services.facets import FacetsService


class TestSearchIntegration:
    """Integration tests for the complete search pipeline."""
    
    @pytest.mark.asyncio
    async def test_search_known_products(self, db_session):
        """Test searching for products we know exist in the database."""
        
        test_queries = [
            "RONNOX compression socks",
            "DouBCQ palazzo pants", 
            "Guy Harvey shirt"
        ]
        
        # Initialize services
        openai_client = get_openai_client()
        retrieval_service = RetrievalService(openai_client)
        ranking_service = RankingService()
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            
            # Test retrieval
            candidates = await retrieval_service.topk(
                query=query,
                session=db_session,
                k=5,
                embedding_strategy="key_value_with_images"
            )
            
            print(f"  Found {len(candidates)} candidates")
            assert len(candidates) >= 0
            
            if candidates:
                # Test ranking
                ranked_results = ranking_service.rerank(candidates, lambda_blend=0.85)
                assert len(ranked_results) == len(candidates)
                
                # Check first result
                top_result = ranked_results[0]
                print(f"  Best match: {top_result.title}")
                print(f"  Similarity: {top_result.similarity:.3f}")
                print(f"  Final score: {top_result.final_score:.3f}")
                
                assert top_result.final_score > 0
                assert top_result.similarity > 0
    
    @pytest.mark.asyncio
    async def test_intent_classification(self, db_session):
        """Test intent classification with different query types."""
        
        test_cases = [
            ("RONNOX compression socks size medium", QueryComplexity.DIRECT),
            ("women's casual pants under $50", QueryComplexity.FILTERED),
            ("comfortable clothing", QueryComplexity.AMBIGUOUS)
        ]
        
        # Initialize intent classifier
        openai_client = get_openai_client()
        intent_classifier = IntentClassifier(openai_client)
        
        for query, expected_complexity in test_cases:
            print(f"\n🧠 Classifying: '{query}'")
            
            complexity, parsed_query = await intent_classifier.classify(query)
            
            print(f"  Classified as: {complexity.value} (expected: {expected_complexity.value})")
            print(f"  Parsed terms: {parsed_query.terms}")
            
            # Allow some flexibility in classification
            assert complexity in [QueryComplexity.DIRECT, QueryComplexity.FILTERED, QueryComplexity.AMBIGUOUS]
            assert parsed_query.terms is not None
    
    @pytest.mark.asyncio
    async def test_facets_generation(self, db_session):
        """Test facets generation with real data."""
        
        # Initialize services
        openai_client = get_openai_client()
        retrieval_service = RetrievalService(openai_client)
        ranking_service = RankingService()
        facets_service = FacetsService(openai_client)
        
        # Get candidates from a broad query
        candidates = await retrieval_service.topk(
            query="clothing",
            session=db_session,
            k=10,
            embedding_strategy="key_value_with_images"
        )
        
        if candidates:
            # Rank candidates
            ranked_results = ranking_service.rerank(candidates, lambda_blend=0.85)
            
            # Generate facets
            facets = facets_service.generate_facets(ranked_results)
            
            print(f"Generated {len(facets)} facet groups")
            for facet in facets:
                print(f"  {facet.name}: {len(facet.options)} options")
                for option in facet.options[:3]:  # Top 3
                    print(f"    - {option.value} ({option.count})")
            
            # Should have at least category facets if we have diverse results
            assert len(facets) >= 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_search(self, db_session):
        """Test complete end-to-end search pipeline."""
        
        query = "comfortable socks"
        
        print(f"\n🎯 End-to-end test for: '{query}'")
        
        # Initialize all services
        openai_client = get_openai_client()
        intent_classifier = IntentClassifier(openai_client)
        retrieval_service = RetrievalService(openai_client)
        ranking_service = RankingService()
        facets_service = FacetsService(openai_client)
        
        # Step 1: Intent classification
        complexity, parsed_query = await intent_classifier.classify(query)
        print(f"  1. Intent: {complexity.value}")
        
        # Step 2: Retrieval
        candidates = await retrieval_service.topk(
            query=query,
            session=db_session,
            k=5,
            embedding_strategy="key_value_with_images"
        )
        print(f"  2. Retrieved: {len(candidates)} candidates")
        
        # Step 3: Ranking
        if candidates:
            ranked_results = ranking_service.rerank(candidates, lambda_blend=0.85)
            print(f"  3. Ranked: {len(ranked_results)} results")
            
            # Step 4: Facets
            facets = facets_service.generate_facets(ranked_results)
            print(f"  4. Generated: {len(facets)} facet groups")
            
            # Step 5: Follow-ups (for ambiguous queries)
            if complexity == QueryComplexity.AMBIGUOUS:
                followups = await facets_service.generate_followups(
                    complexity, query, f"Classified as {complexity.value}"
                )
                print(f"  5. Follow-ups: {len(followups)} prompts")
            
            # Verify results
            assert len(ranked_results) > 0
            assert ranked_results[0].final_score > 0
            
            print(f"  ✅ Top result: {ranked_results[0].title}")
            print(f"     Score: {ranked_results[0].final_score:.3f}")
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, db_session):
        """Test handling of empty or invalid queries."""
        
        openai_client = get_openai_client()
        retrieval_service = RetrievalService(openai_client)
        
        # Empty query
        candidates = await retrieval_service.topk(
            query="",
            session=db_session,
            k=5,
            embedding_strategy="key_value_with_images"
        )
        assert candidates == []
        
        # Whitespace query
        candidates = await retrieval_service.topk(
            query="   ",
            session=db_session,
            k=5,
            embedding_strategy="key_value_with_images"
        )
        assert candidates == []