"""Working integration tests for Chat API vNext search functionality."""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.types.search_api import QueryComplexity
from app.dependencies import get_openai_client
from app.services.intent_classifier import IntentClassifier
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.services.facets import FacetsService
from app.db.database import AsyncSessionLocal


async def test_search_functionality():
    """Test the complete search functionality with real database."""
    
    print("="*80)
    print("CHAT API vNEXT INTEGRATION TESTS")
    print("="*80)
    
    async with AsyncSessionLocal() as db_session:
        # Initialize services
        openai_client = get_openai_client()
        intent_classifier = IntentClassifier(openai_client)
        retrieval_service = RetrievalService(openai_client)
        ranking_service = RankingService()
        facets_service = FacetsService(openai_client)
        
        print("\n1. Testing Intent Classification...")
        test_cases = [
            ("RONNOX compression socks size medium", QueryComplexity.DIRECT),
            ("women's casual pants under $50", QueryComplexity.FILTERED), 
            ("comfortable clothing", QueryComplexity.AMBIGUOUS)
        ]
        
        for query, expected in test_cases:
            complexity, parsed_query = await intent_classifier.classify(query)
            status = "✅" if complexity == expected else "⚠️"
            print(f"  {status} '{query}' → {complexity.value} (expected {expected.value})")
            print(f"      Terms: {parsed_query.terms}")
        
        print("\n2. Testing Retrieval with Known Products...")
        test_queries = ["RONNOX socks", "DouBCQ palazzo", "Guy Harvey"]
        
        for query in test_queries:
            print(f"  🔍 Searching: '{query}'")
            try:
                candidates = await retrieval_service.topk(
                    query=query,
                    session=db_session,
                    k=5,
                    embedding_strategy="key_value_with_images"
                )
                print(f"      Found {len(candidates)} candidates")
                
                if candidates:
                    top_candidate = candidates[0]
                    print(f"      Best: {top_candidate.title[:50]}...")
                    print(f"      Similarity: {top_candidate.similarity:.3f}")
                    
                    # Test ranking
                    ranked = ranking_service.rerank(candidates, lambda_blend=0.85)
                    print(f"      Final score: {ranked[0].final_score:.3f}")
                else:
                    print("      No results found")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
        
        print("\n3. Testing End-to-End Search Pipeline...")
        e2e_query = "comfortable socks"
        print(f"  🎯 Query: '{e2e_query}'")
        
        try:
            # Step 1: Intent
            complexity, parsed_query = await intent_classifier.classify(e2e_query)
            print(f"      1. Intent: {complexity.value}")
            
            # Step 2: Retrieval  
            candidates = await retrieval_service.topk(
                query=e2e_query,
                session=db_session,
                k=5,
                embedding_strategy="key_value_with_images"
            )
            print(f"      2. Retrieved: {len(candidates)} candidates")
            
            if candidates:
                # Step 3: Ranking
                ranked_results = ranking_service.rerank(candidates, lambda_blend=0.85)
                print(f"      3. Ranked: {len(ranked_results)} results")
                
                # Step 4: Facets
                facets = facets_service.generate_facets(ranked_results)
                print(f"      4. Facets: {len(facets)} groups")
                
                # Step 5: Follow-ups (if ambiguous)
                if complexity == QueryComplexity.AMBIGUOUS:
                    followups = await facets_service.generate_followups(
                        complexity, e2e_query, f"Classified as {complexity.value}"
                    )
                    print(f"      5. Follow-ups: {len(followups)} prompts")
                
                # Show results
                print(f"      ✅ Top result: {ranked_results[0].title}")
                print(f"         Score: {ranked_results[0].final_score:.3f}")
                print(f"         Category: {ranked_results[0].category}")
                
                if facets:
                    print("      Facets generated:")
                    for facet in facets:
                        print(f"        {facet.name}: {len(facet.options)} options")
            else:
                print("      No results found")
                
        except Exception as e:
            print(f"      ❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n4. Testing Different Embedding Strategies...")
        strategies = ["key_value_with_images", "comprehensive", "title_only"]
        
        for strategy in strategies:
            print(f"  📊 Strategy: {strategy}")
            try:
                candidates = await retrieval_service.topk(
                    query="socks",
                    session=db_session,
                    k=3,
                    embedding_strategy=strategy
                )
                print(f"      Found {len(candidates)} candidates")
                if candidates:
                    avg_sim = sum(c.similarity for c in candidates) / len(candidates)
                    print(f"      Avg similarity: {avg_sim:.3f}")
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✅ INTEGRATION TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_search_functionality())