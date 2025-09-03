"""
Comprehensive step-by-step validation tests for the Chat Search pipeline.

This test suite uses DataFrame-driven test cases to validate each stage:
1. Query Classification (AgentDecision)  
2. Search Execution (candidates retrieved)
3. Scoring & Blending (ProductMatch scores)
4. Results Ranking (ProductResult ordering)
5. Facet Generation (FacetGroup options)
6. Response Formatting (ChatSearchResponse)
"""

import pytest
import pandas as pd
import asyncio
from typing import List, Dict, Any

from app.types.search_api import SearchRequest, QueryComplexity, ChatSearchResponse
from app.agents.fashion_agent import FashionAgent
from app.dependencies import get_openai_client
from app.db.database import get_async_session


def load_test_cases() -> pd.DataFrame:
    """Load test cases from CSV file."""
    import os
    csv_path = os.path.join(os.path.dirname(__file__), 'search_test_cases.csv')
    df = pd.read_csv(csv_path)
    
    # Parse list fields from CSV strings
    def parse_list_field(field_str):
        if pd.isna(field_str) or field_str == '""""""':
            return []
        # Remove quotes and split by comma
        return [item.strip().strip('"') for item in field_str.strip('"').split(',') if item.strip()]
    
    # Convert string representations to lists
    df['expected_top_result_title_contains'] = df['expected_top_result_title_contains'].apply(parse_list_field)
    df['expected_facets_include'] = df['expected_facets_include'].apply(parse_list_field)
    
    # Convert boolean strings to booleans
    df['should_have_followups'] = df['should_have_followups'].map({'True': True, 'False': False})
    
    return df


class TestSearchPipelineValidation:
    """DataFrame-driven validation tests for search pipeline stages."""
    
    @pytest.fixture(scope="class")
    def test_cases_df(self):
        """Load test cases from CSV file."""
        return load_test_cases()
    
    @pytest.fixture(scope="class")
    def fashion_agent(self):
        """Initialize FashionAgent for testing."""
        openai_client = get_openai_client()
        return FashionAgent(openai_client)
    
    @pytest.mark.asyncio 
    def test_end_to_end_pipeline_validation(self, test_cases_df, fashion_agent, db_session):
        """Test each test case from the DataFrame with step-by-step validation."""
        for _, test_case in test_cases_df.iterrows():
            await self._run_single_test_case(test_case.to_dict(), fashion_agent, db_session)
    
    async def _run_single_test_case(self, test_case: Dict[str, Any], fashion_agent, db_session):
        """
        Test each stage of the search pipeline with step-by-step validation.
        
        Validates:
        1. Query Classification → AgentDecision
        2. Search Execution → Product candidates 
        3. Scoring & Blending → ProductMatch scores
        4. Results Ranking → ProductResult ordering
        5. Facet Generation → FacetGroup options
        6. Response Formatting → Complete ChatSearchResponse
        """
        print(f"\n=== Testing: {test_case['test_id']} - {test_case['query']} ===")
        # Execute search
        response: ChatSearchResponse = await fashion_agent.search_products(
            query=test_case["query"],
            session=db_session,
            topk=test_case["topk"],
            lambda_blend=test_case["lambda_blend"],
            debug=True
        )
        
        # STAGE 1: Query Classification Validation
        self._validate_agent_decision(response.agent, test_case)
        
        # STAGE 2: Search Execution Validation  
        self._validate_search_execution(response.results, test_case)
        
        # STAGE 3: Scoring & Blending Validation
        self._validate_scoring_blending(response.results, test_case)
        
        # STAGE 4: Results Ranking Validation
        self._validate_results_ranking(response.results, test_case)
        
        # STAGE 5: Facet Generation Validation
        self._validate_facet_generation(response.facets, test_case)
        
        # STAGE 6: Response Structure Validation
        self._validate_response_structure(response, test_case)
    
    def _validate_agent_decision(self, agent_decision, test_case):
        """Validate AgentDecision classification results."""
        assert agent_decision.complexity == test_case["expected_complexity"], \
            f"Expected complexity {test_case['expected_complexity']}, got {agent_decision.complexity}"
        
        if agent_decision.parsed:
            if test_case["expected_parsed_category"]:
                assert agent_decision.parsed.category == test_case["expected_parsed_category"], \
                    f"Expected category {test_case['expected_parsed_category']}, got {agent_decision.parsed.category}"
            
            if test_case["expected_parsed_brand"]:
                assert agent_decision.parsed.brand == test_case["expected_parsed_brand"], \
                    f"Expected brand {test_case['expected_parsed_brand']}, got {agent_decision.parsed.brand}"
            
            if test_case["expected_parsed_price_min"] is not None:
                assert agent_decision.parsed.price_min == test_case["expected_parsed_price_min"], \
                    f"Expected price_min {test_case['expected_parsed_price_min']}, got {agent_decision.parsed.price_min}"
            
            if test_case["expected_parsed_price_max"] is not None:
                assert agent_decision.parsed.price_max == test_case["expected_parsed_price_max"], \
                    f"Expected price_max {test_case['expected_parsed_price_max']}, got {agent_decision.parsed.price_max}"
        
        # Validate reasoning exists for complex queries
        if test_case["expected_complexity"] >= 2:
            assert agent_decision.reason is not None, "Expected reasoning for complex queries"
    
    def _validate_search_execution(self, results, test_case):
        """Validate search execution and candidate retrieval."""
        # Should return some results unless it's an edge case
        if test_case["category"] != "AMBIGUOUS" and test_case["test_id"] != "edge_002":
            assert len(results) > 0, f"Expected results for query: {test_case['query']}"
        
        # Check topK constraint
        assert len(results) <= test_case["topk"], \
            f"Results exceed topK limit: {len(results)} > {test_case['topk']}"
    
    def _validate_scoring_blending(self, results, test_case):
        """Validate ProductMatch scoring and blending logic."""
        if not results:
            return  # Skip if no results
        
        for result in results:
            match = result.match
            
            # Validate score ranges
            assert 0 <= match.semantic <= 1, f"Semantic score out of range: {match.semantic}"
            assert 0 <= match.rating <= 1, f"Rating score out of range: {match.rating}"  
            assert 0 <= match.final <= 1, f"Final score out of range: {match.final}"
            
            # Validate lambda usage
            assert match.lambda_used == test_case["lambda_blend"], \
                f"Lambda mismatch: expected {test_case['lambda_blend']}, got {match.lambda_used}"
            
            # Validate blending formula: final = λ * semantic + (1-λ) * rating
            expected_final = (match.lambda_used * match.semantic + 
                            (1 - match.lambda_used) * match.rating)
            assert abs(match.final - expected_final) < 0.001, \
                f"Blending formula error: expected {expected_final}, got {match.final}"
            
            # Check minimum score thresholds
            if test_case["expected_min_semantic_score"]:
                top_semantic = max(r.match.semantic for r in results[:3])  # Top 3 results
                assert top_semantic >= test_case["expected_min_semantic_score"], \
                    f"Top semantic score {top_semantic} below threshold {test_case['expected_min_semantic_score']}"
            
            if test_case["expected_min_rating_score"]:
                top_rating = max(r.match.rating for r in results[:3])
                assert top_rating >= test_case["expected_min_rating_score"], \
                    f"Top rating score {top_rating} below threshold {test_case['expected_min_rating_score']}"
    
    def _validate_results_ranking(self, results, test_case):
        """Validate ProductResult ranking order."""
        if len(results) < 2:
            return
        
        # Results should be sorted by final score (descending)
        for i in range(len(results) - 1):
            assert results[i].match.final >= results[i + 1].match.final, \
                f"Results not properly ranked: {results[i].match.final} < {results[i + 1].match.final}"
        
        # Check top result contains expected terms
        if test_case["expected_top_result_title_contains"] and results:
            top_title = results[0].title.lower()
            for term in test_case["expected_top_result_title_contains"]:
                assert term.lower() in top_title, \
                    f"Top result title '{top_title}' missing expected term '{term}'"
    
    def _validate_facet_generation(self, facets, test_case):
        """Validate FacetGroup generation.""" 
        if not test_case["expected_facets_include"]:
            return
        
        if facets:
            facet_names = [facet.name for facet in facets]
            for expected_facet in test_case["expected_facets_include"]:
                assert expected_facet in facet_names, \
                    f"Missing expected facet: {expected_facet}. Available: {facet_names}"
            
            # Validate facet structure
            for facet in facets:
                assert len(facet.options) > 0, f"Facet {facet.name} has no options"
                for option in facet.options:
                    assert option.count > 0, f"Facet option {option.value} has zero count"
    
    def _validate_response_structure(self, response, test_case):
        """Validate complete ChatSearchResponse structure."""
        # Check followup expectations
        if test_case["should_have_followups"]:
            assert response.followups and len(response.followups) > 0, \
                f"Expected followup questions for ambiguous query: {test_case['query']}"
        
        # UI hints should be present
        assert response.ui is not None, "Missing UI hints"
        assert response.ui.layout in ["grid", "list"], f"Invalid layout: {response.ui.layout}"
        
        # Debug info should be present when requested
        assert response.debug is not None, "Missing debug information"
        if response.debug.timings:
            assert "total_time" in response.debug.timings or len(response.debug.timings) > 0, \
                "Debug timings should contain performance data"


# Additional utility tests
class TestSearchValidationUtils:
    """Utility tests for validation functions."""
    
    def test_test_cases_dataframe_structure(self):
        """Ensure test cases DataFrame has proper structure."""
        df = pd.DataFrame(TEST_CASES_DATA)
        
        required_columns = [
            "test_id", "category", "query", "expected_complexity",
            "lambda_blend", "topk", "description"
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Check for duplicate test IDs
        assert len(df["test_id"].unique()) == len(df), "Duplicate test IDs found"
        
        # Validate complexity values
        valid_complexity = {1, 2, 3}
        assert df["expected_complexity"].isin(valid_complexity).all(), \
            "Invalid complexity values found"
        
        # Validate lambda_blend range
        assert df["lambda_blend"].between(0, 1).all(), "Lambda blend values out of range"
        
        # Validate topK values
        assert (df["topk"] > 0).all(), "TopK values must be positive"
    
    @pytest.mark.asyncio 
    async def test_search_pipeline_performance(self, db_session):
        """Test that search pipeline completes within reasonable time."""
        import time
        
        fashion_agent = FashionAgent(get_openai_client())
        
        start_time = time.time()
        response = await fashion_agent.search_products(
            query="blue jeans",
            session=db_session,
            topk=50,
            lambda_blend=0.8,
            debug=True
        )
        end_time = time.time()
        
        # Should complete within 10 seconds
        assert (end_time - start_time) < 10, \
            f"Search took too long: {end_time - start_time:.2f}s"
        
        # Should return reasonable number of results
        assert len(response.results) > 0, "No results returned"
        assert len(response.results) <= 50, "Too many results returned"


if __name__ == "__main__":
    # Run with: pytest tests/test_search_pipeline_validation.py -v
    pytest.main([__file__])