"""Facets generation and follow-up prompts service."""

import json
import logging
from typing import List, Dict, Optional
from collections import Counter
from openai import AsyncOpenAI

from app.types.search_api import FacetGroup, FacetOption, FollowupPrompt, QueryComplexity
from app.services.ranking import RankedCandidate
from app.prompts.v1.followup_generator import (
    FOLLOWUP_GENERATION_SYSTEM,
    FOLLOWUP_GENERATION_USER_TEMPLATE
)

logger = logging.getLogger(__name__)


class FacetsService:
    """Service for generating facets from search results and follow-up prompts."""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.model = "gpt-4o-mini"
    
    def generate_facets(self, candidates: List[RankedCandidate]) -> List[FacetGroup]:
        """
        Generate facets from top-K results for filtering.
        
        Args:
            candidates: List of ranked candidates
            
        Returns:
            List of facet groups
        """
        if not candidates:
            return []
        
        logger.info(f"Generating facets from {len(candidates)} candidates")
        facets = []
        
        try:
            # Category facets
            category_counts = Counter(c.category for c in candidates if c.category)
            if len(category_counts) > 1:
                category_options = [
                    FacetOption(value=cat, count=count)
                    for cat, count in category_counts.most_common(8)  # Top 8 categories
                ]
                if category_options:
                    facets.append(FacetGroup(name="Category", options=category_options))
            
            # Price range facets
            prices = [c.price for c in candidates if c.price is not None and c.price > 0]
            if prices and len(prices) > 5:  # Only show price facets if we have enough data
                price_ranges = self._create_price_ranges(prices)
                price_options = [
                    FacetOption(value=range_name, count=count)
                    for range_name, count in price_ranges.items()
                    if count > 0
                ]
                if price_options:
                    facets.append(FacetGroup(name="Price Range", options=price_options))
            
            # Brand facets (from store field)
            stores = [
                c.metadata.get('store') 
                for c in candidates 
                if c.metadata.get('store') and c.metadata.get('store').strip()
            ]
            if stores:
                store_counts = Counter(stores)
                if len(store_counts) > 1:
                    store_options = [
                        FacetOption(value=store, count=count)
                        for store, count in store_counts.most_common(10)  # Top 10 brands
                    ]
                    facets.append(FacetGroup(name="Brand", options=store_options))
            
            # Rating facets (group by rating ranges)
            ratings = [c.rating for c in candidates if c.rating is not None and c.rating > 0]
            if ratings and len(ratings) > 10:  # Only show if we have enough rated products
                rating_ranges = self._create_rating_ranges(ratings)
                rating_options = [
                    FacetOption(value=range_name, count=count)
                    for range_name, count in rating_ranges.items()
                    if count > 0
                ]
                if rating_options:
                    facets.append(FacetGroup(name="Customer Rating", options=rating_options))
            
            logger.info(f"Generated {len(facets)} facet groups")
            return facets
            
        except Exception as e:
            logger.error(f"Error generating facets: {e}")
            return []
    
    def _create_price_ranges(self, prices: List[float]) -> Dict[str, int]:
        """Create price range buckets from price list."""
        ranges = {
            "Under $25": 0,
            "$25 - $50": 0, 
            "$50 - $100": 0,
            "$100 - $200": 0,
            "Over $200": 0
        }
        
        for price in prices:
            if price < 25:
                ranges["Under $25"] += 1
            elif price < 50:
                ranges["$25 - $50"] += 1
            elif price < 100:
                ranges["$50 - $100"] += 1
            elif price < 200:
                ranges["$100 - $200"] += 1
            else:
                ranges["Over $200"] += 1
        
        return ranges
    
    def _create_rating_ranges(self, ratings: List[float]) -> Dict[str, int]:
        """Create rating range buckets from ratings list."""
        ranges = {
            "4+ Stars": 0,
            "3+ Stars": 0,
            "2+ Stars": 0,
            "1+ Stars": 0
        }
        
        for rating in ratings:
            if rating >= 4.0:
                ranges["4+ Stars"] += 1
            elif rating >= 3.0:
                ranges["3+ Stars"] += 1
            elif rating >= 2.0:
                ranges["2+ Stars"] += 1
            elif rating >= 1.0:
                ranges["1+ Stars"] += 1
        
        return ranges
    
    async def generate_followups(
        self, 
        complexity: QueryComplexity, 
        query: str, 
        reason: str
    ) -> List[FollowupPrompt]:
        """
        Generate follow-up prompts using LLM for ambiguous queries.
        
        Args:
            complexity: Query complexity level
            query: Original search query
            reason: Classification reason
            
        Returns:
            List of follow-up prompts
        """
        if complexity != QueryComplexity.AMBIGUOUS:
            return []
        
        return await self._llm_generate_followups(query, reason)
    
    async def _llm_generate_followups(self, query: str, reason: str) -> List[FollowupPrompt]:
        """Use LLM to generate contextual follow-up questions."""
        
        user_prompt = FOLLOWUP_GENERATION_USER_TEMPLATE.format(
            query=query,
            reason=reason
        )
        
        try:
            logger.info(f"Generating follow-ups for ambiguous query: {query}")
            
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": FOLLOWUP_GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            followups_data = result.get("followups", [])
            
            followups = [
                FollowupPrompt(
                    text=item["text"],
                    rationale=item.get("rationale", "")
                )
                for item in followups_data
                if item.get("text")  # Ensure text exists
            ]
            
            logger.info(f"Generated {len(followups)} follow-up prompts")
            return followups
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse follow-up generation response: {e}")
            return self._fallback_followups(query)
        except Exception as e:
            logger.error(f"Error in LLM follow-up generation: {e}")
            return self._fallback_followups(query)
    
    def _fallback_followups(self, query: str) -> List[FollowupPrompt]:
        """Fallback follow-ups when LLM generation fails."""
        logger.warning(f"Using fallback follow-ups for query: {query}")
        
        return [
            FollowupPrompt(
                text="Could you be more specific about what you're looking for?",
                rationale="Fallback for ambiguous query"
            ),
            FollowupPrompt(
                text="Are you looking for men's or women's items?",
                rationale="Gender clarification"
            )
        ]