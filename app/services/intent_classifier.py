"""Intent classification service for complexity scoring and query parsing."""

import json
import logging
from typing import Tuple, Dict, Any
from openai import AsyncOpenAI
from app.types.search_api import QueryComplexity, ParsedQuery
from app.prompts.v1.intent_classifier import (
    INTENT_CLASSIFICATION_SYSTEM,
    INTENT_CLASSIFICATION_USER_TEMPLATE
)

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Service for classifying search query complexity and parsing structured elements."""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.model = "gpt-4o-mini"
    
    async def classify(self, query: str) -> Tuple[QueryComplexity, ParsedQuery]:
        """
        Classify query complexity and parse structured elements using OpenAI.
        
        Args:
            query: User search query
            
        Returns:
            Tuple of (complexity, parsed_query)
        """
        if not query or not query.strip():
            return QueryComplexity.AMBIGUOUS, ParsedQuery(terms=[])
        
        return await self._llm_classify_and_parse(query.strip())
    
    async def _llm_classify_and_parse(self, query: str) -> Tuple[QueryComplexity, ParsedQuery]:
        """Use LLM for both classification and structured parsing."""
        
        user_prompt = INTENT_CLASSIFICATION_USER_TEMPLATE.format(query=query)
        
        try:
            logger.info(f"Classifying query: {query}")
            
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.debug(f"LLM classification result: {result}")
            
            # Extract complexity
            complexity = QueryComplexity(result["complexity"])
            
            # Build ParsedQuery from extracted data
            parsed_data = result.get("parsed", {})
            
            # Clean up and validate extracted data
            terms = parsed_data.get("terms")
            if terms and isinstance(terms, list):
                # Remove empty strings and duplicates
                terms = list(dict.fromkeys([t.strip() for t in terms if t and t.strip()]))
            
            colors = parsed_data.get("colors")
            if colors and isinstance(colors, list):
                colors = [c.strip().lower() for c in colors if c and c.strip()]
            
            # Build filters dictionary
            filters = {}
            if parsed_data.get("size"):
                filters["size"] = parsed_data["size"]
            if parsed_data.get("gender"):
                filters["gender"] = parsed_data["gender"]
            if parsed_data.get("occasion"):
                filters["occasion"] = parsed_data["occasion"]
            
            parsed = ParsedQuery(
                terms=terms if terms else None,
                category=parsed_data.get("category"),
                brand=parsed_data.get("brand"),
                colors=colors if colors else None,
                price_min=parsed_data.get("price_min"),
                price_max=parsed_data.get("price_max"),
                filters=filters if filters else None
            )
            
            logger.info(f"Query classified as complexity {complexity.value}: {result.get('reason', '')}")
            return complexity, parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_classification(query)
        except ValueError as e:
            logger.error(f"Invalid complexity value in LLM response: {e}")
            return self._fallback_classification(query)
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> Tuple[QueryComplexity, ParsedQuery]:
        """Fallback classification when LLM fails."""
        logger.warning(f"Using fallback classification for query: {query}")
        
        # Simple fallback: split query into terms
        terms = [term.strip() for term in query.split() if term.strip()]
        
        return QueryComplexity.AMBIGUOUS, ParsedQuery(terms=terms)