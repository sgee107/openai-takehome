import json
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.search import SemanticSearchTool
from app.prompts.agent_prompt import FASHION_AGENT_PROMPT


class FashionAgent:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.search_tool = SemanticSearchTool(openai_client)
        self.system_prompt = FASHION_AGENT_PROMPT
    
    def format_products_for_llm(self, products: List[Dict[str, Any]]) -> str:
        """Format product search results for the LLM to understand."""
        if not products:
            return "No products found matching the query."
        
        formatted = []
        for i, product in enumerate(products[:5], 1):  # Limit to top 5 for context
            formatted.append(f"""
Product {i}:
- Title: {product['title']}
- Price: ${product['price']:.2f} if product['price'] else 'N/A'
- Rating: {product['average_rating']:.1f}/5.0 ({product['rating_number']} reviews) if product['average_rating'] else 'No ratings'
- Category: {product['main_category']}
- Store: {product['store'] or 'N/A'}
- Match Score: {product['similarity_score']:.2%}
- Features: {', '.join(product['features'][:3]) if product['features'] else 'N/A'}
""")
        
        return "\n".join(formatted)
    
    async def process_query(
        self,
        user_query: str,
        session: AsyncSession,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process a user query about fashion products.
        
        Args:
            user_query: The user's question or request
            session: Database session for product search
            conversation_history: Optional previous messages for context
        
        Returns:
            Agent's response as a string
        """
        # Search for relevant products
        search_results = await self.search_tool.search_products(
            query=user_query,
            session=session,
            limit=10
        )
        
        # Format search results
        products_context = self.format_products_for_llm(search_results)
        
        # Build messages for the LLM
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query with search results
        enhanced_query = f"""User Query: {user_query}

Available Products from Search:
{products_context}

Please provide a helpful response based on these search results."""
        
        messages.append({"role": "user", "content": enhanced_query})
        
        # Get response from LLM
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Using a more cost-effective model for the agent
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    async def process_with_filters(
        self,
        user_query: str,
        session: AsyncSession,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        category: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process a user query with additional filters.
        
        Args:
            user_query: The user's question or request
            session: Database session for product search
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_rating: Minimum rating filter
            category: Category filter
            conversation_history: Optional previous messages for context
        
        Returns:
            Agent's response as a string
        """
        # Search with filters
        search_results = await self.search_tool.search_with_filters(
            query=user_query,
            session=session,
            limit=10,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            category=category
        )
        
        # Format search results
        products_context = self.format_products_for_llm(search_results)
        
        # Build filter context
        filter_context = []
        if min_price is not None:
            filter_context.append(f"minimum price: ${min_price:.2f}")
        if max_price is not None:
            filter_context.append(f"maximum price: ${max_price:.2f}")
        if min_rating is not None:
            filter_context.append(f"minimum rating: {min_rating}/5.0")
        if category:
            filter_context.append(f"category: {category}")
        
        filter_str = f" (Filters applied: {', '.join(filter_context)})" if filter_context else ""
        
        # Build messages for the LLM
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current query with search results
        enhanced_query = f"""User Query: {user_query}{filter_str}

Available Products from Search:
{products_context}

Please provide a helpful response based on these filtered search results."""
        
        messages.append({"role": "user", "content": enhanced_query})
        
        # Get response from LLM
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content