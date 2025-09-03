import json
import logging
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.tools.comprehensive_search import ComprehensiveSearchTool
from app.types.search_api import ChatSearchResponse
from app.prompts.agent_prompt import FASHION_AGENT_PROMPT

logger = logging.getLogger(__name__)


class FashionAgent:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.search_tool = ComprehensiveSearchTool(openai_client)
        self.system_prompt = FASHION_AGENT_PROMPT
    
    async def search_products(
        self,
        query: str,
        session: AsyncSession,
        topk: Optional[int] = None,
        lambda_blend: Optional[float] = None,
        debug: bool = False
    ) -> ChatSearchResponse:
        """
        Execute comprehensive product search and return structured response.
        
        Args:
            query: User search query
            session: Database session
            topk: Number of candidates to retrieve
            lambda_blend: Semantic vs rating blend weight
            debug: Whether to include debug information
            
        Returns:
            ChatSearchResponse with structured search results
        """
        logger.info(f"FashionAgent: Starting search for query='{query}', topk={topk}, lambda={lambda_blend}")
        
        try:
            search_result = await self.search_tool.search_products(
                query=query,
                session=session,
                topk=topk,
                lambda_blend=lambda_blend,
                debug=debug
            )
            
            logger.info(f"FashionAgent: Search completed - found {len(search_result.get('results', []))} results")
            logger.debug(f"FashionAgent: Agent decision - complexity={search_result['agent'].complexity}")
            
        except Exception as e:
            logger.error(f"FashionAgent: Error during search - {str(e)}", exc_info=True)
            raise
        
        response = ChatSearchResponse(
            agent=search_result["agent"],
            ui=search_result["ui"],
            results=search_result["results"],
            facets=search_result["facets"],
            followups=search_result["followups"],
            debug=search_result["debug"]
        )
        
        logger.info(f"FashionAgent: Returning response with {len(response.results)} results")
        return response
