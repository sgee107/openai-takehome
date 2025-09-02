"""Chat API router using non-streaming completions (no SSE)."""

from fastapi import APIRouter, HTTPException, Depends
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional, List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from app.settings import settings
from app.dependencies import get_openai_client
from app.db.database import get_async_session
from app.agents.fashion_agent import FashionAgent


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None
    use_agent: bool = True  # Use fashion agent by default
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    category: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: Optional[int] = None


@router.post("")
async def chat(
    request: ChatRequest, 
    client: AsyncOpenAI = Depends(get_openai_client),
    session: AsyncSession = Depends(get_async_session)
):
    """Chat endpoint that returns a full, non-streaming completion."""
    try:
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4()}"
        
        # Use fashion agent if requested
        if request.use_agent:
            agent = FashionAgent(client)
            
            # Use filtered search if filters are provided
            if any([request.min_price, request.max_price, request.min_rating, request.category]):
                response_text = await agent.process_with_filters(
                    user_query=request.message,
                    session=session,
                    min_price=request.min_price,
                    max_price=request.max_price,
                    min_rating=request.min_rating,
                    category=request.category,
                    conversation_history=request.conversation_history
                )
            else:
                response_text = await agent.process_query(
                    user_query=request.message,
                    session=session,
                    conversation_history=request.conversation_history
                )
            
            # For agent responses, we can't easily calculate tokens
            # You could implement token counting if needed
            return ChatResponse(
                response=response_text,
                conversation_id=conversation_id,
                tokens_used=None
            )
        
        # Fall back to direct OpenAI chat if agent is disabled
        else:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.message})

            response = await client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                max_completion_tokens=1000,
            )

            return ChatResponse(
                response=response.choices[0].message.content,
                conversation_id=conversation_id,
                tokens_used=(response.usage.total_tokens if getattr(response, "usage", None) else None),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@router.post("/agent")
async def chat_with_agent(
    request: ChatRequest,
    client: AsyncOpenAI = Depends(get_openai_client),
    session: AsyncSession = Depends(get_async_session)
):
    """Fashion agent endpoint for semantic product search and recommendations."""
    try:
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4()}"
        agent = FashionAgent(client)
        
        # Use filtered search if filters are provided
        if any([request.min_price, request.max_price, request.min_rating, request.category]):
            response_text = await agent.process_with_filters(
                user_query=request.message,
                session=session,
                min_price=request.min_price,
                max_price=request.max_price,
                min_rating=request.min_rating,
                category=request.category,
                conversation_history=request.conversation_history
            )
        else:
            response_text = await agent.process_query(
                user_query=request.message,
                session=session,
                conversation_history=request.conversation_history
            )
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            tokens_used=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing agent request: {str(e)}")
    
