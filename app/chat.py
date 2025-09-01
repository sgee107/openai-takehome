"""Chat API router using non-streaming completions (no SSE)."""

from fastapi import APIRouter, HTTPException, Depends
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Optional
import uuid

from app.settings import settings
from app.dependencies import get_openai_client


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: Optional[int] = None


@router.post("")
async def chat(request: ChatRequest, client: AsyncOpenAI = Depends(get_openai_client)):
    """Chat endpoint that returns a full, non-streaming completion."""
    try:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})

        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
        )

        conversation_id = request.conversation_id or f"conv_{uuid.uuid4()}"

        return ChatResponse(
            response=response.choices[0].message.content,
            conversation_id=conversation_id,
            tokens_used=(response.usage.total_tokens if getattr(response, "usage", None) else None),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")
    
