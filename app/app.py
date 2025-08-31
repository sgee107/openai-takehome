from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from app.settings import settings
from app.dependencies import OpenAIClient
from app.search import search_reviews_endpoint


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tokens_used: Optional[int] = None


@app.get("/")
async def root():
    return {"message": "Chat API is running", "version": settings.app_version}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, client: OpenAIClient):
    try:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})
        
        response = await client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        conversation_id = request.conversation_id or f"conv_{id(request)}"
        
        return ChatResponse(
            response=response.choices[0].message.content,
            conversation_id=conversation_id,
            tokens_used=response.usage.total_tokens if response.usage else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")