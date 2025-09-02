from fastapi import APIRouter, HTTPException, Depends
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_openai_client
from app.db.database import get_async_session
from app.search import search_products
from app.prompts.simple_prompt import SIMPLE_AGENT_PROMPT

router = APIRouter(prefix="/simple", tags=["simple"])

class SimpleRequest(BaseModel):
    query: str

class SimpleResponse(BaseModel):
    response: str
    products: list

@router.post("/search")
async def simple_search(
    request: SimpleRequest,
    client: AsyncOpenAI = Depends(get_openai_client),
    session: AsyncSession = Depends(get_async_session)
):
    """Simple product search endpoint."""
    try:
        # Search for products
        products = await search_products(request.query, client, session)
        
        # Create context for LLM
        context = f"User query: {request.query}\n\nFound products:\n"
        for i, product in enumerate(products, 1):
            context += f"{i}. {product['title']} - ${product['price']} (similarity: {product['similarity']:.2f})\n"
        
        # Get LLM response
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SIMPLE_AGENT_PROMPT},
                {"role": "user", "content": context}
            ],
            max_tokens=500
        )
        
        return SimpleResponse(
            response=response.choices[0].message.content,
            products=products
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))