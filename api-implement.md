# Chat API vNext Implementation Plan

## Current State Analysis

### Existing Architecture
- **FastAPI Backend**: Located in `app/api/chat.py` with basic chat endpoints
- **Fashion Agent**: `app/agents/fashion_agent.py` with LLM-powered responses
- **Semantic Search**: `app/agents/tools/search.py` using pgvector cosine similarity
- **Frontend**: Next.js with mock API in `frontend/app/api/chat/route.ts`
- **Database**: PostgreSQL with pgvector extension, product embeddings already generated

### Current Capabilities
✅ Product embeddings stored in database  
✅ Semantic search with cosine similarity  
✅ Basic filtering (price, rating, category)  
✅ Frontend with search interface and debug mode  
✅ LLM agent for conversational responses  

### Gaps to Address
❌ No complexity classification (direct/filtered/ambiguous)  
❌ No structured query parsing  
❌ No Bayesian rating adjustment  
❌ No linear-blend re-ranking  
❌ No facets generation  
❌ No follow-up prompts for ambiguous queries  
❌ No structured response format matching frontend needs  

## Implementation Plan

### Phase 1: New API Schema & Types

#### 1.1 Backend Types (`app/types/search_api.py`)
```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union, Literal
from enum import IntEnum

class QueryComplexity(IntEnum):
    DIRECT = 1      # SKU/brand+model, quoted terms
    FILTERED = 2    # category + constraints  
    AMBIGUOUS = 3   # broad terms, conflicting signals

class SearchRequest(BaseModel):
    query: str
    topK: Optional[int] = 200
    lambda_blend: Optional[float] = 0.85  # semantic weight
    debug: Optional[bool] = False

class ParsedQuery(BaseModel):
    terms: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    colors: Optional[List[str]] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None

class AgentDecision(BaseModel):
    complexity: QueryComplexity
    parsed: Optional[ParsedQuery] = None
    reason: Optional[str] = None

class ProductMatch(BaseModel):
    final: float        # 0-1 final score
    semantic: float     # 0-1 normalized similarity
    rating: float       # 0-1 normalized Bayesian rating
    lambda_used: float  # λ used in blend
    explanation: Optional[str] = None

class ProductResult(BaseModel):
    id: str
    title: str
    image: Optional[str] = None
    url: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    ratingCount: Optional[int] = None
    match: ProductMatch

class FacetOption(BaseModel):
    value: str
    count: int

class FacetGroup(BaseModel):
    name: str
    options: List[FacetOption]

class FollowupPrompt(BaseModel):
    text: str
    rationale: Optional[str] = None

class UIHints(BaseModel):
    layout: Literal["grid", "list"] = "grid"
    showRating: bool = True
    showFacets: bool = True
    emptyStateCopy: Optional[str] = None

class DebugTrace(BaseModel):
    timings: Optional[Dict[str, float]] = None
    plan: Optional[str] = None
    rawScores: Optional[List[Dict[str, Any]]] = None

class ChatSearchResponse(BaseModel):
    agent: AgentDecision
    ui: UIHints
    results: List[ProductResult]
    facets: Optional[List[FacetGroup]] = None
    followups: Optional[List[FollowupPrompt]] = None
    debug: Optional[DebugTrace] = None
```

#### 1.2 Frontend Types Update (`frontend/lib/types.ts`)
```typescript
// Add new interfaces matching backend schema
export interface AgentDecision {
  complexity: 1 | 2 | 3;
  parsed?: ParsedQuery;
  reason?: string;
}

export interface ParsedQuery {
  terms?: string[];
  filters?: Record<string, any>;
  category?: string;
  brand?: string;
  colors?: string[];
  price_min?: number;
  price_max?: number;
}

export interface ProductMatch {
  final: number;
  semantic: number;
  rating: number;
  lambda_used: number;
  explanation?: string;
}

// Updated ProductResult for new API
export interface ProductResultVNext {
  id: string;
  title: string;
  image?: string;
  url?: string;
  price?: number;
  rating?: number;
  ratingCount?: number;
  match: ProductMatch;
}

export interface ChatSearchResponse {
  agent: AgentDecision;
  ui: UIHints;
  results: ProductResultVNext[];
  facets?: FacetGroup[];
  followups?: FollowupPrompt[];
  debug?: DebugTrace;
}

export interface UIHints {
  layout: "grid" | "list";
  showRating: boolean;
  showFacets: boolean;
  emptyStateCopy?: string;
}

export interface FacetGroup {
  name: string;
  options: Array<{ value: string; count: number }>;
}

export interface FollowupPrompt {
  text: string;
  rationale?: string;
}

export interface DebugTrace {
  timings?: Record<string, number>;
  plan?: string;
  rawScores?: any[];
}
```

### Phase 2: Core Pipeline Components

#### 2.1 Intent Classifier (`app/services/intent_classifier.py`)
```python
import json
from typing import Tuple, Dict, Any
from openai import AsyncOpenAI
from app.types.search_api import QueryComplexity, ParsedQuery
from app.prompts.v1.intent_classifier import (
    INTENT_CLASSIFICATION_SYSTEM,
    INTENT_CLASSIFICATION_USER_TEMPLATE
)

class IntentClassifier:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
    
    async def classify(self, query: str) -> Tuple[QueryComplexity, ParsedQuery]:
        """Classify query complexity and parse structured elements using OpenAI."""
        return await self._llm_classify_and_parse(query)
    
    async def _llm_classify_and_parse(self, query: str) -> Tuple[QueryComplexity, ParsedQuery]:
        """Use LLM for both classification and structured parsing."""
        
        user_prompt = INTENT_CLASSIFICATION_USER_TEMPLATE.format(query=query)
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extract complexity
            complexity = QueryComplexity(result["complexity"])
            
            # Build ParsedQuery from extracted data
            parsed_data = result.get("parsed", {})
            parsed = ParsedQuery(
                terms=parsed_data.get("terms"),
                category=parsed_data.get("category"),
                brand=parsed_data.get("brand"),
                colors=parsed_data.get("colors"),
                price_min=parsed_data.get("price_min"),
                price_max=parsed_data.get("price_max"),
                filters={
                    "size": parsed_data.get("size"),
                    "gender": parsed_data.get("gender"),
                    "occasion": parsed_data.get("occasion")
                }
            )
            
            return complexity, parsed
            
        except Exception as e:
            # Fallback to ambiguous classification
            return QueryComplexity.AMBIGUOUS, ParsedQuery(terms=query.split())
```

#### 2.2 Retrieval Service (`app/services/retrieval.py`)
```python
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from openai import AsyncOpenAI
from dataclasses import dataclass
from app.db.models import Product, ProductEmbedding

@dataclass
class Candidate:
    product_id: str
    title: str
    price: Optional[float]
    rating: Optional[float]
    rating_count: Optional[int]
    category: str
    similarity: float
    metadata: Dict[str, Any]

class RetrievalService:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.embedding_model = "text-embedding-3-small"
    
    async def get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        response = await self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding
    
    async def topk(
        self, 
        query: str,
        session: AsyncSession,
        k: int = 200,
        category_filter: Optional[str] = None
    ) -> List[Candidate]:
        """Retrieve top-K candidates using semantic similarity."""
        
        # Get query embedding
        query_embedding = await self.get_query_embedding(query)
        
        # Build similarity query
        query_stmt = (
            select(
                ProductEmbedding,
                Product,
                (1 - ProductEmbedding.embedding.cosine_distance(query_embedding)).label('similarity')
            )
            .join(Product, ProductEmbedding.product_id == Product.id)
        )
        
        # Apply category pre-filter if strong signal
        if category_filter:
            query_stmt = query_stmt.where(Product.main_category.ilike(f"%{category_filter}%"))
        
        # Order by similarity and limit
        query_stmt = query_stmt.order_by(text('similarity DESC')).limit(k)
        
        # Execute query
        result = await session.execute(query_stmt)
        rows = result.all()
        
        # Convert to candidates
        candidates = []
        for row in rows:
            embedding_record, product, similarity = row
            candidates.append(Candidate(
                product_id=product.id,
                title=product.title,
                price=product.price,
                rating=product.average_rating,
                rating_count=product.rating_number,
                category=product.main_category,
                similarity=float(similarity),
                metadata={
                    'parent_asin': product.parent_asin,
                    'store': product.store,
                    'features': product.features or [],
                    'images': getattr(product, 'images', [])
                }
            ))
        
        return candidates
```

#### 2.3 Ranking Service (`app/services/ranking.py`)
```python
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, replace
from app.services.retrieval import Candidate

@dataclass
class RankedCandidate(Candidate):
    final_score: float
    semantic_norm: float
    rating_norm: float
    lambda_used: float

class RankingService:
    def __init__(self, mu: float = 4.0, w: float = 20.0):
        self.mu = mu  # Prior rating mean
        self.w = w    # Prior weight for Bayesian shrinkage
    
    def bayes_rating(self, rating: Optional[float], count: Optional[int]) -> float:
        """Compute Bayesian-adjusted rating."""
        if rating is None or count is None:
            return self.mu  # Default to prior mean
        
        return (self.w * self.mu + rating * count) / (self.w + count)
    
    def rerank(
        self, 
        candidates: List[Candidate], 
        lambda_blend: float = 0.85
    ) -> List[RankedCandidate]:
        """Re-rank candidates using linear blend of semantic + rating scores."""
        
        if not candidates:
            return []
        
        # Extract similarity scores
        similarities = np.array([c.similarity for c in candidates])
        
        # Normalize similarities per-query
        sim_min = similarities.min()
        sim_max = similarities.max()
        sim_range = sim_max - sim_min
        
        if sim_range < 1e-9:
            # Zero variance case - use rank-based scoring
            s_prime = 1.0 - np.arange(len(candidates)) / (len(candidates) + 1)
        else:
            s_prime = (similarities - sim_min) / sim_range
        
        # Compute Bayesian ratings and normalize
        bayes_ratings = np.array([
            self.bayes_rating(c.rating, c.rating_count) 
            for c in candidates
        ])
        r_prime = bayes_ratings / 5.0  # Normalize to [0,1]
        
        # Linear blend
        final_scores = lambda_blend * s_prime + (1 - lambda_blend) * r_prime
        
        # Create ranked candidates
        ranked = []
        for i, candidate in enumerate(candidates):
            ranked.append(RankedCandidate(
                **candidate.__dict__,
                final_score=final_scores[i],
                semantic_norm=s_prime[i],
                rating_norm=r_prime[i],
                lambda_used=lambda_blend
            ))
        
        # Sort by final score (descending), then by rating_count (descending), then by price (ascending)
        ranked.sort(
            key=lambda x: (-x.final_score, -(x.rating_count or 0), x.price or float('inf'))
        )
        
        return ranked
```

#### 2.4 Facets & Follow-ups (`app/services/facets.py`)
```python
import json
from typing import List, Dict, Counter
from collections import defaultdict
from openai import AsyncOpenAI
from app.types.search_api import FacetGroup, FacetOption, FollowupPrompt, QueryComplexity
from app.services.ranking import RankedCandidate

class FacetsService:
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
    
    def generate_facets(self, candidates: List[RankedCandidate]) -> List[FacetGroup]:
        """Generate facets from top-K results."""
        
        if not candidates:
            return []
        
        facets = []
        
        # Category facets
        category_counts = Counter(c.category for c in candidates)
        if len(category_counts) > 1:
            category_options = [
                FacetOption(value=cat, count=count)
                for cat, count in category_counts.most_common(5)
            ]
            facets.append(FacetGroup(name="Category", options=category_options))
        
        # Price range facets
        prices = [c.price for c in candidates if c.price is not None]
        if prices:
            price_ranges = self._create_price_ranges(prices)
            price_options = [
                FacetOption(value=range_name, count=count)
                for range_name, count in price_ranges.items()
            ]
            facets.append(FacetGroup(name="Price Range", options=price_options))
        
        # Brand facets (from store field)
        stores = [c.metadata.get('store') for c in candidates if c.metadata.get('store')]
        store_counts = Counter(stores)
        if len(store_counts) > 1:
            store_options = [
                FacetOption(value=store, count=count)
                for store, count in store_counts.most_common(5)
            ]
            facets.append(FacetGroup(name="Brand", options=store_options))
        
        return facets
    
    def _create_price_ranges(self, prices: List[float]) -> Dict[str, int]:
        """Create price range buckets."""
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
        
        return {k: v for k, v in ranges.items() if v > 0}
    
    async def generate_followups(self, complexity: QueryComplexity, query: str, reason: str) -> List[FollowupPrompt]:
        """Generate follow-up prompts using LLM for ambiguous queries."""
        
        if complexity != QueryComplexity.AMBIGUOUS:
            return []
        
        return await self._llm_generate_followups(query, reason)
    
    async def _llm_generate_followups(self, query: str, reason: str) -> List[FollowupPrompt]:
        """Use LLM to generate contextual follow-up questions."""
        from app.prompts.v1.followup_generator import (
            FOLLOWUP_GENERATION_SYSTEM,
            FOLLOWUP_GENERATION_USER_TEMPLATE
        )
        from openai import AsyncOpenAI
        
        # This would need to be injected via constructor in real implementation
        # For now, showing the pattern
        user_prompt = FOLLOWUP_GENERATION_USER_TEMPLATE.format(
            query=query,
            reason=reason
        )
        
        try:
            # This would use the injected OpenAI client
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
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
            
            return [
                FollowupPrompt(
                    text=item["text"],
                    rationale=item["rationale"]
                )
                for item in followups_data
            ]
            
        except Exception:
            # Fallback to basic follow-ups
            return [
                FollowupPrompt(
                    text="Could you be more specific about what you're looking for?",
                    rationale="Fallback for ambiguous query"
                )
            ]
```

### Phase 3: Main API Endpoint

#### 3.1 Search Router (`app/api/search.py`)
```python
import time
from fastapi import APIRouter, HTTPException, Depends
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.types.search_api import (
    SearchRequest, ChatSearchResponse, AgentDecision, UIHints, 
    ProductResult, ProductMatch, DebugTrace
)
from app.services.intent_classifier import IntentClassifier
from app.services.retrieval import RetrievalService
from app.services.ranking import RankingService
from app.services.facets import FacetsService
from app.dependencies import get_openai_client
from app.db.database import get_async_session

router = APIRouter(prefix="/chat", tags=["search"])

@router.post("/search")
async def search_products(
    request: SearchRequest,
    client: AsyncOpenAI = Depends(get_openai_client),
    session: AsyncSession = Depends(get_async_session)
) -> ChatSearchResponse:
    """
    Product search endpoint with intent classification and linear-blend ranking.
    """
    start_time = time.time()
    debug_trace = DebugTrace(timings={}) if request.debug else None
    
    try:
        # Initialize services
        classifier = IntentClassifier(client)
        retriever = RetrievalService(client) 
        ranker = RankingService()
        facets_service = FacetsService(client)
        
        # Step 1: Classify intent and parse query
        classify_start = time.time()
        complexity, parsed = await classifier.classify(request.query)
        if debug_trace:
            debug_trace.timings["classify"] = time.time() - classify_start
        
        classification_reason = f"Classified as complexity {complexity.value}"
        agent_decision = AgentDecision(
            complexity=complexity,
            parsed=parsed,
            reason=classification_reason
        )
        
        # Step 2: Retrieve top-K candidates
        retrieve_start = time.time()
        category_filter = parsed.category if parsed else None
        candidates = await retriever.topk(
            query=request.query,
            session=session,
            k=request.topK,
            category_filter=category_filter
        )
        if debug_trace:
            debug_trace.timings["retrieve"] = time.time() - retrieve_start
        
        # Step 3: Re-rank with linear blend
        rank_start = time.time()
        ranked_candidates = ranker.rerank(candidates, request.lambda_blend)
        if debug_trace:
            debug_trace.timings["rank"] = time.time() - rank_start
        
        # Step 4: Generate facets and follow-ups
        facets_start = time.time()
        facets = facets_service.generate_facets(ranked_candidates)
        followups = await facets_service.generate_followups(complexity, request.query, classification_reason)
        if debug_trace:
            debug_trace.timings["facets"] = time.time() - facets_start
        
        # Step 5: Format response
        results = []
        for candidate in ranked_candidates:
            # Get primary image
            images = candidate.metadata.get('images', [])
            primary_image = None
            if images:
                for img in images:
                    if isinstance(img, dict) and img.get('hi_res'):
                        primary_image = img['hi_res']
                        break
                    elif isinstance(img, dict) and img.get('large'):
                        primary_image = img['large']
                        break
            
            product_result = ProductResult(
                id=candidate.product_id,
                title=candidate.title,
                image=primary_image,
                url=f"/product/{candidate.metadata.get('parent_asin', candidate.product_id)}",
                price=candidate.price,
                rating=candidate.rating,
                ratingCount=candidate.rating_count,
                match=ProductMatch(
                    final=candidate.final_score,
                    semantic=candidate.semantic_norm,
                    rating=candidate.rating_norm,
                    lambda_used=candidate.lambda_used,
                    explanation=f"Semantic: {candidate.semantic_norm:.3f}, Rating: {candidate.rating_norm:.3f}"
                )
            )
            results.append(product_result)
        
        # UI hints based on complexity
        ui_hints = UIHints(
            layout="grid" if complexity in [QueryComplexity.FILTERED, QueryComplexity.AMBIGUOUS] else "list",
            showRating=True,
            showFacets=complexity != QueryComplexity.DIRECT,
            emptyStateCopy="No products found. Try a broader search or different terms." if not results else None
        )
        
        # Debug information
        if debug_trace:
            debug_trace.timings["total"] = time.time() - start_time
            debug_trace.plan = f"Complexity {complexity.value} → TopK({request.topK}) → LinearBlend(λ={request.lambda_blend})"
            if request.debug:
                debug_trace.rawScores = [
                    {
                        "product_id": c.product_id,
                        "title": c.title,
                        "similarity": c.similarity,
                        "final_score": c.final_score
                    }
                    for c in ranked_candidates[:10]  # Top 10 for debug
                ]
        
        return ChatSearchResponse(
            agent=agent_decision,
            ui=ui_hints,
            results=results,
            facets=facets if ui_hints.showFacets else None,
            followups=followups if followups else None,
            debug=debug_trace
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search pipeline error: {str(e)}"
        )
```

#### 3.2 Update Main App (`app/app.py`)
```python
# Add import for new search router
from app.api.search import router as search_router

# Register the new router
app.include_router(search_router)
```

### Phase 4: Frontend Integration

#### 4.1 Update Frontend API Route (`frontend/app/api/chat/route.ts`)
```typescript
import { NextRequest, NextResponse } from 'next/server';
import { ChatSearchResponse } from '../../../lib/types';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { query, topK = 20, lambda = 0.85, debug = false } = body;
    
    // Forward to backend API
    const response = await fetch(`${BACKEND_URL}/chat/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        topK,
        lambda_blend: lambda,
        debug
      })
    });
    
    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`);
    }
    
    const data: ChatSearchResponse = await response.json();
    return NextResponse.json(data);
    
  } catch (error) {
    console.error('API error:', error);
    return NextResponse.json(
      { error: 'Failed to process search request' },
      { status: 500 }
    );
  }
}
```

#### 4.2 Update Product Grid Component
Update `frontend/components/ProductGrid.tsx` to use new response structure and display facets, follow-ups, and enhanced debug information.

### Phase 5: Testing Strategy

#### 5.1 Unit Tests (`tests/test_search_pipeline.py`)
```python
import pytest
from app.services.intent_classifier import IntentClassifier, QueryComplexity
from app.services.ranking import RankingService, Candidate

def test_intent_classification():
    """Test intent classification heuristics."""
    classifier = IntentClassifier(None)  # Mock client for heuristic tests
    
    # Test direct queries
    complexity, parsed = classifier._heuristic_classify("Levi 511 jeans 32x32")
    assert complexity == QueryComplexity.DIRECT
    
    # Test filtered queries  
    complexity, parsed = classifier._heuristic_classify("blue jeans under $50")
    assert complexity == QueryComplexity.FILTERED
    assert parsed.price_max == 50.0
    assert "blue" in parsed.colors
    
    # Test ambiguous queries
    complexity, parsed = classifier._heuristic_classify("work clothes")
    assert complexity == QueryComplexity.AMBIGUOUS

def test_bayesian_ranking():
    """Test Bayesian rating adjustment."""
    ranker = RankingService()
    
    # High rating, low count (should shrink toward prior)
    rating = ranker.bayes_rating(5.0, 1)
    assert 4.0 < rating < 5.0
    
    # High rating, high count (should stay close to original)
    rating = ranker.bayes_rating(5.0, 100)  
    assert rating > 4.8

def test_linear_blend_ranking():
    """Test linear blend re-ranking."""
    ranker = RankingService()
    
    candidates = [
        Candidate("1", "Product A", 100.0, 5.0, 10, "Category", 0.9, {}),
        Candidate("2", "Product B", 50.0, 3.0, 100, "Category", 0.8, {}),
    ]
    
    # High semantic weight should favor Product A
    ranked = ranker.rerank(candidates, lambda_blend=0.9)
    assert ranked[0].product_id == "1"
    
    # Low semantic weight should favor Product B (better rating with more reviews)
    ranked = ranker.rerank(candidates, lambda_blend=0.1) 
    assert ranked[0].product_id == "2"
```

#### 5.2 Integration Tests (`tests/test_search_api.py`)
```python
import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_search_endpoint_direct():
    """Test search endpoint with direct query."""
    response = client.post("/chat/search", json={
        "query": "Nike Air Max 270",
        "debug": True
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["agent"]["complexity"] == 1  # Direct
    assert data["ui"]["layout"] == "list"
    assert data["ui"]["showFacets"] == False
    assert len(data["results"]) > 0
    assert "debug" in data

def test_search_endpoint_filtered():
    """Test search endpoint with filtered query."""
    response = client.post("/chat/search", json={
        "query": "blue jeans under $100",
        "topK": 50
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["agent"]["complexity"] == 2  # Filtered
    assert data["ui"]["layout"] == "grid"
    assert data["ui"]["showFacets"] == True
    assert "facets" in data
    assert len(data["results"]) > 0

def test_search_endpoint_ambiguous():
    """Test search endpoint with ambiguous query."""
    response = client.post("/chat/search", json={
        "query": "work clothes"
    })
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["agent"]["complexity"] == 3  # Ambiguous
    assert "followups" in data
    assert len(data["followups"]) > 0
```

## Implementation Timeline

### Week 1: Core Pipeline (Days 1-5)
- **Day 1**: API schema and types (backend + frontend)
- **Day 2**: Intent classifier with heuristic rules
- **Day 3**: Retrieval service integration 
- **Day 4**: Bayesian ranking and linear blend
- **Day 5**: Facets and follow-ups generation

### Week 2: API & Integration (Days 6-10)  
- **Day 6**: Main search endpoint implementation
- **Day 7**: Frontend API integration and UI updates
- **Day 8**: Debug tracing and logging
- **Day 9**: Unit and integration tests
- **Day 10**: End-to-end testing and bug fixes

## Success Metrics

### Functional Requirements
✅ **Complexity Classification**: 95%+ accuracy on test queries  
✅ **Response Time**: <500ms p95 for Top-200 retrieval  
✅ **Ranking Quality**: A/B test shows improvement over current agent  
✅ **Schema Compatibility**: Frontend renders all response fields correctly  

### Technical Requirements  
✅ **Test Coverage**: >80% code coverage for pipeline components  
✅ **Error Handling**: Graceful degradation for edge cases  
✅ **Debug Mode**: Complete tracing for performance optimization  
✅ **Backwards Compatibility**: Existing `/chat` endpoints remain functional  

## Risk Mitigation

### Performance Risks
- **Top-K scaling**: Monitor query times, add caching if needed
- **LLM fallback latency**: Set aggressive timeouts for classification
- **Database load**: Consider read replicas for search queries

### Quality Risks  
- **Complexity classification errors**: Implement confidence scoring
- **Poor ranking**: A/B test against existing agent, tune λ parameter
- **Missing facets**: Fallback to basic category/price facets

### Integration Risks
- **Frontend breaking changes**: Maintain backwards compatibility
- **Database schema changes**: Use feature flags for gradual rollout
- **API versioning**: Consider `/v1/chat/search` endpoint naming

This implementation plan provides a structured path from the current LLM-agent architecture to the new linear-blend search pipeline while maintaining system reliability and user experience.