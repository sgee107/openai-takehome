# Fashion Search Architecture Analysis

## Complete Search Flow Mermaid Diagram

```mermaid
graph TD
    A[User Query] --> B[POST chat/search API Endpoint]
    B --> C[SearchRequest Validation]
    C --> D[Initialize Services]
    
    %% Services Initialization
    D --> E[IntentClassifier Service]
    D --> F[RetrievalService]  
    D --> G[RankingService]
    D --> H[FacetsService]
    
    %% Step 1: Intent Classification
    E --> I[LLM Classification GPT-4o-mini]
    I --> J[Query Complexity Analysis DIRECT/AMBIGUOUS/COMPLEX]
    J --> K[Structured Query Parsing Extract category brand price colors]
    
    %% Step 2: Retrieval
    F --> L[Generate Query Embedding text-embedding-3-small]
    L --> M[Database Query Builder]
    M --> N[(PostgreSQL + pgvector)]
    N --> O[Top-K Candidates Cosine Similarity]
    
    %% Step 3: Re-ranking
    G --> P[Bayesian Rating Adjustment μ=4.0 w=20.0]
    P --> Q[Score Normalization similarity and rating norms]
    Q --> R[Linear Blend Formula λ * semantic + 1-λ * rating]
    R --> S[Final Ranked Results]
    
    %% Step 4: Enhancement
    H --> T[Generate Facets Category/Brand/Price]
    H --> U[Generate Followups Based on complexity]
    
    %% Step 5: Response Assembly
    S --> V[Format ProductResults]
    T --> V
    U --> V
    K --> W[Generate UI Hints Layout/Facets/Rating]
    W --> V
    V --> X[ChatSearchResponse]
    
    %% Database Schema
    N --> Y[(Products Table)]
    N --> Z[(ProductEmbeddings Table)]
    Y -.-> Z
    
    %% Filters Applied
    M --> M1[Category Filter]
    M --> M2[Brand Filter] 
    M --> M3[Price Range Filter]
    M --> M4[Embedding Strategy Filter]
    
    %% Debug & Monitoring
    X --> DEBUG[Debug Traces Timings and Raw Scores]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style X fill:#e8f5e8
    style N fill:#fff3e0
    style I fill:#fce4ec
```

## Architecture Components Breakdown

### 1. API Layer (`app/api/search.py`)
- **Endpoint**: `POST /chat/search`
- **Input**: `SearchRequest` (query, topK, lambda_blend, debug)
- **Output**: `ChatSearchResponse` with structured product results
- **Features**: Debug tracing, error handling, performance monitoring

### 2. Intent Classification (`app/services/intent_classifier.py`)
- **Model**: GPT-4o-mini with JSON response format
- **Classification**: DIRECT/AMBIGUOUS/COMPLEX query complexity
- **Parsing**: Extracts category, brand, colors, price ranges, filters
- **Fallback**: Rule-based classification if LLM fails

### 3. Retrieval Service (`app/services/retrieval.py`)
- **Embedding Model**: text-embedding-3-small
- **Search Strategy**: Cosine similarity using pgvector
- **Default Strategy**: "key_value_with_images" embeddings
- **Filters**: Category, brand, price range with ILIKE matching
- **Top-K**: Retrieves up to 200 candidates by default

### 4. Ranking Service (`app/services/ranking.py`)
- **Bayesian Rating**: Shrinkage toward prior (μ=4.0, w=20.0)
- **Normalization**: Per-query min-max scaling for similarity scores
- **Linear Blend**: `final_score = λ * s' + (1-λ) * r'`
- **Default Lambda**: 0.85 (favors semantic similarity)
- **Tie Breaking**: By rating count, then price

### 5. Database Architecture
- **Products Table**: Core product metadata
- **ProductEmbeddings Table**: Vector embeddings with strategy labels
- **Vector Extension**: pgvector for cosine distance operations
- **Indexes**: Optimized for similarity search performance

### 6. Agent Tools Layer (`app/agents/tools/search.py`)
- **SemanticSearchTool**: Lower-level database interface
- **Direct Queries**: Raw cosine similarity search
- **Filter Support**: Price, rating, category filters
- **Multiple Strategies**: Different embedding approaches

## Key Technical Decisions

### Embedding Strategy
- **Choice**: text-embedding-3-small
- **Strategy**: "key_value_with_images" as default
- **Rationale**: Balance of performance and quality for fashion search

### Ranking Formula
- **Linear Blend**: Combines semantic similarity with quality signals
- **Bayesian Adjustment**: Handles rating reliability and new products
- **Lambda Parameter**: Tunable semantic vs. quality trade-off

### Search Pipeline
- **Multi-Stage**: Intent → Retrieve → Re-rank → Enhance
- **Scalable**: Top-K retrieval with post-filtering
- **Flexible**: Query complexity drives UI adaptation

### Data Flow
```
User Query → LLM Classification → Vector Search → Bayesian Re-ranking → Faceted Response
