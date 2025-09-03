# Fashion Search System Architecture

```mermaid
flowchart TD
    %% Data Sources
    DS[Amazon Fashion Dataset<br/>JSON with products, images, metadata]
    
    %% Data Processing Layer
    subgraph "Data Processing Pipeline"
        PL[Product Loader<br/>app/process/core/product_loader.py]
        EG[Embedding Generator<br/>OpenAI text-embedding-3-small]
        IA[Image Analysis<br/>GPT-4o Vision API]
        TS[Text Strategies<br/>Multiple embedding strategies]
    end
    
    %% Database Layer
    subgraph "PostgreSQL Database with pgvector"
        P[(Products Table)]
        PE[(ProductEmbeddings Table<br/>Vector similarity search)]
        PI[(ProductImages Table)]
        PIA[(ProductImageAnalysis Table)]
    end
    
    %% API Layer
    subgraph "FastAPI Backend"
        CR[Chat Router<br/>app/api/chat.py]
        SR[Search Router<br/>app/api/search.py]
        MR[Mock Router<br/>app/api/mock.py]
    end
    
    %% AI Agent Layer
    subgraph "Fashion Agent"
        FA[FashionAgent<br/>app/agents/fashion_agent.py]
        CST[ComprehensiveSearchTool<br/>app/agents/tools/comprehensive_search.py]
    end
    
    %% Search Pipeline
    subgraph "Search Processing Pipeline"
        IC[Intent Classification<br/>GPT-4o-mini]
        QP[Query Parsing<br/>Extract filters, categories]
        VS[Vector Similarity<br/>Cosine distance search]
        BR[Bayesian Rating<br/>Adjustment with priors]
        LB[Linear Blending<br/>λ * semantic + (1-λ) * rating]
        FG[Facet Generation<br/>Price & rating ranges]
    end
    
    %% Frontend Layer
    subgraph "Next.js Frontend"
        SB[Search Bar<br/>components/SearchBar.tsx]
        PG[Product Grid<br/>components/ProductGrid.tsx]
        PM[Product Modal<br/>components/ProductModal.tsx]
        DT[Debug Toggle<br/>components/DebugToggle.tsx]
    end
    
    %% User
    U[👤 User Query]
    
    %% Data Flow Connections
    DS --> PL
    DS --> IA
    DS --> EG
    
    PL --> P
    EG --> PE
    TS --> PE
    IA --> PIA
    P --> PI
    
    PE --> CST
    P --> CST
    PI --> CST
    PIA --> CST
    
    CR --> FA
    SR --> FA
    FA --> CST
    
    CST --> IC
    IC --> QP
    QP --> VS
    VS --> BR
    BR --> LB
    LB --> FG
    
    FG --> CR
    CR --> SB
    SB --> PG
    PG --> PM
    PM --> DT
    
    U --> SB
    PG --> U
    
    %% Styling
    classDef dataSource fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef database fill:#f1f8e9,stroke:#388e3c,stroke-width:2px
    classDef api fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef agent fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef frontend fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef user fill:#fffde7,stroke:#f9a825,stroke-width:3px
    
    class DS dataSource
    class PL,EG,IA,TS,IC,QP,VS,BR,LB,FG processing
    class P,PE,PI,PIA database
    class CR,SR,MR api
    class FA,CST agent
    class SB,PG,PM,DT frontend
    class U user
```

## System Data Flow Overview

### 1. **Data Ingestion & Processing**
- Amazon Fashion Dataset (JSON) contains product metadata, images, ratings
- Product Loader parses and validates data into structured format
- Embedding Generator creates vector representations using OpenAI's text-embedding-3-small
- Image Analysis uses GPT-4o Vision to extract visual features
- Multiple text strategies generate embeddings for different search approaches

### 2. **Database Storage**
- PostgreSQL with pgvector extension for efficient vector operations
- Products table stores core product information
- ProductEmbeddings table stores vector representations with cosine similarity search
- ProductImages and ProductImageAnalysis tables store visual data and AI-generated insights

### 3. **API Layer**
- FastAPI backend with multiple endpoints (/chat, /search, /agent)
- Chat router handles conversational queries
- Search router provides direct search functionality
- Mock router for testing and development

### 4. **AI Agent Processing**
- FashionAgent orchestrates the search pipeline
- ComprehensiveSearchTool handles the complete search workflow:
  - Intent classification using GPT-4o-mini
  - Query parsing to extract filters and categories
  - Vector similarity search using cosine distance
  - Bayesian rating adjustment with confidence intervals
  - Linear blending of semantic and rating scores (λ parameter)
  - Facet generation for filtering options

### 5. **Search Pipeline Details**
- **Intent Classification**: Determines query complexity and structure
- **Vector Similarity**: Finds semantically similar products using embeddings
- **Bayesian Rating**: Adjusts ratings based on review count confidence
- **Linear Blending**: Combines semantic relevance with product quality
- **Facet Generation**: Creates price and rating filter options

### 6. **Frontend Interface**
- Next.js React application with TypeScript
- Search bar for query input
- Product grid displays results with images and metadata
- Product modal shows detailed information
- Debug toggle for development insights

### 7. **User Interaction Flow**
1. User enters search query
2. Frontend sends request to FastAPI backend
3. FashionAgent processes query through search pipeline
4. Database returns ranked results with similarity scores
5. Results formatted and returned to frontend
6. User sees product grid with relevant items