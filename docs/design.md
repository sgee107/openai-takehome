# Fashion Semantic Search System Design

## Overview

This document provides a comprehensive design overview of the Fashion Semantic Search microservice, which enables natural language queries for fashion products using semantic search capabilities powered by OpenAI embeddings and PostgreSQL with pgvector.

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Database Schema](#database-schema)
3. [API Flow](#api-flow)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Search Flow](#search-flow)
6. [Component Interactions](#component-interactions)
7. [Deployment Architecture](#deployment-architecture)

## System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Web UI/CLI]
        CLI[Data Load CLI]
    end
    
    subgraph "API Layer"
        FAPI[FastAPI Server]
        CHAT[Chat Endpoints]
        SIMPLE[Simple Search Endpoint]
    end
    
    subgraph "Business Logic Layer"
        FA[Fashion Agent]
        ST[Semantic Search Tool]
        EMB[Embedding Generator]
    end
    
    subgraph "Data Processing Pipeline"
        DL[Data Loader<br/>load_products.py]
        AM[Metadata Analyzer<br/>analyze_metadata.py]
        ES[Embedding Strategies<br/>- Title Only<br/>- Title + Categories<br/>- Title + Features<br/>- Full Context]
        EXP[Experiment Runner<br/>run_experiments.py]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL + pgvector)]
        CACHE[Redis Cache<br/>Future Enhancement]
        JSON[Amazon Fashion<br/>JSON Dataset]
    end
    
    subgraph "External Services"
        OAI[OpenAI API<br/>- Embeddings API<br/>- Chat Completions API]
        MLF[MLflow Server<br/>+ MinIO Storage]
    end
    
    %% Client connections
    UI --> FAPI
    CLI --> DL
    
    %% API Layer connections
    FAPI --> CHAT
    FAPI --> SIMPLE
    CHAT --> FA
    SIMPLE --> ST
    
    %% Business Logic connections
    FA --> ST
    ST --> EMB
    EMB --> OAI
    ST --> PG
    FA --> OAI
    
    %% Data Processing Pipeline connections
    JSON --> DL
    DL --> AM
    DL --> PG
    AM --> ES
    ES --> OAI
    ES --> PG
    EXP --> ES
    EXP --> MLF
    DL --> MLF
    
    %% MLflow tracking
    FAPI --> MLF
    
    style FAPI fill:#f9f,stroke:#333,stroke-width:4px
    style PG fill:#9ff,stroke:#333,stroke-width:4px
    style OAI fill:#ff9,stroke:#333,stroke-width:4px
    style DL fill:#9f9,stroke:#333,stroke-width:4px
```

## Database Schema

### Entity Relationship Diagram

```mermaid
erDiagram
    PRODUCT {
        int id PK
        string parent_asin UK
        string main_category
        string title
        float average_rating
        int rating_number
        float price
        string store
        json features
        json description
        json categories
        json details
        text bought_together
        datetime created_at
        datetime updated_at
    }
    
    PRODUCT_IMAGE {
        int id PK
        int product_id FK
        string thumb
        string large
        string hi_res
        string variant
        datetime created_at
        datetime updated_at
    }
    
    PRODUCT_VIDEO {
        int id PK
        int product_id FK
        string url
        string title
        datetime created_at
        datetime updated_at
    }
    
    PRODUCT_EMBEDDING {
        int id PK
        int product_id FK
        string strategy
        text embedding_text
        vector embedding
        string model
        datetime created_at
        datetime updated_at
    }
    
    PRODUCT ||--o{ PRODUCT_IMAGE : has
    PRODUCT ||--o{ PRODUCT_VIDEO : has
    PRODUCT ||--o{ PRODUCT_EMBEDDING : has
```

## API Flow

### Request Processing Sequence

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI
    participant Agent as Fashion Agent
    participant Search as Semantic Search
    participant DB as PostgreSQL
    participant OpenAI as OpenAI API
    
    User->>API: POST /chat {"message": "beach outfit for summer"}
    API->>Agent: process_query(message)
    Agent->>Search: search_products(query)
    Search->>OpenAI: create_embedding(query)
    OpenAI-->>Search: query_embedding
    Search->>DB: cosine_similarity_search(embedding)
    DB-->>Search: matched_products
    Search-->>Agent: search_results
    Agent->>Agent: format_products_for_llm()
    Agent->>OpenAI: chat.completions.create(context + results)
    OpenAI-->>Agent: formatted_response
    Agent-->>API: response_text
    API-->>User: {"response": "Here are some great beach outfits..."}
```

## Data Processing Pipeline

### Product Loading and Embedding Generation

```mermaid
flowchart LR
    subgraph "Data Source"
        JSON[Amazon Fashion<br/>JSON Dataset]
    end
    
    subgraph "Data Loader"
        LOAD[load_products.py]
        ANALYZE[analyze_metadata.py]
    end
    
    subgraph "Embedding Strategies"
        S1[Title Only]
        S2[Title + Categories]
        S3[Title + Features]
        S4[Full Context]
    end
    
    subgraph "Storage"
        PROD[(Products Table)]
        EMB[(Embeddings Table)]
    end
    
    JSON --> LOAD
    LOAD --> ANALYZE
    ANALYZE --> S1
    ANALYZE --> S2
    ANALYZE --> S3
    ANALYZE --> S4
    S1 --> EMB
    S2 --> EMB
    S3 --> EMB
    S4 --> EMB
    LOAD --> PROD
```

## Search Flow

### Semantic Search Process

```mermaid
flowchart TD
    START([User Query]) --> PARSE{Query Type?}
    
    PARSE -->|Natural Language| NLP[Process with Agent]
    PARSE -->|Simple Search| DIRECT[Direct Search]
    
    NLP --> EMBED[Generate Query Embedding]
    DIRECT --> EMBED
    
    EMBED --> VECTOR[Vector Similarity Search]
    
    VECTOR --> FILTER{Apply Filters?}
    FILTER -->|Yes| FILTERED[Price, Rating, Category Filters]
    FILTER -->|No| RESULTS[Raw Results]
    
    FILTERED --> RANK[Rank by Similarity Score]
    RESULTS --> RANK
    
    RANK --> FORMAT{Response Type?}
    FORMAT -->|Agent| LLM[Format with LLM]
    FORMAT -->|Simple| JSON[Return JSON]
    
    LLM --> RESPONSE([User Response])
    JSON --> RESPONSE
```

## Component Interactions

### Class and Module Dependencies

```mermaid
classDiagram
    class FastAPI {
        +lifespan()
        +root()
        +health_check()
    }
    
    class ChatRouter {
        +chat()
        +chat_with_agent()
    }
    
    class SimpleRouter {
        +simple_search()
    }
    
    class FashionAgent {
        -openai_client
        -search_tool
        -system_prompt
        +process_query()
        +process_with_filters()
        +format_products_for_llm()
    }
    
    class SemanticSearchTool {
        -openai_client
        -embedding_model
        +get_query_embedding()
        +search_products()
        +search_with_filters()
    }
    
    class Product {
        +id
        +parent_asin
        +title
        +price
        +images
        +embeddings
    }
    
    class ProductEmbedding {
        +product_id
        +strategy
        +embedding_text
        +embedding
        +model
    }
    
    class MLflowClient {
        +create_experiment()
        +start_run()
        +log_metrics()
        +log_params()
    }
    
    FastAPI --> ChatRouter
    FastAPI --> SimpleRouter
    ChatRouter --> FashionAgent
    SimpleRouter --> SemanticSearchTool
    FashionAgent --> SemanticSearchTool
    SemanticSearchTool --> Product
    Product --> ProductEmbedding
    FashionAgent --> MLflowClient
```

## Deployment Architecture

### Docker Compose Services

```mermaid
graph TB
    subgraph "Docker Network"
        subgraph "Application Services"
            APP[FastAPI App<br/>:8000]
            MLFLOW[MLflow Server<br/>:5000]
        end
        
        subgraph "Data Services"
            PG[(PostgreSQL<br/>+ pgvector<br/>:5432)]
            MINIO[MinIO<br/>Object Storage<br/>:9000]
        end
        
        subgraph "Volumes"
            PGDATA[postgres_data]
            MLDATA[mlflow_data]
            MINIODATA[minio_data]
        end
    end
    
    subgraph "External"
        OPENAI[OpenAI API]
    end
    
    APP --> PG
    APP --> MLFLOW
    APP --> OPENAI
    MLFLOW --> PG
    MLFLOW --> MINIO
    PG --> PGDATA
    MLFLOW --> MLDATA
    MINIO --> MINIODATA
    
    style APP fill:#f9f,stroke:#333,stroke-width:2px
    style PG fill:#9ff,stroke:#333,stroke-width:2px
    style MLFLOW fill:#ff9,stroke:#333,stroke-width:2px
```

## Key Design Decisions

### 1. Vector Database Choice
- **PostgreSQL + pgvector**: Chosen for simplicity and unified data storage
- Supports cosine similarity search with good performance for 300-item dataset
- Easy integration with existing relational data

### 2. Embedding Strategy
- **text-embedding-3-small (1536 dimensions)**: Balance between quality and cost
- Multiple embedding strategies tested via MLflow experiments
- Normalized vectors for stable cosine similarity

### 3. Two-Stage Query Processing
- **Stage 1**: Semantic search for relevant products
- **Stage 2**: LLM formatting for natural language response
- Provides both explainability and conversational interface

### 4. Modular Architecture
- Separate concerns: API, Agent, Search Tool, Database
- Easy to test and modify individual components
- Support for both simple search and agent-based queries

## Performance Considerations

1. **Caching Strategy** (Future Enhancement)
   - Redis for query embedding cache
   - Product result caching for popular queries

2. **Index Optimization**
   - IVFFlat or HNSW indexes for larger datasets
   - Currently using exact search for 300 items

3. **Batch Processing**
   - Bulk embedding generation during data loading
   - Async processing for non-blocking operations

## Security Considerations

1. **API Key Management**
   - Environment variables for sensitive data
   - Separate `.env` files for different environments

2. **Input Validation**
   - Pydantic models for request/response validation
   - SQL injection prevention via SQLAlchemy ORM

3. **Rate Limiting** (Future Enhancement)
   - Prevent API abuse
   - Cost control for OpenAI API calls

## Future Enhancements

1. **Advanced Search Features**
   - Multi-modal search (image + text)
   - Outfit composition using graph relationships
   - Personalization based on user history

2. **Scalability Improvements**
   - Distributed vector search with dedicated vector DB
   - Horizontal scaling with load balancer
   - Event-driven architecture for real-time updates

3. **Analytics and Monitoring**
   - Search quality metrics dashboard
   - A/B testing framework for embedding strategies
   - Real-time performance monitoring

## Conclusion

This semantic search system provides a foundation for natural language fashion product discovery. The modular architecture allows for easy experimentation and enhancement while maintaining production stability. The use of industry-standard tools (FastAPI, PostgreSQL, OpenAI) ensures maintainability and community support.
