# Semantic Search Approaches for Fashion E-commerce

## Overview
This document outlines different approaches for implementing semantic search to handle natural language queries like "I need an outfit to go to the beach this summer" against a fashion product dataset.

## Approach 1: Two-Stage LLM Pipeline

### Architecture
1. **Stage 1 - Query Understanding**: LLM parses natural language into structured intent
2. **Stage 2 - Structured Search**: Use extracted attributes for targeted retrieval

### Implementation
```python
# Stage 1: Parse query
def parse_query(user_input):
    # LLM extracts: {occasion: "beach", season: "summer", 
    #                items: ["swimwear", "coverup", "sandals"],
    #                style: "casual", weather: "hot"}
    
# Stage 2: Execute searches
def search_products(parsed_intent):
    # Use structured fields to query database
    # Can mix keyword, categorical, and semantic search
```

### Pros & Cons
- ✅ Explainable and debuggable
- ✅ Handles complex multi-item outfit requests
- ✅ Leverages existing categorical structure
- ❌ Two LLM calls add latency
- ❌ Rigid structure might miss nuanced requests

## Approach 2: Hybrid Embedding + Re-ranking

### Architecture
1. **Indexing**: Pre-compute embeddings from weighted field combinations
2. **Retrieval**: Fast vector similarity search
3. **Re-ranking**: LLM scores top candidates with full context

### Implementation with PostgreSQL + pgvector
```sql
-- Create embeddings table
CREATE TABLE product_embeddings (
    product_id INTEGER REFERENCES products(id),
    embedding_type TEXT, -- 'content', 'context', 'composite'
    embedding vector(1536)
);

-- Search query
SELECT p.*, pe.embedding <=> query_embedding::vector AS distance
FROM products p
JOIN product_embeddings pe ON p.id = pe.product_id
ORDER BY distance
LIMIT 50;
```

### Embedding Strategy
```python
# Weighted composite embeddings
title_emb = embed(title) * 0.35
category_emb = embed(categories) * 0.25
description_emb = embed(description) * 0.20
features_emb = embed(features) * 0.15
details_emb = embed(brand, material) * 0.05

final_embedding = normalize(sum([title_emb, category_emb, ...]))
```

### Pros & Cons
- ✅ Fast initial retrieval
- ✅ Handles ambiguous/poetic queries
- ✅ Single-stage for simple queries
- ❌ Embedding space might not capture all relationships
- ❌ Struggles with complementary items (outfit building)

## Approach 3: Graph-Enhanced Semantic Search

### Architecture
1. **Knowledge Graph**: Build relationships between products, occasions, styles
2. **Hybrid Search**: Combine graph traversal with embedding similarity
3. **Outfit Composition**: Use "bought_together" data for compatible items

### Implementation
```python
# Graph relationships
- Product → Occasions (beach, formal, casual)
- Product → Season (summer, winter)
- Product → Compatible Items (from bought_together)

# Query execution
def search_outfit(query):
    # 1. Identify key entities in query
    # 2. Traverse graph for related products
    # 3. Use embeddings for style matching
    # 4. Combine results into outfit
```

### Pros & Cons
- ✅ Naturally handles outfit composition
- ✅ Explainable recommendation paths
- ✅ Can incorporate business rules
- ❌ Complex to build and maintain
- ❌ Requires significant preprocessing

## Approach 4: Specialized Outfit Search (Hybrid Solution)

### Architecture
Combines the best of above approaches for outfit-specific queries:

```python
def outfit_search(query):
    # 1. Identify outfit components needed
    components = identify_components(query)  # {top: [...], bottom: [...], shoes: [...]}
    
    # 2. Search each component with context-aware embeddings
    results = {}
    for component_type, keywords in components.items():
        # Embed query with component context
        component_query = f"{query} {component_type}"
        
        # SQL with pgvector + category boosting
        sql = """
        SELECT p.*, 
               (embedding <=> %s::vector) * 
               CASE WHEN categories @> %s THEN 0.8 ELSE 1.0 END AS score
        FROM products p
        JOIN product_embeddings pe ON p.id = pe.product_id
        ORDER BY score
        LIMIT 5
        """
        results[component_type] = execute(sql, [query_embedding, keywords])
    
    # 3. Apply outfit coherence scoring
    return score_outfit_combinations(results)
```

## Recommended MVP Approach

For initial implementation, recommend **Approach 1 (Two-Stage LLM)** with elements of **Approach 2 (Embeddings)**:

1. **Start Simple**: Use OpenAI's text-embedding-3-small for all products
2. **Query Processing**: Use GPT-4 to parse intent and identify outfit components
3. **Storage**: PostgreSQL with pgvector extension
4. **Search**: Combine embedding similarity with SQL filters
5. **Progressive Enhancement**: Add re-ranking, fine-tuning, and graph relationships as needed

### Sample Implementation Flow
```python
# 1. User query
query = "I need an outfit to go to the beach this summer"

# 2. Parse with LLM
intent = {
    "occasion": "beach",
    "season": "summer",
    "types": ["swimwear", "coverup", "footwear", "accessories"]
}

# 3. Execute targeted searches
for item_type in intent["types"]:
    context_query = f"{intent['occasion']} {intent['season']} {item_type}"
    embedding = create_embedding(context_query)
    results = vector_search(embedding, filter={"category": item_type})

# 4. Return coherent outfit
```

## Key Considerations

### Embedding Model Choice
- **OpenAI text-embedding-3-small**: Good general performance, 1536 dimensions
- **sentence-transformers**: Open source alternative, 384-768 dimensions
- **Custom fine-tuned**: Best performance but requires training data

### Vector Database Options
- **PostgreSQL + pgvector**: Simple, unified database
- **Pinecone/Weaviate**: Dedicated vector DBs with advanced features
- **FAISS**: Fast, in-memory option for smaller datasets

### Performance Optimizations
- Pre-compute and cache embeddings
- Use IVFFlat or HNSW indexes for vector search
- Implement pagination and result limits
- Consider embedding dimensionality reduction for speed