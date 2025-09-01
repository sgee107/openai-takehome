# Project TODO: Embeddings Experimentation Framework

## Phase 0: Environment Setup
### Already Complete
- [x] Python virtual environment (.venv with Python 3.12)
- [x] PostgreSQL with pgvector extension (via Docker)
- [x] Base dependencies installed (FastAPI, SQLAlchemy, OpenAI, asyncpg, pgvector)
- [x] Environment configuration (.env file structure)
- [x] Docker Compose setup with pgvector/pgvector:pg16

### Required Setup Tasks - MVP Approach
- [ ] **Phase 0.1: Core Infrastructure**
  - [x] Add MinIO service to docker-compose-dev.yml (if not already there)
  - [x] Add Bitnami MLflow service to docker-compose-dev.yml with environment variables:
    - [ ] `MLFLOW_BACKEND_STORE_URI=postgresql://postgres:postgres@postgres:5432/chatdb`
    - [ ] `MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts/`
    - [ ] `MLFLOW_S3_ENDPOINT_URL=http://minio:9000`
    - [ ] `AWS_ACCESS_KEY_ID=minioadmin`
    - [ ] `AWS_SECRET_ACCESS_KEY=minioadmin`
  - [x] Update app settings.py with MLflow client configuration
  - [ ] Verify pgvector extension is enabled in database
  - [x] Add MinIO bucket auto-create in data loader CLI
- [ ] Install core dependencies:
  - [x] numpy for vector operations
  - [x] scikit-learn for PCA (reduce 1536D → 2D/3D for visualization)
  - [x] matplotlib/plotly for creating interactive plots
  - [x] UMAP-learn for embedding space visualization:
    - **Why UMAP?** Better than t-SNE for preserving global structure
    - **Visualizations it enables:**
      - 2D/3D scatter plots of your 300 products colored by category
      - Cluster identification (are shoes grouping together?)
      - Query-result paths (visualize search trajectory)
      - Before/after preprocessing comparisons
      - Outlier detection (products far from others)
- [ ] Add testing dependencies:
  - [x] pytest-benchmark for performance testing
  - [x] memory-profiler for memory usage tracking
- [ ] Configure OpenAI embedding model:
  - [ ] OpenAI text-embedding-3-small (1536 dims) - primary model
  - [ ] API key already configured in .env ✅
  - [ ] Add retry logic for API calls
- [ ] Set up experiment tracking with MLflow:
  - **Why MLflow?** Track experiments across different preprocessing strategies
  - **What it provides:**
    - [ ] Automatic logging of parameters (normalization, preprocessing type)
    - [ ] Metrics tracking over time (precision, recall, latency)
    - [ ] Model versioning (which embedding config was used)
    - [ ] Comparison UI (side-by-side metrics for normalized vs non-normalized)
    - [ ] Artifact storage (save embeddings, confusion matrices, plots)
    - [ ] Reproducibility (exact config to recreate any experiment)
  - **Dashboard gives you:**
    - Parallel coordinates plot comparing all experiments
    - Metric trends across preprocessing strategies
    - Best performing configuration at a glance
  - **Setup Steps:**
    - [x] Use official MLflow Docker image (ghcr.io/mlflow/mlflow:latest)
    - [x] Add MLflow service to docker-compose-dev.yml with official image configuration
    - [ ] Configure PostgreSQL as backend store (shares same DB as app)
    - [ ] **Use MinIO for artifact storage** (you already have it configured!):
      - [ ] Point MLflow artifacts to MinIO bucket instead of local volume
      - [ ] Benefits: Scalable, S3-compatible, web interface for browsing artifacts
      - [ ] Store: UMAP plots, embedding visualizations, model artifacts, confusion matrices
      - [ ] MLflow artifact URI: `s3://mlflow-artifacts/` (MinIO S3-compatible endpoint)
    - [x] Add MLflow client to Python dependencies: `mlflow`
      
    - [ ] **No manual configuration needed!** Bitnami image handles:
      - PostgreSQL backend connection
      - MinIO S3 artifact storage  
      - All MLflow server setup
    - [ ] Client connects to: `http://localhost:5001`

- [ ] **Phase 0.2: MVP Validation Script**
  - [ ] Create `scripts/validate_setup.py` that tests:
    - [ ] **PostgreSQL connection**: Can connect to database
    - [ ] **pgvector extension**: `CREATE EXTENSION IF NOT EXISTS vector`
    - [ ] **OpenAI API**: Test embedding generation with sample text
    - [ ] **MinIO connection**: Can create bucket and upload test file
    - [ ] **MLflow tracking**: Can create experiment and log test metrics
    - [ ] **Full pipeline test**: Load 1 product → generate embedding → store in DB → log to MLflow
  - [ ] Script outputs: ✅ PASS / ❌ FAIL for each component
  - [ ] **Run with**: `python scripts/validate_setup.py`

- [ ] **Phase 0.3: Baseline Data Pipeline**
  - [ ] Create `scripts/baseline_test.py` that:
    - [ ] Loads first 5 products from JSON data
    - [ ] Tests both simple concatenation and one enrichment strategy
    - [ ] Generates embeddings (normalized and non-normalized)
    - [ ] Stores in database with different table names
    - [ ] Creates MLflow experiment with basic metrics
    - [ ] Generates simple UMAP plot and logs as artifact
  - [ ] **Success criteria**: 5 products processed, 2 embedding strategies, 1 UMAP plot in MLflow
  - **UMAP Integration with MLflow:**
    - [ ] Generate UMAP plots as matplotlib/plotly artifacts
    - [ ] Log plots as MLflow artifacts (PNG/HTML files)
    - [ ] Tag experiments with preprocessing strategy for filtering
    - [ ] Create custom metrics for cluster quality
  - **Preprocessing Tagging Strategy:**
    - [ ] `preprocessing_strategy`: "simple_concat" | "image_enriched" | "llm_expanded" 
    - [ ] `normalization`: "l2_normalized" | "raw"
    - [ ] `fields_used`: "title_desc_features" | "title_only" | etc.
    - [ ] `image_analysis`: true/false
    - [ ] `llm_expansion`: true/false
    - [ ] `embedding_model`: "text-embedding-3-small"
  - **Dependencies Note:**
    - MLflow has built-in plotting but matplotlib/plotly still needed for UMAP
    - MLflow can display custom plots but you create them with matplotlib/plotly
    - Both libraries work together: you generate → MLflow stores & displays
- [ ] Initialize database schema variations:
  - [ ] Create migration for multiple embedding table schemas
  - [ ] Add indexes for vector similarity search

## Phase 1: Data Loading Infrastructure
### Data Source
- Working with 300 Amazon Fashion product samples from `/data` directory
- Each item contains: title, description, features, price, ratings, images metadata
- Multiple image URLs per product (main, variants, different angles)
- Rich metadata: ratings, price, category that could inform embeddings

### Core Components
- [ ] Implement data loader for Amazon Fashion JSON data
- [ ] Create text preprocessing pipeline:
  - [ ] Concatenate relevant text fields (title, description, features)
  - [ ] Handle missing/empty fields gracefully
  - [ ] Create consistent text format for embedding
- [ ] Command-line interface with parameters:
  - [ ] `--table-name`: Target table name for experiments
  - [ ] `--embedding-model`: "text-embedding-3-small" (default)
  - [ ] `--normalize`: Boolean flag for L2 normalization
  - [ ] `--batch-size`: Control embedding batch processing
  - [ ] `--preprocessing-strategy`: Choose enrichment approach

### Agentic Preprocessing Strategies
- [ ] **Image-Enriched Embeddings**
  - [ ] Use GPT-4V or similar to analyze product images
  - [ ] Extract visual attributes not in metadata:
    - [ ] Color patterns, textures, style elements
    - [ ] Fit type (slim, regular, loose)
    - [ ] Material appearance (shiny, matte, textured)
    - [ ] Design details (buttons, zippers, patterns)
  - [ ] Generate descriptive text from images
  - [ ] Combine with existing metadata for richer embeddings
  - [ ] Store image descriptions separately for analysis

- [ ] **Context-Aware Text Generation**
  - [ ] Use LLM to expand product descriptions:
    - [ ] Input: "YUEDGE 5 Pairs Men's Moisture Control Cushioned Dry Fit Casual Athletic Crew Socks"
    - [ ] Output: Enriched description with use cases, occasions, similar items
  - [ ] Generate category-specific attributes:
    - [ ] Shoes: comfort level, terrain suitability, formality
    - [ ] Clothing: season, occasion, style category
  - [ ] Create structured attributes from unstructured text

- [ ] **Metadata Weighting Strategies**
  - [ ] **Rating-weighted embeddings**: Boost popular items
    ```
    text = title + description
    if rating > 4.5:
        text = text + " highly rated popular choice"
    ```
  - [ ] **Price-tier embeddings**: Include price context
    ```
    text = f"{title} {price_tier} {description}"
    price_tier = "budget" | "mid-range" | "premium"
    ```
  - [ ] **Category-hierarchical embeddings**:
    ```
    text = f"{main_category} > {sub_category} > {title}"
    ```

### Embedding Combination Experiments
- [ ] **Multi-field Strategies**:
  1. **Simple Concatenation**: `title + " " + description + " " + features`
  2. **Weighted Fields**: `title*2 + description + features*0.5`
  3. **Structured Format**: `"Product: {title}\nDetails: {description}\nFeatures: {features}"`
  4. **Image-Enriched**: `title + description + image_analysis`
  
- [ ] **Hybrid Embeddings**:
  - [ ] Text embedding from metadata
  - [ ] Visual embedding from product images (CLIP)
  - [ ] Combined embedding: weighted average or concatenation
  - [ ] Store both for A/B testing retrieval quality

### Embedding Variations (Initial Focus)
- [ ] **Normalized embeddings table**: L2-normalized vectors for cosine similarity
  - Table: `products_embeddings_normalized`
  - Use case: When magnitude doesn't matter, only direction
  - Benefits: Faster cosine similarity, consistent scale
- [ ] **Non-normalized embeddings table**: Raw OpenAI embeddings
  - Table: `products_embeddings_raw`  
  - Use case: When embedding magnitude carries semantic meaning
  - Benefits: Preserves original information, flexible distance metrics

### Storage Schema
- [ ] Base table structure:
  ```sql
  - id (UUID)
  - product_id (original item identifier)
  - title (TEXT)
  - full_text (TEXT - concatenated fields)
  - enriched_text (TEXT - with image/LLM additions)
  - embedding (VECTOR(1536))
  - embedding_model (VARCHAR)
  - preprocessing_strategy (VARCHAR)
  - is_normalized (BOOLEAN)
  - created_at (TIMESTAMP)
  - metadata (JSONB - store original item data)
  - image_analysis (JSONB - store vision model outputs)
  - processing_metadata (JSONB - track enrichment methods used)
  ```
- [ ] Create indexes:
  - [ ] Vector similarity index (ivfflat or hnsw)
  - [ ] B-tree on product_id for lookups
  - [ ] Hash index on embedding_model

## Phase 2: Testing Infrastructure
### Key Metrics & Their Importance

#### 1. **Retrieval Quality Metrics**
- [ ] **Precision@K** (K=1,5,10)
  - *Why*: Measures relevance of top results
  - *Decision Impact*: Low precision → users see irrelevant results → need better embeddings
  - *Test*: Create query-result pairs based on product categories/features
  
- [ ] **Recall@K** (K=10,20,50)
  - *Why*: Measures completeness of relevant results found
  - *Decision Impact*: Low recall → missing relevant products → adjust embedding model or normalization
  - *Test*: Use product variants/similar items as ground truth

- [ ] **Mean Reciprocal Rank (MRR)**
  - *Why*: Emphasizes position of first relevant result
  - *Decision Impact*: Critical for user experience - first result matters most
  - *Test*: Fashion queries with known best matches

#### 2. **Performance Metrics**
- [ ] **Query Latency** (p50, p95, p99)
  - *Why*: User experience depends on speed
  - *Decision Impact*: >100ms p95 → need optimization (index type, vector size)
  - *Test*: Benchmark with concurrent queries
  
- [ ] **Index Build Time**
  - *Why*: Affects deployment and update frequency
  - *Decision Impact*: Slow builds → can't update catalog frequently
  - *Comparison*: IVFFlat vs HNSW indexes

- [ ] **Memory Footprint**
  - *Why*: Determines infrastructure costs
  - *Decision Impact*: High memory → expensive scaling
  - *Test*: Measure with different index types and parameters

#### 3. **Semantic Quality Metrics**
- [ ] **Intra-category Similarity**
  - *Why*: Similar products should cluster together
  - *Decision Impact*: Low clustering → poor semantic understanding
  - *Test*: Average cosine similarity within product categories
  
- [ ] **Inter-category Distinction**
  - *Why*: Different categories should be separable
  - *Decision Impact*: Poor separation → confused search results
  - *Test*: Silhouette score across categories

- [ ] **Neighborhood Consistency**
  - *Why*: K-nearest neighbors should make semantic sense
  - *Decision Impact*: Inconsistent neighbors → unpredictable search
  - *Test*: Manual review of top-5 neighbors for sample products

### Normalized vs Non-normalized Comparison Tests
- [ ] **Cosine vs Euclidean Distance Performance**
  - Normalized: Optimized for cosine (dot product with unit vectors)
  - Non-normalized: Flexible distance metrics
  - Test: Compare retrieval quality with both metrics
  
- [ ] **Magnitude Sensitivity Analysis**
  - Test if embedding magnitude correlates with:
    - Product popularity/ratings
    - Text length
    - Semantic importance
  - Decision: If magnitude meaningful → use non-normalized

- [ ] **Storage & Computation Trade-offs**
  - Normalized: Faster dot product operations
  - Non-normalized: Preserves all information
  - Test: Benchmark query performance difference

## Phase 3: Semantic Search Testing
### Test Query Categories for Fashion Domain
- [ ] **Attribute-based Queries**
  - "Blue cotton shirts for summer"
  - "Waterproof hiking boots size 10"
  - Test: Precision of attribute matching
  
- [ ] **Style/Occasion Queries**
  - "Business casual outfit pieces"
  - "Beach vacation essentials"
  - Test: Semantic understanding beyond keywords
  
- [ ] **Comparison Queries**
  - "Alternatives to [specific product]"
  - "Similar to Nike but cheaper"
  - Test: Neighborhood quality in embedding space
  
- [ ] **Negative Queries**
  - "Shirts but not polo"
  - "Shoes excluding heels"
  - Test: Embedding space navigation

### Search Evaluation Framework
- [ ] Create golden dataset:
  - [ ] 50 diverse fashion queries
  - [ ] Human-annotated relevant products (1-5 scale)
  - [ ] Expected failure cases (out-of-domain queries)
  
- [ ] A/B Testing Protocol:
  - [ ] Route 50% traffic to normalized embeddings
  - [ ] Route 50% traffic to non-normalized embeddings
  - [ ] Measure: CTR, dwell time, conversion
  
- [ ] Query Expansion Testing:
  - [ ] Test with exact query
  - [ ] Test with synonyms/related terms
  - [ ] Measure improvement in recall

### Decision Framework for Embedding Strategies

#### **When to use Normalized Embeddings**:
- Cosine similarity is primary metric
- Consistent scale needed across queries
- Speed is critical (dot product optimization)
- All products should be treated equally regardless of text length

#### **When to use Non-normalized Embeddings**:
- Multiple distance metrics needed
- Magnitude carries semantic weight
- Flexibility for post-processing
- Embedding magnitude correlates with importance/relevance

#### **When to use Image-Enriched Embeddings**:
- Visual attributes are critical for retrieval (fashion, home decor)
- Text metadata is sparse or generic
- Users search by visual concepts ("bohemian style", "minimalist")
- Worth the extra processing cost for accuracy gains

#### **When to use LLM-Expanded Embeddings**:
- Products have minimal descriptions
- Need to capture implicit attributes (occasions, use cases)
- Building conversational search experiences
- Category-specific expertise needed

### Metrics Decision Guide

#### **If Precision@K is Low (<0.7)**:
- Problem: Top results aren't relevant
- Solutions:
  - Try image-enriched embeddings for better context
  - Adjust preprocessing to emphasize key fields
  - Consider different embedding model

#### **If Recall@K is Low (<0.5)**:
- Problem: Missing relevant products
- Solutions:
  - Use LLM expansion to add synonyms/related terms
  - Try non-normalized to preserve magnitude information
  - Reduce embedding dimensions for broader matching

#### **If Query Latency >100ms (p95)**:
- Problem: Too slow for production
- Solutions:
  - Use normalized embeddings (faster dot product)
  - Switch from HNSW to IVFFlat index
  - Reduce vector dimensions
  - Add caching layer

#### **If Neighborhood Consistency is Poor**:
- Problem: Similar products not clustering
- Solutions:
  - Add image analysis for visual similarity
  - Use structured format for consistent parsing
  - Weight important fields more heavily

## Phase 4: Frontend & Visualization
### Baseline Interface
- [ ] Design minimal frontend architecture
- [ ] Implement search interface
- [ ] Create visualization components:
  - [ ] Embedding space visualizer (t-SNE/UMAP)
  - [ ] Search result comparisons
  - [ ] Performance metrics dashboard
  - [ ] A/B test results viewer
- [ ] Add interactive testing capabilities
- [ ] Implement human-in-the-loop evaluation tools

## Artifacts to Generate
### Documentation
- [ ] System architecture diagram
- [ ] Data flow diagrams
- [ ] API documentation
- [ ] Testing methodology document

### Code Artifacts
- [ ] Configuration templates
- [ ] Example queries and expected results
- [ ] Performance baseline reports
- [ ] Deployment scripts

## Success Criteria
- [ ] Define measurable goals for each phase
- [ ] Establish performance benchmarks
- [ ] Create evaluation rubrics
- [ ] Set up continuous monitoring

## Notes & Considerations
- Consider versioning strategy for experiments
- Plan for scalability from the start
- Keep modularity for easy component swapping
- Maintain reproducibility of all experiments
