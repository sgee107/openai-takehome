# Project TODO: Embeddings Experimentation Framework

## Phase 0: Environment Setup
### Already Complete
- [x] Python virtual environment (.venv with Python 3.12)
- [x] PostgreSQL with pgvector extension (via Docker)
- [x] Base dependencies installed (FastAPI, SQLAlchemy, OpenAI, asyncpg, pgvector)
- [x] Environment configuration (.env file structure)
- [x] Docker Compose setup with pgvector/pgvector:pg16

### Required Setup Tasks - MVP Approach
- [x] **Phase 0.1: Core Infrastructure** ✅ MOSTLY COMPLETE
  - [x] Add MinIO service to docker-compose-dev.yml 
  - [x] MLflow setup - using simplified approach:
    - [x] MLflow already added as dependency in pyproject.toml
    - [x] Created MLflow client (app/mlflow_client.py)
    - [x] Verified MLflow UI works (runs on http://127.0.0.1:5000)
    - [ ] Add MLflow service to docker-compose-dev.yml (optional for production)
  - [x] Update app settings.py with MLflow client configuration
  - [x] ⚠️ Verify pgvector extension is enabled in database (CREATE EXTENSION IF NOT EXISTS vector) ✅ CONFIRMED WORKING
  - [x] Data loader creates MinIO bucket if needed
- [x] Install core dependencies: ✅ COMPLETE
  - [x] numpy for vector operations (in use)
  - [x] scikit-learn for PCA (imported in semantic metrics)
  - [x] matplotlib for plots (used in distribution.py)
  - [ ] ~~UMAP-learn for embedding space visualization (not yet installed)~~ **[DEPRECATED]**
- [x] Add testing dependencies:
  - [x] pytest-benchmark for performance testing
  - [x] memory-profiler for memory usage tracking
- [x] Configure OpenAI embedding model: ✅ COMPLETE
  - [x] OpenAI text-embedding-3-small (1536 dims) implemented
  - [x] API key configured in .env
  - [ ] ~~Add retry logic for API calls (nice to have)~~ **[DEPRECATED]**
- [x] Set up experiment tracking with MLflow:
  - **Why MLflow?** Track experiments across different preprocessing strategies
  - **What it provides:**
    - [x] Automatic logging of parameters (normalization, preprocessing type)
    - [x] Metrics tracking over time (precision, recall, latency)
    - [x] Model versioning (which embedding config was used)
    - [x] Comparison UI (side-by-side metrics for normalized vs non-normalized)
    - [x] Artifact storage (save embeddings, confusion matrices, plots)
    - [x] Reproducibility (exact config to recreate any experiment)
  - **Dashboard gives you:**
    - Parallel coordinates plot comparing all experiments
    - Metric trends across preprocessing strategies
    - Best performing configuration at a glance
  - **Simplified Setup (Working):**
    - [x] MLflow installed via uv as dependency
    - [x] Local tracking with file store (./mlruns)
    - [x] UI accessible at http://127.0.0.1:5000 via `mlflow ui`
    - [x] Verified with test script (scripts/test_mlflow.py)
  - **Optional Docker Setup:**
      - [ ] Point MLflow artifacts to MinIO bucket instead of local volume
      - [ ] Benefits: Scalable, S3-compatible, web interface for browsing artifacts
      - [ ] Store: UMAP plots, embedding visualizations, model artifacts, confusion matrices
      - [ ] MLflow artifact URI: `s3://mlflow-artifacts/` (MinIO S3-compatible endpoint)
    - [x] Add MLflow client to Python dependencies: `mlflow`
      
  - **Current Status:** ✅ MLflow working locally with file-based tracking

- [ ] ~~**Phase 0.2: MVP Validation Script**~~ **[DEPRECATED - Setup already validated through working experiments]**

- [x] **Phase 0.3: Baseline Data Pipeline** ✅ FUNCTIONALITY EXISTS
  - **Note**: Instead of `baseline_test.py`, we have comprehensive implementations:
    - [x] `scripts/data_loader.py`: Full data loader with 5 embedding strategies
    - [x] `scripts/load_products.py`: Product loader with TextStrategy class
    - [x] Multiple text strategies already implemented:
      - title_only
      - title_features  
      - title_category_store
      - title_details
      - comprehensive (all_text)
    - [x] Database schema with pgvector support
    - [x] HNSW index for similarity search
  - [ ] ⚠️ Still need to add:
    - [ ] ~~L2 normalization option for embeddings~~ **[DEPRECATED - Not needed]**
    - [ ] UMAP visualization generation **[DEPRECATED - Not needed]**
    - [x] Integration with MLflow experiments for baseline metrics

## Phase 0.4: Embedding Strategy Experiments ✅ MOSTLY COMPLETE

### Current Implementation Status
The experiments module is already implemented with most core functionality:

```
experiments/
├── __init__.py ✅
├── base.py ✅                   # Base experiment class with MLflow integration
├── run_experiments.py ✅         # Main CLI to run experiments
├── runners/
│   ├── __init__.py ✅
│   ├── embedding_strategy.py ✅  # Strategy comparison experiment (IMPLEMENTED)
│   ├── normalization.py ❌       # [DEPRECATED - Embedding model handles normalization]
│   └── retrieval_quality.py ❌   # Future: Search quality tests
├── metrics/
│   ├── __init__.py ✅
│   ├── semantic.py ✅           # Semantic quality metrics (IMPLEMENTED)
│   ├── performance.py ❌        # [DEPRECATED]
│   └── storage.py ❌            # [DEPRECATED]
├── visualizations/
│   ├── __init__.py ✅
│   ├── distribution.py ✅       # Text length distributions (IMPLEMENTED)
│   ├── umap_plots.py ❌         # [DEPRECATED - UMAP incompatible with Python >3.10]
│   └── comparison.py ❌         # [DEPRECATED - MLflow provides comparison]
└── debug.py ✅                  # Debug utilities
```

### What's Already Working:
- ✅ CLI interface: `python -m app.experiments.run_experiments --experiment strategy`
- ✅ Compares all 5 text strategies with MLflow tracking
- ✅ Generates embeddings and saves to database (optional)
- ✅ Tracks metrics: time, text length, failures
- ✅ Creates text length distribution visualizations
- ✅ Calculates semantic quality metrics (clustering, similarity)
- ✅ Saves artifacts to MLflow

### What's Still Needed:
- [x] Token count analysis using tiktoken ✅ IMPLEMENTED
- [ ] ~~UMAP visualizations of embedding space~~ **[DEPRECATED - UMAP package incompatible with Python >3.10]**
- [ ] ~~Performance profiling (memory, latency)~~ **[DEPRECATED]**
- [ ] ~~L2 normalization experiments~~ **[DEPRECATED - Embedding model handles normalization]**
- [ ] ~~Storage efficiency analysis~~ **[DEPRECATED]**

### Implementation Plan

#### **1. Base Experiment Class (experiments/base.py)**
```python
class BaseExperiment:
    """Base class for all experiments with MLflow tracking"""
    
    def __init__(self, experiment_name: str, tracking_uri: str = None):
        self.mlflow_client = mlflow_client
        self.experiment_name = experiment_name
        
    async def setup(self):
        """Setup experiment, create MLflow experiment"""
        
    async def run(self):
        """Main experiment logic - override in subclasses"""
        
    async def log_metrics(self, metrics: Dict):
        """Log metrics to MLflow"""
        
    async def log_artifacts(self, artifacts: Dict):
        """Save and log artifacts"""
        
    async def cleanup(self):
        """Cleanup after experiment"""
```

#### **2. Embedding Strategy Experiment (experiments/runners/embedding_strategy.py)**
```python
class EmbeddingStrategyExperiment(BaseExperiment):
    """Compare different text strategies for embeddings"""
    
    def __init__(self, strategies: List[str], num_products: int = 50):
        super().__init__("embedding_strategy_comparison")
        self.strategies = strategies
        self.num_products = num_products
        
    async def run_single_strategy(self, strategy: str, products: List[Dict]):
        """Run experiment for one strategy"""
        # Generate embeddings
        # Track metrics
        # Create visualizations
        
    async def compare_strategies(self):
        """Run all strategies and generate comparison"""
        # Run each strategy
        # Generate comparison plots
        # Create summary report
```

#### **3. Metrics Modules (experiments/metrics/)**
- **semantic.py**: Intra-category similarity, clustering quality, nearest neighbor accuracy
- **performance.py**: Embedding generation time, query latency, batch processing speed
- **storage.py**: Database size, index size, memory usage

#### **4. Visualization Utilities (experiments/visualizations/)**
- **umap_plots.py**: Generate UMAP plots colored by category, strategy
- **distribution.py**: 
  - [ ] **Token Count Histograms**:
    - [ ] Create histograms for each embedding strategy showing token distribution
    - [ ] Compare token counts across strategies (title_only vs comprehensive)
    - [ ] Identify optimal token ranges (sweet spot: 100-500 tokens)
    - [ ] Flag products with extremely low (<50) or high (>1000) token counts
  - [ ] **Text Length Analysis**:
    - [ ] Character count distributions
    - [ ] Word count distributions
    - [ ] Correlation between text length and embedding quality
  - [ ] **Token Count Estimation**:
    ```python
    def estimate_tokens(text: str) -> int:
        # Rough estimation: 1 token ≈ 0.75 words
        word_count = len(text.split())
        return int(word_count / 0.75)
    
    def create_token_histogram(products: List[Dict], strategy: str):
        token_counts = []
        for product in products:
            text = generate_text_for_strategy(product, strategy)
            tokens = estimate_tokens(text)
            token_counts.append(tokens)
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(token_counts, bins=50, alpha=0.7)
        plt.axvline(x=100, color='r', linestyle='--', label='Min optimal')
        plt.axvline(x=500, color='r', linestyle='--', label='Max optimal')
        plt.xlabel('Estimated Token Count')
        plt.ylabel('Number of Products')
        plt.title(f'Token Distribution for {strategy} Strategy')
        plt.legend()
        
        # Log to MLflow
        mlflow.log_figure(plt.gcf(), f"token_histogram_{strategy}.png")
    ```
- **comparison.py**: Side-by-side strategy comparisons, radar charts

#### **5. Main Runner Script (experiments/run_experiments.py)**
```python
# CLI interface
import click

@click.command()
@click.option('--experiment', type=click.Choice(['strategy', 'normalization', 'retrieval']))
@click.option('--num-products', default=50)
@click.option('--strategies', multiple=True)
async def run_experiment(experiment, num_products, strategies):
    """Run embedding experiments with MLflow tracking"""
    
    if experiment == 'strategy':
        exp = EmbeddingStrategyExperiment(strategies, num_products)
        await exp.run()
        
# Usage: python experiments/run_experiments.py --experiment strategy --num-products 50 --strategies title_only --strategies comprehensive
```

### Available Embedding Strategies to Test
1. **title_only** - Baseline, minimal text
2. **title_features** - Title + product features  
3. **title_category_store** - Title + category context + brand
4. **title_details** - Title + key product details (Brand, Material, Style, etc.)
5. **comprehensive** - Everything: title + brand + category + features + details + description

### Remaining Implementation Tasks

- [x] **Step 1-2: Module Structure & Base Framework** ✅ COMPLETE
  - Module structure created
  - BaseExperiment class implemented with MLflow integration
  - CLI interface working

- [x] **Step 3: Embedding Strategy Experiment** ✅ COMPLETE
  - Loads products (default 300, configurable)
  - Generates embeddings for all strategies
  - Tracks metrics and saves to MLflow
  - Token count analysis with tiktoken implemented

- [x] **Step 4: Semantic Quality Metrics** ✅ COMPLETE
  - Cosine similarity calculations implemented
  - Intra-category clustering metrics working
  - Missing: UMAP visualizations

- [ ] ~~**Step 5: Performance & Storage Analysis**~~ **[DEPRECATED]**

- [x] **Step 6: Enhanced Features** ✅ COMPLETE
  - [x] Can run experiments with: `python -m app.experiments.run_experiments --experiment strategy`
  - [x] Token count analysis implemented with tiktoken
  - [ ] ~~Add UMAP visualizations~~ **[DEPRECATED]**
  - [ ] ~~Create comprehensive comparison dashboard~~ **[DEPRECATED - MLflow provides this]**

### Next Priority Tasks:
1. [x] ~~Install tiktoken and add token count analysis to embedding_strategy.py~~ ✅ ALREADY DONE
2. [ ] ~~Install umap-learn and create UMAP visualizations~~ **[DEPRECATED]**
3. [ ] ~~Implement performance.py and storage.py modules~~ **[DEPRECATED]**
4. [ ] ~~Add L2 normalization option to experiments~~ **[DEPRECATED]**
5. [ ] ~~Create validate_setup.py script for Phase 0.2~~ **[DEPRECATED]**

### Success Metrics
- [x] All 5 strategies tested on same product set ✅
- [x] MLflow tracking captures all metrics ✅
- [x] Clear performance comparison table ✅
- [ ] ~~UMAP visualizations showing clustering quality~~ **[DEPRECATED]**
- [ ] Recommendation on which strategy to use when

### Next Steps After This Phase
- [ ] ~~Add L2 normalization comparison~~ **[DEPRECATED]**
- [x] Test with larger dataset (all 300 products) ✅
- [ ] Implement image enrichment (Phase 1)
- [ ] Add retrieval quality metrics with test queries

## Phase 1: Data Loading Infrastructure ✅ COMPLETE
### Data Source
- Working with 300 Amazon Fashion product samples from `/data` directory
- Each item contains: title, description, features, price, ratings, images metadata
- Multiple image URLs per product (main, variants, different angles)
- Rich metadata: ratings, price, category that could inform embeddings

### Core Components
- [x] Implement data loader for Amazon Fashion JSON data ✅ COMPLETE
  - **Note: Duplication exists** - Two implementations:
    - `data_loader.py` - Direct asyncpg implementation
    - `load_products.py` - SQLAlchemy implementation (recommended)
- [x] Create text preprocessing pipeline: ✅ COMPLETE
  - [x] Concatenate relevant text fields (title, description, features)
  - [x] Handle missing/empty fields gracefully
  - [x] Create consistent text format for embedding
  - [x] 9 strategies implemented: 
    - Original: title_only, title_features, title_category_store, title_details, comprehensive
    - Key-Value: key_value_basic, key_value_detailed, key_value_with_images, key_value_comprehensive
- [ ] ~~Command-line interface with parameters:~~ **[DEPRECATED - Scripts work well without CLI params]**
  - [ ] ~~`--table-name`: Target table name for experiments~~
  - [ ] ~~`--embedding-model`: "text-embedding-3-small" (default)~~
  - [ ] ~~`--normalize`: Boolean flag for L2 normalization~~
  - [ ] ~~`--batch-size`: Control embedding batch processing~~
  - [ ] ~~`--preprocessing-strategy`: Choose enrichment approach~~

### Explicit Key-Value Text Generation Strategy

#### **Core Strategy: Structured Key-Value Format**
Transform unstructured product data into explicit, searchable key-value pairs that enhance embedding quality.

##### **1. Base Product Information Template**
```python
def generate_structured_text(product):
    """Generate explicit key-value text for embeddings"""
    
    # Core product identity
    text_parts = [
        f"Product: {product['title']}",
        f"Category: {product['main_category']}",
        f"Store: {product.get('store', 'Unknown Brand')}",
        f"ASIN: {product['parent_asin']}"
    ]
    
    # Price information (only 13.7% have this)
    if product.get('price'):
        text_parts.append(f"Price: ${product['price']}")
        # Add price tier
        if product['price'] < 20:
            text_parts.append("Price Tier: Budget")
        elif product['price'] < 50:
            text_parts.append("Price Tier: Mid-range")
        else:
            text_parts.append("Price Tier: Premium")
    
    return " | ".join(text_parts)
```

##### **2. Details Field Extraction Strategy**
The `details` field (100% coverage) contains rich structured data that needs extraction:

```python
def extract_product_details(details: dict) -> List[str]:
    """Extract key-value pairs from details JSON"""
    
    key_mappings = {
        'Brand': 'Brand',
        'Color': 'Color',
        'Material': 'Material',
        'Style': 'Style',
        'Department': 'Department',
        'Closure Type': 'Closure',
        'Country of Origin': 'Made in',
        'Age Range (Description)': 'Age Group',
        'Item Weight': 'Weight',
        'Package Dimensions': 'Size'
    }
    
    extracted = []
    for original_key, display_key in key_mappings.items():
        if original_key in details:
            value = details[original_key]
            # Clean up the value (remove extra metadata)
            if isinstance(value, str):
                value = value.replace(' ‏ : ‎ ', '').strip()
            extracted.append(f"{display_key}: {value}")
    
    return extracted
```

##### **3. Features List Processing**
Features are lists that need to be parsed and categorized:

```python
def process_features(features: List[str]) -> List[str]:
    """Convert feature list into categorized key-value pairs"""
    
    processed = []
    feature_patterns = {
        'material': ['cotton', 'polyester', 'wool', 'leather', 'synthetic'],
        'size': ['fits', 'sizing', 'length', 'width'],
        'care': ['wash', 'dry clean', 'iron', 'bleach'],
        'occasion': ['casual', 'formal', 'sport', 'work', 'party'],
        'season': ['summer', 'winter', 'spring', 'fall', 'all-season']
    }
    
    for feature in features:
        feature_lower = feature.lower()
        
        # Categorize the feature
        for category, keywords in feature_patterns.items():
            if any(keyword in feature_lower for keyword in keywords):
                processed.append(f"Feature-{category}: {feature}")
                break
        else:
            # Generic feature if no category matched
            processed.append(f"Feature: {feature}")
            
    return processed
```

##### **4. Category Hierarchy Extraction**
Categories field contains hierarchical information:

```python
def extract_category_hierarchy(categories: List[List[str]]) -> str:
    """Extract and format category hierarchy"""
    
    if not categories:
        return ""
        
    # Take the most specific (longest) category path
    longest_path = max(categories, key=len) if categories else []
    
    if longest_path:
        # Create hierarchical representation
        hierarchy = " > ".join(longest_path)
        # Also create individual category tags
        tags = [f"Category-Level-{i}: {cat}" for i, cat in enumerate(longest_path)]
        
        return f"Category-Hierarchy: {hierarchy} | " + " | ".join(tags)
    
    return ""
```

##### **5. Image Analysis Integration Plan**
For handling images alongside text:

```python
def plan_image_enrichment(product_images: List[dict]) -> List[str]:
    """Plan for image analysis to enrich text embeddings"""
    
    enrichment_tasks = []
    
    # Identify what to extract from images
    for idx, image in enumerate(product_images):
        if image.get('variant'):
            enrichment_tasks.append(f"Image-{idx}-Variant: {image['variant']}")
        
        # Plan for vision model analysis
        enrichment_tasks.extend([
            f"Image-{idx}-Analysis: [Extract colors, patterns, style]",
            f"Image-{idx}-Details: [Identify buttons, zippers, logos]",
            f"Image-{idx}-Material: [Detect fabric texture, finish]"
        ])
    
    return enrichment_tasks
```

#### **Complete Text Generation Pipeline**

```python
class StructuredTextGenerator:
    """Generate structured key-value text for embeddings"""
    
    def __init__(self, include_images: bool = False):
        self.include_images = include_images
        
    def generate(self, product: dict) -> str:
        """Generate complete structured text"""
        
        text_parts = []
        
        # 1. Base product info
        text_parts.append(self.generate_base_info(product))
        
        # 2. Extract from details (100% coverage)
        if product.get('details'):
            details_text = extract_product_details(product['details'])
            text_parts.extend(details_text)
        
        # 3. Process features (53% coverage when present)
        if product.get('features'):
            features_text = process_features(product['features'])
            text_parts.extend(features_text)
        
        # 4. Category hierarchy
        if product.get('categories'):
            cat_text = extract_category_hierarchy(product['categories'])
            if cat_text:
                text_parts.append(cat_text)
        
        # 5. Description (only 7% have this)
        if product.get('description'):
            # Limit description length to avoid token explosion
            desc_text = ' '.join(product['description'][:2])  # First 2 sentences
            text_parts.append(f"Description: {desc_text[:200]}...")
        
        # 6. Related products
        if product.get('bought_together'):
            text_parts.append(f"Often-Bought-With: {product['bought_together']}")
        
        # 7. Image enrichment placeholders
        if self.include_images and product.get('images'):
            image_tasks = plan_image_enrichment(product['images'])
            text_parts.extend(image_tasks)
        
        # Join with delimiter for clarity
        return " | ".join(text_parts)
```

#### **Implementation Steps**

1. **Create Text Strategy Classes** ✅ COMPLETE
   - [x] `BaseTextStrategy`: Abstract base class
   - [x] `KeyValueStrategy`: Implements structured key-value generation
   - [x] `KeyValueWithImagesStrategy`: Adds image analysis placeholders
   - [x] `ComprehensiveKeyValueStrategy`: Maximum extraction with all fields

2. **Add to Existing Strategies** ✅ COMPLETE
   - [x] Update `app/scripts/load_products.py` to include new strategies:
     - [x] `key_value_basic`: Core key-value pairs only
     - [x] `key_value_detailed`: Include all details extraction
     - [x] `key_value_with_images`: Add image enrichment plans
     - [x] `key_value_comprehensive`: Maximum field extraction

3. **Token Optimization** ✅ COMPLETE
   - [x] Set max tokens per field to avoid explosion
   - [x] Prioritize fields by importance:
     1. Title, Category, Brand (essential)
     2. Key details (Color, Material, Style)
     3. Features (categorized)
     4. Price information
     5. Description (truncated)

4. **Validation and Testing** ✅ COMPLETE
   - [x] Generate sample outputs for review
   - [x] Check token counts stay within limits (target: 200-500 tokens)
     - key_value_basic: avg 76 tokens ✓
     - key_value_detailed: avg 136 tokens ✓
     - key_value_with_images: avg 251 tokens ✓
     - key_value_comprehensive: avg 182 tokens ✓
   - [x] Ensure consistent formatting across products
   - [x] Test with products missing various fields

#### **Expected Output Example**

```
Product: Men's Cotton Blend Polo Shirt | Category: AMAZON FASHION | Store: Tommy Hilfiger | ASIN: B07XYZ123 | Brand: Tommy Hilfiger | Color: Navy Blue | Material: 60% Cotton, 40% Polyester | Style: Classic Fit | Department: Mens | Closure: Button | Feature-material: Soft cotton-polyester blend for comfort | Feature-care: Machine washable, tumble dry low | Feature-occasion: Perfect for casual and semi-formal occasions | Category-Hierarchy: Clothing > Men > Shirts > Polos | Category-Level-0: Clothing | Category-Level-1: Men | Category-Level-2: Shirts | Category-Level-3: Polos | Description: Classic polo shirt with signature logo embroidery. Comfortable fit suitable for...
```

#### **Benefits of This Approach**

1. **Explicit Search Terms**: Each key-value pair becomes a searchable term
2. **Consistent Structure**: Same format across all products aids embedding quality
3. **Missing Data Handling**: Graceful degradation when fields are absent
4. **Semantic Clarity**: Clear relationships between attributes and values
5. **Token Efficiency**: Structured format reduces redundancy
6. **Future Image Integration**: Placeholders ready for vision model outputs

### Agentic Preprocessing Strategies
- [ ] **Context and Synonym Injection**
  - [ ] **Structured Attribute Expansion**:
    - [ ] Transform: "Blue Cotton T-Shirt" → "Blue Cotton T-Shirt. Color: blue. Material: cotton. Type: t-shirt, casual wear, top, clothing"
    - [ ] Add hierarchical categories: "Clothing > Men's > Tops > T-Shirts > Casual"
    - [ ] Include style descriptors: "casual, everyday, comfortable, relaxed fit"
  - [ ] **Synonym Enrichment**:
    - [ ] Build domain-specific synonym mappings:
      - [ ] "Jeans" → "jeans, denim, pants, trousers, bottoms, denims"
      - [ ] "Sneakers" → "sneakers, trainers, athletic shoes, sports shoes, tennis shoes, kicks"
      - [ ] "Jacket" → "jacket, coat, outerwear, layer, top layer"
    - [ ] Context-aware synonyms (e.g., "running shoes" vs "dress shoes")
  - [ ] **Use Case and Occasion Injection**:
    - [ ] Add contextual usage: "suitable for: casual wear, lounging, layering, weekend activities"
    - [ ] Season mapping: "spring, summer, all-season, warm weather"
    - [ ] Activity mapping: "gym, running, hiking, casual office, date night"
  - [ ] **Template-Based Augmentation**:
    ```python
    def augment_product_text(product):
        template = f"""
        {product['title']}. 
        Category: {product['category']}. 
        Type: {extract_product_type(product)}. 
        Style: {product.get('style', 'casual')}. 
        Material: {extract_materials(product)}. 
        Features: {', '.join(product.get('features', []))}. 
        Colors: {extract_colors(product)}. 
        Suitable for: {generate_use_cases(product)}.
        Season: {determine_season(product)}.
        Similar to: {', '.join(find_similar_items(product))}.
        Synonyms: {', '.join(generate_synonyms(product))}.
        """
        return template
    ```

- [ ] **Image-Enriched Embeddings**
  - [ ] Use GPT-4V or similar to analyze product images
  - [ ] Extract visual attributes not in metadata:
    - [ ] Color patterns, textures, style elements
    - [ ] Fit type (slim, regular, loose)
    - [ ] Material appearance (shiny, matte, textured)
    - [ ] Design details (buttons, zippers, patterns)
    - [ ] Pattern recognition (stripes, polka dots, floral, geometric)
    - [ ] Style classification (vintage, modern, classic, trendy)
  - [ ] **Multi-Image Analysis**:
    - [ ] Analyze all available product images (main + variants)
    - [ ] Extract angle-specific details (front buttons, back pockets, side zippers)
    - [ ] Identify color variations across images
    - [ ] Detect styling suggestions from lifestyle images
  - [ ] **Visual-to-Text Generation Pipeline**:
    ```python
    async def extract_image_features(image_urls: List[str]) -> Dict:
        features = {
            'primary_colors': [],
            'patterns': [],
            'style_attributes': [],
            'material_appearance': [],
            'design_details': [],
            'fit_type': None,
            'occasions': []
        }
        # Process each image
        for url in image_urls:
            image_analysis = await analyze_with_vision_model(url)
            features = merge_features(features, image_analysis)
        return features
    ```
  - [ ] Generate descriptive text from images
  - [ ] Combine with existing metadata for richer embeddings
  - [ ] Store image descriptions separately for analysis
  - [ ] Create confidence scores for visual attributes

- [ ] **Context-Aware Text Generation**
  - [ ] Use LLM to expand product descriptions:
    - [ ] Input: "YUEDGE 5 Pairs Men's Moisture Control Cushioned Dry Fit Casual Athletic Crew Socks"
    - [ ] Output: Enriched description with use cases, occasions, similar items
  - [ ] Generate category-specific attributes:
    - [ ] Shoes: comfort level, terrain suitability, formality
    - [ ] Clothing: season, occasion, style category
  - [ ] Create structured attributes from unstructured text

- [ ] **Metadata Weighting Strategies**
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
- [x] **Intra-category Similarity** ✅ IMPLEMENTED
  - *Why*: Similar products should cluster together
  - *Decision Impact*: Low clustering → poor semantic understanding
  - *Test*: Average cosine similarity within product categories
  - *Status*: Implemented in `app/experiments/metrics/semantic.py`
  
- [x] **Inter-category Distinction** ✅ IMPLEMENTED
  - *Why*: Different categories should be separable
  - *Decision Impact*: Poor separation → confused search results
  - *Test*: Silhouette score across categories
  - *Status*: Implemented with category separation score and silhouette score

- [x] **Neighborhood Consistency** ✅ IMPLEMENTED
  - *Why*: K-nearest neighbors should make semantic sense
  - *Decision Impact*: Inconsistent neighbors → unpredictable search
  - *Test*: Manual review of top-5 neighbors for sample products
  - *Status*: Implemented as avg_neighbor_consistency metric

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

## Phase 4: Frontend & Visualization - Next.js/TypeScript Web Application

### Architecture Overview
**Tech Stack Decision**: Next.js + TypeScript + Tailwind CSS for SSR, skeleton loading, and rapid styling

**Key Requirements**:
- Interactive search interface with faded product image overlay background
- Mock API integration returning ranked fashion products with similarity scores
- Product grid with skeleton loading states
- Modal carousel for enhanced product views
- Developer debug mode with semantic search insights
- Completely frontend-independent with mock data
- Docker containerization for easy deployment

---

### Phase 4.1: Project Setup & Core Architecture ✅ READY TO IMPLEMENT

#### **Frontend Foundation**
- [ ] **Project Initialization**
  - [ ] Create Next.js project in `/frontend` directory: `npx create-next-app@latest frontend --typescript --tailwind --eslint --app`
  - [ ] Configure development environment:
    - [ ] ESLint + Prettier for code quality
    - [ ] Tailwind CSS (pre-configured)
    - [ ] TypeScript strict mode
  - [ ] Project structure:
    ```
    frontend/
    ├── app/
    │   ├── globals.css          # Global styles and background
    │   ├── layout.tsx           # Root layout component
    │   ├── page.tsx            # Home page with search
    │   └── api/
    │       └── chat/
    │           └── route.ts     # Mock chat endpoint
    ├── components/
    │   ├── SearchBar.tsx        # Main search input
    │   ├── ProductGrid.tsx      # Results grid with skeletons
    │   ├── ProductCard.tsx      # Individual product thumbnails
    │   ├── ProductModal.tsx     # Enhanced view modal
    │   └── DebugPanel.tsx       # Developer magnifying glass
    ├── lib/
    │   ├── mockData.ts         # Mock API responses
    │   ├── types.ts            # TypeScript interfaces
    │   └── utils.ts            # Helper functions
    ├── public/
    │   └── backgrounds/        # Generated background images
    └── Dockerfile
    ```

- [ ] **Core Dependencies Installation**
  - [ ] UI Components: `@headlessui/react` + `@heroicons/react` for modals and icons
  - [ ] Image handling: `next/image` for optimized loading
  - [ ] Animations: `framer-motion` for smooth transitions
  - [ ] HTTP Client: Built-in `fetch` with Next.js
  - [ ] State: React built-in `useState`/`useEffect` (keep it simple)

#### **Mock API Integration**
- [ ] **Mock Chat Endpoint** (`/frontend/app/api/chat/route.ts`)
  ```typescript
  // Mock endpoint returning ranked fashion products
  export async function POST(request: Request) {
    const { query } = await request.json()
    
    // Return 20 mock products with similarity scores
    return Response.json({
      results: mockProducts.map((product, index) => ({
        ...product,
        similarity_score: 0.95 - (index * 0.03), // Decreasing scores
        rank: index + 1
      })).slice(0, 20)
    })
  }
  ```

- [ ] **TypeScript Interfaces**
  ```typescript
  // lib/types.ts - Based on Amazon Fashion sample data
  export interface ProductResult {
    parent_asin: string
    title: string
    main_category: string
    store?: string
    images: Array<{
      hi_res?: string
      large?: string
      thumb?: string
      variant?: string
    }>
    price?: string
    rating?: number
    rating_number?: number
    features?: string[]
    details?: Record<string, any>
    categories?: string[][]
    similarity_score: number
    rank: number
  }
  
  export interface ChatResponse {
    results: ProductResult[]
    query: string
    strategy: string
  }
  ```

---

### Phase 4.2: Search Interface & Background - Core User Experience

#### **Home Page with Faded Background** (`/app/page.tsx`)
- [ ] **Background Image Overlay**
  - [ ] Extract product images from `data/amazon_fashion_sample.json`
  - [ ] Create faded mosaic/collage background using CSS:
    ```css
    .bg-overlay {
      background-image: url('/backgrounds/fashion-collage.webp');
      background-size: cover;
      background-position: center;
      opacity: 0.1;
      filter: blur(2px);
    }
    ```
  - [ ] Generate optimized background images during build
  - [ ] Responsive background for mobile/desktop

- [ ] **Central Search Bar Component**
  ```tsx
  // components/SearchBar.tsx
  interface SearchBarProps {
    onSearch: (query: string) => void;
    loading: boolean;
  }
  
  export default function SearchBar({ onSearch, loading }: SearchBarProps) {
    // Clean, centered search input
    // Submit on Enter or search button click
    // Loading spinner integration
  }
  ```

#### **Product Grid with Skeleton Loading**
- [ ] **Product Card Component**
  ```tsx
  // components/ProductCard.tsx
  interface ProductCardProps {
    product: ProductResult;
    onClick: (product: ProductResult) => void;
    showDebug?: boolean;
  }
  
  export default function ProductCard({ product, onClick, showDebug }: ProductCardProps) {
    return (
      <div className="group cursor-pointer transform hover:scale-105 transition-transform">
        {/* Product thumbnail image */}
        {/* Title (truncated) */}
        {/* Price if available */}
        {/* "Coming Soon" buy button (disabled) */}
        {/* Debug magnifying glass icon (conditional) */}
        {showDebug && <DebugPanel similarity={product.similarity_score} rank={product.rank} />}
      </div>
    )
  }
  ```

- [ ] **Skeleton Loading States**
  ```tsx
  // components/ProductGrid.tsx - Show skeletons while loading
  function ProductSkeleton() {
    return (
      <div className="animate-pulse">
        <div className="bg-gray-300 aspect-square rounded-lg mb-2"></div>
        <div className="h-4 bg-gray-300 rounded mb-1"></div>
        <div className="h-3 bg-gray-300 rounded w-3/4"></div>
      </div>
    )
  }
  ```

- [ ] **Grid Layout**
  - [ ] Responsive grid: 2 cols mobile, 4 cols tablet, 6 cols desktop
  - [ ] Proper aspect ratios for product images
  - [ ] Smooth transitions and hover effects

#### **Developer Debug Mode**
- [ ] **Debug Toggle**
  - [ ] Floating action button or keyboard shortcut (Cmd/Ctrl + D)
  - [ ] Toggle visibility of similarity scores and metadata
  
- [ ] **Debug Panel Component**
  ```tsx
  // components/DebugPanel.tsx
  interface DebugPanelProps {
    similarity: number;
    rank: number;
  }
  
  export default function DebugPanel({ similarity, rank }: DebugPanelProps) {
    return (
      <div className="absolute top-2 right-2 bg-black/80 text-white text-xs p-2 rounded">
        <div>Score: {similarity.toFixed(3)}</div>
        <div>Rank: #{rank}</div>
      </div>
    )
  }
  ```

---

### Phase 4.3: Product Modal & Enhanced View

#### **Modal Carousel Component** (`components/ProductModal.tsx`)
- [ ] **Modal Structure**
  ```tsx
  interface ProductModalProps {
    product: ProductResult;
    isOpen: boolean;
    onClose: () => void;
  }
  
  export default function ProductModal({ product, isOpen, onClose }: ProductModalProps) {
    // Headless UI Dialog for accessibility
    // Image carousel with navigation
    // Product details and metadata
    // Disabled "Coming Soon" buy button
    // Enhanced debug info (if debug mode enabled)
  }
  ```

- [ ] **Image Carousel**
  - [ ] Display all available product images (hi_res, large, variants)
  - [ ] Arrow navigation and dot indicators
  - [ ] Zoom functionality on image click
  - [ ] Fallback for missing images
  - [ ] Optimized loading with Next.js Image component

- [ ] **Enhanced Product Details**
  - [ ] Full product title (no truncation)
  - [ ] Complete feature list with proper formatting
  - [ ] Structured details display (Brand, Material, Style, etc.)
  - [ ] Category hierarchy breadcrumb
  - [ ] Price and rating information (if available)

- [ ] **Modal Actions**
  - [ ] "Coming Soon" button (disabled with tooltip)
  - [ ] "Find Similar" functionality (searches for similar products)
  - [ ] Share button (copy link to product)
  - [ ] Close button (X) and overlay click to close
  - [ ] Escape key support

#### **Background Image Generation**
- [ ] **Image Processing Script**
  ```typescript
  // lib/generateBackground.ts
  async function generateFashionCollage() {
    // Read amazon_fashion_sample.json
    // Extract all product images
    // Create mosaic/collage layout
    // Generate optimized WebP background
    // Save to public/backgrounds/
  }
  ```

- [ ] **Build Integration**
  - [ ] Add background generation to Next.js build process
  - [ ] Create multiple background variants
  - [ ] Optimize for different screen sizes
  - [ ] Implement lazy loading for backgrounds

---

### Phase 4.4: Docker Setup & Deployment

#### **Dockerfile Configuration**
- [ ] **Next.js Dockerfile**
  ```dockerfile
  # frontend/Dockerfile
  FROM node:18-alpine AS base
  WORKDIR /app
  
  # Dependencies
  COPY package*.json ./
  RUN npm ci --only=production
  
  # Build
  COPY . .
  RUN npm run build
  
  # Production
  FROM node:18-alpine AS runner
  WORKDIR /app
  RUN addgroup --system --gid 1001 nodejs
  RUN adduser --system --uid 1001 nextjs
  
  COPY --from=base /app/public ./public
  COPY --from=base /app/.next/standalone ./
  COPY --from=base /app/.next/static ./.next/static
  
  USER nextjs
  EXPOSE 3000
  ENV PORT 3000
  CMD ["node", "server.js"]
  ```

- [ ] **Docker Compose Integration**
  ```yaml
  # docker-compose.yml (add to existing)
  services:
    frontend:
      build: ./frontend
      ports:
        - "3000:3000"
      environment:
        - NODE_ENV=production
      depends_on:
        - api
  ```

#### **Production Optimizations**
- [ ] **Next.js Configuration**
  ```typescript
  // next.config.js
  module.exports = {
    output: 'standalone',
    images: {
      domains: ['m.media-amazon.com'], // For product images
      formats: ['image/webp', 'image/avif']
    },
    experimental: {
      optimizeCss: true
    }
  }
  ```

- [ ] **Build Performance**
  - [ ] Enable static generation where possible
  - [ ] Optimize image loading and caching
  - [ ] Bundle analysis and tree shaking
  - [ ] Service Worker for offline capability

---

### Phase 4.5: Future Extensions (Post-MVP)

#### **Advanced Features** (Phase 2)
- [ ] **Real Backend Integration**
  - [ ] Replace mock `/chat` endpoint with actual FastAPI integration
  - [ ] Connect to real embedding search
  - [ ] MLflow experiment dashboard integration

- [ ] **Enhanced UX**
  - [ ] Search suggestions and autocomplete
  - [ ] Search history and favorites
  - [ ] Keyboard shortcuts and accessibility
  - [ ] Mobile-optimized responsive design

#### **Analytics & Monitoring** (Phase 3)
- [ ] **User Analytics**
  - [ ] Search query tracking
  - [ ] Click-through rates
  - [ ] User session insights
  - [ ] A/B testing framework

- [ ] **Performance Monitoring**
  - [ ] Core Web Vitals tracking
  - [ ] API response time monitoring
  - [ ] Error tracking and alerting

---

### MVP Implementation Timeline

#### **Day 1-2: Foundation Setup** (Phase 4.1)
1. Create Next.js project in `/frontend` directory
2. Set up TypeScript, Tailwind, and core dependencies
3. Create basic project structure and mock API endpoint
4. Generate background collage from sample data

#### **Day 3-4: Core Search Interface** (Phase 4.2)
1. Build SearchBar component with centered layout
2. Implement ProductGrid with skeleton loading
3. Create ProductCard components with hover effects
4. Add faded background overlay and responsive design

#### **Day 5-6: Product Modal & Debug Mode** (Phase 4.3)
1. Build ProductModal with image carousel
2. Implement debug toggle and similarity score display
3. Add "Coming Soon" disabled buy buttons
4. Test modal interactions and accessibility

#### **Day 7: Docker & Polish** (Phase 4.4)
1. Create Dockerfile and docker-compose integration
2. Optimize build process and image loading
3. Add error boundaries and loading states
4. Final testing and bug fixes

#### **Expected Deliverables (Week 1)**
- ✅ Fully functional Next.js frontend with mock data
- ✅ Search interface with skeleton loading
- ✅ Product grid with modal carousel views
- ✅ Developer debug mode with similarity scores
- ✅ Docker containerization ready for deployment
- ✅ Responsive design for mobile/desktop

---

### Mock Data Implementation Details

#### **Sample Data Processing**
```typescript
// lib/mockData.ts
import fashionSample from '../data/amazon_fashion_sample.json';

// Transform raw Amazon data to ProductResult format
export const mockProducts: ProductResult[] = fashionSample.map((item, index) => ({
  parent_asin: item.parent_asin,
  title: item.title,
  main_category: item.main_category || 'Fashion',
  store: item.store,
  images: item.images || [],
  price: item.price,
  rating: item.rating,
  rating_number: item.rating_number,
  features: item.features || [],
  details: item.details || {},
  categories: item.categories || [],
  // Mock fields for MVP
  similarity_score: 0.95 - (index * 0.02), // Decreasing scores
  rank: index + 1
}));
```

#### **API Endpoint Structure**
```typescript
// app/api/chat/route.ts
export async function POST(request: Request) {
  const { query, limit = 20 } = await request.json();
  
  // Simple text matching for demo
  const filteredProducts = mockProducts
    .filter(product => 
      product.title.toLowerCase().includes(query.toLowerCase()) ||
      product.main_category.toLowerCase().includes(query.toLowerCase())
    )
    .slice(0, limit);
  
  // Add some delay to show skeleton loading
  await new Promise(resolve => setTimeout(resolve, 800));
  
  return Response.json({
    results: filteredProducts.length > 0 ? filteredProducts : mockProducts.slice(0, limit),
    query,
    strategy: 'mock',
    total: filteredProducts.length || mockProducts.length
  });
}
```

#### **Key Features Summary**
- **Search Interface**: Clean, centered search bar with faded product image background
- **Product Grid**: Responsive grid (2/4/6 columns) with skeleton loading states
- **Product Cards**: Hover effects, similarity scores, disabled "Coming Soon" buttons
- **Modal Carousel**: Full-screen product view with image navigation and enhanced details
- **Debug Mode**: Toggle to show similarity scores and ranking information
- **Mock API**: Returns ranked fashion products with realistic similarity scores
- **Docker Ready**: Containerized for easy deployment and development

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
