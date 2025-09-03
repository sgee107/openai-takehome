# Call Stack Trace: Pipeline Execution Flows

## **Pipeline Command Examples**

```bash
# Experiment Mode
uv run pipeline run --experiment -n 10 -s key_value_with_images

# Normal Mode
uv run pipeline run -n 100 -s key_value_with_images
```

---

## **🧪 EXPERIMENT PIPELINE FLOW**

**Purpose:** Compare strategies without persisting to database, with MLflow tracking

### **Abstraction Levels & Order of Operations:**

```python
🎯 LEVEL 1: CLI Interface
├─► app/process/cli/pipeline.py:run()
    └─► Parses arguments, detects --experiment flag

🎯 LEVEL 2: Pipeline Orchestration  
├─► app/process/cli/pipeline.py:_run_async()
    ├─► Experiment Mode Branch: 
    └─► result = await ExperimentRunner.run_embedding_experiment(...)

🎯 LEVEL 3: Experiment Management
├─► app/process/experiments/runner.py:ExperimentRunner.run_embedding_experiment()
    ├─► Creates EmbeddingExperiment instance
    └─► return await experiment.execute()

🎯 LEVEL 4: Experiment Execution Framework
├─► app/process/experiments/runner.py:EmbeddingExperiment.execute()
    ├─► Setup MLflow tracking
    ├─► Start MLflow run
    ├─► results = await self.run()
    ├─► Log artifacts & metrics
    └─► Return experiment results

🎯 LEVEL 5: Strategy Comparison Logic
├─► app/process/experiments/runner.py:EmbeddingExperiment.run()
    ├─► Load raw JSON data (no DB interaction)
    ├─► For each strategy in self.strategies:
    │   └─► await self.run_single_strategy(strategy, session)
    ├─► Generate comparison metrics
    └─► Create visualizations

🎯 LEVEL 6: Individual Strategy Processing  
├─► app/process/experiments/runner.py:EmbeddingExperiment.run_single_strategy()
    ├─► Get strategy instance
    ├─► Initialize metrics collection
    ├─► For each product in batch:
    │   ├─► 🎯 IMAGE ANALYSIS (if key_value_with_images):
    │   │   └─► await extract_enhanced_fashion_analysis()
    │   │       └─► 🚀 OPENAI VISION API CALL
    │   ├─► text = strategy_instance.generate(product_data, image_analysis=analysis)
    │   └─► Store text & metrics
    └─► Generate embeddings for all texts

🎯 LEVEL 7: Strategy Text Generation
├─► app/process/strategies/text_strategies.py:KeyValueWithImagesStrategy.generate()
    ├─► base_text = super().generate(product, **kwargs)
    ├─► image_analysis = kwargs.get('image_analysis')
    ├─► If image_analysis: enrich text with visual data
    └─► Return enriched text

🎯 LEVEL 8: Embedding Generation (Batch)
├─► app/process/core/embedding_generator.py:generate_embeddings_batch()
    ├─► For each text in batch:
    │   └─► embedding = await self.generate_embedding(text)
    └─► Return list of embeddings

🎯 LEVEL 9: Individual Embedding API
├─► app/process/core/embedding_generator.py:generate_embedding()
    └─► response = await self.client.embeddings.create(...)
        └─► 🚀 OPENAI EMBEDDING API CALL
```

---

## **🏭 NORMAL PIPELINE FLOW**

**Purpose:** Process and persist products to database for production use

### **Abstraction Levels & Order of Operations:**

```python
🎯 LEVEL 1: CLI Interface
├─► app/process/cli/pipeline.py:run()
    └─► Parses arguments, no --experiment flag

🎯 LEVEL 2: Pipeline Orchestration
├─► app/process/cli/pipeline.py:_run_async()
    ├─► Normal Mode Branch:
    ├─► config = PipelineConfig(...)
    └─► result = await run_pipeline(config)

🎯 LEVEL 3: Core Pipeline Management
├─► app/process/core/pipeline.py:run_pipeline()
    ├─► Create DataPipeline instance
    └─► return await pipeline.execute()

🎯 LEVEL 4: Data Pipeline Execution
├─► app/process/core/pipeline.py:DataPipeline.execute()
    ├─► Load raw JSON data
    ├─► Initialize database
    ├─► Create ProductLoader instance
    └─► await loader.load_products_batch(products_data)

🎯 LEVEL 5: Product Batch Loading
├─► app/process/core/product_loader.py:ProductLoader.load_products_batch()
    ├─► For each batch of products:
    │   ├─► First Pass: Load products to DB
    │   │   └─► For each product: await self.load_product(product_data)
    │   ├─► await session.commit()  # Get product IDs
    │   ├─► Second Pass: Generate embeddings
    │   │   └─► For each product: await self.generate_embeddings_for_product()
    │   └─► await session.commit()  # Save embeddings
    └─► Return (loaded_count, failed_count)

🎯 LEVEL 6: Individual Product Processing
├─► app/process/core/product_loader.py:ProductLoader.load_product()
    ├─► Check if product exists in DB
    ├─► Create Product model instance
    ├─► Add ProductImage & ProductVideo instances
    └─► session.add(product)

🎯 LEVEL 7: Embedding Generation Per Product
├─► app/process/core/product_loader.py:generate_embeddings_for_product()
    ├─► 🎯 IMAGE ANALYSIS (always performed):
    │   └─► image_analysis = await self.analyze_product_images(product)
    │       └─► await extract_enhanced_fashion_analysis()
    │           └─► 🚀 OPENAI VISION API CALL
    ├─► For each strategy in strategies:
    │   ├─► strategy = get_strategy(strategy_name)
    │   ├─► If strategy needs images: text = strategy.generate(data, image_analysis=analysis)
    │   ├─► Else: text = strategy.generate(data)
    │   ├─► embedding = await embedding_generator.generate_embedding(text)
    │   │   └─► 🚀 OPENAI EMBEDDING API CALL
    │   └─► Create ProductEmbedding model & session.add()
    └─► Store analysis in database

🎯 LEVEL 8: Strategy Text Generation (Same as Experiment)
├─► app/process/strategies/text_strategies.py:Strategy.generate()
    └─► Generate structured text based on strategy type

🎯 LEVEL 9: Database Persistence
├─► SQLAlchemy Models (Product, ProductEmbedding, ProductImage, etc.)
    └─► session.commit() → PostgreSQL database
```

---

## **📊 KEY DIFFERENCES**

| Aspect | **Experiment Pipeline** | **Normal Pipeline** |
|--------|------------------------|-------------------|
| **Purpose** | Strategy comparison & analysis | Production data loading |
| **Data Source** | Raw JSON (in-memory) | Raw JSON → Database models |
| **Database** | Read-only session | Full CRUD operations |
| **Image Analysis** | Only for strategies that need it | Always performed & stored |
| **MLflow Tracking** | Full experiment tracking | Optional metrics only |
| **Persistence** | Temporary artifacts only | Full database persistence |
| **Batch Processing** | In-memory batches | Database transaction batches |
| **Error Handling** | Continue with failures | Robust transaction rollback |
| **Performance Focus** | Comparison metrics | Throughput & reliability |
| **Output** | Experiment reports & visualizations | Database records |

## **🔍 ISSUE DISCOVERED AND RESOLVED: Image Analysis Integration in Experiments**

**PROBLEM IDENTIFIED:**
Looking at the code flow, the OpenAI image analysis (`extract_enhanced_fashion_analysis`) was only called in:

**File:** `app/process/core/product_loader.py:generate_embeddings_for_product()`

But this method was only called when using `ProductLoader.load_products_batch()` for **actual database loading**, NOT during experiments!

The experiment runner was using a different path:
- `EmbeddingExperiment.run_single_strategy()` 
- Called `strategy.generate(product_data)` directly
- **No image analysis passed** (no `image_analysis` kwarg)

**WHY KeyValueWithImages Strategy Showed Misleading Performance:**
It was fast because **no actual image analysis was happening** in experiments - it just fell back to the base KeyValue strategy text without image enrichment!

**SOLUTION IMPLEMENTED:**
Modified `app/process/experiments/runner.py` to:

1. **Detect strategies that need image analysis**: Check if strategy is `key_value_with_images`
2. **Initialize OpenAI client when needed**: Create client for image analysis
3. **Call image analysis for each product**: Extract enhanced fashion analysis from product images
4. **Pass analysis to strategy**: Provide `image_analysis` parameter to strategy.generate()
5. **Track image analysis metrics**: Log success rate, API call count, and confidence scores

**VERIFICATION:**
✅ Ran experiment with `uv run pipeline run --experiment -n 10 -s key_value_with_images`
✅ **10/10 image analyses successful (100.0%)**
✅ **Average text length: 783 chars (much higher with image data)**
✅ **Average tokens: 216 (showing image enrichment is working)**
✅ **Proper API call tracking and confidence logging**

The expensive OpenAI vision calls now happen during **both** experiment comparisons AND real data loading, providing accurate performance comparisons!
