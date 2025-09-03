# Implementation Plan

[Overview]
Consolidate embedding creation and experiment management logic from scattered scripts and experiments into a unified process directory with two streamlined scripts.

The current codebase has significant duplication between `app/scripts/` and `app/experiments/` directories, with overlapping functionality for embedding generation, product loading, and text strategies. This refactoring will create a clean separation between database schema management and data processing pipelines, while eliminating redundant code and providing a single interface for both simple data loading and experimental runs.

[Types]
Define unified interfaces for embedding strategies and pipeline configuration.

```python
# app/process/types.py
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass

class EmbeddingStrategy(Protocol):
    """Protocol for embedding text generation strategies."""
    def generate(self, product: Dict[str, Any], **kwargs) -> str: ...

@dataclass
class PipelineConfig:
    """Configuration for the data processing pipeline."""
    num_products: Optional[int] = None
    batch_size: int = 20
    strategies: Optional[List[str]] = None
    save_to_db: bool = True
    enable_experiments: bool = False
    experiment_name: Optional[str] = None

@dataclass
class ProcessingResults:
    """Results from pipeline execution."""
    products_loaded: int
    embeddings_created: int
    strategies_processed: List[str]
    duration: float
    experiment_run_id: Optional[str] = None
```

[Files]
Create new consolidated process directory structure and remove redundant files.

**New files to be created:**
- `app/process/__init__.py` - Package initialization
- `app/process/types.py` - Type definitions and protocols
- `app/process/strategies/` - Directory for embedding strategies
- `app/process/strategies/__init__.py` - Strategy package init
- `app/process/strategies/text_strategies.py` - Consolidated text generation strategies
- `app/process/strategies/registry.py` - Strategy registration and management
- `app/process/core/` - Core processing logic
- `app/process/core/__init__.py` - Core package init  
- `app/process/core/embedding_generator.py` - OpenAI embedding generation
- `app/process/core/product_loader.py` - Product loading and database operations
- `app/process/core/pipeline.py` - Main pipeline orchestration
- `app/process/experiments/` - Experiment-specific logic
- `app/process/experiments/__init__.py` - Experiments package init
- `app/process/experiments/runner.py` - Experiment execution and MLflow integration
- `app/process/experiments/metrics.py` - Moved metrics calculation
- `app/process/cli/` - CLI interfaces
- `app/process/cli/__init__.py` - CLI package init
- `app/process/cli/db_manager.py` - Database management CLI
- `app/process/cli/pipeline.py` - Main pipeline CLI

**Existing files to be modified:**
- `app/db/models.py` - No changes needed, just imports will change
- `app/settings.py` - No changes needed

**Files to be deleted or moved:**
- `app/scripts/load_products.py` - Delete (logic moved to process/)
- `app/scripts/data_loader.py` - Delete (legacy, unused)
- `app/scripts/structured_text_strategies.py` - Delete (moved to process/)
- `app/experiments/run_experiments.py` - Delete (replaced by process CLI)
- `app/experiments/runners/embedding_strategy.py` - Delete (moved to process/)
- `app/experiments/metrics/semantic.py` - Move to `app/process/experiments/metrics.py`
- `app/experiments/visualizations/` - Move entire directory to `app/process/experiments/visualizations/`
- `app/scripts/db_management.py` - Move to `app/process/cli/db_manager.py`

[Functions]
Consolidate and refactor core processing functions with clear separation of concerns.

**New functions:**
- `app/process/strategies/registry.py::get_strategy(name: str) -> EmbeddingStrategy` - Get strategy by name
- `app/process/strategies/registry.py::list_strategies() -> List[str]` - List available strategies
- `app/process/core/pipeline.py::run_pipeline(config: PipelineConfig) -> ProcessingResults` - Main pipeline execution
- `app/process/core/product_loader.py::load_products_batch(products: List[Dict], session: AsyncSession) -> Tuple[int, int]` - Batch product loading
- `app/process/experiments/runner.py::run_experiment(config: PipelineConfig) -> Dict[str, Any]` - Execute with MLflow tracking

**Modified functions:**
- Consolidate `TextStrategy` methods and `KeyValueStrategy` classes into unified strategy classes
- Merge `EmbeddingGenerator` classes from scripts and experiments into single implementation
- Combine product loading logic from scripts/load_products.py and experiments/runners/embedding_strategy.py

**Removed functions:**
- `app/scripts/load_products.py::main()` - Replaced by CLI
- `app/experiments/run_experiments.py::run_experiment()` - Merged into new pipeline

[Classes]
Unify duplicate classes and create clean inheritance hierarchy.

**New classes:**
- `app/process/strategies/text_strategies.py::BaseTextStrategy` - Abstract base for all strategies
- `app/process/strategies/text_strategies.py::TitleOnlyStrategy` - Simple title-based strategy  
- `app/process/strategies/text_strategies.py::ComprehensiveStrategy` - Full text generation
- `app/process/strategies/text_strategies.py::KeyValueStrategy` - Structured key-value approach
- `app/process/core/pipeline.py::DataPipeline` - Main pipeline orchestrator
- `app/process/experiments/runner.py::ExperimentRunner` - MLflow experiment management

**Modified classes:**
- Consolidate `TextStrategy` (from load_products.py) and `KeyValueStrategy` classes (from structured_text_strategies.py) into unified hierarchy
- Merge `EmbeddingGenerator` classes from both scripts and experiments
- Combine `ProductLoader` functionality into single implementation

**Removed classes:**
- `app/scripts/load_products.py::TextStrategy` - Merged into new hierarchy
- `app/scripts/structured_text_strategies.py::KeyValueStrategy` and subclasses - Consolidated
- `app/experiments/runners/embedding_strategy.py::EmbeddingStrategyExperiment` - Replaced

[Dependencies]
No new external dependencies required, reorganize internal imports.

All existing dependencies (asyncio, sqlalchemy, openai, mlflow, click, etc.) remain the same. Internal imports will be updated to reference the new `app.process` module structure instead of scattered `app.scripts` and `app.experiments` imports.

[Testing]
Minimal testing changes, update import paths in existing tests.

- Update `tests/test_embeddings.py` to import from `app.process` instead of `app.scripts`
- Update any experiment-related tests to use new process module
- No new test files needed initially, existing test coverage should transfer

[Implementation Order]
Implement in careful order to avoid breaking existing functionality.

1. **Create process directory structure** - Set up all new directories and __init__.py files
2. **Move and consolidate strategies** - Create unified text strategy classes combining both TextStrategy and KeyValueStrategy functionality 
3. **Create core processing classes** - Build EmbeddingGenerator, ProductLoader, and Pipeline classes by consolidating existing logic
4. **Build experiment runner** - Move MLflow integration and metrics from experiments/ to process/experiments/
5. **Create CLI interfaces** - Build db_manager.py and pipeline.py CLI commands
6. **Update imports across codebase** - Change all internal imports to use new process structure
7. **Test pipeline functionality** - Verify both basic loading and experimental modes work
8. **Remove old files** - Delete redundant files from scripts/ and experiments/
9. **Update documentation** - Update any docs or README files to reference new CLI commands
