"""
Main pipeline orchestration for data processing and embedding generation.

This module provides the central Pipeline class that coordinates product loading,
embedding generation, and optional experiment tracking.
"""
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.db.database import init_db
from app.db.models import Product, ProductEmbedding
from app.process.types import PipelineConfig, ProcessingResults
from app.process.core.product_loader import ProductLoader
from app.process.strategies.registry import list_strategies


class DataPipeline:
    """Main pipeline orchestrator for data processing and embedding generation."""
    
    def __init__(self, session: AsyncSession):
        """Initialize the pipeline.
        
        Args:
            session: Async database session
        """
        self.session = session
        self.product_loader = ProductLoader(session)
    
    async def load_data_file(self, data_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Load product data from JSON file.
        
        Args:
            data_path: Path to JSON data file. If None, uses default sample data.
            
        Returns:
            List of product data dictionaries
        """
        if data_path is None:
            # Default to sample data
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "amazon_fashion_sample.json"
        
        print(f"📂 Loading data from: {data_path}")
        
        with open(data_path, 'r') as f:
            products_data = json.load(f)
        
        print(f"📦 Found {len(products_data)} products to process")
        return products_data
    
    async def run_pipeline(self, config: PipelineConfig) -> ProcessingResults:
        """Run the main data processing pipeline.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Processing results
        """
        start_time = time.time()
        
        print("🚀 Starting data processing pipeline")
        print(f"📍 Configuration:")
        print(f"  Products: {config.num_products or 'all'}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Strategies: {config.strategies or 'all'}")
        print(f"  Save to DB: {config.save_to_db}")
        print(f"  Experiments enabled: {config.enable_experiments}")
        
        # Initialize database if needed
        if config.save_to_db:
            print("\n🔧 Initializing database...")
            await init_db()
        
        # Load data
        products_data = await self.load_data_file()
        
        # Limit number of products if specified
        if config.num_products:
            products_data = products_data[:config.num_products]
            print(f"📦 Limited to {len(products_data)} products")
        
        # Determine strategies to use
        strategies_to_use = config.strategies or list_strategies()
        print(f"\n🎯 Will generate {len(strategies_to_use)} embedding strategies:")
        for strategy in strategies_to_use:
            print(f"    - {strategy}")
        
        # Process products
        print(f"\n⏳ Processing products...")
        if config.save_to_db:
            loaded, failed = await self.product_loader.load_products_batch(
                products_data, 
                config.batch_size,
                strategies_to_use
            )
        else:
            # Dry run mode - just process without saving
            loaded, failed = await self._dry_run_processing(
                products_data, 
                config.batch_size,
                strategies_to_use
            )
        
        # Calculate results
        duration = time.time() - start_time
        embeddings_created = loaded * len(strategies_to_use) if loaded > 0 else 0
        
        results = ProcessingResults(
            products_loaded=loaded,
            embeddings_created=embeddings_created,
            strategies_processed=strategies_to_use,
            duration=duration
        )
        
        print(f"\n✅ Pipeline completed!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Products processed: {loaded}")
        print(f"  Products failed: {failed}")
        print(f"  Embeddings created: {embeddings_created}")
        print(f"  Strategies processed: {len(strategies_to_use)}")
        
        return results
    
    async def _dry_run_processing(
        self,
        products_data: List[Dict[str, Any]],
        batch_size: int,
        strategies: List[str]
    ) -> tuple[int, int]:
        """Process products without saving to database (dry run).
        
        Args:
            products_data: List of product data dictionaries
            batch_size: Batch size for processing
            strategies: List of strategy names
            
        Returns:
            Tuple of (processed_count, failed_count)
        """
        from app.process.strategies.registry import get_strategy
        
        processed = 0
        failed = 0
        
        print("🔍 Running in dry-run mode (no database saves)")
        
        for i, product_data in enumerate(products_data):
            try:
                # Test each strategy on the product
                for strategy_name in strategies:
                    strategy = get_strategy(strategy_name)
                    text = strategy.generate(product_data)
                    
                    if not text:
                        print(f"  ⚠️ Strategy {strategy_name} produced empty text for product {i}")
                
                processed += 1
                
                # Progress update
                if (i + 1) % batch_size == 0:
                    print(f"  Processed {i + 1}/{len(products_data)} products...")
                    
            except Exception as e:
                print(f"  ❌ Error processing product {i}: {e}")
                failed += 1
        
        return processed, failed
    
    async def analyze_results(self) -> Dict[str, Any]:
        """Analyze the loaded products and embeddings in the database.
        
        Returns:
            Dictionary with analysis results
        """
        print("\n📊 Analyzing results...")
        
        # Count products
        result = await self.session.execute(select(func.count(Product.id)))
        total_products = result.scalar()
        
        # Count embeddings by strategy
        result = await self.session.execute(
            select(ProductEmbedding.strategy, func.count(ProductEmbedding.id))
            .group_by(ProductEmbedding.strategy)
            .order_by(ProductEmbedding.strategy)
        )
        embeddings_by_strategy = result.all()
        
        # Count products with at least one embedding
        result = await self.session.execute(
            select(func.count(func.distinct(ProductEmbedding.product_id)))
        )
        products_with_embeddings = result.scalar()
        
        # Products by category
        result = await self.session.execute(
            select(Product.main_category, func.count(Product.id))
            .group_by(Product.main_category)
            .order_by(Product.main_category)
        )
        categories = result.all()
        
        # Build analysis results
        analysis = {
            'total_products': total_products,
            'products_with_embeddings': products_with_embeddings,
            'coverage_percentage': (products_with_embeddings/total_products*100) if total_products > 0 else 0,
            'embeddings_by_strategy': {strategy: count for strategy, count in embeddings_by_strategy},
            'products_by_category': {category: count for category, count in categories}
        }
        
        # Print summary
        print(f"  Total products: {total_products}")
        print(f"  Products with embeddings: {products_with_embeddings}")
        print(f"  Coverage: {analysis['coverage_percentage']:.1f}%" if total_products > 0 else "N/A")
        
        print(f"\n📈 Embeddings by Strategy:")
        total_embeddings = 0
        for strategy, count in embeddings_by_strategy:
            print(f"    {strategy:25} {count:>6} embeddings")
            total_embeddings += count
        print(f"    {'TOTAL':25} {total_embeddings:>6} embeddings")
        
        print(f"\n📦 Products by Category:")
        for category, count in categories:
            print(f"    {category:30} {count:>6} products")
        
        return analysis
    
    async def update_embeddings(
        self, 
        strategies: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> int:
        """Update embeddings for existing products with new or missing strategies.
        
        Args:
            strategies: List of strategy names to generate. If None, uses all strategies.
            limit: Maximum number of products to update. If None, updates all.
            
        Returns:
            Number of products updated
        """
        print(f"\n🔄 Updating embeddings for existing products...")
        
        updated_count = await self.product_loader.update_existing_product_embeddings(
            strategies=strategies,
            limit=limit
        )
        
        return updated_count
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available embedding strategies.
        
        Returns:
            List of strategy names
        """
        return self.product_loader.get_available_strategies()


async def run_pipeline(config: PipelineConfig) -> ProcessingResults:
    """Convenience function to run the pipeline with a session.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Processing results
    """
    from app.db.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        pipeline = DataPipeline(session)
        return await pipeline.run_pipeline(config)
