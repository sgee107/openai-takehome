"""
Experiment to compare different embedding strategies.
"""
import time
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from app.experiments.base import BaseExperiment
from app.scripts.load_products import TextStrategy, EmbeddingGenerator, ProductLoader
from app.db.database import AsyncSessionLocal, init_db
from app.db.models import Product, ProductEmbedding
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession


class EmbeddingStrategyExperiment(BaseExperiment):
    """Compare different text strategies for embeddings."""
    
    def __init__(
        self, 
        strategies: Optional[List[str]] = None,
        num_products: int = 300,
        batch_size: int = 20,
        save_to_db: bool = False
    ):
        super().__init__(
            experiment_name="embedding_strategy_comparison",
            description=f"Compare embedding strategies on {num_products} products"
        )
        
        # Use all strategies if none specified
        self.strategies = strategies or list(ProductLoader.STRATEGIES.keys())
        self.num_products = num_products
        self.batch_size = batch_size
        self.save_to_db = save_to_db
        self.products = []
        self.results = defaultdict(dict)
        
        # Add parameters
        self.params = {
            "num_products": num_products,
            "batch_size": batch_size,
            "num_strategies": len(self.strategies),
            "strategies": ",".join(self.strategies),
            "save_to_db": save_to_db
        }
        
    async def load_products_data(self) -> List[Dict[str, Any]]:
        """Load product data from JSON file."""
        data_path = Path(__file__).parent.parent.parent.parent / "data" / "amazon_fashion_sample.json"
        
        with open(data_path, 'r') as f:
            all_products = json.load(f)
        
        # Take specified number of products
        self.products = all_products[:self.num_products]
        
        print(f"📦 Loaded {len(self.products)} products for testing")
        
        # Log product categories distribution
        categories = defaultdict(int)
        for p in self.products:
            categories[p.get('main_category', 'Unknown')] += 1
        
        await self.save_artifact(dict(categories), "category_distribution.json")
        
        return self.products
    
    async def run_single_strategy(
        self, 
        strategy: str, 
        session: AsyncSession,
        save_to_db: bool = False
    ) -> Dict[str, Any]:
        """Run experiment for one strategy."""
        print(f"\n📊 Testing strategy: {strategy}")
        
        strategy_func = ProductLoader.STRATEGIES[strategy]
        embedding_generator = EmbeddingGenerator()
        
        # Metrics for this strategy
        text_lengths = []
        generation_times = []
        embeddings = []
        sample_texts = []
        products_saved = 0
        embeddings_saved = 0
        
        # Track overall time
        strategy_start = time.time()
        
        # Process products in batches
        for i in range(0, len(self.products), self.batch_size):
            batch = self.products[i:i + self.batch_size]
            batch_start = time.time()
            
            # Generate texts and embeddings for batch
            batch_data = []  # Store for database saving
            texts = []
            
            for product_data in batch:
                text = strategy_func(product_data)
                texts.append(text)
                text_lengths.append(len(text))
                
                # Save samples
                if len(sample_texts) < 10:
                    sample_texts.append({
                        "title": product_data.get('title', 'Unknown')[:100],
                        "strategy": strategy,
                        "text": text[:500],  # First 500 chars
                        "full_length": len(text)
                    })
                
                # Store product data and text for DB saving
                batch_data.append((product_data, text))
            
            # Generate embeddings for batch
            batch_embeddings = await embedding_generator.generate_embeddings_batch(texts, self.batch_size)
            
            # Save to database if requested
            if save_to_db:
                for (product_data, text), embedding in zip(batch_data, batch_embeddings):
                    if embedding is not None:
                        try:
                            # Check if product exists
                            result = await session.execute(
                                select(Product).where(Product.parent_asin == product_data.get('parent_asin'))
                            )
                            product = result.scalar_one_or_none()
                            
                            # Create product if it doesn't exist
                            if not product:
                                product = Product(
                                    parent_asin=product_data.get('parent_asin', ''),
                                    main_category=product_data.get('main_category', ''),
                                    title=product_data.get('title', ''),
                                    average_rating=product_data.get('average_rating'),
                                    rating_number=product_data.get('rating_number'),
                                    price=product_data.get('price'),
                                    store=product_data.get('store'),
                                    features=product_data.get('features', []),
                                    description=product_data.get('description', []),
                                    categories=product_data.get('categories', []),
                                    details=product_data.get('details', {}),
                                    bought_together=product_data.get('bought_together')
                                )
                                session.add(product)
                                await session.flush()  # Get the product ID
                                products_saved += 1
                            
                            # Check if embedding for this strategy exists
                            emb_result = await session.execute(
                                select(ProductEmbedding).where(
                                    ProductEmbedding.product_id == product.id,
                                    ProductEmbedding.strategy == strategy
                                )
                            )
                            existing_embedding = emb_result.scalar_one_or_none()
                            
                            if existing_embedding:
                                # Update existing embedding
                                existing_embedding.embedding = embedding
                                existing_embedding.embedding_text = text
                            else:
                                # Create new embedding
                                product_embedding = ProductEmbedding(
                                    product_id=product.id,
                                    strategy=strategy,
                                    embedding_text=text,
                                    embedding=embedding,
                                    model="text-embedding-3-small"
                                )
                                session.add(product_embedding)
                                embeddings_saved += 1
                            
                        except Exception as e:
                            print(f"  ⚠️  Error saving to DB: {e}")
                
                # Commit batch
                await session.commit()
            
            # Track time
            batch_time = time.time() - batch_start
            generation_times.append(batch_time)
            
            # Store embeddings (filter out None values)
            valid_embeddings = [e for e in batch_embeddings if e is not None]
            embeddings.extend(valid_embeddings)
            
            # Progress update
            if (i + len(batch)) % 50 == 0:
                print(f"  Processed {i + len(batch)}/{len(self.products)} products...")
                if save_to_db:
                    print(f"    Saved {products_saved} new products, {embeddings_saved} embeddings to DB")
        
        # Calculate metrics
        total_time = time.time() - strategy_start
        
        results = {
            "strategy": strategy,
            "total_time": total_time,
            "avg_time_per_product": total_time / len(self.products),
            "avg_time_per_batch": np.mean(generation_times),
            "text_length_mean": np.mean(text_lengths),
            "text_length_std": np.std(text_lengths),
            "text_length_min": min(text_lengths),
            "text_length_max": max(text_lengths),
            "text_length_median": np.median(text_lengths),
            "num_embeddings": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "failed_embeddings": len(self.products) - len(embeddings),
            "products_saved_to_db": products_saved,
            "embeddings_saved_to_db": embeddings_saved
        }
        
        # Log metrics for this strategy
        for key, value in results.items():
            if isinstance(value, (int, float)):
                await self.log_metrics({f"{strategy}_{key}": value})
        
        # Save artifacts for this strategy
        await self.save_artifact(sample_texts, f"{strategy}_sample_texts.json")
        await self.save_artifact({
            "lengths": text_lengths,
            "mean": results["text_length_mean"],
            "std": results["text_length_std"],
            "min": results["text_length_min"],
            "max": results["text_length_max"]
        }, f"{strategy}_text_lengths.json")
        
        # Store embeddings as numpy array for later analysis
        if embeddings:
            embeddings_array = np.array(embeddings)
            np.save(self.artifacts_dir / f"{strategy}_embeddings.npy", embeddings_array)
            
            # Store in results for comparison
            self.results[strategy]["embeddings"] = embeddings_array
        
        self.results[strategy]["metrics"] = results
        
        print(f"  ✅ Strategy {strategy} completed in {total_time:.2f}s")
        print(f"     Avg text length: {results['text_length_mean']:.0f} chars (±{results['text_length_std']:.0f})")
        print(f"     Generated {results['num_embeddings']} embeddings")
        if save_to_db:
            print(f"     Saved to DB: {products_saved} products, {embeddings_saved} embeddings")
        
        return results
    
    async def compare_strategies(self) -> Dict[str, Any]:
        """Generate comparison metrics across all strategies."""
        print("\n📈 Generating strategy comparison...")
        
        comparison = {
            "strategies": {},
            "rankings": {}
        }
        
        # Collect all metrics
        for strategy, data in self.results.items():
            comparison["strategies"][strategy] = data["metrics"]
        
        # Rank strategies by different criteria
        metrics_to_rank = [
            ("text_length_mean", "ascending"),  # Shorter is better for efficiency
            ("total_time", "ascending"),  # Faster is better
            ("failed_embeddings", "ascending")  # Fewer failures is better
        ]
        
        for metric, order in metrics_to_rank:
            strategy_scores = [
                (s, data["metrics"].get(metric, float('inf'))) 
                for s, data in self.results.items()
            ]
            
            # Sort based on order
            reverse = (order == "descending")
            strategy_scores.sort(key=lambda x: x[1], reverse=reverse)
            
            comparison["rankings"][metric] = [
                {"rank": i+1, "strategy": s, "value": v}
                for i, (s, v) in enumerate(strategy_scores)
            ]
        
        # Save comparison
        await self.save_artifact(comparison, "strategy_comparison.json")
        
        # Create summary table
        summary = []
        for strategy in self.strategies:
            if strategy in self.results:
                metrics = self.results[strategy]["metrics"]
                summary.append({
                    "Strategy": strategy,
                    "Avg Text Length": f"{metrics['text_length_mean']:.0f}",
                    "Total Time (s)": f"{metrics['total_time']:.2f}",
                    "Time/Product (s)": f"{metrics['avg_time_per_product']:.4f}",
                    "Failed": metrics['failed_embeddings']
                })
        
        await self.save_artifact(summary, "strategy_summary_table.json")
        
        # Print summary
        print("\n📊 Strategy Comparison Summary:")
        print("-" * 80)
        for item in summary:
            print(f"{item['Strategy']:20} | "
                  f"Text: {item['Avg Text Length']:>8} chars | "
                  f"Time: {item['Total Time (s)']:>8}s | "
                  f"Failed: {item['Failed']:>3}")
        print("-" * 80)
        
        return comparison
    
    async def run(self) -> Dict[str, Any]:
        """Run the embedding strategy comparison experiment."""
        
        # Load products
        await self.load_products_data()
        
        # Initialize database
        await init_db()
        
        # Run each strategy
        async with AsyncSessionLocal() as session:
            for strategy in self.strategies:
                await self.run_single_strategy(strategy, session, save_to_db=self.save_to_db)
        
        # Generate comparisons
        comparison = await self.compare_strategies()
        
        # Calculate semantic metrics if we have embeddings
        if any("embeddings" in data for data in self.results.values()):
            print("\n🧮 Calculating semantic metrics...")
            from app.experiments.metrics.semantic import calculate_semantic_metrics
            
            for strategy, data in self.results.items():
                if "embeddings" in data:
                    semantic_metrics = await calculate_semantic_metrics(
                        embeddings=data["embeddings"],
                        labels=[p.get('main_category', 'Unknown') for p in self.products],
                        strategy_name=strategy
                    )
                    
                    # Log semantic metrics
                    for key, value in semantic_metrics.items():
                        if isinstance(value, (int, float)):
                            await self.log_metrics({f"{strategy}_semantic_{key}": value})
                    
                    # Save semantic analysis
                    await self.save_artifact(semantic_metrics, f"{strategy}_semantic_metrics.json")
        
        # Generate visualizations if matplotlib is available
        try:
            from app.experiments.visualizations.distribution import create_text_length_comparison
            
            print("\n📊 Creating visualizations...")
            
            # Text length distribution comparison
            fig = create_text_length_comparison(self.results)
            fig.savefig(self.artifacts_dir / "text_length_comparison.png", dpi=100, bbox_inches='tight')
            
            print("  ✅ Visualizations created")
        except ImportError:
            print("  ⚠️  Matplotlib not available, skipping visualizations")
        
        return {
            "num_strategies": len(self.strategies),
            "num_products": self.num_products,
            "comparison": comparison
        }