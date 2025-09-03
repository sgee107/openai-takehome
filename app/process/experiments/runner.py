"""
Experiment runner with MLflow integration.

Consolidates experiment functionality from app/experiments/ into a unified runner
that integrates with the new process structure.
"""
import time
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
from abc import ABC, abstractmethod
import tiktoken

from app.mlflow_client import mlflow_client
from app.db.database import AsyncSessionLocal, init_db
from app.db.models import Product, ProductEmbedding
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.process.core.pipeline import DataPipeline
from app.process.types import PipelineConfig, ProcessingResults
from app.process.strategies.registry import list_strategies, get_strategy
from app.process.experiments.metrics import calculate_semantic_metrics
from app.process.core.product_loader import ProductLoader
from app.agents.tools.image_extraction import extract_enhanced_fashion_analysis
from app.settings import settings
from openai import AsyncOpenAI


class BaseExperiment(ABC):
    """Base class for all experiments with MLflow tracking."""
    
    def __init__(
        self, 
        experiment_name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None
    ):
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or {}
        self.start_time = None
        self.metrics = {}
        self.params = {}
        self.artifacts_dir = Path("/tmp") / f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    async def setup(self):
        """Setup experiment, create MLflow experiment."""
        print(f"\n🔬 Setting up experiment: {self.experiment_name}")
        print(f"📝 Description: {self.description}")
        
        # Create artifacts directory
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set MLflow experiment
        mlflow_client.set_experiment(self.experiment_name)
        
        # Add default tags
        self.tags.update({
            "experiment_type": self.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        })
        
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Main experiment logic - override in subclasses."""
        pass
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the full experiment lifecycle."""
        try:
            # Setup
            await self.setup()
            
            # Start MLflow run
            with mlflow_client.start_run(run_name=f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log tags
                mlflow_client.add_tags(self.tags)
                
                # Log parameters
                if self.params:
                    mlflow_client.log_params(self.params)
                
                # Record start time
                self.start_time = time.time()
                
                # Run the experiment
                print(f"\n🚀 Running experiment...")
                results = await self.run()
                
                # Calculate duration
                duration = time.time() - self.start_time
                mlflow_client.log_metric("experiment_duration_seconds", duration)
                
                # Log any collected metrics
                if self.metrics:
                    mlflow_client.log_metrics(self.metrics)
                
                # Log artifacts
                await self.log_artifacts()
                
                print(f"\n✅ Experiment completed in {duration:.2f} seconds")
                
                return {
                    "status": "success",
                    "duration": duration,
                    "results": results,
                    "metrics": self.metrics,
                    "params": self.params
                }
                
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            mlflow_client.log_metric("experiment_failed", 1)
            mlflow_client.add_tags({"error": str(e)})
            
            return {
                "status": "failed",
                "error": str(e)
            }
        finally:
            await self.cleanup()
    
    async def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow and store locally."""
        for key, value in metrics.items():
            if step is not None:
                mlflow_client.log_metric(key, value, step)
            else:
                mlflow_client.log_metric(key, value)
            # Store locally for summary
            self.metrics[key] = value
    
    async def log_artifacts(self):
        """Save and log artifacts to MLflow."""
        # Save metrics summary
        metrics_file = self.artifacts_dir / "metrics_summary.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "experiment": self.experiment_name,
                "metrics": self.metrics,
                "params": self.params,
                "duration": time.time() - self.start_time if self.start_time else None
            }, f, indent=2)
        
        # Log all artifacts in directory
        if self.artifacts_dir.exists() and any(self.artifacts_dir.iterdir()):
            mlflow_client.log_artifacts(str(self.artifacts_dir))
    
    async def save_artifact(self, data: Any, filename: str, format: str = "json"):
        """Save data as an artifact."""
        filepath = self.artifacts_dir / filename
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "text":
            with open(filepath, 'w') as f:
                f.write(str(data))
        elif format == "csv":
            # Assume data is a pandas DataFrame
            data.to_csv(filepath, index=False)
        
        return filepath
    
    async def cleanup(self):
        """Cleanup after experiment."""
        print(f"📁 Artifacts saved to: {self.artifacts_dir}")
    
    def add_param(self, key: str, value: Any):
        """Add a parameter to track."""
        self.params[key] = value
        mlflow_client.log_param(key, value)
    
    def add_metric(self, key: str, value: float):
        """Add a metric to track."""
        self.metrics[key] = value
        mlflow_client.log_metric(key, value)


class EmbeddingExperiment(BaseExperiment):
    """Experiment to compare different embedding strategies using the new pipeline."""
    
    def __init__(
        self, 
        strategies: Optional[List[str]] = None,
        num_products: int = 300,
        batch_size: int = 20,
        save_to_db: bool = False
    ):
        super().__init__(
            experiment_name="embedding_strategy_comparison",
            description=f"Compare embedding strategies on {num_products} products using new pipeline"
        )
        
        # Use all strategies if none specified
        self.strategies = strategies or list_strategies()
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
    
    async def run(self) -> Dict[str, Any]:
        """Run the embedding strategy comparison experiment using the new pipeline."""
        
        # Initialize database if needed
        if self.save_to_db:
            await init_db()
        
        async with AsyncSessionLocal() as session:
            pipeline = DataPipeline(session)
            
            # Load products data for analysis
            products_data = await pipeline.load_data_file()
            self.products = products_data[:self.num_products]
            
            print(f"📦 Loaded {len(self.products)} products for testing")
            
            # Log product categories distribution
            categories = defaultdict(int)
            for p in self.products:
                categories[p.get('main_category', 'Unknown')] += 1
            
            await self.save_artifact(dict(categories), "category_distribution.json")
            
            # Process each strategy individually to collect detailed metrics
            for strategy in self.strategies:
                await self.run_single_strategy(strategy, session)
            
            # Generate comparisons
            comparison = await self.compare_strategies()
            
            # Calculate semantic metrics if we have embeddings
            if any("embeddings" in data for data in self.results.values()):
                print("\n🧮 Calculating semantic metrics...")
                
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
            
            # Generate visualizations if available
            await self.generate_visualizations()
            
            return {
                "num_strategies": len(self.strategies),
                "num_products": self.num_products,
                "comparison": comparison
            }
    
    async def run_single_strategy(self, strategy: str, session: AsyncSession) -> Dict[str, Any]:
        """Run experiment for one strategy using the new pipeline components."""
        print(f"\n📊 Testing strategy: {strategy}")
        
        # Get strategy instance
        strategy_instance = get_strategy(strategy)
        
        # Initialize tiktoken encoder for token counting
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Metrics for this strategy
        text_lengths = []
        token_counts = []
        generation_times = []
        embeddings = []
        sample_texts = []
        products_saved = 0
        embeddings_saved = 0
        image_analysis_count = 0
        image_analysis_success = 0
        
        # Track overall time
        strategy_start = time.time()
        
        # Check if this strategy requires image analysis
        needs_image_analysis = strategy == 'key_value_with_images'
        
        # Initialize OpenAI client for image analysis if needed
        openai_client = None
        if needs_image_analysis:
            openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            print(f"  🎯 Strategy requires image analysis - initializing OpenAI client")
        
        # Process products using the strategy
        from app.process.core.embedding_generator import EmbeddingGenerator
        embedding_generator = EmbeddingGenerator()
        
        # Process products in batches
        for i in range(0, len(self.products), self.batch_size):
            batch = self.products[i:i + self.batch_size]
            batch_start = time.time()
            
            # Generate texts and embeddings for batch
            texts = []
            
            for product_data in batch:
                # Handle image analysis for strategies that need it
                image_analysis = None
                if needs_image_analysis and product_data.get('images'):
                    try:
                        image_analysis_count += 1
                        # Get the first high-resolution image
                        target_image = None
                        for img in product_data['images']:
                            if isinstance(img, dict) and img.get('large'):
                                target_image = img['large']
                                break
                        
                        if target_image:
                            print(f"    🎯 [API CALL #{image_analysis_count}] Analyzing image for: {product_data.get('title', 'Unknown')[:40]}...")
                            
                            # Call OpenAI image analysis
                            image_analysis = await extract_enhanced_fashion_analysis(
                                image_url=target_image,
                                client=openai_client,
                                prompt_version="v1"
                            )
                            
                            if image_analysis and image_analysis.confidence > 0.5:
                                image_analysis_success += 1
                                print(f"    ✅ [API CALL #{image_analysis_count}] Image analysis completed (confidence: {image_analysis.confidence:.2f})")
                            else:
                                print(f"    ⚠️ [API CALL #{image_analysis_count}] Low confidence analysis (confidence: {image_analysis.confidence:.2f if image_analysis else 0:.2f})")
                                
                    except Exception as e:
                        print(f"    ❌ [API CALL #{image_analysis_count}] Error analyzing image: {e}")
                        image_analysis = None
                
                # Generate text with or without image analysis
                if needs_image_analysis and image_analysis:
                    text = strategy_instance.generate(product_data, image_analysis=image_analysis)
                else:
                    text = strategy_instance.generate(product_data)
                
                texts.append(text)
                text_lengths.append(len(text))
                
                # Count tokens
                tokens = encoder.encode(text)
                token_counts.append(len(tokens))
                
                # Save samples with image analysis info
                if len(sample_texts) < 10:
                    sample_data = {
                        "title": product_data.get('title', 'Unknown')[:100],
                        "strategy": strategy,
                        "text": text[:500],  # First 500 chars
                        "full_length": len(text),
                        "token_count": len(tokens),
                        "has_images": bool(product_data.get('images')),
                        "image_analysis_used": image_analysis is not None,
                        "image_analysis_confidence": image_analysis.confidence if image_analysis else None
                    }
                    sample_texts.append(sample_data)
            
            # Generate embeddings for batch
            batch_embeddings = await embedding_generator.generate_embeddings_batch(texts, self.batch_size)
            
            # Store embeddings (filter out None values)
            valid_embeddings = [e for e in batch_embeddings if e is not None]
            embeddings.extend(valid_embeddings)
            
            # Track time
            batch_time = time.time() - batch_start
            generation_times.append(batch_time)
            
            # Progress update
            if (i + len(batch)) % 50 == 0:
                processed = i + len(batch)
                print(f"  Processed {processed}/{len(self.products)} products...")
                if needs_image_analysis:
                    print(f"    Image analysis: {image_analysis_success}/{image_analysis_count} successful")
        
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
            "token_count_mean": np.mean(token_counts),
            "token_count_std": np.std(token_counts),
            "token_count_min": min(token_counts),
            "token_count_max": max(token_counts),
            "token_count_median": np.median(token_counts),
            "num_embeddings": len(embeddings),
            "embedding_dimension": len(embeddings[0]) if embeddings else 0,
            "failed_embeddings": len(self.products) - len(embeddings),
            "image_analysis_attempted": image_analysis_count,
            "image_analysis_successful": image_analysis_success,
            "image_analysis_success_rate": image_analysis_success / image_analysis_count if image_analysis_count > 0 else 0
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
        
        # Save token count distribution
        await self.save_artifact({
            "counts": token_counts,
            "mean": results["token_count_mean"],
            "std": results["token_count_std"],
            "min": results["token_count_min"],
            "max": results["token_count_max"],
            "median": results["token_count_median"]
        }, f"{strategy}_token_counts.json")
        
        # Store embeddings as numpy array for later analysis
        if embeddings:
            embeddings_array = np.array(embeddings)
            np.save(self.artifacts_dir / f"{strategy}_embeddings.npy", embeddings_array)
            
            # Store in results for comparison
            self.results[strategy]["embeddings"] = embeddings_array
        
        self.results[strategy]["metrics"] = results
        
        print(f"  ✅ Strategy {strategy} completed in {total_time:.2f}s")
        print(f"     Avg text length: {results['text_length_mean']:.0f} chars (±{results['text_length_std']:.0f})")
        print(f"     Avg token count: {results['token_count_mean']:.0f} tokens (±{results['token_count_std']:.0f})")
        print(f"     Token range: {results['token_count_min']}-{results['token_count_max']} tokens")
        print(f"     Generated {results['num_embeddings']} embeddings")
        
        # Print image analysis metrics if applicable
        if needs_image_analysis:
            success_rate = results['image_analysis_success_rate'] * 100
            print(f"     Image analysis: {results['image_analysis_successful']}/{results['image_analysis_attempted']} successful ({success_rate:.1f}%)")
        
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
            ("token_count_mean", "ascending"),  # Fewer tokens is better for cost
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
                    "Avg Token Count": f"{metrics['token_count_mean']:.0f}",
                    "Token Range": f"{metrics['token_count_min']:.0f}-{metrics['token_count_max']:.0f}",
                    "Total Time (s)": f"{metrics['total_time']:.2f}",
                    "Time/Product (s)": f"{metrics['avg_time_per_product']:.4f}",
                    "Failed": metrics['failed_embeddings']
                })
        
        await self.save_artifact(summary, "strategy_summary_table.json")
        
        # Print summary
        print("\n📊 Strategy Comparison Summary:")
        print("-" * 100)
        print(f"{'Strategy':20} | {'Text (chars)':>12} | {'Tokens':>12} | {'Token Range':>15} | {'Time (s)':>10} | {'Failed':>6}")
        print("-" * 100)
        for item in summary:
            print(f"{item['Strategy']:20} | "
                  f"{item['Avg Text Length']:>12} | "
                  f"{item['Avg Token Count']:>12} | "
                  f"{item['Token Range']:>15} | "
                  f"{item['Total Time (s)']:>10} | "
                  f"{item['Failed']:>6}")
        print("-" * 100)
        
        return comparison
    
    async def generate_visualizations(self):
        """Generate visualizations if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            # Import visualization functions from the new process structure
            try:
                from app.process.experiments.visualizations.distribution import create_text_length_comparison
                from app.process.experiments.visualizations.token_analysis import (
                    create_token_comparison, 
                    create_token_cost_analysis,
                    identify_outliers
                )
                visualizations_available = True
            except ImportError:
                # Fallback to old structure if new one not available
                try:
                    from app.experiments.visualizations.distribution import create_text_length_comparison
                    from app.experiments.visualizations.token_analysis import (
                        create_token_comparison, 
                        create_token_cost_analysis,
                        identify_outliers
                    )
                    visualizations_available = True
                    print("  ℹ️  Using visualization modules from legacy location")
                except ImportError:
                    print("  ⚠️  Visualization modules not available, skipping charts")
                    visualizations_available = False
            
            if visualizations_available:
                print("\n📊 Creating visualizations...")
                
                # Text length distribution comparison
                fig = create_text_length_comparison(self.results)
                fig.savefig(self.artifacts_dir / "text_length_comparison.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Token count comparison
                fig = create_token_comparison(self.results, self.artifacts_dir / "token_count_comparison.png")
                plt.close(fig)
                
                # Token cost analysis
                fig = create_token_cost_analysis(self.results, self.artifacts_dir / "token_cost_analysis.png")
                plt.close(fig)
                
                print("\n  ✅ All visualizations created successfully")
                print("     - text_length_comparison.png")
                print("     - token_count_comparison.png") 
                print("     - token_cost_analysis.png")
            else:
                print("\n  ⚠️  Skipping visualizations (modules not available)")
            
        except ImportError as e:
            print(f"  ⚠️  Import error: {e}, skipping visualizations")
        except Exception as e:
            print(f"  ⚠️  Error creating visualizations: {e}")


class ExperimentRunner:
    """Main experiment runner that coordinates different types of experiments."""
    
    @staticmethod
    async def run_embedding_experiment(
        strategies: Optional[List[str]] = None,
        num_products: int = 300,
        batch_size: int = 20,
        save_to_db: bool = False
    ) -> Dict[str, Any]:
        """Run an embedding strategy comparison experiment.
        
        Args:
            strategies: List of strategy names to test
            num_products: Number of products to test
            batch_size: Batch size for processing
            save_to_db: Whether to save results to database
            
        Returns:
            Experiment results
        """
        experiment = EmbeddingExperiment(
            strategies=strategies,
            num_products=num_products,
            batch_size=batch_size,
            save_to_db=save_to_db
        )
        
        return await experiment.execute()
    
    @staticmethod
    async def run_pipeline_experiment(config: PipelineConfig) -> Dict[str, Any]:
        """Run a pipeline experiment with experiment tracking.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Experiment results
        """
        # Enable experiments in config
        config.enable_experiments = True
        config.experiment_name = config.experiment_name or "pipeline_experiment"
        
        # Create a simple experiment wrapper
        class PipelineExperiment(BaseExperiment):
            def __init__(self, pipeline_config: PipelineConfig):
                super().__init__(
                    experiment_name=pipeline_config.experiment_name,
                    description=f"Pipeline experiment with {len(pipeline_config.strategies or list_strategies())} strategies"
                )
                self.config = pipeline_config
                self.params = {
                    "num_products": pipeline_config.num_products,
                    "batch_size": pipeline_config.batch_size,
                    "strategies": ",".join(pipeline_config.strategies or list_strategies()),
                    "save_to_db": pipeline_config.save_to_db
                }
            
            async def run(self) -> Dict[str, Any]:
                from app.process.core.pipeline import run_pipeline
                results = await run_pipeline(self.config)
                
                # Log key metrics
                await self.log_metrics({
                    "products_loaded": results.products_loaded,
                    "embeddings_created": results.embeddings_created,
                    "processing_duration": results.duration,
                    "strategies_count": len(results.strategies_processed)
                })
                
                return {
                    "processing_results": results,
                    "experiment_run_id": mlflow_client.get_current_run_id()
                }
        
        experiment = PipelineExperiment(config)
        result = await experiment.execute()
        
        # Add experiment run ID to processing results if successful
        if result.get("status") == "success" and "processing_results" in result["results"]:
            processing_results = result["results"]["processing_results"]
            processing_results.experiment_run_id = result["results"].get("experiment_run_id")
        
        return result
