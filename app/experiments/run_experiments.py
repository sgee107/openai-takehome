#!/usr/bin/env python
"""
CLI interface for running embedding experiments.

Usage:
    python -m app.experiments.run_experiments --experiment strategy --num-products 300
    python -m app.experiments.run_experiments --experiment strategy --strategies title_only --strategies comprehensive
"""
import asyncio
import click
from typing import List, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.experiments.runners.embedding_strategy import EmbeddingStrategyExperiment


@click.command()
@click.option(
    '--experiment', 
    type=click.Choice(['strategy', 'normalization', 'retrieval']),
    default='strategy',
    help='Type of experiment to run'
)
@click.option(
    '--num-products', 
    default=300,
    type=int,
    help='Number of products to test (default: 300)'
)
@click.option(
    '--strategies', 
    multiple=True,
    help='Specific strategies to test (can specify multiple). If none, tests all.'
)
@click.option(
    '--batch-size',
    default=20,
    type=int,
    help='Batch size for embedding generation (default: 20)'
)
@click.option(
    '--save-to-db',
    is_flag=True,
    help='Save embeddings to PostgreSQL database'
)
def run_experiment(
    experiment: str, 
    num_products: int, 
    strategies: tuple,
    batch_size: int,
    save_to_db: bool
):
    """Run embedding experiments with MLflow tracking."""
    
    print("=" * 80)
    print(f"🔬 EMBEDDING EXPERIMENTS")
    print("=" * 80)
    print(f"Experiment Type: {experiment}")
    print(f"Number of Products: {num_products}")
    print(f"Batch Size: {batch_size}")
    print(f"Save to Database: {save_to_db}")
    
    if experiment == 'strategy':
        # Convert tuple to list, use None if empty
        strategy_list = list(strategies) if strategies else None
        
        if strategy_list:
            print(f"Strategies: {', '.join(strategy_list)}")
        else:
            print("Strategies: ALL (title_only, title_features, title_category_store, title_details, comprehensive)")
        
        if save_to_db:
            print("💾 Embeddings will be saved to PostgreSQL database")
        
        print("=" * 80)
        
        # Create and run experiment
        exp = EmbeddingStrategyExperiment(
            strategies=strategy_list,
            num_products=num_products,
            batch_size=batch_size,
            save_to_db=save_to_db
        )
        
        # Run async experiment
        results = asyncio.run(exp.execute())
        
        # Print results summary
        print("\n" + "=" * 80)
        print("📊 EXPERIMENT RESULTS")
        print("=" * 80)
        
        if results["status"] == "success":
            print(f"✅ Experiment completed successfully!")
            print(f"⏱️  Total duration: {results['duration']:.2f} seconds")
            
            if "results" in results and "comparison" in results["results"]:
                print("\n📈 Strategy Rankings:")
                rankings = results["results"]["comparison"].get("rankings", {})
                
                for metric, ranking in rankings.items():
                    print(f"\n{metric}:")
                    for item in ranking[:3]:  # Show top 3
                        print(f"  {item['rank']}. {item['strategy']}: {item['value']:.2f}")
            
            print(f"\n📁 Artifacts saved to: {exp.artifacts_dir}")
            print("\n🔍 View detailed results in MLflow UI:")
            print("   Run: mlflow ui --port 5000")
            print("   Then open: http://localhost:5000")
            
        else:
            print(f"❌ Experiment failed: {results.get('error', 'Unknown error')}")
    
    elif experiment == 'normalization':
        print("⚠️  Normalization experiment not yet implemented")
        
    elif experiment == 'retrieval':
        print("⚠️  Retrieval quality experiment not yet implemented")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_experiment()