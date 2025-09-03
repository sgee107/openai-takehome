#!/usr/bin/env python3
"""
Unified pipeline CLI with experiment flag.

This provides a single interface for both data loading and experimental runs,
replacing the scattered CLIs from scripts and experiments directories.
"""

import asyncio
import sys
import click
from typing import List, Optional
from pathlib import Path

from app.process.types import PipelineConfig, ProcessingResults
from app.process.core.pipeline import run_pipeline
from app.process.experiments.runner import ExperimentRunner
from app.process.strategies.registry import list_strategies


def async_wrapper(f):
    """Wrapper to handle async functions in Click commands."""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@click.group()
def cli():
    """Data processing pipeline CLI commands."""
    pass


@cli.command()
@click.option(
    '--num-products', '-n',
    type=int,
    help='Number of products to process (default: all products in dataset)'
)
@click.option(
    '--batch-size', '-b',
    type=int,
    default=20,
    help='Batch size for processing (default: 20)'
)
@click.option(
    '--strategies', '-s',
    multiple=True,
    type=click.Choice(list_strategies()),
    help='Specific strategies to use (can specify multiple). If none, uses all strategies.'
)
@click.option(
    '--save-to-db/--no-save-to-db',
    default=True,
    help='Save results to PostgreSQL database (default: True)'
)
@click.option(
    '--experiment', '-e',
    is_flag=True,
    help='Enable experiment mode with MLflow tracking and detailed metrics'
)
@click.option(
    '--experiment-name',
    type=str,
    help='Custom experiment name (only used with --experiment flag)'
)
@click.option(
    '--data-file',
    type=click.Path(exists=True, path_type=Path),
    help='Custom JSON data file (default: uses sample dataset)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Process without saving to database (useful for testing)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def run(
    num_products: Optional[int],
    batch_size: int,
    strategies: tuple,
    save_to_db: bool,
    experiment: bool,
    experiment_name: Optional[str],
    data_file: Optional[Path],
    dry_run: bool,
    verbose: bool
):
    """
    Unified data processing pipeline with optional experiment tracking.
    
    This tool consolidates both data loading and experiment functionality into
    a single interface. Use --experiment flag to enable MLflow tracking and
    detailed metrics collection.
    
    \b
    Examples:
      # Simple data loading (first 100 products)
      pipeline run -n 100
      
      # Run experiment with specific strategies
      pipeline run --experiment -n 300 -s title_only -s comprehensive
      
      # Dry run (no database saves)
      pipeline run --dry-run -n 50
      
      # Full dataset with all strategies
      pipeline run --experiment
    """
    
    async def _run_async():
        # Override save_to_db if dry_run is specified
        if dry_run:
            save_to_db_final = False
        else:
            save_to_db_final = save_to_db
        
        # Convert strategies tuple to list
        strategy_list = list(strategies) if strategies else None
        
        # Display configuration
        click.echo("=" * 80)
        click.echo("🚀 DATA PROCESSING PIPELINE")
        click.echo("=" * 80)
        
        if experiment:
            click.echo(f"🔬 Mode: EXPERIMENT (with MLflow tracking)")
            if experiment_name:
                click.echo(f"📝 Experiment name: {experiment_name}")
        else:
            click.echo(f"📊 Mode: DATA LOADING")
        
        click.echo(f"📦 Products: {num_products or 'all'}")
        click.echo(f"🔄 Batch size: {batch_size}")
        
        if strategy_list:
            click.echo(f"🎯 Strategies: {', '.join(strategy_list)}")
        else:
            available_strategies = list_strategies()
            click.echo(f"🎯 Strategies: ALL ({len(available_strategies)} strategies)")
            if verbose:
                for strategy in available_strategies:
                    click.echo(f"    - {strategy}")
        
        click.echo(f"💾 Save to database: {save_to_db_final}")
        if data_file:
            click.echo(f"📂 Data file: {data_file}")
        if dry_run:
            click.echo("🔍 DRY RUN MODE: No database operations will be performed")
        
        click.echo("=" * 80)
        
        # Create pipeline configuration
        config = PipelineConfig(
            num_products=num_products,
            batch_size=batch_size,
            strategies=strategy_list,
            save_to_db=save_to_db_final,
            enable_experiments=experiment,
            experiment_name=experiment_name
        )
        
        try:
            if experiment:
                # Run in experiment mode with MLflow tracking
                click.echo("🔬 Starting experiment mode...")
                
                # Use the experiment runner
                result = await ExperimentRunner.run_embedding_experiment(
                    strategies=strategy_list,
                    num_products=num_products or 300,  # Default for experiments
                    batch_size=batch_size,
                    save_to_db=save_to_db_final
                )
                
                if result["status"] == "success":
                    click.echo("\n" + "=" * 80)
                    click.echo("📊 EXPERIMENT RESULTS")
                    click.echo("=" * 80)
                    
                    duration = result["duration"]
                    click.echo(f"✅ Experiment completed successfully!")
                    click.echo(f"⏱️  Total duration: {duration:.2f} seconds")
                    
                    if "results" in result and "comparison" in result["results"]:
                        comparison = result["results"]["comparison"]
                        click.echo("\n📈 Strategy Rankings (Top 3):")
                        
                        rankings = comparison.get("rankings", {})
                        for metric, ranking in rankings.items():
                            if metric in ["text_length_mean", "token_count_mean"]:
                                click.echo(f"\n{metric}:")
                                for item in ranking[:3]:
                                    click.echo(f"  {item['rank']}. {item['strategy']}: {item['value']:.2f}")
                    
                    # Show MLflow information
                    click.echo(f"\n🔍 View detailed results in MLflow UI:")
                    click.echo(f"   Run: mlflow ui --port 5000")
                    click.echo(f"   Then open: http://localhost:5000")
                    
                else:
                    click.echo(f"❌ Experiment failed: {result.get('error', 'Unknown error')}")
                    sys.exit(1)
                    
            else:
                # Run in simple pipeline mode
                click.echo("📊 Starting data processing pipeline...")
                
                # TODO: Handle custom data file if provided
                # For now, the pipeline uses the default data file
                
                result = await run_pipeline(config)
                
                click.echo("\n" + "=" * 80)
                click.echo("📊 PIPELINE RESULTS")
                click.echo("=" * 80)
                
                click.echo(f"✅ Pipeline completed successfully!")
                click.echo(f"⏱️  Duration: {result.duration:.2f} seconds")
                click.echo(f"📦 Products processed: {result.products_loaded}")
                click.echo(f"🎯 Embeddings created: {result.embeddings_created}")
                click.echo(f"🔧 Strategies processed: {len(result.strategies_processed)}")
                
                if verbose:
                    click.echo(f"\n📋 Strategies used:")
                    for strategy in result.strategies_processed:
                        click.echo(f"    - {strategy}")
            
            click.echo("\n" + "=" * 80)
            
        except KeyboardInterrupt:
            click.echo("\n⚠️  Operation cancelled by user")
            sys.exit(1)
        except Exception as e:
            click.echo(f"\n❌ Operation failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    asyncio.run(_run_async())


@cli.command(name="list-strategies")
def list_strategies_cmd():
    """List all available embedding strategies."""
    strategies = list_strategies()
    
    click.echo(f"📋 Available embedding strategies ({len(strategies)}):")
    click.echo()
    
    for strategy in strategies:
        # Add brief descriptions based on strategy name
        description = ""
        if strategy == "title_only":
            description = "Simple baseline using only product title"
        elif strategy == "title_features":
            description = "Title combined with product features"
        elif strategy == "title_category_store":
            description = "Title with category and brand information"
        elif strategy == "title_details":
            description = "Title with selected product details"
        elif strategy == "comprehensive":
            description = "Comprehensive text with multiple fields"
        elif strategy.startswith("key_value"):
            if "basic" in strategy:
                description = "Structured key-value format (essential fields)"
            elif "detailed" in strategy:
                description = "Structured key-value format (all fields)"
            elif "images" in strategy:
                description = "Key-value format with image analysis"
            elif "comprehensive" in strategy:
                description = "Maximum extraction key-value format"
        
        click.echo(f"  • {strategy:30} {description}")


@cli.command()
@click.option('--host', default='localhost', help='MLflow tracking server host')
@click.option('--port', default=5000, type=int, help='MLflow UI port')
def mlflow_ui(host, port):
    """Launch MLflow UI to view experiment results."""
    import subprocess
    
    click.echo(f"🚀 Starting MLflow UI on {host}:{port}")
    click.echo(f"🔍 Open your browser to: http://{host}:{port}")
    
    try:
        subprocess.run(['mlflow', 'ui', '--host', host, '--port', str(port)], check=True)
    except FileNotFoundError:
        click.echo("❌ MLflow not found. Install with: pip install mlflow")
    except KeyboardInterrupt:
        click.echo("\n⚠️  MLflow UI stopped")




if __name__ == "__main__":
    cli()
