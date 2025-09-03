"""
Startup embedding generation functionality.

Simple wrapper around existing pipeline functionality to enable
automatic embedding generation on application startup.
"""
import asyncio
from typing import Optional

from app.settings import settings
from app.process.types import PipelineConfig
from app.process.core.pipeline import run_pipeline
from app.process.strategies.registry import list_strategies


async def run_startup_embeddings() -> bool:
    """Run startup embedding generation if enabled.
    
    This function reuses the existing pipeline functionality to generate
    embeddings for a single strategy on startup if the feature flag is enabled.
    
    Returns:
        True if successful or disabled, False if failed
    """
    if not settings.startup_embedding_enabled:
        print("🔕 Startup embeddings disabled (set STARTUP_EMBEDDING_ENABLED=true to enable)")
        return True
    
    print("🚀 Startup embedding generation enabled!")
    print(f"📍 Strategy: {settings.startup_embedding_strategy}")
    print(f"📦 Max products: {settings.startup_embedding_max_products}")
    
    try:
        # Validate the configured strategy exists
        available_strategies = list_strategies()
        if settings.startup_embedding_strategy not in available_strategies:
            print(f"❌ Invalid startup strategy: {settings.startup_embedding_strategy}")
            print(f"   Available strategies: {', '.join(available_strategies)}")
            return False
        
        # Create simple pipeline configuration
        config = PipelineConfig(
            num_products=settings.startup_embedding_max_products,
            batch_size=settings.startup_embedding_batch_size,
            strategies=[settings.startup_embedding_strategy],
            save_to_db=True,  # Save to database for app functionality
            enable_experiments=False,  # No experiments on startup
            experiment_name=None
        )
        
        # Reuse existing pipeline functionality
        results = await run_pipeline(config)
        
        print(f"✅ Startup embeddings completed!")
        print(f"📦 Processed {results.products_loaded} products")
        print(f"🎯 Generated {results.embeddings_created} embeddings")
        
        return True
        
    except Exception as e:
        print(f"❌ Startup embedding generation failed: {e}")
        print("⚠️  Application will continue without startup embeddings")
        return False


def get_startup_config_summary() -> str:
    """Get a summary of startup embedding configuration for logging.
    
    Returns:
        Configuration summary string
    """
    if not settings.startup_embedding_enabled:
        return "Startup embeddings: DISABLED"
    
    return (
        f"Startup embeddings: ENABLED | "
        f"Strategy: {settings.startup_embedding_strategy} | "
        f"Max products: {settings.startup_embedding_max_products} | "
        f"Batch size: {settings.startup_embedding_batch_size}"
    )
