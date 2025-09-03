#!/usr/bin/env python3
"""
Quick test to verify the new consolidated process structure works.
"""

async def test_imports():
    """Test that all new process modules can be imported correctly."""
    print("🧪 Testing process structure imports...")
    
    try:
        # Test types
        from app.process.types import PipelineConfig, ProcessingResults, EmbeddingStrategy
        print("✅ Types imported successfully")
        
        # Test strategies
        from app.process.strategies.registry import get_strategy, list_strategies, get_all_strategies
        from app.process.strategies.text_strategies import TitleOnlyStrategy, ComprehensiveStrategy
        print("✅ Strategies imported successfully")
        
        # Test core classes
        from app.process.core.embedding_generator import EmbeddingGenerator
        from app.process.core.product_loader import ProductLoader
        from app.process.core.pipeline import DataPipeline, run_pipeline
        print("✅ Core classes imported successfully")
        
        # Test experiments
        from app.process.experiments.runner import ExperimentRunner, BaseExperiment
        from app.process.experiments.metrics import calculate_semantic_metrics
        print("✅ Experiment modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


async def test_strategy_registry():
    """Test strategy registration and retrieval."""
    print("\n🧪 Testing strategy registry...")
    
    try:
        from app.process.strategies.registry import get_strategy, list_strategies
        
        # List strategies
        strategies = list_strategies()
        print(f"✅ Found {len(strategies)} strategies: {', '.join(strategies)}")
        
        # Test getting a specific strategy
        title_only = get_strategy('title_only')
        test_product = {
            'title': 'Test Product',
            'main_category': 'Test Category',
            'store': 'Test Store'
        }
        
        result = title_only.generate(test_product)
        print(f"✅ title_only strategy generated: '{result}'")
        
        # Test comprehensive strategy
        comprehensive = get_strategy('comprehensive')
        result = comprehensive.generate(test_product)
        print(f"✅ comprehensive strategy generated: '{result[:100]}...'")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False


async def test_pipeline_config():
    """Test pipeline configuration."""
    print("\n🧪 Testing pipeline configuration...")
    
    try:
        from app.process.types import PipelineConfig
        
        config = PipelineConfig(
            num_products=10,
            batch_size=5,
            strategies=['title_only', 'comprehensive'],
            save_to_db=False,
            enable_experiments=True
        )
        
        print(f"✅ Pipeline config created: {config}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline config test failed: {e}")
        return False


async def test_core_classes():
    """Test core class instantiation."""
    print("\n🧪 Testing core classes...")
    
    try:
        from app.process.core.embedding_generator import EmbeddingGenerator
        
        # Test EmbeddingGenerator instantiation
        generator = EmbeddingGenerator()
        model_info = generator.get_model_info()
        print(f"✅ EmbeddingGenerator created with model: {model_info['model']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core classes test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing consolidated process structure")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_strategy_registry,
        test_pipeline_config,
        test_core_classes
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("🎉 Process structure is working correctly!")
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        print("🔧 Please check the errors above")
    
    return passed == total


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
