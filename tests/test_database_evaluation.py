"""Test script to evaluate current database state and table populations."""

import pytest
from sqlalchemy import text


class TestDatabaseEvaluation:
    """Evaluate the current state of database tables."""

    @pytest.mark.asyncio
    async def test_evaluate_table_counts(self, db_session):
        """Check row counts in all major tables."""
        session = db_session
        
        # Products table
        result = await session.execute(text("SELECT COUNT(*) FROM products"))
        product_count = result.scalar()
        print(f"\n📊 Products table: {product_count} rows")
        
        # Product embeddings table
        result = await session.execute(text("SELECT COUNT(*) FROM product_embeddings"))
        embedding_count = result.scalar()
        print(f"📊 Product embeddings table: {embedding_count} rows")
        
        # Product images table
        result = await session.execute(text("SELECT COUNT(*) FROM product_images"))
        image_count = result.scalar()
        print(f"📊 Product images table: {image_count} rows")
        
        # Product image analysis table
        result = await session.execute(text("SELECT COUNT(*) FROM product_image_analysis"))
        analysis_count = result.scalar()
        print(f"📊 Product image analysis table: {analysis_count} rows")
        
        # Store results for other tests
        self.counts = {
            'products': product_count,
            'embeddings': embedding_count,
            'images': image_count,
            'analyses': analysis_count
        }
        
        # Assert we have some data to work with
        assert product_count > 0, "No products found in database"

    @pytest.mark.asyncio
    async def test_evaluate_embedding_strategies(self, db_session):
        """Check what embedding strategies are available."""
        session = db_session
        
        result = await session.execute(
            text("SELECT strategy, COUNT(*) FROM product_embeddings GROUP BY strategy ORDER BY COUNT(*) DESC")
        )
        strategies = result.fetchall()
        
        print(f"\n🔍 Available embedding strategies:")
        for strategy, count in strategies:
            print(f"  - {strategy}: {count} products")
        
        # Check if our target strategy exists
        strategy_names = [s[0] for s in strategies]
        has_image_strategy = any('image' in s.lower() for s in strategy_names)
        
        # Store for other tests
        self.strategies = strategies
        self.has_image_strategy = has_image_strategy
        
        return strategies, has_image_strategy

    @pytest.mark.asyncio
    async def test_evaluate_data_quality(self, db_session):
        """Check data quality in key fields."""
        session = db_session
        
        # Check products with prices
        result = await session.execute(
            text("SELECT COUNT(*) FROM products WHERE price IS NOT NULL")
        )
        products_with_price = result.scalar()
        
        # Check products with ratings
        result = await session.execute(
            text("SELECT COUNT(*) FROM products WHERE average_rating IS NOT NULL")
        )
        products_with_rating = result.scalar()
        
        # Check products with categories
        result = await session.execute(
            text("SELECT COUNT(*) FROM products WHERE main_category IS NOT NULL")
        )
        products_with_category = result.scalar()
        
        # Check products with stores/brands
        result = await session.execute(
            text("SELECT COUNT(*) FROM products WHERE store IS NOT NULL")
        )
        products_with_store = result.scalar()
        
        print(f"\n📈 Data quality metrics:")
        print(f"  - Products with price: {products_with_price}")
        print(f"  - Products with rating: {products_with_rating}")
        print(f"  - Products with category: {products_with_category}")
        print(f"  - Products with store/brand: {products_with_store}")
        
        quality_metrics = {
            'price_coverage': products_with_price,
            'rating_coverage': products_with_rating,
            'category_coverage': products_with_category,
            'store_coverage': products_with_store
        }
        
        self.quality_metrics = quality_metrics
        return quality_metrics

    @pytest.mark.asyncio
    async def test_sample_product_data(self, db_session):
        """Sample a few products to understand data structure."""
        session = db_session
        
        # Get 3 sample products with their embeddings
        result = await session.execute(
            text("""
            SELECT p.title, p.price, p.average_rating, p.main_category, p.store,
                   pe.strategy, substring(pe.embedding_text, 1, 100) as text_preview
            FROM products p 
            LEFT JOIN product_embeddings pe ON p.id = pe.product_id
            LIMIT 3
            """)
        )
        samples = result.fetchall()
        
        print(f"\n📋 Sample product data:")
        for i, (title, price, rating, category, store, strategy, text_preview) in enumerate(samples, 1):
            print(f"  Product {i}:")
            print(f"    Title: {title}")
            print(f"    Price: ${price}" if price else "    Price: N/A")
            print(f"    Rating: {rating}/5.0" if rating else "    Rating: N/A")
            print(f"    Category: {category}")
            print(f"    Store: {store}")
            print(f"    Embedding Strategy: {strategy}")
            print(f"    Text Preview: {text_preview}...")
            print()
        
        return samples

    @pytest.mark.asyncio
    async def test_check_search_readiness(self, db_session):
        """Check if we're ready for search testing."""
        session = db_session
        
        # Check for products with embeddings that have all required fields for ranking
        result = await session.execute(
            text("""
            SELECT COUNT(*) as searchable_products
            FROM products p
            INNER JOIN product_embeddings pe ON p.id = pe.product_id
            WHERE p.title IS NOT NULL 
            AND pe.embedding IS NOT NULL
            AND pe.strategy = 'key_value_with_images'
            """)
        )
        searchable_count = result.scalar() or 0
        
        # Check for products suitable for ranking tests (with ratings)
        result = await session.execute(
            text("""
            SELECT COUNT(*) as rankable_products
            FROM products p
            INNER JOIN product_embeddings pe ON p.id = pe.product_id
            WHERE p.title IS NOT NULL 
            AND pe.embedding IS NOT NULL
            AND p.average_rating IS NOT NULL
            AND p.rating_number IS NOT NULL
            AND pe.strategy = 'key_value_with_images'
            """)
        )
        rankable_count = result.scalar() or 0
        
        # Also check for any embeddings at all (fallback)
        result = await session.execute(
            text("""
            SELECT COUNT(*) as any_searchable
            FROM products p
            INNER JOIN product_embeddings pe ON p.id = pe.product_id
            WHERE p.title IS NOT NULL 
            AND pe.embedding IS NOT NULL
            """)
        )
        any_searchable = result.scalar() or 0
        
        print(f"\n🎯 Search readiness assessment:")
        print(f"  - Products searchable with key_value_with_images: {searchable_count}")
        print(f"  - Products suitable for ranking tests: {rankable_count}")
        print(f"  - Products searchable with any strategy: {any_searchable}")
        
        search_ready = searchable_count >= 10
        ranking_ready = rankable_count >= 5
        fallback_ready = any_searchable >= 10
        
        print(f"  - Ready for image-enhanced search tests: {'✅' if search_ready else '❌'}")
        print(f"  - Ready for ranking tests: {'✅' if ranking_ready else '❌'}")
        print(f"  - Ready for fallback search tests: {'✅' if fallback_ready else '❌'}")
        
        readiness = {
            'searchable_count': searchable_count,
            'rankable_count': rankable_count,
            'any_searchable': any_searchable,
            'search_ready': search_ready,
            'ranking_ready': ranking_ready,
            'fallback_ready': fallback_ready
        }
        
        self.readiness = readiness
        return readiness

    @pytest.mark.asyncio
    async def test_comprehensive_evaluation(self, db_session):
        """Run comprehensive evaluation and print summary."""
        # Run all evaluations
        await self.test_evaluate_table_counts(db_session)
        strategies, has_images = await self.test_evaluate_embedding_strategies(db_session)
        quality_metrics = await self.test_evaluate_data_quality(db_session)
        await self.test_sample_product_data(db_session)
        readiness = await self.test_check_search_readiness(db_session)
        
        print("\n" + "="*60)
        print("📊 DATABASE EVALUATION SUMMARY")
        print("="*60)
        print(f"Total products: {getattr(self, 'counts', {}).get('products', 0)}")
        print(f"Total embeddings: {getattr(self, 'counts', {}).get('embeddings', 0)}")
        print(f"Image-based embeddings available: {'✅' if has_images else '❌'}")
        print(f"Search testing ready: {'✅' if readiness['search_ready'] else '❌'}")
        print(f"Ranking testing ready: {'✅' if readiness['ranking_ready'] else '❌'}")
        print(f"Fallback testing ready: {'✅' if readiness['fallback_ready'] else '❌'}")
        
        # Recommend test approach based on data availability
        print(f"\n🧪 RECOMMENDED TESTING APPROACH:")
        if readiness['search_ready']:
            print("  1. ✅ Full vNext pipeline testing with image-enhanced embeddings")
            print("  2. ✅ Comprehensive ranking algorithm testing")
        elif readiness['fallback_ready']:
            print("  1. ⚠️  Modified pipeline testing with available embedding strategies")
            print("  2. ⚠️  Limited ranking testing (may need to adjust strategy filter)")
        else:
            print("  1. ❌ Unit testing only - insufficient data for integration tests")
            print("  2. ❌ Mock data testing recommended")
        
        return {
            'counts': getattr(self, 'counts', {}),
            'strategies': strategies,
            'has_images': has_images,
            'quality_metrics': quality_metrics,
            'readiness': readiness
        }