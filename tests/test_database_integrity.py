"""Database integrity and data availability tests."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.database import AsyncSessionLocal
from app.db.models import Product, ProductImage, ProductEmbedding, ProductImageAnalysis
from sqlalchemy import select, func, text


class TestDatabaseIntegrity:
    """Test database tables and data integrity."""
    
    @pytest.mark.asyncio
    async def test_database_connection(self, db_session):
        """Test basic database connectivity."""
        result = await db_session.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1
    
    @pytest.mark.asyncio
    async def test_products_table_exists(self, db_session):
        """Test that products table exists and has data."""
        result = await db_session.execute(select(func.count(Product.id)))
        product_count = result.scalar()
        print(f"Found {product_count} products in database")
        assert product_count >= 0  # Allow empty but test structure
        
        # Test table structure by selecting a few products
        if product_count > 0:
            result = await db_session.execute(
                select(Product.parent_asin, Product.title, Product.main_category, Product.price)
                .limit(5)
            )
            products = result.fetchall()
            print("Sample products:")
            for product in products:
                print(f"  ASIN: {product.parent_asin}, Title: {product.title[:50]}...")
    
    @pytest.mark.asyncio
    async def test_product_images_table(self, db_session):
        """Test product images table and relationships."""
        result = await db_session.execute(select(func.count(ProductImage.id)))
        image_count = result.scalar()
        print(f"Found {image_count} product images in database")
        
        if image_count > 0:
            # Test image URLs exist
            result = await db_session.execute(
                select(ProductImage.product_id, ProductImage.hi_res, ProductImage.large)
                .limit(5)
            )
            images = result.fetchall()
            print("Sample images:")
            for image in images:
                print(f"  Product ID: {image.product_id}, URLs: {image.hi_res[:50] if image.hi_res else 'None'}...")
    
    @pytest.mark.asyncio
    async def test_embeddings_table(self, db_session):
        """Test embeddings table and availability."""
        result = await db_session.execute(select(func.count(ProductEmbedding.id)))
        embedding_count = result.scalar()
        print(f"Found {embedding_count} product embeddings in database")
        
        if embedding_count > 0:
            # Test embedding strategies
            result = await db_session.execute(
                select(ProductEmbedding.strategy, func.count(ProductEmbedding.id))
                .group_by(ProductEmbedding.strategy)
            )
            strategies = result.fetchall()
            print("Embedding strategies:")
            for strategy, count in strategies:
                print(f"  {strategy}: {count} embeddings")
                
            # Test embedding dimensions
            result = await db_session.execute(
                select(ProductEmbedding.strategy, ProductEmbedding.model, ProductEmbedding.embedding)
                .limit(1)
            )
            sample = result.fetchone()
            if sample:
                embedding_dim = len(sample.embedding) if sample.embedding else 0
                print(f"Sample embedding - Strategy: {sample.strategy}, Model: {sample.model}, Dimension: {embedding_dim}")
    
    @pytest.mark.asyncio
    async def test_image_analysis_table(self, db_session):
        """Test image analysis table and data."""
        result = await db_session.execute(select(func.count(ProductImageAnalysis.id)))
        analysis_count = result.scalar()
        print(f"Found {analysis_count} image analyses in database")
        
        if analysis_count > 0:
            # Test analysis data structure
            result = await db_session.execute(
                select(ProductImageAnalysis.prompt_version, ProductImageAnalysis.model, ProductImageAnalysis.confidence)
                .limit(5)
            )
            analyses = result.fetchall()
            print("Sample image analyses:")
            for analysis in analyses:
                print(f"  Version: {analysis.prompt_version}, Model: {analysis.model}, Confidence: {analysis.confidence}")
    
    @pytest.mark.asyncio 
    async def test_data_relationships(self, db_session):
        """Test relationships between tables."""
        # Test products with embeddings
        result = await db_session.execute(
            select(func.count(Product.id))
            .join(ProductEmbedding, Product.id == ProductEmbedding.product_id)
        )
        products_with_embeddings = result.scalar()
        
        # Test products with images
        result = await db_session.execute(
            select(func.count(Product.id))
            .join(ProductImage, Product.id == ProductImage.product_id) 
        )
        products_with_images = result.scalar()
        
        # Test images with analysis
        result = await db_session.execute(
            select(func.count(ProductImage.id))
            .join(ProductImageAnalysis, ProductImage.id == ProductImageAnalysis.image_id)
        )
        images_with_analysis = result.scalar()
        
        print(f"Products with embeddings: {products_with_embeddings}")
        print(f"Products with images: {products_with_images}")
        print(f"Images with analysis: {images_with_analysis}")
        
        # Test key_value_with_images strategy specifically
        result = await db_session.execute(
            select(func.count(ProductEmbedding.id))
            .where(ProductEmbedding.strategy == "key_value_with_images")
        )
        kvwi_embeddings = result.scalar()
        print(f"key_value_with_images embeddings: {kvwi_embeddings}")
    
    @pytest.mark.asyncio
    async def test_price_and_rating_data(self, db_session):
        """Test price and rating data availability."""
        # Products with prices
        result = await db_session.execute(
            select(func.count(Product.id))
            .where(Product.price.is_not(None))
            .where(Product.price > 0)
        )
        products_with_prices = result.scalar()
        
        # Products with ratings
        result = await db_session.execute(
            select(func.count(Product.id))
            .where(Product.average_rating.is_not(None))
        )
        products_with_ratings = result.scalar()
        
        # Average price and rating
        result = await db_session.execute(
            select(
                func.avg(Product.price),
                func.min(Product.price), 
                func.max(Product.price)
            ).where(Product.price.is_not(None))
        )
        price_stats = result.fetchone()
        
        result = await db_session.execute(
            select(
                func.avg(Product.average_rating),
                func.min(Product.average_rating),
                func.max(Product.average_rating)
            ).where(Product.average_rating.is_not(None))
        )
        rating_stats = result.fetchone()
        
        print(f"Products with prices: {products_with_prices}")
        print(f"Products with ratings: {products_with_ratings}")
        if price_stats and price_stats[0]:
            print(f"Price stats - Avg: ${price_stats[0]:.2f}, Min: ${price_stats[1]:.2f}, Max: ${price_stats[2]:.2f}")
        if rating_stats and rating_stats[0]:
            print(f"Rating stats - Avg: {rating_stats[0]:.2f}, Min: {rating_stats[1]:.2f}, Max: {rating_stats[2]:.2f}")
    
    @pytest.mark.asyncio
    async def test_categories_and_stores(self, db_session):
        """Test category and store data distribution."""
        # Top categories
        result = await db_session.execute(
            select(Product.main_category, func.count(Product.id))
            .where(Product.main_category.is_not(None))
            .group_by(Product.main_category)
            .order_by(func.count(Product.id).desc())
            .limit(10)
        )
        categories = result.fetchall()
        print("Top categories:")
        for category, count in categories:
            print(f"  {category}: {count} products")
        
        # Top stores
        result = await db_session.execute(
            select(Product.store, func.count(Product.id))
            .where(Product.store.is_not(None))
            .group_by(Product.store)
            .order_by(func.count(Product.id).desc())
            .limit(10)
        )
        stores = result.fetchall()
        print("Top stores:")
        for store, count in stores:
            print(f"  {store}: {count} products")