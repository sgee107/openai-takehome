"""
Test script to validate enhanced image analysis with 5 products.
"""
import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.db.database import AsyncSessionLocal, init_db
from app.scripts.load_products import ProductLoader
from sqlalchemy import select, func
from app.db.models import Product, ProductImage, ProductImageAnalysis, ProductEmbedding


async def test_image_analysis_workflow():
    """Test the enhanced image analysis workflow with 5 products."""
    print("🧪 Testing Enhanced Fashion Image Analysis")
    print("=" * 60)
    
    # Initialize database
    await init_db()
    
    # Load sample data
    data_path = Path(__file__).parent.parent.parent / "data" / "amazon_fashion_sample.json"
    with open(data_path, 'r') as f:
        all_products = json.load(f)
    
    # Select 5 products with images for testing
    test_products = []
    for product in all_products:
        if product.get('images') and len(product['images']) > 0:
            # Check if any image has a 'large' URL
            if any(img.get('large') for img in product['images']):
                test_products.append(product)
                if len(test_products) >= 5:
                    break
    
    if len(test_products) < 5:
        print(f"⚠️ Only found {len(test_products)} products with images")
    
    print(f"📦 Testing with {len(test_products)} products:")
    for i, product in enumerate(test_products, 1):
        print(f"  {i}. {product['title'][:60]}")
    
    print(f"\n⏳ Processing products with image analysis...")
    
    # Process products
    async with AsyncSessionLocal() as session:
        loader = ProductLoader(session)
        
        # Process each product individually for detailed feedback
        for i, product_data in enumerate(test_products, 1):
            print(f"\n--- Processing Product {i}/{len(test_products)} ---")
            print(f"Title: {product_data['title']}")
            print(f"Category: {product_data.get('main_category', 'Unknown')}")
            print(f"Images: {len(product_data.get('images', []))} available")
            
            # Load product
            product = await loader.load_product(product_data)
            if product:
                await session.commit()  # Get product ID
                
                # Generate embeddings (which includes image analysis)
                await loader.generate_embeddings_for_product(product, product_data)
                await session.commit()
                
                print(f"✅ Product {i} processed successfully")
            else:
                print(f"⚠️ Product {i} already exists or failed to load")
    
    # Analyze results
    print(f"\n📊 Analysis Results:")
    print("=" * 60)
    
    async with AsyncSessionLocal() as session:
        # Count total products processed
        result = await session.execute(select(func.count(Product.id)))
        total_products = result.scalar()
        print(f"Total products in database: {total_products}")
        
        # Count image analyses
        result = await session.execute(select(func.count(ProductImageAnalysis.id)))
        total_analyses = result.scalar()
        print(f"Total image analyses: {total_analyses}")
        
        # Count embeddings by strategy
        result = await session.execute(
            select(ProductEmbedding.strategy, func.count(ProductEmbedding.id))
            .group_by(ProductEmbedding.strategy)
            .order_by(ProductEmbedding.strategy)
        )
        embeddings_by_strategy = result.all()
        
        print(f"\n📈 Embeddings Generated:")
        for strategy, count in embeddings_by_strategy:
            print(f"  {strategy:25} {count:>3} embeddings")
        
        # Show sample image analyses
        result = await session.execute(
            select(ProductImageAnalysis)
            .order_by(ProductImageAnalysis.confidence.desc())
            .limit(3)
        )
        analyses = result.scalars().all()
        
        if analyses:
            print(f"\n🔍 Sample Image Analyses (Top 3 by Confidence):")
            for i, analysis in enumerate(analyses, 1):
                data = analysis.analysis_data
                print(f"\n  Analysis {i} (Confidence: {analysis.confidence:.2f}):")
                print(f"    Overview: {data.get('overview', 'N/A')[:80]}...")
                if 'visual_attributes' in data:
                    colors = data['visual_attributes'].get('primary_colors', [])
                    print(f"    Colors: {', '.join(colors) if colors else 'N/A'}")
                if 'style_analysis' in data:
                    style = data['style_analysis'].get('style_classification', 'N/A')
                    print(f"    Style: {style}")
        
        # Show enhanced embedding example
        result = await session.execute(
            select(ProductEmbedding)
            .where(ProductEmbedding.strategy == 'key_value_with_images')
            .limit(1)
        )
        enhanced_embedding = result.scalar_one_or_none()
        
        if enhanced_embedding:
            print(f"\n📝 Sample Enhanced Embedding Text:")
            print(f"Strategy: {enhanced_embedding.strategy}")
            print(f"Text (first 200 chars): {enhanced_embedding.embedding_text[:200]}...")
            
            # Check if it contains visual analysis
            if 'Visual-' in enhanced_embedding.embedding_text:
                print("✅ Contains visual analysis data!")
            else:
                print("⚠️ No visual analysis data found in embedding")
    
    print(f"\n🎉 Test completed successfully!")
    print("=" * 60)
    
    return True


async def main():
    """Run the validation test."""
    try:
        await test_image_analysis_workflow()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
