"""
Test script to run image analysis on existing products in the database.
"""
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from openai import AsyncOpenAI

from app.db.database import AsyncSessionLocal, init_db
from app.db.models import Product, ProductImage, ProductImageAnalysis
from app.settings import settings
from app.agents.tools.image_extraction import (
    extract_enhanced_fashion_analysis,
    store_image_analysis
)


async def test_image_analysis_on_existing_products():
    """Test image analysis on existing products."""
    print("🔍 Testing Image Analysis on Existing Products")
    print("=" * 60)
    
    # Initialize database
    await init_db()
    
    async with AsyncSessionLocal() as session:
        # Get 3 products with images
        stmt = (
            select(Product)
            .join(ProductImage)
            .where(ProductImage.large.isnot(None))
            .limit(3)
        )
        result = await session.execute(stmt)
        products = result.scalars().unique().all()
        
        print(f"📦 Found {len(products)} products with images")
        
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        for i, product in enumerate(products, 1):
            print(f"\n--- Testing Product {i} ---")
            print(f"Title: {product.title}")
            print(f"ASIN: {product.parent_asin}")
            
            # Load images relationship
            await session.refresh(product, ['images'])
            print(f"Images available: {len(product.images)}")
            
            # Find first image with large URL
            target_image = None
            for image in product.images:
                if image.large:
                    target_image = image
                    print(f"Using image URL: {image.large[:60]}...")
                    break
            
            if not target_image:
                print("❌ No large image URL found")
                continue
            
            try:
                print("🎯 Calling OpenAI vision analysis...")
                
                # Extract fashion analysis
                analysis = await extract_enhanced_fashion_analysis(
                    image_url=target_image.large,
                    client=client,
                    prompt_version="test_v1"
                )
                
                if analysis:
                    print(f"✅ Analysis completed! Confidence: {analysis.confidence:.2f}")
                    print(f"Overview: {analysis.overview}")
                    print(f"Colors: {', '.join(analysis.visual_attributes.primary_colors) if analysis.visual_attributes.primary_colors else 'None'}")
                    print(f"Style: {analysis.style_analysis.style_classification}")
                    
                    # Store the analysis
                    stored = await store_image_analysis(
                        session=session,
                        image=target_image,
                        analysis=analysis,
                        prompt_version="test_v1"
                    )
                    print(f"💾 Analysis stored in database with ID: {stored.id}")
                else:
                    print("❌ Analysis returned None")
                    
            except Exception as e:
                print(f"❌ Error during analysis: {e}")
                import traceback
                traceback.print_exc()
            
            await session.commit()
    
    # Check results
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ProductImageAnalysis)
            .where(ProductImageAnalysis.prompt_version == "test_v1")
        )
        analyses = result.scalars().all()
        
        print(f"\n📊 Final Results:")
        print(f"Total analyses stored: {len(analyses)}")
        
        for i, analysis in enumerate(analyses, 1):
            print(f"\nAnalysis {i}:")
            print(f"  Confidence: {analysis.confidence:.2f}")
            data = analysis.analysis_data
            print(f"  Overview: {data.get('overview', 'N/A')[:60]}...")
            if 'visual_attributes' in data:
                colors = data['visual_attributes'].get('primary_colors', [])
                print(f"  Colors: {', '.join(colors) if colors else 'N/A'}")
    
    print(f"\n🎉 Image analysis test completed!")


if __name__ == "__main__":
    asyncio.run(test_image_analysis_on_existing_products())
