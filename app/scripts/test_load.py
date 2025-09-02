"""
Test data loader with 5 products using title_only strategy.
"""
import json
import asyncio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from openai import AsyncOpenAI
from sqlalchemy import select, func
from app.db.database import AsyncSessionLocal, init_db
from app.db.models import Product, ProductImage, ProductEmbedding
from app.settings import settings


async def test_load():
    """Test loading 5 products with title embeddings."""
    print("🧪 TEST: Loading 5 products with title-only embeddings")
    print(f"📍 Database: {settings.database_url}")
    print(f"🤖 Model: {settings.openai_embedding_model}")
    
    # Initialize database
    print("\n🔧 Initializing database...")
    await init_db()
    
    # Load test data
    data_path = Path(__file__).parent.parent.parent / "data" / "amazon_fashion_sample.json"
    with open(data_path, 'r') as f:
        all_products = json.load(f)
    
    # Take only first 5 products
    test_products = all_products[:5]
    print(f"\n📦 Testing with {len(test_products)} products")
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    async with AsyncSessionLocal() as session:
        loaded_count = 0
        
        for i, product_data in enumerate(test_products, 1):
            print(f"\n[{i}/5] Processing: {product_data.get('title', 'Unknown')[:60]}...")
            
            # Check if already exists
            existing = await session.execute(
                select(Product).where(Product.parent_asin == product_data.get('parent_asin'))
            )
            if existing.scalar_one_or_none():
                print(f"  ⚠️  Product already exists, skipping")
                continue
            
            # Create product
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
            
            # Add images (just first 3 for test)
            if product_data.get('images'):
                for img_data in product_data['images'][:3]:
                    if isinstance(img_data, dict):
                        image = ProductImage(
                            thumb=img_data.get('thumb'),
                            large=img_data.get('large'),
                            hi_res=img_data.get('hi_res'),
                            variant=img_data.get('variant')
                        )
                        product.images.append(image)
            
            session.add(product)
            await session.commit()  # Commit to get product ID
            
            print(f"  ✅ Product saved (ID: {product.id})")
            
            # Generate embedding for title
            title = product_data.get('title', '')
            if title:
                print(f"  🔄 Generating embedding for title...")
                try:
                    response = await client.embeddings.create(
                        model=settings.openai_embedding_model,
                        input=title,
                        dimensions=settings.openai_embedding_dimension
                    )
                    embedding_vector = response.data[0].embedding
                    
                    # Save embedding
                    product_embedding = ProductEmbedding(
                        product_id=product.id,
                        strategy='title_only',
                        embedding_text=title[:500],
                        embedding=embedding_vector,
                        model=settings.openai_embedding_model
                    )
                    session.add(product_embedding)
                    await session.commit()
                    
                    print(f"  ✅ Embedding saved (dimension: {len(embedding_vector)})")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error generating embedding: {e}")
        
        # Show results
        print("\n" + "="*60)
        print("📊 TEST RESULTS:")
        
        # Count products
        result = await session.execute(select(func.count(Product.id)))
        total_products = result.scalar()
        
        # Count embeddings
        result = await session.execute(select(func.count(ProductEmbedding.id)))
        total_embeddings = result.scalar()
        
        print(f"  Products in database: {total_products}")
        print(f"  Embeddings in database: {total_embeddings}")
        print(f"  New products loaded in this test: {loaded_count}")
        
        # Sample query: Find similar products
        if total_embeddings > 0:
            print("\n🔍 Testing similarity search...")
            
            # Get first product with embedding
            result = await session.execute(
                select(ProductEmbedding).limit(1)
            )
            sample_embedding = result.scalar_one_or_none()
            
            if sample_embedding:
                # Find similar products using SQLAlchemy query
                from sqlalchemy import text
                
                query = text("""
                    SELECT p.title, pe.strategy, 
                           1 - (pe.embedding <=> :embedding) as similarity
                    FROM product_embeddings pe
                    JOIN products p ON p.id = pe.product_id
                    WHERE pe.id != :id
                    ORDER BY pe.embedding <=> :embedding
                    LIMIT 3
                """)
                
                # Convert embedding to list if it's a numpy array
                embedding_vector = sample_embedding.embedding
                if hasattr(embedding_vector, 'tolist'):
                    embedding_vector = embedding_vector.tolist()
                
                result = await session.execute(
                    query,
                    {
                        'embedding': embedding_vector,
                        'id': sample_embedding.id
                    }
                )
                
                similar = result.fetchall()
                
                print(f"  Query product: {sample_embedding.embedding_text[:60]}...")
                if similar:
                    print(f"  Similar products:")
                    for title, strategy, similarity in similar:
                        print(f"    - {title[:50]:50} (similarity: {similarity:.3f})")
                else:
                    print(f"  No similar products found (need more data)")
        
        print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_load())