"""
Load Amazon Fashion products with multiple embedding strategies using SQLAlchemy models.
"""
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from dotenv import load_dotenv

from app.db.database import AsyncSessionLocal, init_db
from app.db.models import Product, ProductImage, ProductVideo, ProductEmbedding
from app.settings import settings

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using OpenAI's embedding model."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimension = settings.openai_embedding_dimension
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tasks = [self.generate_embedding(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) < len(texts):
                await asyncio.sleep(0.5)  # Rate limiting
        
        return embeddings


class TextStrategy:
    """Different strategies for combining text fields."""
    
    @staticmethod
    def title_only(product: Dict[str, Any]) -> str:
        """Use title only - baseline strategy."""
        return product.get('title', '')
    
    @staticmethod
    def title_features(product: Dict[str, Any]) -> str:
        """Combine title and features."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('features') and isinstance(product['features'], list):
            features_text = '. '.join(product['features'][:5])  # Limit to first 5 features
            if features_text:
                parts.append(f"Features: {features_text}")
        
        return ' '.join(parts)
    
    @staticmethod
    def title_category_store(product: Dict[str, Any]) -> str:
        """Combine title, category, and store - good coverage."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('main_category'):
            parts.append(f"Category: {product['main_category']}")
        
        if product.get('store'):
            parts.append(f"Brand: {product['store']}")
        
        return ' '.join(parts)
    
    @staticmethod
    def title_details(product: Dict[str, Any]) -> str:
        """Title with selected product details."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('details') and isinstance(product['details'], dict):
            important_keys = ['Brand', 'Department', 'Material', 'Style', 'Color', 'Item model number']
            detail_parts = []
            for key in important_keys:
                if key in product['details']:
                    detail_parts.append(f"{key}: {product['details'][key]}")
            if detail_parts:
                parts.append('. '.join(detail_parts[:4]))
        
        return ' '.join(parts)
    
    @staticmethod
    def comprehensive(product: Dict[str, Any]) -> str:
        """Comprehensive text including title, features, description, and key details."""
        parts = []
        
        # Title (most important)
        if product.get('title'):
            parts.append(product['title'])
        
        # Store/Brand
        if product.get('store'):
            parts.append(f"Brand: {product['store']}")
        
        # Category
        if product.get('main_category'):
            parts.append(f"Category: {product['main_category']}")
        
        # Top features
        if product.get('features') and isinstance(product['features'], list):
            features = product['features'][:3]  # Top 3 features
            if features:
                parts.append('. '.join(features))
        
        # Key details
        if product.get('details') and isinstance(product['details'], dict):
            important_keys = ['Brand', 'Department', 'Material', 'Style', 'Color']
            detail_parts = []
            for key in important_keys:
                if key in product['details']:
                    detail_parts.append(f"{key}: {product['details'][key]}")
            if detail_parts:
                parts.append('. '.join(detail_parts[:3]))
        
        # Description (if short)
        if product.get('description') and isinstance(product['description'], list):
            desc = ' '.join(product['description'])
            if len(desc) < 200:  # Only include if short
                parts.append(desc)
        
        return ' '.join(parts)


class ProductLoader:
    """Load products using SQLAlchemy models with multiple embedding strategies."""
    
    # Define all strategies to generate
    STRATEGIES = {
        'title_only': TextStrategy.title_only,
        'title_features': TextStrategy.title_features,
        'title_category_store': TextStrategy.title_category_store,
        'title_details': TextStrategy.title_details,
        'comprehensive': TextStrategy.comprehensive
    }
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.embedding_generator = EmbeddingGenerator()
    
    async def load_product(self, product_data: Dict[str, Any]) -> Optional[Product]:
        """Load a single product without embeddings."""
        try:
            # Check if product already exists
            existing = await self.session.execute(
                select(Product).where(Product.parent_asin == product_data.get('parent_asin'))
            )
            if existing.scalar_one_or_none():
                return None
            
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
            
            # Add images
            if product_data.get('images'):
                for img_data in product_data['images']:
                    if isinstance(img_data, dict):
                        image = ProductImage(
                            thumb=img_data.get('thumb'),
                            large=img_data.get('large'),
                            hi_res=img_data.get('hi_res'),
                            variant=img_data.get('variant')
                        )
                        product.images.append(image)
            
            # Add videos
            if product_data.get('videos'):
                for video_data in product_data['videos']:
                    if isinstance(video_data, dict):
                        video = ProductVideo(
                            url=video_data.get('url', ''),
                            title=video_data.get('title')
                        )
                        product.videos.append(video)
            
            self.session.add(product)
            return product
            
        except Exception as e:
            print(f"Error loading product: {e}")
            return None
    
    async def generate_embeddings_for_product(self, product: Product, product_data: Dict[str, Any]):
        """Generate all embedding strategies for a product."""
        for strategy_name, strategy_func in self.STRATEGIES.items():
            try:
                # Generate text for this strategy
                embedding_text = strategy_func(product_data)
                
                if not embedding_text:
                    continue
                
                # Generate embedding
                embedding_vector = await self.embedding_generator.generate_embedding(embedding_text)
                
                if not embedding_vector:
                    continue
                
                # Create ProductEmbedding
                product_embedding = ProductEmbedding(
                    product_id=product.id,
                    strategy=strategy_name,
                    embedding_text=embedding_text[:1000],  # Store first 1000 chars for reference
                    embedding=embedding_vector,
                    model=settings.openai_embedding_model
                )
                
                self.session.add(product_embedding)
                
            except Exception as e:
                print(f"Error generating {strategy_name} embedding for product {product.id}: {e}")
    
    async def load_products_batch(self, products_data: List[Dict[str, Any]], batch_size: int = 10):
        """Load products in batches with embeddings."""
        total = len(products_data)
        loaded = 0
        failed = 0
        
        for i in range(0, total, batch_size):
            batch = products_data[i:i + batch_size]
            batch_products = []
            
            # First pass: Load all products
            for product_data in batch:
                product = await self.load_product(product_data)
                if product:
                    batch_products.append((product, product_data))
                    loaded += 1
                else:
                    failed += 1
            
            # Commit products to get IDs
            await self.session.commit()
            
            # Second pass: Generate embeddings for all products
            for product, product_data in batch_products:
                await self.generate_embeddings_for_product(product, product_data)
            
            # Commit embeddings
            await self.session.commit()
            
            # Progress update
            processed = i + len(batch)
            print(f"  Processed {processed}/{total} products (loaded: {loaded}, skipped/failed: {failed})")
        
        return loaded, failed


async def analyze_results(session: AsyncSession):
    """Analyze the loaded products and embeddings."""
    # Count products
    result = await session.execute(select(func.count(Product.id)))
    total_products = result.scalar()
    
    # Count embeddings by strategy
    result = await session.execute(
        select(ProductEmbedding.strategy, func.count(ProductEmbedding.id))
        .group_by(ProductEmbedding.strategy)
        .order_by(ProductEmbedding.strategy)
    )
    embeddings_by_strategy = result.all()
    
    # Count products with at least one embedding
    result = await session.execute(
        select(func.count(func.distinct(ProductEmbedding.product_id)))
    )
    products_with_embeddings = result.scalar()
    
    # Products by category
    result = await session.execute(
        select(Product.main_category, func.count(Product.id))
        .group_by(Product.main_category)
        .order_by(Product.main_category)
    )
    categories = result.all()
    
    print("\n📊 Data Loading Results:")
    print(f"  Total products: {total_products}")
    print(f"  Products with embeddings: {products_with_embeddings}")
    print(f"  Coverage: {(products_with_embeddings/total_products*100):.1f}%" if total_products > 0 else "N/A")
    
    print(f"\n📈 Embeddings by Strategy:")
    total_embeddings = 0
    for strategy, count in embeddings_by_strategy:
        print(f"    {strategy:25} {count:>6} embeddings")
        total_embeddings += count
    print(f"    {'TOTAL':25} {total_embeddings:>6} embeddings")
    
    print(f"\n📦 Products by Category:")
    for category, count in categories:
        print(f"    {category:30} {count:>6} products")


async def main():
    """Main function to load products with multiple embedding strategies."""
    print("🚀 Starting product data loader with multiple embedding strategies")
    print(f"📍 Using database: {settings.database_url}")
    print(f"🤖 Embedding model: {settings.openai_embedding_model}")
    print(f"📐 Embedding dimension: {settings.openai_embedding_dimension}")
    
    # Initialize database
    print("\n🔧 Initializing database...")
    await init_db()
    
    # Load data file
    data_path = Path(__file__).parent.parent.parent / "data" / "amazon_fashion_sample.json"
    print(f"📂 Loading data from: {data_path}")
    
    with open(data_path, 'r') as f:
        products_data = json.load(f)
    
    print(f"📦 Found {len(products_data)} products to load")
    
    # Strategies that will be generated
    print(f"\n🎯 Will generate {len(ProductLoader.STRATEGIES)} embedding strategies:")
    for strategy in ProductLoader.STRATEGIES:
        print(f"    - {strategy}")
    
    # Load products
    async with AsyncSessionLocal() as session:
        loader = ProductLoader(session)
        
        print(f"\n⏳ Loading products with embeddings...")
        loaded, failed = await loader.load_products_batch(products_data, batch_size=10)
        
        print(f"\n✅ Loading complete!")
        print(f"  Successfully loaded: {loaded} products")
        print(f"  Skipped/Failed: {failed}")
        
        # Analyze results
        await analyze_results(session)


if __name__ == "__main__":
    asyncio.run(main())