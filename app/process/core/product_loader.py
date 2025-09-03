"""
Product loading and database operations.

Consolidates ProductLoader functionality from scripts and experiments
into a single, unified implementation with strategy support.
"""
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from openai import AsyncOpenAI

from app.db.models import Product, ProductImage, ProductVideo, ProductEmbedding
from app.settings import settings
from app.agents.tools.image_extraction import (
    extract_enhanced_fashion_analysis,
    store_image_analysis,
)
from app.process.core.embedding_generator import EmbeddingGenerator
from app.process.strategies.registry import get_strategy, list_strategies


class ProductLoader:
    """Load products using SQLAlchemy models with multiple embedding strategies."""
    
    def __init__(self, session: AsyncSession):
        """Initialize the product loader.
        
        Args:
            session: Async database session
        """
        self.session = session
        self.embedding_generator = EmbeddingGenerator()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._api_call_count = 0
    
    async def load_product(self, product_data: Dict[str, Any]) -> Optional[Product]:
        """Load a single product without embeddings.
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            Product instance if loaded successfully, None if already exists or failed
        """
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
    
    async def analyze_product_images(self, product: Product) -> Optional[Any]:
        """Analyze the first high-quality image of a product using OpenAI vision.
        
        Args:
            product: Product instance with images
            
        Returns:
            Image analysis result or None if failed
        """
        if not product.images:
            return None
        
        # Use the first image with a large URL
        target_image = None
        for image in product.images:
            if image.large:
                target_image = image
                break
        
        if not target_image:
            return None
        
        try:
            self._api_call_count += 1
            print(f"    🎯 [API CALL #{self._api_call_count}] Analyzing image for: {product.title[:40]}...")
            
            # Extract fashion analysis
            analysis = await extract_enhanced_fashion_analysis(
                image_url=target_image.large,
                client=self.openai_client,
                prompt_version="v1"
            )
            
            if analysis and analysis.confidence > 0.5:
                # Store the analysis in database
                await store_image_analysis(
                    session=self.session,
                    image=target_image,
                    analysis=analysis,
                    prompt_version="v1"
                )
                print(f"    ✅ [API CALL #{self._api_call_count}] Image analysis completed (confidence: {analysis.confidence:.2f})")
                return analysis
            else:
                print(f"    ⚠️ [API CALL #{self._api_call_count}] Low confidence analysis (confidence: {analysis.confidence:.2f})")
                return None
                
        except Exception as e:
            print(f"    ❌ [API CALL #{self._api_call_count}] Error analyzing image: {e}")
            return None

    async def generate_embeddings_for_product(
        self, 
        product: Product, 
        product_data: Dict[str, Any],
        strategies: Optional[List[str]] = None
    ):
        """Generate embeddings for a product using specified strategies.
        
        Args:
            product: Product instance
            product_data: Original product data dictionary
            strategies: List of strategy names to use. If None, uses all strategies.
        """
        # Use all strategies if none specified
        if strategies is None:
            strategies = list_strategies()
        
        # First, analyze images if available
        image_analysis = None
        print(f"🔍 Checking product images for {product.title[:40]}...")
        print(f"    Product has {len(product.images)} images")
        
        if product.images:
            print(f"    🎯 Calling image analysis...")
            image_analysis = await self.analyze_product_images(product)
            if image_analysis:
                print(f"    ✅ Image analysis successful with confidence {image_analysis.confidence:.2f}")
            else:
                print(f"    ❌ Image analysis failed or returned None")
        else:
            print(f"    ⚠️ No images found for product")
        
        # Generate embeddings for each strategy
        for strategy_name in strategies:
            try:
                # Get strategy instance
                strategy = get_strategy(strategy_name)
                
                # Generate text for this strategy
                print(f"    📝 Processing {strategy_name} strategy...")
                
                # Pass image analysis to strategies that can use it
                if strategy_name == 'key_value_with_images' and image_analysis:
                    print(f"    🔗 Passing image analysis to {strategy_name}")
                    embedding_text = strategy.generate(product_data, image_analysis=image_analysis)
                else:
                    embedding_text = strategy.generate(product_data)
                
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
    
    async def load_products_batch(
        self, 
        products_data: List[Dict[str, Any]], 
        batch_size: int = 10,
        strategies: Optional[List[str]] = None
    ) -> Tuple[int, int]:
        """Load products in batches with embeddings.
        
        Args:
            products_data: List of product data dictionaries
            batch_size: Number of products to process in each batch
            strategies: List of strategy names to use for embeddings
            
        Returns:
            Tuple of (products_loaded, products_failed)
        """
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
                await self.generate_embeddings_for_product(product, product_data, strategies)
            
            # Commit embeddings
            await self.session.commit()
            
            # Progress update
            processed = i + len(batch)
            print(f"  Processed {processed}/{total} products (loaded: {loaded}, skipped/failed: {failed})")
        
        return loaded, failed
    
    async def update_existing_product_embeddings(
        self,
        strategies: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> int:
        """Update embeddings for existing products with new strategies.
        
        Args:
            strategies: List of strategy names to generate. If None, uses all strategies.
            limit: Maximum number of products to update. If None, updates all.
            
        Returns:
            Number of products updated
        """
        # Use all strategies if none specified
        if strategies is None:
            strategies = list_strategies()
        
        # Get existing products
        query = select(Product)
        if limit:
            query = query.limit(limit)
        
        result = await self.session.execute(query)
        products = result.scalars().all()
        
        updated_count = 0
        
        for product in products:
            # Convert product to dictionary for strategy processing
            product_data = {
                'parent_asin': product.parent_asin,
                'main_category': product.main_category,
                'title': product.title,
                'average_rating': product.average_rating,
                'rating_number': product.rating_number,
                'price': product.price,
                'store': product.store,
                'features': product.features,
                'description': product.description,
                'categories': product.categories,
                'details': product.details,
                'bought_together': product.bought_together,
                'images': [
                    {
                        'thumb': img.thumb,
                        'large': img.large,
                        'hi_res': img.hi_res,
                        'variant': img.variant
                    } for img in product.images
                ]
            }
            
            # Generate embeddings for missing strategies
            await self.generate_embeddings_for_product(product, product_data, strategies)
            updated_count += 1
            
            # Commit in smaller batches
            if updated_count % 10 == 0:
                await self.session.commit()
                print(f"  Updated embeddings for {updated_count}/{len(products)} products...")
        
        # Final commit
        await self.session.commit()
        print(f"✅ Updated embeddings for {updated_count} products")
        
        return updated_count
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available embedding strategies.
        
        Returns:
            List of strategy names
        """
        return list_strategies()
