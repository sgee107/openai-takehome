"""
Mock data service for frontend development.
Loads and transforms Amazon fashion sample data for API responses.
"""

import json
import random
import time
from pathlib import Path
from typing import List, Optional

from app.types.frontend import ProductResult, ProductImage


class MockDataService:
    """Service for handling mock fashion product data"""
    
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            # Default to sample data in the project
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / "data" / "amazon_fashion_sample.json"
        
        self.data_path = Path(data_path)
        self._products: List[ProductResult] = []
        self._load_data()
    
    def _load_data(self):
        """Load and transform raw Amazon data to ProductResult format"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Sample data not found at {self.data_path}")
        
        with open(self.data_path) as f:
            raw_data = json.load(f)
        
        self._products = []
        for index, item in enumerate(raw_data):
            # Transform images
            images = []
            for img in item.get("images", []):
                images.append(ProductImage(
                    thumb=img.get("thumb"),
                    large=img.get("large"),
                    hi_res=img.get("hi_res"),
                    variant=img.get("variant")
                ))
            
            # Create ProductResult with mock similarity data
            product = ProductResult(
                parent_asin=item["parent_asin"],
                title=item["title"],
                main_category=item.get("main_category", "Fashion"),
                store=item.get("store"),
                images=images,
                price=item.get("price"),
                average_rating=item.get("average_rating"),
                rating_number=item.get("rating_number"),
                features=item.get("features", []),
                description=item.get("description", []),
                details=item.get("details", {}),
                categories=item.get("categories", []),
                videos=item.get("videos", []),
                bought_together=item.get("bought_together"),
                # Mock fields for ranking
                similarity_score=0.95 - (index * 0.01),  # Decreasing scores
                rank=index + 1
            )
            
            self._products.append(product)
    
    def get_all_products(self) -> List[ProductResult]:
        """Get all products"""
        return self._products
    
    def search_products(self, query: str, limit: int = 20) -> List[ProductResult]:
        """
        Mock search functionality with simple text matching.
        In a real system, this would use embedding similarity.
        """
        if not query or query.strip() == "":
            # Return top products for empty query
            return self._products[:limit]
        
        query_lower = query.lower()
        matched_products = []
        
        # Simple text matching across multiple fields
        for product in self._products:
            score = 0.0
            
            # Check title (highest weight)
            if query_lower in product.title.lower():
                score += 0.6
            
            # Check category
            if query_lower in product.main_category.lower():
                score += 0.3
            
            # Check store
            if product.store and query_lower in product.store.lower():
                score += 0.2
            
            # Check features
            for feature in product.features:
                if query_lower in feature.lower():
                    score += 0.1
                    break
            
            # Check details
            for key, value in product.details.items():
                if isinstance(value, str) and query_lower in value.lower():
                    score += 0.05
                    break
            
            if score > 0:
                # Update similarity score based on match quality
                product.similarity_score = min(0.95, 0.5 + score)
                matched_products.append(product)
        
        # Sort by similarity score (descending) and take top results
        matched_products.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Update ranks
        for i, product in enumerate(matched_products):
            product.rank = i + 1
        
        return matched_products[:limit]
    
    def get_random_products(self, count: int = 20) -> List[ProductResult]:
        """Get random products for testing"""
        products = random.sample(self._products, min(count, len(self._products)))
        
        # Randomize similarity scores
        for i, product in enumerate(products):
            product.similarity_score = random.uniform(0.7, 0.95)
            product.rank = i + 1
        
        # Sort by similarity score
        products.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return products
    
    def get_product_by_asin(self, asin: str) -> Optional[ProductResult]:
        """Get a specific product by ASIN"""
        for product in self._products:
            if product.parent_asin == asin:
                return product
        return None
    
    def get_sample_images(self, count: int = 50) -> List[str]:
        """Get sample image URLs for background generation"""
        image_urls = []
        for product in self._products[:count]:
            for image in product.images:
                if image.thumb:
                    image_urls.append(image.thumb)
                elif image.large:
                    image_urls.append(image.large)
                
                if len(image_urls) >= count:
                    break
            
            if len(image_urls) >= count:
                break
        
        return image_urls
    
    @property
    def total_products(self) -> int:
        """Total number of products in the dataset"""
        return len(self._products)


# Global instance for the application
mock_data_service = MockDataService()