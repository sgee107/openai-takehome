"""
Consolidated text generation strategies for embeddings.

This module unifies the TextStrategy static methods from load_products.py
and the KeyValueStrategy classes from structured_text_strategies.py into
a single, clean inheritance hierarchy.
"""
from typing import Dict, List, Any, Optional
import re
from abc import ABC, abstractmethod


class BaseTextStrategy(ABC):
    """Abstract base class for all text generation strategies."""
    
    @abstractmethod
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Generate text from product data.
        
        Args:
            product: Product data dictionary
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Generated text for embedding
        """
        pass


class TitleOnlyStrategy(BaseTextStrategy):
    """Simple strategy using only the product title."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Use title only - baseline strategy."""
        return product.get('title', '')


class TitleFeaturesStrategy(BaseTextStrategy):
    """Strategy combining title and features."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Combine title and features."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('features') and isinstance(product['features'], list):
            features_text = '. '.join(product['features'][:5])  # Limit to first 5 features
            if features_text:
                parts.append(f"Features: {features_text}")
        
        return ' '.join(parts)


class TitleCategoryStoreStrategy(BaseTextStrategy):
    """Strategy combining title, category, and store."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Combine title, category, and store - good coverage."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('main_category'):
            parts.append(f"Category: {product['main_category']}")
        
        if product.get('store'):
            parts.append(f"Brand: {product['store']}")
        
        return ' '.join(parts)


class TitleDetailsStrategy(BaseTextStrategy):
    """Strategy using title with selected product details."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
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


class ComprehensiveStrategy(BaseTextStrategy):
    """Comprehensive text including title, features, description, and key details."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
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


class KeyValueStrategy(BaseTextStrategy):
    """Base class for structured key-value text generation strategies."""
    
    def __init__(self):
        self.key_mappings = {
            'Brand': 'Brand',
            'Color': 'Color',
            'Material': 'Material',
            'Style': 'Style',
            'Department': 'Department',
            'Closure Type': 'Closure',
            'Country of Origin': 'Made in',
            'Age Range (Description)': 'Age Group',
            'Item Weight': 'Weight',
            'Package Dimensions': 'Size'
        }
        
        self.feature_patterns = {
            'material': ['cotton', 'polyester', 'wool', 'leather', 'synthetic', 'fabric', 'nylon', 'silk'],
            'size': ['fits', 'sizing', 'length', 'width', 'small', 'medium', 'large'],
            'care': ['wash', 'dry clean', 'iron', 'bleach', 'machine wash', 'hand wash'],
            'occasion': ['casual', 'formal', 'sport', 'work', 'party', 'business', 'everyday'],
            'season': ['summer', 'winter', 'spring', 'fall', 'all-season', 'weather']
        }
    
    def generate_base_info(self, product: Dict[str, Any]) -> str:
        """Generate base product information."""
        text_parts = [
            f"Product: {product['title']}",
            f"Category: {product.get('main_category', 'Unknown')}",
            f"Store: {product.get('store', 'Unknown Brand')}",
            f"ASIN: {product.get('parent_asin', 'Unknown')}"
        ]
        
        # Price information with tier
        if product.get('price'):
            price = product['price']
            text_parts.append(f"Price: ${price}")
            
            # Add price tier
            if price < 20:
                text_parts.append("Price Tier: Budget")
            elif price < 50:
                text_parts.append("Price Tier: Mid-range")
            else:
                text_parts.append("Price Tier: Premium")
        
        return " | ".join(text_parts)
    
    def extract_product_details(self, details: Dict[str, Any]) -> List[str]:
        """Extract key-value pairs from details JSON."""
        extracted = []
        
        for original_key, display_key in self.key_mappings.items():
            if original_key in details:
                value = details[original_key]
                # Clean up the value
                if isinstance(value, str):
                    value = value.replace(' ‏ : ‎ ', '').strip()
                    value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                extracted.append(f"{display_key}: {value}")
        
        return extracted
    
    def process_features(self, features: List[str]) -> List[str]:
        """Convert feature list into categorized key-value pairs."""
        processed = []
        
        for feature in features:
            feature_lower = feature.lower()
            categorized = False
            
            # Try to categorize the feature
            for category, keywords in self.feature_patterns.items():
                if any(keyword in feature_lower for keyword in keywords):
                    processed.append(f"Feature-{category}: {feature}")
                    categorized = True
                    break
            
            # Generic feature if no category matched
            if not categorized:
                processed.append(f"Feature: {feature}")
        
        return processed
    
    def extract_category_hierarchy(self, categories: List[List[str]]) -> str:
        """Extract and format category hierarchy."""
        if not categories:
            return ""
        
        # Take the most specific (longest) category path
        longest_path = max(categories, key=len) if categories else []
        
        if longest_path:
            # Create hierarchical representation
            hierarchy = " > ".join(longest_path)
            # Also create individual category tags
            tags = [f"Category-Level-{i}: {cat}" for i, cat in enumerate(longest_path)]
            
            return f"Category-Hierarchy: {hierarchy} | " + " | ".join(tags)
        
        return ""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Generate complete structured text."""
        text_parts = []
        
        # 1. Base product info
        text_parts.append(self.generate_base_info(product))
        
        # 2. Extract from details (100% coverage)
        if product.get('details'):
            details_text = self.extract_product_details(product['details'])
            text_parts.extend(details_text)
        
        # 3. Process features (53% coverage when present)
        if product.get('features'):
            features_text = self.process_features(product['features'][:5])  # Limit to 5 features
            text_parts.extend(features_text)
        
        # 4. Category hierarchy
        if product.get('categories'):
            cat_text = self.extract_category_hierarchy(product['categories'])
            if cat_text:
                text_parts.append(cat_text)
        
        # 5. Description (only 7% have this)
        if product.get('description') and isinstance(product['description'], list):
            # Limit description length
            desc_text = ' '.join(product['description'][:2])  # First 2 items
            if len(desc_text) > 200:
                desc_text = desc_text[:200] + "..."
            text_parts.append(f"Description: {desc_text}")
        
        # 6. Related products
        if product.get('bought_together'):
            text_parts.append(f"Often-Bought-With: {product['bought_together']}")
        
        # Join with delimiter for clarity
        return " | ".join(text_parts)


class KeyValueBasicStrategy(KeyValueStrategy):
    """Basic key-value strategy with core fields only."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Generate basic key-value text with essential fields only."""
        text_parts = []
        
        # Base info
        text_parts.append(self.generate_base_info(product))
        
        # Only essential details
        if product.get('details'):
            essential_keys = ['Brand', 'Color', 'Material', 'Department']
            extracted = []
            for key in essential_keys:
                if key in product['details']:
                    value = product['details'][key]
                    if isinstance(value, str):
                        value = value.replace(' ‏ : ‎ ', '').strip()
                    extracted.append(f"{key}: {value}")
            text_parts.extend(extracted)
        
        # Top 3 features only
        if product.get('features'):
            features = product['features'][:3]
            for feature in features:
                text_parts.append(f"Feature: {feature}")
        
        return " | ".join(text_parts)


class KeyValueDetailedStrategy(KeyValueStrategy):
    """Detailed key-value strategy with all available fields."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Generate detailed key-value text with all fields."""
        # Use the parent class implementation which is already comprehensive
        return super().generate(product, **kwargs)


class KeyValueWithImagesStrategy(KeyValueStrategy):
    """Key-value strategy with mandatory image analysis integration."""
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Generate text with enhanced image analysis integration.
        
        Args:
            product: Product data dictionary
            image_analysis: FashionImageAnalysis object from vision analysis (required)
        """
        # Get base text from parent
        base_text = super().generate(product, **kwargs)
        
        # Image analysis is REQUIRED for this strategy
        image_analysis = kwargs.get('image_analysis')
        if not image_analysis:
            return base_text + " | Visual-Analysis: NOT_AVAILABLE"
        
        enrichments = []
        
        # Core visual analysis
        if image_analysis.overview:
            enrichments.append(f"Visual-Overview: {image_analysis.overview}")
        
        if image_analysis.visual_attributes.primary_colors:
            colors = ", ".join(image_analysis.visual_attributes.primary_colors)
            enrichments.append(f"Visual-Colors: {colors}")
        
        if image_analysis.style_analysis.style_classification:
            enrichments.append(f"Visual-Style: {image_analysis.style_analysis.style_classification}")
        
        # Enhanced styling context
        if image_analysis.seasonal_appropriateness:
            seasons = ", ".join(image_analysis.seasonal_appropriateness)
            enrichments.append(f"Seasonal-Perfect: {seasons}")
        
        if image_analysis.enhanced_styling_suggestions:
            styling = ", ".join(image_analysis.enhanced_styling_suggestions[:2])
            enrichments.append(f"Styling-Tips: {styling}")
        
        if image_analysis.complementary_items:
            complements = ", ".join(image_analysis.complementary_items[:3])
            enrichments.append(f"Pairs-With: {complements}")
        
        if image_analysis.wardrobe_role:
            enrichments.append(f"Wardrobe-Role: {image_analysis.wardrobe_role}")
        
        # Enhanced target audience
        if image_analysis.demographic_fit:
            enrichments.append(f"Target-Demo: {image_analysis.demographic_fit}")
        
        if image_analysis.lifestyle_alignment:
            enrichments.append(f"Lifestyle-Match: {image_analysis.lifestyle_alignment}")
        
        if image_analysis.brand_personality:
            enrichments.append(f"Brand-Vibe: {image_analysis.brand_personality}")
        
        # Technical details
        if image_analysis.fit_description:
            enrichments.append(f"Fit-Profile: {image_analysis.fit_description}")
        
        if image_analysis.silhouette_analysis:
            enrichments.append(f"Silhouette-Effect: {image_analysis.silhouette_analysis}")
        
        if image_analysis.functional_features:
            features = ", ".join(image_analysis.functional_features[:3])
            enrichments.append(f"Functional-Features: {features}")
        
        # Add confidence score
        enrichments.append(f"Visual-Confidence: {image_analysis.confidence:.2f}")
        
        return base_text + " | " + " | ".join(enrichments) if enrichments else base_text


class KeyValueComprehensiveStrategy(KeyValueStrategy):
    """Maximum extraction strategy with all fields and enhanced processing."""
    
    def __init__(self):
        super().__init__()
        # Extended key mappings
        self.extended_key_mappings = {
            **self.key_mappings,
            'Product Dimensions': 'Dimensions',
            'Manufacturer': 'Manufacturer',
            'ASIN': 'ASIN',
            'Item model number': 'Model',
            'Customer Reviews': 'Customer Feedback',
            'Best Sellers Rank': 'Popularity Rank',
            'Date First Available': 'Available Since'
        }
    
    def extract_all_details(self, details: Dict[str, Any]) -> List[str]:
        """Extract all possible key-value pairs from details."""
        extracted = []
        
        # Process known mappings first
        for original_key, display_key in self.extended_key_mappings.items():
            if original_key in details:
                value = details[original_key]
                if isinstance(value, str):
                    value = value.replace(' ‏ : ‎ ', '').strip()
                    value = re.sub(r'\s+', ' ', value)
                extracted.append(f"{display_key}: {value}")
        
        # Process any remaining keys
        processed_keys = set(self.extended_key_mappings.keys())
        for key, value in details.items():
            if key not in processed_keys and value:
                # Clean the key
                clean_key = key.replace('_', ' ').title()
                if isinstance(value, str):
                    value = value.replace(' ‏ : ‎ ', '').strip()
                extracted.append(f"{clean_key}: {value}")
        
        return extracted
    
    def generate(self, product: Dict[str, Any], **kwargs) -> str:
        """Generate comprehensive text with maximum field extraction."""
        text_parts = []
        
        # 1. Base product info
        text_parts.append(self.generate_base_info(product))
        
        # 2. All details
        if product.get('details'):
            details_text = self.extract_all_details(product['details'])
            text_parts.extend(details_text)
        
        # 3. All features (up to 10)
        if product.get('features'):
            features_text = self.process_features(product['features'][:10])
            text_parts.extend(features_text)
        
        # 4. Category hierarchy
        if product.get('categories'):
            cat_text = self.extract_category_hierarchy(product['categories'])
            if cat_text:
                text_parts.append(cat_text)
        
        # 5. Full description (if available)
        if product.get('description') and isinstance(product['description'], list):
            desc_text = ' '.join(product['description'])
            if len(desc_text) > 500:
                desc_text = desc_text[:500] + "..."
            text_parts.append(f"Description: {desc_text}")
        
        # 6. Related products
        if product.get('bought_together'):
            text_parts.append(f"Often-Bought-With: {product['bought_together']}")
        
        return " | ".join(text_parts)
