"""
Structured Key-Value text generation strategies for embeddings.
"""
from typing import Dict, List, Any, Optional
import re


class BaseTextStrategy:
    """Abstract base class for text generation strategies."""
    
    def generate(self, product: Dict[str, Any]) -> str:
        """Generate text from product data."""
        raise NotImplementedError


class KeyValueStrategy(BaseTextStrategy):
    """Generate explicit key-value text for embeddings."""
    
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
    
    def generate(self, product: Dict[str, Any]) -> str:
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
        
        # 6. Rating information
        if product.get('average_rating'):
            text_parts.append(f"Rating: {product['average_rating']}/5")
            if product.get('rating_number'):
                text_parts.append(f"Reviews: {product['rating_number']} reviews")
        
        # 7. Related products
        if product.get('bought_together'):
            text_parts.append(f"Often-Bought-With: {product['bought_together']}")
        
        # Join with delimiter for clarity
        return " | ".join(text_parts)


class KeyValueBasicStrategy(KeyValueStrategy):
    """Basic key-value strategy with core fields only."""
    
    def generate(self, product: Dict[str, Any]) -> str:
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
    
    def generate(self, product: Dict[str, Any]) -> str:
        """Generate detailed key-value text with all fields."""
        # Use the parent class implementation which is already comprehensive
        return super().generate(product)


class KeyValueWithImagesStrategy(KeyValueStrategy):
    """Key-value strategy with actual image analysis using OpenAI vision."""
    
    def __init__(self):
        super().__init__()
    
    def generate(self, product: Dict[str, Any], image_analysis=None) -> str:
        """Generate text with image analysis integration.
        
        Args:
            product: Product data dictionary
            image_analysis: Optional FashionImageAnalysis object from vision analysis
        """
        # Get base text from parent
        base_text = super().generate(product)
        
        # Add image analysis if available
        if image_analysis and image_analysis.confidence > 0.7:
            enrichments = []
            
            # Add visual overview
            if hasattr(image_analysis, 'overview') and image_analysis.overview:
                enrichments.append(f"Visual-Overview: {image_analysis.overview}")
            
            # Add primary colors
            if hasattr(image_analysis, 'visual_attributes') and image_analysis.visual_attributes.primary_colors:
                colors = ", ".join(image_analysis.visual_attributes.primary_colors)
                enrichments.append(f"Visual-Colors: {colors}")
            
            # Add style classification
            if hasattr(image_analysis, 'style_analysis') and image_analysis.style_analysis.style_classification:
                enrichments.append(f"Visual-Style: {image_analysis.style_analysis.style_classification}")
            
            # Add key occasions
            if hasattr(image_analysis, 'usage_context') and image_analysis.usage_context.occasions:
                occasions = ", ".join(image_analysis.usage_context.occasions[:3])
                enrichments.append(f"Visual-Occasions: {occasions}")
            
            # Add design details
            if hasattr(image_analysis, 'style_analysis') and image_analysis.style_analysis.design_details:
                details = ", ".join(image_analysis.style_analysis.design_details[:3])
                enrichments.append(f"Visual-Details: {details}")
            
            if enrichments:
                return base_text + " | " + " | ".join(enrichments)
        
        # Fallback: add basic image info if no analysis
        elif product.get('images') and isinstance(product['images'], list):
            enrichments = []
            for idx, image in enumerate(product['images'][:2]):
                if image.get('variant'):
                    enrichments.append(f"Image-{idx}-Variant: {image['variant']}")
            
            if enrichments:
                return base_text + " | " + " | ".join(enrichments)
        
        return base_text


class ComprehensiveKeyValueStrategy(KeyValueStrategy):
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
    
    def generate(self, product: Dict[str, Any]) -> str:
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
