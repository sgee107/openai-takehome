"""
Pydantic models for fashion image analysis structured outputs.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class VisualAttributes(BaseModel):
    """Visual characteristics of the fashion item."""
    primary_colors: List[str] = Field(max_length=3, description="Most dominant colors")
    secondary_colors: List[str] = Field(max_length=2, description="Accent or detail colors")
    patterns: List[str] = Field(max_length=3, description="Visual patterns like stripes, floral, geometric")
    textures: List[str] = Field(max_length=3, description="Fabric textures visible in image")


class StyleAnalysis(BaseModel):
    """Style and design characteristics."""
    silhouette: str = Field(description="Overall shape/cut - fitted, loose, A-line, straight, etc.")
    fit_type: str = Field(description="How it fits - slim, regular, relaxed, oversized, etc.")
    design_details: List[str] = Field(max_length=4, description="Specific design elements")
    style_classification: str = Field(description="Fashion category - casual, business, formal, athletic, etc.")


class UsageContext(BaseModel):
    """When and how the item would be used."""
    occasions: List[str] = Field(max_length=4, description="Appropriate occasions or events")
    seasons: List[str] = Field(max_length=2, description="Suitable seasons")
    styling_suggestions: List[str] = Field(max_length=3, description="How to wear or style this item")


class TargetAudience(BaseModel):
    """Who this item is designed for."""
    age_group: str = Field(description="Target age demographic")
    lifestyle: str = Field(description="Target lifestyle - professional, casual, active, etc.")




class FashionImageAnalysis(BaseModel):
    """Complete fashion analysis of a product image."""
    overview: str = Field(description="One sentence summary of the item's key characteristics")
    detailed_description: str = Field(description="2-3 sentences with fashion terminology for styling context")
    visual_attributes: VisualAttributes
    style_analysis: StyleAnalysis
    usage_context: UsageContext
    target_audience: TargetAudience
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the analysis")
    
    # Enhanced styling context fields
    seasonal_appropriateness: Optional[List[str]] = Field(None, max_length=3, description="Seasons this item works best in")
    enhanced_styling_suggestions: Optional[List[str]] = Field(None, max_length=4, description="Detailed styling tips and combinations")
    complementary_items: Optional[List[str]] = Field(None, max_length=4, description="Items that pair well with this piece")
    wardrobe_role: Optional[str] = Field(None, description="Role in wardrobe - statement piece, basic, accent, etc.")
    
    # Enhanced target audience fields
    demographic_fit: Optional[str] = Field(None, description="Primary demographic - age, gender, lifestyle descriptors")
    lifestyle_alignment: Optional[str] = Field(None, description="Lifestyle match - professional, casual, active, creative, etc.")
    brand_personality: Optional[str] = Field(None, description="Brand personality this item conveys")
    
    # Enhanced technical details fields
    fit_description: Optional[str] = Field(None, description="Detailed fit characteristics and body interaction")
    silhouette_analysis: Optional[str] = Field(None, description="Silhouette impact and visual effect")
    functional_features: Optional[List[str]] = Field(None, max_length=4, description="Functional aspects and practical benefits")


# For backwards compatibility with existing code
class SimpleFashionAnalysis(BaseModel):
    """Simplified fashion analysis matching current schema."""
    description: str
    primary_colors: List[str] = Field(max_length=3)
    style_attributes: List[str] = Field(max_length=5)
    occasions: List[str] = Field(max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
