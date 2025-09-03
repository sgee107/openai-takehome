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


# For backwards compatibility with existing code
class SimpleFashionAnalysis(BaseModel):
    """Simplified fashion analysis matching current schema."""
    description: str
    primary_colors: List[str] = Field(max_length=3)
    style_attributes: List[str] = Field(max_length=5)
    occasions: List[str] = Field(max_length=3)
    confidence: float = Field(ge=0.0, le=1.0)
