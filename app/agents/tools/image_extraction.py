"""
Image extraction using OpenAI's vision models with structured outputs.
"""
from typing import Optional
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.settings import settings
from app.prompts import IMAGE_EXTRACTION_PROMPT
from app.types.fashion_analysis import FashionImageAnalysis, SimpleFashionAnalysis
from app.db.models import ProductImage, ProductImageAnalysis


async def extract_enhanced_fashion_analysis(
    image_url: str, 
    client: Optional[AsyncOpenAI] = None,
    prompt_version: str = "v1"
) -> FashionImageAnalysis:
    """
    Extract comprehensive fashion analysis from a product image using Pydantic models.
    
    Args:
        image_url: URL of the product image to analyze
        client: Optional AsyncOpenAI client (creates new one if not provided)
        prompt_version: Version of the prompt being used
        
    Returns:
        FashionImageAnalysis Pydantic model instance
    """
    if not client:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0,
            response_format=FashionImageAnalysis,
            messages=[{
                "role": "system",
                "content": IMAGE_EXTRACTION_PROMPT
            }, {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this fashion product image and provide a comprehensive structured description."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }]
        )
        
        return response.choices[0].message.parsed
        
    except Exception as e:
        # Return a fallback analysis with minimal data
        from app.types.fashion_analysis import VisualAttributes, StyleAnalysis, UsageContext, TargetAudience
        
        return FashionImageAnalysis(
            overview="[Image analysis unavailable due to error]",
            detailed_description=f"Error occurred during analysis: {str(e)[:100]}...",
            visual_attributes=VisualAttributes(
                primary_colors=[],
                secondary_colors=[],
                patterns=[],
                textures=[]
            ),
            style_analysis=StyleAnalysis(
                silhouette="Unknown",
                fit_type="Unknown",
                design_details=[],
                style_classification="Unknown"
            ),
            usage_context=UsageContext(
                occasions=[],
                seasons=[],
                styling_suggestions=[]
            ),
            target_audience=TargetAudience(
                age_group="Unknown",
                lifestyle="Unknown"
            ),
            confidence=0.0
        )


async def extract_simple_fashion_analysis(
    image_url: str, 
    client: Optional[AsyncOpenAI] = None
) -> SimpleFashionAnalysis:
    """
    Extract basic fashion analysis for backwards compatibility.
    
    Args:
        image_url: URL of the product image to analyze
        client: Optional AsyncOpenAI client (creates new one if not provided)
        
    Returns:
        SimpleFashionAnalysis Pydantic model instance
    """
    if not client:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
    
    try:
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            temperature=0,
            response_format=SimpleFashionAnalysis,
            messages=[{
                "role": "system",
                "content": IMAGE_EXTRACTION_PROMPT
            }, {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this fashion product image and provide a structured description."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }]
        )
        
        return response.choices[0].message.parsed
        
    except Exception as e:
        # Return a fallback structure
        return SimpleFashionAnalysis(
            description=f"[Image analysis unavailable: {str(e)[:50]}...]",
            primary_colors=[],
            style_attributes=[],
            occasions=[],
            confidence=0.0
        )


async def store_image_analysis(
    session: AsyncSession,
    image: ProductImage,
    analysis: FashionImageAnalysis,
    prompt_version: str = "v1"
) -> ProductImageAnalysis:
    """
    Store fashion image analysis in the database.
    
    Args:
        session: Database session
        image: ProductImage instance
        analysis: FashionImageAnalysis to store
        prompt_version: Version of the prompt used
        
    Returns:
        Created ProductImageAnalysis instance
    """
    # Check if analysis already exists for this image/prompt combination
    stmt = select(ProductImageAnalysis).where(
        ProductImageAnalysis.image_id == image.id,
        ProductImageAnalysis.prompt_version == prompt_version
    )
    existing = await session.execute(stmt)
    existing_analysis = existing.scalar_one_or_none()
    
    if existing_analysis:
        # Update existing analysis
        existing_analysis.analysis_data = analysis.model_dump()
        existing_analysis.confidence = analysis.confidence
        await session.commit()
        return existing_analysis
    else:
        # Create new analysis
        new_analysis = ProductImageAnalysis(
            image_id=image.id,
            prompt_version=prompt_version,
            analysis_data=analysis.model_dump(),
            confidence=analysis.confidence
        )
        session.add(new_analysis)
        await session.commit()
        return new_analysis


async def get_image_analysis(
    session: AsyncSession,
    image: ProductImage,
    prompt_version: str = "v1"
) -> Optional[FashionImageAnalysis]:
    """
    Retrieve stored fashion image analysis from database.
    
    Args:
        session: Database session
        image: ProductImage instance
        prompt_version: Version of the prompt to retrieve
        
    Returns:
        FashionImageAnalysis instance if found, None otherwise
    """
    stmt = select(ProductImageAnalysis).where(
        ProductImageAnalysis.image_id == image.id,
        ProductImageAnalysis.prompt_version == prompt_version
    )
    result = await session.execute(stmt)
    analysis = result.scalar_one_or_none()
    
    if analysis and analysis.analysis_data:
        try:
            return FashionImageAnalysis.model_validate(analysis.analysis_data)
        except Exception:
            return None
    
    return None


async def enhance_product_text_with_analysis(
    product_text: str, 
    analysis: FashionImageAnalysis,
    min_confidence: float = 0.5
) -> str:
    """
    Enhance product text with fashion analysis results.
    
    Args:
        product_text: Original product text
        analysis: FashionImageAnalysis instance
        min_confidence: Minimum confidence threshold to use analysis
        
    Returns:
        Enhanced text with image information integrated
    """
    if analysis.confidence < min_confidence:
        return product_text
    
    # Build enrichments from structured analysis
    enrichments = []
    
    # Add overview
    if analysis.overview:
        enrichments.append(f"Visual Overview: {analysis.overview}")
    
    # Add primary colors
    if analysis.visual_attributes.primary_colors:
        colors = ", ".join(analysis.visual_attributes.primary_colors)
        enrichments.append(f"Colors: {colors}")
    
    # Add style classification
    if analysis.style_analysis.style_classification:
        enrichments.append(f"Style: {analysis.style_analysis.style_classification}")
    
    # Add key occasions (limit to 2)
    if analysis.usage_context.occasions:
        occasions = ", ".join(analysis.usage_context.occasions[:2])
        enrichments.append(f"Occasions: {occasions}")
    
    # Append enrichments to original text
    if enrichments:
        return product_text + " | " + " | ".join(enrichments)
    
    return product_text


# Legacy function for backwards compatibility
async def extract_fashion_image_data(image_url: str, client: Optional[AsyncOpenAI] = None):
    """Legacy function - use extract_simple_fashion_analysis instead."""
    analysis = await extract_simple_fashion_analysis(image_url, client)
    return analysis.model_dump()
