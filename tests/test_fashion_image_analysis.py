"""
Simple tests for fashion image analysis functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from app.types.fashion_analysis import (
    FashionImageAnalysis, 
    SimpleFashionAnalysis,
    VisualAttributes,
    StyleAnalysis, 
    UsageContext,
    TargetAudience
)
from app.agents.tools.image_extraction import (
    extract_enhanced_fashion_analysis,
    store_image_analysis,
    get_image_analysis,
    enhance_product_text_with_analysis
)
from app.db.models import Product, ProductImage


class TestFashionAnalysisBasics:
    """Basic tests for fashion analysis functionality."""
    
    @pytest.fixture
    def sample_analysis(self):
        """Simple sample analysis for testing."""
        return FashionImageAnalysis(
            overview="A blue cotton t-shirt",
            detailed_description="Casual t-shirt perfect for everyday wear.",
            visual_attributes=VisualAttributes(
                primary_colors=["Blue"],
                secondary_colors=[],
                patterns=["Solid"],
                textures=["Cotton"]
            ),
            style_analysis=StyleAnalysis(
                silhouette="Regular",
                fit_type="Standard",
                design_details=["Round neck"],
                style_classification="Casual"
            ),
            usage_context=UsageContext(
                occasions=["Casual", "Weekend"],
                seasons=["Summer"],
                styling_suggestions=["Pair with jeans"]
            ),
            target_audience=TargetAudience(
                age_group="Adult",
                lifestyle="Casual"
            ),
            confidence=0.8
        )
    
    def test_pydantic_models_work(self):
        """Test that Pydantic models validate correctly."""
        attrs = VisualAttributes(
            primary_colors=["Red", "Blue"],
            secondary_colors=["White"],
            patterns=["Solid"],
            textures=["Cotton"]
        )
        
        assert len(attrs.primary_colors) == 2
        assert attrs.secondary_colors == ["White"]
        
        # Test serialization works
        data = attrs.model_dump()
        reconstructed = VisualAttributes(**data)
        assert reconstructed.primary_colors == attrs.primary_colors
    
    @pytest.mark.asyncio
    async def test_extract_analysis_works(self, sample_analysis):
        """Test that image analysis extraction works."""
        mock_client = AsyncMock()
        
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.parsed = sample_analysis
        mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        # Test extraction
        result = await extract_enhanced_fashion_analysis(
            image_url="https://example.com/test.jpg",
            client=mock_client
        )
        
        # Basic assertions
        assert isinstance(result, FashionImageAnalysis)
        assert result.confidence == 0.8
        assert "blue" in result.overview.lower()
        assert result.style_analysis.style_classification == "Casual"
    
    @pytest.mark.asyncio
    async def test_database_storage_works(self, db_session, sample_analysis):
        """Test that storing and retrieving analysis works."""
        # Create test product and image
        product = Product(
            parent_asin="TEST_FASHION",
            main_category="Testing",
            title="Test Item"
        )
        db_session.add(product)
        await db_session.flush()
        
        image = ProductImage(
            product_id=product.id,
            large="https://example.com/test.jpg"
        )
        db_session.add(image)
        await db_session.flush()
        
        # Store analysis
        stored = await store_image_analysis(
            session=db_session,
            image=image,
            analysis=sample_analysis,
            prompt_version="test_v1"
        )
        
        # Retrieve analysis
        retrieved = await get_image_analysis(
            session=db_session,
            image=image,
            prompt_version="test_v1"
        )
        
        # Basic assertions
        assert stored is not None
        assert retrieved is not None
        assert retrieved.confidence == 0.8
        assert retrieved.overview == sample_analysis.overview
        assert "Blue" in retrieved.visual_attributes.primary_colors
    
    def test_text_enhancement_works(self, sample_analysis):
        """Test that text enhancement works."""
        original_text = "Blue t-shirt"
        
        enhanced = enhance_product_text_with_analysis(
            product_text=original_text,
            analysis=sample_analysis
        )
        
        # Should be enhanced and contain original text
        assert enhanced != original_text
        assert original_text in enhanced
        assert "Blue" in enhanced
        assert "Casual" in enhanced
    
    def test_confidence_threshold_works(self, sample_analysis):
        """Test that confidence threshold prevents low-confidence enhancement."""
        # Low confidence - should not enhance
        sample_analysis.confidence = 0.3
        original_text = "Test shirt"
        
        enhanced = enhance_product_text_with_analysis(
            product_text=original_text,
            analysis=sample_analysis,
            min_confidence=0.5
        )
        
        assert enhanced == original_text  # Should remain unchanged
    
    @pytest.mark.asyncio
    async def test_prompt_versions_work(self, db_session, sample_analysis):
        """Test that multiple prompt versions can be stored."""
        # Create test data
        product = Product(parent_asin="PROMPT_TEST", main_category="Test", title="Test")
        db_session.add(product)
        await db_session.flush()
        
        image = ProductImage(product_id=product.id, large="https://example.com/test.jpg")
        db_session.add(image)
        await db_session.flush()
        
        # Store v1 analysis
        await store_image_analysis(db_session, image, sample_analysis, "v1")
        
        # Store v2 analysis (different confidence)
        v2_analysis = sample_analysis.model_copy()
        v2_analysis.confidence = 0.95
        await store_image_analysis(db_session, image, v2_analysis, "v2")
        
        # Retrieve both
        v1_retrieved = await get_image_analysis(db_session, image, "v1")
        v2_retrieved = await get_image_analysis(db_session, image, "v2")
        
        assert v1_retrieved.confidence == 0.8
        assert v2_retrieved.confidence == 0.95


@pytest.mark.asyncio
async def test_end_to_end_workflow(db_session):
    """Simple end-to-end test of the basic workflow."""
    from app.types.fashion_analysis import VisualAttributes, StyleAnalysis, UsageContext, TargetAudience
    
    # Create test data
    product = Product(parent_asin="E2E_TEST", main_category="Clothing", title="E2E Test Shirt")
    db_session.add(product)
    await db_session.flush()
    
    image = ProductImage(product_id=product.id, large="https://example.com/e2e-test.jpg")
    db_session.add(image)
    await db_session.flush()
    
    # Create analysis
    analysis = FashionImageAnalysis(
        overview="End-to-end test shirt",
        detailed_description="Testing the complete workflow",
        visual_attributes=VisualAttributes(
            primary_colors=["Green"], secondary_colors=[], patterns=["Solid"], textures=["Cotton"]
        ),
        style_analysis=StyleAnalysis(
            silhouette="Regular", fit_type="Standard", design_details=["Basic"], style_classification="Casual"
        ),
        usage_context=UsageContext(
            occasions=["Testing"], seasons=["All"], styling_suggestions=["Perfect for workflows"]
        ),
        target_audience=TargetAudience(age_group="Developer", lifestyle="Testing"),
        confidence=0.9
    )
    
    # Store, retrieve, and enhance
    stored = await store_image_analysis(
