from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, func, String, Float, Integer, Text, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Product(Base):
    __tablename__ = "products"
    
    parent_asin: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    main_category: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(String(500))
    average_rating: Mapped[Optional[float]] = mapped_column(Float)
    rating_number: Mapped[Optional[int]] = mapped_column(Integer)
    price: Mapped[Optional[float]] = mapped_column(Float)
    store: Mapped[Optional[str]] = mapped_column(String(200))
    features: Mapped[Optional[list]] = mapped_column(JSON)
    description: Mapped[Optional[list]] = mapped_column(JSON)
    categories: Mapped[Optional[list]] = mapped_column(JSON)
    details: Mapped[Optional[dict]] = mapped_column(JSON)
    bought_together: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    images: Mapped[list["ProductImage"]] = relationship("ProductImage", back_populates="product", cascade="all, delete-orphan")
    videos: Mapped[list["ProductVideo"]] = relationship("ProductVideo", back_populates="product", cascade="all, delete-orphan")
    embeddings: Mapped[list["ProductEmbedding"]] = relationship("ProductEmbedding", back_populates="product", cascade="all, delete-orphan")


class ProductImage(Base):
    __tablename__ = "product_images"
    
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    thumb: Mapped[Optional[str]] = mapped_column(String(500))
    large: Mapped[Optional[str]] = mapped_column(String(500))
    hi_res: Mapped[Optional[str]] = mapped_column(String(500))
    variant: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Relationships
    product: Mapped[Product] = relationship("Product", back_populates="images")


class ProductVideo(Base):
    __tablename__ = "product_videos"
    
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    url: Mapped[str] = mapped_column(String(500))
    title: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Relationships
    product: Mapped[Product] = relationship("Product", back_populates="videos")


class ProductEmbedding(Base):
    __tablename__ = "product_embeddings"
    __table_args__ = (
        UniqueConstraint('product_id', 'strategy', name='_product_strategy_uc'),
    )
    
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    strategy: Mapped[str] = mapped_column(String(50), index=True)
    embedding_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))
    model: Mapped[str] = mapped_column(String(50), default="text-embedding-3-small")
    
    # Relationships
    product: Mapped[Product] = relationship("Product", back_populates="embeddings")