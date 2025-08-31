from sqlalchemy import Column, String, Text, DateTime, Float, Integer, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON)
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    files = relationship("File", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(1536))
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    
    conversation = relationship("Conversation", back_populates="messages")
    files = relationship("File", back_populates="message", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"))
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"))
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(100))
    minio_object_name = Column(String(500), nullable=False)
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)
    
    conversation = relationship("Conversation", back_populates="files")
    message = relationship("Message", back_populates="files")


class Review(Base):
    __tablename__ = "reviews"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asin = Column(String(50))
    user_id = Column(String(100))
    rating = Column(Float)
    title = Column(Text)
    text = Column(Text)
    parent_asin = Column(String(50))
    timestamp = Column(Integer)
    helpful_vote = Column(Integer)
    verified_purchase = Column(String(10))
    embedding = Column(VECTOR(1536))
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON)