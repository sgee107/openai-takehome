-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS messages_embedding_idx ON messages 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for conversation lookup
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages(conversation_id);

-- Create files table for blob storage references
CREATE TABLE IF NOT EXISTS files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(100),
    minio_object_name VARCHAR(500) NOT NULL,
    file_size BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create reviews table for Amazon reviews data
CREATE TABLE IF NOT EXISTS reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asin VARCHAR(50),
    user_id VARCHAR(100),
    rating FLOAT,
    title TEXT,
    text TEXT,
    parent_asin VARCHAR(50),
    timestamp BIGINT,
    helpful_vote INTEGER,
    verified_purchase VARCHAR(10),
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Create indexes for reviews
CREATE INDEX IF NOT EXISTS reviews_asin_idx ON reviews(asin);
CREATE INDEX IF NOT EXISTS reviews_user_idx ON reviews(user_id);
CREATE INDEX IF NOT EXISTS reviews_rating_idx ON reviews(rating);
CREATE INDEX IF NOT EXISTS reviews_embedding_idx ON reviews 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);