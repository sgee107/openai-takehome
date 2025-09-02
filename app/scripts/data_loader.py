"""
Data loader for Amazon Fashion products with OpenAI embeddings.
"""
import json
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings using OpenAI's text-embedding-3-small model."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"
        self.dimension = 1536  # Default dimension for text-embedding-3-small
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimension
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Return None for failed items
                embeddings.extend([None] * len(batch))
        
        return embeddings


class TextCombiner:
    """Combine product fields into text for embeddings."""
    
    @staticmethod
    def title_only(product: Dict[str, Any]) -> str:
        """Use title only."""
        return product.get('title', '')
    
    @staticmethod
    def title_features(product: Dict[str, Any]) -> str:
        """Combine title and features."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('features') and isinstance(product['features'], list):
            features_text = '. '.join(product['features'])
            if features_text:
                parts.append(f"Features: {features_text}")
        
        return ' '.join(parts)
    
    @staticmethod
    def title_description(product: Dict[str, Any]) -> str:
        """Combine title and description."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('description') and isinstance(product['description'], list):
            description_text = ' '.join(product['description'])
            if description_text:
                parts.append(f"Description: {description_text}")
        
        return ' '.join(parts)
    
    @staticmethod
    def title_category_store(product: Dict[str, Any]) -> str:
        """Combine title, category, and store."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('main_category'):
            parts.append(f"Category: {product['main_category']}")
        
        if product.get('store'):
            parts.append(f"Store: {product['store']}")
        
        return ' '.join(parts)
    
    @staticmethod
    def title_details(product: Dict[str, Any]) -> str:
        """Combine title with selected product details."""
        parts = []
        
        if product.get('title'):
            parts.append(product['title'])
        
        if product.get('details') and isinstance(product['details'], dict):
            # Select important detail fields
            important_keys = ['Brand', 'Department', 'Material', 'Style', 'Color', 
                            'Item model number', 'Manufacturer']
            
            detail_parts = []
            for key in important_keys:
                if key in product['details']:
                    detail_parts.append(f"{key}: {product['details'][key]}")
            
            if detail_parts:
                parts.append(' '.join(detail_parts))
        
        return ' '.join(parts)
    
    @staticmethod
    def all_text(product: Dict[str, Any]) -> str:
        """Combine all available text fields."""
        parts = []
        
        # Title
        if product.get('title'):
            parts.append(product['title'])
        
        # Features
        if product.get('features') and isinstance(product['features'], list):
            features_text = '. '.join(product['features'])
            if features_text:
                parts.append(f"Features: {features_text}")
        
        # Description
        if product.get('description') and isinstance(product['description'], list):
            description_text = ' '.join(product['description'])
            if description_text:
                parts.append(f"Description: {description_text}")
        
        # Category
        if product.get('main_category'):
            parts.append(f"Category: {product['main_category']}")
        
        # Store
        if product.get('store'):
            parts.append(f"Store: {product['store']}")
        
        # Selected details
        if product.get('details') and isinstance(product['details'], dict):
            important_keys = ['Brand', 'Department', 'Material', 'Style', 'Color']
            for key in important_keys:
                if key in product['details']:
                    parts.append(f"{key}: {product['details'][key]}")
        
        return ' '.join(parts)


class DataLoader:
    """Load Amazon Fashion data into PostgreSQL with embeddings."""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.embedding_generator = EmbeddingGenerator()
        self.text_combiner = TextCombiner()
        self.conn = None
    
    async def connect(self):
        """Establish database connection."""
        self.conn = await asyncpg.connect(**self.db_config)
    
    async def disconnect(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
    
    async def create_tables(self):
        """Create database tables for products and embeddings."""
        
        # Drop existing tables
        await self.conn.execute("DROP TABLE IF EXISTS product_embeddings CASCADE")
        await self.conn.execute("DROP TABLE IF EXISTS products CASCADE")
        
        # Create products table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                parent_asin VARCHAR(50) UNIQUE NOT NULL,
                title TEXT NOT NULL,
                main_category VARCHAR(100),
                store VARCHAR(200),
                average_rating FLOAT,
                rating_number INTEGER,
                price FLOAT,
                features JSONB,
                description JSONB,
                details JSONB,
                images JSONB,
                categories JSONB,
                bought_together JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create embeddings table with pgvector
        await self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS product_embeddings (
                id SERIAL PRIMARY KEY,
                product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
                embedding_type VARCHAR(50) NOT NULL,
                embedding_text TEXT NOT NULL,
                embedding vector(1536) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(product_id, embedding_type)
            )
        """)
        
        # Create indexes
        await self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_parent_asin ON products(parent_asin)
        """)
        
        await self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_category ON products(main_category)
        """)
        
        await self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_product_embeddings_type ON product_embeddings(embedding_type)
        """)
        
        # Create HNSW index for similarity search
        await self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_product_embeddings_vector 
            ON product_embeddings USING hnsw (embedding vector_cosine_ops)
        """)
        
        print("✅ Database tables created successfully")
    
    async def insert_product(self, product: Dict[str, Any]) -> int:
        """Insert a product into the database."""
        query = """
            INSERT INTO products (
                parent_asin, title, main_category, store, 
                average_rating, rating_number, price,
                features, description, details, images, 
                categories, bought_together
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
        """
        
        product_id = await self.conn.fetchval(
            query,
            product.get('parent_asin', ''),
            product.get('title', ''),
            product.get('main_category'),
            product.get('store'),
            product.get('average_rating'),
            product.get('rating_number'),
            product.get('price'),
            json.dumps(product.get('features', [])),
            json.dumps(product.get('description', [])),
            json.dumps(product.get('details', {})),
            json.dumps(product.get('images', [])),
            json.dumps(product.get('categories', [])),
            json.dumps(product.get('bought_together'))
        )
        
        return product_id
    
    async def insert_embedding(self, product_id: int, embedding_type: str, 
                             embedding_text: str, embedding: List[float]):
        """Insert an embedding into the database."""
        query = """
            INSERT INTO product_embeddings (
                product_id, embedding_type, embedding_text, embedding
            ) VALUES ($1, $2, $3, $4)
            ON CONFLICT (product_id, embedding_type) DO UPDATE
            SET embedding_text = $3, embedding = $4
        """
        
        await self.conn.execute(
            query,
            product_id,
            embedding_type,
            embedding_text,
            embedding
        )
    
    async def load_data(self, data_path: str, limit: Optional[int] = None):
        """Load products and generate embeddings."""
        
        # Load data
        with open(data_path, 'r') as f:
            products = json.load(f)
        
        if limit:
            products = products[:limit]
        
        print(f"📊 Loading {len(products)} products...")
        
        # Define embedding strategies
        embedding_strategies = [
            ('title_only', self.text_combiner.title_only),
            ('title_features', self.text_combiner.title_features),
            ('title_category_store', self.text_combiner.title_category_store),
            ('title_details', self.text_combiner.title_details),
            ('all_text', self.text_combiner.all_text)
        ]
        
        # Process products
        for i, product in enumerate(products):
            try:
                # Insert product
                product_id = await self.insert_product(product)
                
                # Generate and store embeddings for each strategy
                for strategy_name, strategy_func in embedding_strategies:
                    # Generate text
                    text = strategy_func(product)
                    
                    if text:  # Only process if text is not empty
                        # Generate embedding
                        embedding = await self.embedding_generator.generate_embedding(text)
                        
                        if embedding:
                            # Store embedding
                            await self.insert_embedding(
                                product_id, 
                                strategy_name, 
                                text[:500],  # Store first 500 chars of text for reference
                                embedding
                            )
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(products)} products...")
                    
            except Exception as e:
                print(f"Error processing product {i}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(products)} products with embeddings")
    
    async def get_statistics(self):
        """Get statistics about loaded data."""
        stats = {}
        
        # Product count
        stats['total_products'] = await self.conn.fetchval(
            "SELECT COUNT(*) FROM products"
        )
        
        # Embeddings count by type
        embedding_counts = await self.conn.fetch("""
            SELECT embedding_type, COUNT(*) as count
            FROM product_embeddings
            GROUP BY embedding_type
            ORDER BY embedding_type
        """)
        
        stats['embeddings_by_type'] = {
            row['embedding_type']: row['count'] 
            for row in embedding_counts
        }
        
        # Total embeddings
        stats['total_embeddings'] = await self.conn.fetchval(
            "SELECT COUNT(*) FROM product_embeddings"
        )
        
        return stats


async def main():
    """Main function to load data."""
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'takehome'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password')
    }
    
    # Initialize loader
    loader = DataLoader(db_config)
    
    try:
        # Connect to database
        await loader.connect()
        print("🔗 Connected to database")
        
        # Create tables
        await loader.create_tables()
        
        # Load data
        data_path = Path(__file__).parent.parent.parent / "data" / "amazon_fashion_sample.json"
        await loader.load_data(str(data_path), limit=300)  # Load all 300 products
        
        # Get statistics
        stats = await loader.get_statistics()
        print("\n📈 Loading Statistics:")
        print(f"  Total products: {stats['total_products']}")
        print(f"  Total embeddings: {stats['total_embeddings']}")
        print("  Embeddings by type:")
        for embed_type, count in stats['embeddings_by_type'].items():
            print(f"    - {embed_type}: {count}")
        
    finally:
        await loader.disconnect()
        print("\n✅ Data loading complete!")


if __name__ == "__main__":
    asyncio.run(main())