import asyncio
from app.db.database import get_async_session
from sqlalchemy import text

async def check_embedding_table():
    """Check what's actually in the embedding table."""
    async for session in get_async_session():
        # Check table structure
        result = await session.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'product_embeddings'
            ORDER BY ordinal_position
        """))
        
        print("Table structure:")
        for row in result:
            print(f"  {row[0]}: {row[1]}")
        
        # Check available strategies/types
        result = await session.execute(text("""
            SELECT strategy, COUNT(*) as count
            FROM product_embeddings 
            GROUP BY strategy
        """))
        
        print("\nAvailable strategies:")
        for row in result:
            print(f"  {row[0]}: {row[1]}")
        
        # Sample a few records
        result = await session.execute(text("""
            SELECT strategy, LEFT(embedding_text, 100) 
            FROM product_embeddings 
            LIMIT 3
        """))
        
        print("\nSample embedding texts:")
        for row in result:
            print(f"  {row[0]}: {row[1]}...")
        
        break

if __name__ == "__main__":
    asyncio.run(check_embedding_table())