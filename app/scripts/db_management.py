#!/usr/bin/env python3
"""
Database management script for development operations.
Handles schema destruction, creation, and data reloading.
"""

import asyncio
import sys
import argparse
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.settings import settings
from app.db.models import Base
from app.scripts.data_loader import DataLoader


async def get_engine_and_session():
    """Create database engine and session factory."""
    engine = create_async_engine(
        settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
        echo=True
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    return engine, async_session


async def drop_all_tables(engine):
    """Drop all tables in the database."""
    print("🗑️  Dropping all tables...")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        print("✅ All tables dropped successfully!")
        
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        raise


async def create_all_tables(engine):
    """Create all tables from the models."""
    print("🏗️  Creating all tables...")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ All tables created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise


async def reset_schema():
    """Drop and recreate all tables (full reset)."""
    engine, _ = await get_engine_and_session()
    
    try:
        await drop_all_tables(engine)
        await create_all_tables(engine)
        print("🔄 Schema reset completed successfully!")
        
    finally:
        await engine.dispose()


async def create_schema():
    """Create all tables (without dropping first)."""
    engine, _ = await get_engine_and_session()
    
    try:
        await create_all_tables(engine)
        print("🏗️  Schema creation completed successfully!")
        
    finally:
        await engine.dispose()


async def drop_schema():
    """Drop all tables."""
    engine, _ = await get_engine_and_session()
    
    try:
        await drop_all_tables(engine)
        print("🗑️  Schema destruction completed successfully!")
        
    finally:
        await engine.dispose()


async def load_sample_data():
    """Load sample product data using the existing data loader."""
    print("📥 Loading sample product data...")
    
    try:
        # Import and run the load_products script
        from app.scripts.load_products import main as load_main
        await load_main()
        print("✅ Sample data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        raise


async def reset_with_data():
    """Full reset: drop schema, create schema, and load sample data."""
    print("🔄 Starting full database reset with data...")
    
    await reset_schema()
    await load_sample_data()
    
    print("🎉 Full database reset with data completed successfully!")


async def check_tables():
    """Check what tables exist in the database."""
    print("🔍 Checking existing tables...")
    
    engine, _ = await get_engine_and_session()
    
    try:
        async with engine.begin() as conn:
            # Get table names
            result = await conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            tables = [row[0] for row in result.fetchall()]
            
            if tables:
                print(f"📊 Found {len(tables)} tables:")
                for table in sorted(tables):
                    print(f"  • {table}")
            else:
                print("📊 No tables found in the database.")
                
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        raise
    
    finally:
        await engine.dispose()


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Database management script for development operations"
    )
    
    parser.add_argument(
        "action",
        choices=[
            "reset",           # Drop and recreate all tables
            "create",          # Create all tables
            "drop",            # Drop all tables
            "load-data",       # Load sample data
            "reset-with-data", # Full reset + data load
            "check",           # Check existing tables
        ],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    # Map actions to functions
    actions = {
        "reset": reset_schema,
        "create": create_schema,
        "drop": drop_schema,
        "load-data": load_sample_data,
        "reset-with-data": reset_with_data,
        "check": check_tables,
    }
    
    print(f"🚀 Starting database management: {args.action}")
    
    try:
        asyncio.run(actions[args.action]())
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
