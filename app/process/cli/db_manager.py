#!/usr/bin/env python3
"""
Database management CLI for schema operations.

Moved from app/scripts/db_management.py and adapted for the new process structure.
Handles schema destruction, creation, and reset operations only.
"""

import asyncio
import sys
import click
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.settings import settings
from app.db.models import Base


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
    click.echo("🗑️  Dropping all tables...")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        click.echo("✅ All tables dropped successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error dropping tables: {e}")
        raise


async def create_all_tables(engine):
    """Create all tables from the models."""
    click.echo("🏗️  Creating all tables...")
    
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        click.echo("✅ All tables created successfully!")
        
    except Exception as e:
        click.echo(f"❌ Error creating tables: {e}")
        raise


async def reset_schema():
    """Drop and recreate all tables (full reset)."""
    engine, _ = await get_engine_and_session()
    
    try:
        await drop_all_tables(engine)
        await create_all_tables(engine)
        click.echo("🔄 Schema reset completed successfully!")
        
    finally:
        await engine.dispose()


async def create_schema():
    """Create all tables (without dropping first)."""
    engine, _ = await get_engine_and_session()
    
    try:
        await create_all_tables(engine)
        click.echo("🏗️  Schema creation completed successfully!")
        
    finally:
        await engine.dispose()


async def drop_schema():
    """Drop all tables."""
    engine, _ = await get_engine_and_session()
    
    try:
        await drop_all_tables(engine)
        click.echo("🗑️  Schema destruction completed successfully!")
        
    finally:
        await engine.dispose()


async def check_tables():
    """Check what tables exist in the database."""
    click.echo("🔍 Checking existing tables...")
    
    engine, _ = await get_engine_and_session()
    
    try:
        async with engine.begin() as conn:
            # Get table names
            result = await conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
            tables = [row[0] for row in result.fetchall()]
            
            if tables:
                click.echo(f"📊 Found {len(tables)} tables:")
                for table in sorted(tables):
                    click.echo(f"  • {table}")
            else:
                click.echo("📊 No tables found in the database.")
                
    except Exception as e:
        click.echo(f"❌ Error checking tables: {e}")
        raise
    
    finally:
        await engine.dispose()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """Database schema management for the data processing pipeline.
    
    This tool handles database schema operations only. For data loading and 
    processing, use the pipeline CLI instead:
    
      python -m app.process.cli.pipeline --help
    """
    if verbose:
        click.echo(f"📍 Database: {settings.database_url}")


@cli.command()
def reset():
    """Drop and recreate all tables (full reset)."""
    try:
        click.echo("🚀 Starting schema reset...")
        asyncio.run(reset_schema())
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Operation failed: {e}")
        sys.exit(1)


@cli.command()
def create():
    """Create all tables (without dropping existing ones)."""
    try:
        click.echo("🚀 Starting schema creation...")
        asyncio.run(create_schema())
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Operation failed: {e}")
        sys.exit(1)


@cli.command()
def drop():
    """Drop all tables (destructive operation)."""
    if click.confirm('⚠️  This will drop all tables and data. Continue?'):
        try:
            click.echo("🚀 Starting schema destruction...")
            asyncio.run(drop_schema())
        except KeyboardInterrupt:
            click.echo("\n⚠️  Operation cancelled by user")
            sys.exit(1)
        except Exception as e:
            click.echo(f"\n❌ Operation failed: {e}")
            sys.exit(1)
    else:
        click.echo("Operation cancelled.")


@cli.command()
def check():
    """Check existing tables in the database."""
    try:
        asyncio.run(check_tables())
    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
