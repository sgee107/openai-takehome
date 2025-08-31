from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.settings import settings
from app.models import Base

# Async engine for FastAPI
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Sync engine for data loading scripts
sync_database_url = settings.database_url.replace("+asyncpg", "")
sync_engine = create_engine(
    sync_database_url,
    echo=settings.debug
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=sync_engine
)

async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

def get_sync_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)