from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.settings import settings
from app.db.models import Base

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

async def get_async_session():
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with async_engine.begin() as conn:
        # Create pgvector extension
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
        # Create tables
        await conn.run_sync(Base.metadata.create_all)