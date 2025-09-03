from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.api.chat import router as chat_router
from app.api.chat_simple import router as simple_router
from app.api.search import router as search_router
from app.db.database import init_db, async_engine
from app.process.startup import run_startup_embeddings, get_startup_config_summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    if settings.startup_embedding_enabled:
        await run_startup_embeddings()
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Fashion API...")
    await async_engine.dispose()
    print("✅ Database connections closed")


app = FastAPI(
    title="Fashion API",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(simple_router)
app.include_router(search_router)  # vNext search API


@app.get("/")
async def root():
    return {"message": "API is running", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
