from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.chat import router as chat_router
from app.chat_simple import router as simple_router
from app.db.database import init_db, async_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    yield
    # Shutdown
    await async_engine.dispose()


app = FastAPI(
    title="Takehome API",
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


@app.get("/")
async def root():
    return {"message": "API is running", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
