"""
Minimal FastAPI app for testing mock endpoints without database dependencies.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.mock import router as mock_router


def create_mock_app():
    """Create a minimal FastAPI app with only mock routes"""
    app = FastAPI(
        title="Mock Fashion API (Testing)",
        version="1.0.0-test"
    )
    
    # CORS for frontend testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include only the mock router
    app.include_router(mock_router)
    
    @app.get("/")
    async def root():
        return {"message": "Mock API for testing", "version": "1.0.0-test"}
    
    return app