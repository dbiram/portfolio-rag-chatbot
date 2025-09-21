import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.models.schemas import HealthResponse
from app.services.vector_store import vector_store
from app.api import chat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown tasks."""
    # Startup
    logger.info("Starting Portfolio RAG API...")
    
    try:
        # Load vector store if it exists
        if os.path.exists(settings.faiss_index_path) and os.path.exists(settings.chunks_path):
            logger.info("Loading vector store from disk...")
            vector_store.load_index()
            logger.info(f"Vector store loaded successfully: {vector_store.get_stats()}")
        else:
            logger.warning("Vector store files not found. Run ingestion script first.")
            logger.info(f"Looking for: {settings.faiss_index_path} and {settings.chunks_path}")
    
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        logger.warning("API will start but chat functionality will be limited")
    
    logger.info("Portfolio RAG API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Portfolio RAG API...")
    
    try:
        # Clean up any resources
        from app.services.mistral_client import mistral_client
        await mistral_client.close()
        logger.info("Mistral client closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Portfolio RAG API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Portfolio RAG API",
    description="RAG-powered API for answering questions about Moez's career and experience",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origin,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, tags=["chat"])


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {"message": "Portfolio RAG API", "docs": "/docs", "health": "/health"}


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check the health status of the API and its dependencies"
)
async def health() -> HealthResponse:
    """
    Health check endpoint that returns service status and system information.
    """
    try:
        # Get vector store stats
        vector_stats = vector_store.get_stats()
        
        # Determine overall status
        status = "ok"
        if not vector_stats.get("loaded", False):
            status = "degraded"  # API works but vector store not loaded
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            vector_store=vector_stats
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            timestamp=datetime.now().isoformat(),
            vector_store={"error": str(e)}
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": f"The endpoint {request.url.path} was not found.",
            "available_endpoints": ["/", "/health", "/chat", "/docs"]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # For local development
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )