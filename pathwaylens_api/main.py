"""
PathwayLens API - Main FastAPI application.

This module sets up the FastAPI application with all routes, middleware,
and configuration for the PathwayLens 2.0 web API.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pathwaylens_api.routes import (
    auth,
    normalize,
    analyze,
    compare,
    visualize,
    jobs,
    config,
    info
)
from pathwaylens_api.middleware import auth as auth_middleware
from pathwaylens_api.middleware import rate_limit
from pathwaylens_api.middleware import logging as logging_middleware
from pathwaylens_api.utils.exceptions import PathwayLensException
from pathwaylens_core.utils.config import get_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    print("🚀 Starting PathwayLens API...")
    
    # Initialize database connections
    # Initialize cache
    # Load configuration
    config = get_config()
    print(f"📋 Configuration loaded: {config.get('version', 'unknown')}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down PathwayLens API...")
    # Close database connections
    # Clean up resources


# Create FastAPI application
app = FastAPI(
    title="PathwayLens 2.0 API",
    description="Next-generation computational biology platform for multi-omics pathway analysis",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Custom middleware
app.add_middleware(logging_middleware.RequestLoggingMiddleware)
app.add_middleware(rate_limit.RateLimitMiddleware)
app.add_middleware(auth_middleware.AuthenticationMiddleware)

# Include API routes
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    normalize.router,
    prefix="/api/v1/normalize",
    tags=["Normalization"]
)

app.include_router(
    analyze.router,
    prefix="/api/v1/analyze",
    tags=["Analysis"]
)

app.include_router(
    compare.router,
    prefix="/api/v1/compare",
    tags=["Comparison"]
)

app.include_router(
    visualize.router,
    prefix="/api/v1/visualize",
    tags=["Visualization"]
)

app.include_router(
    jobs.router,
    prefix="/api/v1/jobs",
    tags=["Job Management"]
)

app.include_router(
    config.router,
    prefix="/api/v1/config",
    tags=["Configuration"]
)

app.include_router(
    info.router,
    prefix="/api/v1/info",
    tags=["System Information"]
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "PathwayLens 2.0 API",
        "version": "2.0.0",
        "description": "Next-generation computational biology platform",
        "docs": "/api/docs",
        "health": "/api/v1/info/health"
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.exception_handler(PathwayLensException)
async def pathwaylens_exception_handler(request: Request, exc: PathwayLensException):
    """Handle PathwayLens-specific exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An internal server error occurred",
            "details": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    config = get_config()
    
    # Run the application
    uvicorn.run(
        "pathwaylens_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.get("debug", False),
        log_level="info"
    )
