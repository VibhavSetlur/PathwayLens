"""
System information routes for PathwayLens API.

This module provides endpoints for system information and health checks.
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from pathwaylens_api.utils.dependencies import get_optional_user, get_database, get_storage
from pathwaylens_core.utils.config import get_config

router = APIRouter()


# Response models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


class SystemInfoResponse(BaseModel):
    """System information response model."""
    version: str
    status: str
    databases: Dict[str, Any]
    cache: Dict[str, Any]
    storage: Dict[str, Any]


class DatabaseInfoResponse(BaseModel):
    """Database information response model."""
    databases: list


# Route handlers
@router.get("/health", response_model=HealthResponse)
async def health_check(
    db = Depends(get_database),
    storage = Depends(get_storage)
):
    """Health check endpoint."""
    try:
        # Check database health
        db_health = "healthy" if await db.health_check() else "unhealthy"
        
        # Check storage health
        storage_info = await storage.get_storage_info()
        storage_health = "healthy" if storage_info else "unhealthy"
        
        # Get configuration
        config = get_config()
        
        return HealthResponse(
            status="healthy" if db_health == "healthy" and storage_health == "healthy" else "unhealthy",
            version=config.get("version", "2.0.0"),
            timestamp=datetime.utcnow().isoformat(),
            services={
                "database": db_health,
                "storage": storage_health
            }
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version="2.0.0",
            timestamp=datetime.utcnow().isoformat(),
            services={
                "database": "unhealthy",
                "storage": "unhealthy",
                "error": str(e)
            }
        )


@router.get("/", response_model=SystemInfoResponse)
async def get_system_info(
    user = Depends(get_optional_user),
    db = Depends(get_database),
    storage = Depends(get_storage)
):
    """Get system information."""
    try:
        # Get configuration
        config = get_config()
        
        # Get database information
        databases = config.get("databases", {})
        db_info = {}
        for db_name, db_config in databases.items():
            db_info[db_name] = {
                "enabled": db_config.get("enabled", True),
                "rate_limit": db_config.get("rate_limit", 10),
                "base_url": db_config.get("base_url", "")
            }
        
        # Get cache information
        cache_config = config.get("cache", {})
        cache_info = {
            "enabled": cache_config.get("enabled", True),
            "base_dir": cache_config.get("base_dir", ".pathwaylens/cache"),
            "max_size_mb": cache_config.get("max_size_mb", 1000),
            "ttl_days": cache_config.get("ttl_days", 90)
        }
        
        # Get storage information
        storage_info = await storage.get_storage_info()
        
        return SystemInfoResponse(
            version=config.get("version", "2.0.0"),
            status="healthy",
            databases=db_info,
            cache=cache_info,
            storage=storage_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system info: {e}"
        )


@router.get("/databases", response_model=DatabaseInfoResponse)
async def get_database_info():
    """Get database information."""
    try:
        # Get configuration
        config = get_config()
        databases = config.get("databases", {})
        
        db_list = []
        for db_name, db_config in databases.items():
            db_list.append({
                "name": db_name,
                "display_name": db_config.get("display_name", db_name.title()),
                "enabled": db_config.get("enabled", True),
                "rate_limit": db_config.get("rate_limit", 10),
                "base_url": db_config.get("base_url", ""),
                "description": db_config.get("description", "")
            })
        
        return DatabaseInfoResponse(databases=db_list)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database info: {e}"
        )


@router.get("/version")
async def get_version():
    """Get version information."""
    try:
        config = get_config()
        
        return {
            "version": config.get("version", "2.0.0"),
            "build_date": "2024-01-01",
            "git_commit": "unknown",
            "python_version": "3.8+",
            "api_version": "v1"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get version info: {e}"
        )
