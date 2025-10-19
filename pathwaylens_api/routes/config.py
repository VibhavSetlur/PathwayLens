"""
Configuration routes for PathwayLens API.

This module provides endpoints for managing configuration.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from pathwaylens_api.utils.dependencies import get_current_user
from pathwaylens_core.utils.config import get_config

router = APIRouter()


# Request/Response models
class ConfigUpdateRequest(BaseModel):
    """Configuration update request model."""
    config: Dict[str, Any]


class ConfigResponse(BaseModel):
    """Configuration response model."""
    config: Dict[str, Any]


# Route handlers
@router.get("/", response_model=ConfigResponse)
async def get_configuration(
    user = Depends(get_current_user)
):
    """Get user configuration."""
    try:
        # Get configuration
        config = get_config()
        
        # Filter sensitive information
        safe_config = {
            "version": config.get("version"),
            "databases": config.get("databases", {}),
            "analysis": config.get("analysis", {}),
            "cache": config.get("cache", {}),
            "output": config.get("output", {})
        }
        
        return ConfigResponse(config=safe_config)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {e}"
        )


@router.put("/", response_model=ConfigResponse)
async def update_configuration(
    request: ConfigUpdateRequest,
    user = Depends(get_current_user)
):
    """Update user configuration."""
    try:
        # Update configuration
        config = get_config()
        config.update(request.config)
        
        # Save configuration
        # TODO: Implement configuration persistence
        
        return ConfigResponse(config=request.config)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {e}"
        )


@router.get("/databases")
async def get_database_config():
    """Get database configuration."""
    try:
        config = get_config()
        databases = config.get("databases", {})
        
        return {
            "databases": [
                {
                    "name": db_name,
                    "enabled": db_config.get("enabled", True),
                    "rate_limit": db_config.get("rate_limit", 10),
                    "base_url": db_config.get("base_url", "")
                }
                for db_name, db_config in databases.items()
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database configuration: {e}"
        )


@router.get("/analysis")
async def get_analysis_config():
    """Get analysis configuration."""
    try:
        config = get_config()
        analysis = config.get("analysis", {})
        
        return {
            "analysis": {
                "significance_threshold": analysis.get("significance_threshold", 0.05),
                "correction_method": analysis.get("correction_method", "fdr_bh"),
                "min_pathway_size": analysis.get("min_pathway_size", 5),
                "max_pathway_size": analysis.get("max_pathway_size", 500),
                "gsea_permutations": analysis.get("gsea_permutations", 1000)
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analysis configuration: {e}"
        )
