"""
Configuration schemas for PathwayLens API.
"""

from typing import Dict, Any
from pydantic import BaseModel


class ConfigUpdateRequest(BaseModel):
    """Configuration update request model."""
    config: Dict[str, Any]


class ConfigResponse(BaseModel):
    """Configuration response model."""
    config: Dict[str, Any]
