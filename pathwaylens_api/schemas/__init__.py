"""
PathwayLens API schemas.

This module provides Pydantic schemas for API requests and responses.
"""

from .auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from .jobs import JobCreateRequest, JobResponse, JobListResponse, JobStatusUpdate
from .normalize import NormalizeRequest, NormalizeResponse, BatchNormalizeRequest, BatchNormalizeResponse
from .analyze import AnalyzeRequest, AnalyzeResponse, GSEARequest, GSEAResponse, BatchAnalyzeRequest, BatchAnalyzeResponse
from .compare import CompareRequest, CompareResponse
from .visualize import VisualizeRequest, VisualizeResponse
from .config import ConfigUpdateRequest, ConfigResponse

__all__ = [
    # Auth schemas
    "LoginRequest",
    "RegisterRequest", 
    "TokenResponse",
    "UserResponse",
    
    # Job schemas
    "JobCreateRequest",
    "JobResponse",
    "JobListResponse",
    "JobStatusUpdate",
    
    # Normalize schemas
    "NormalizeRequest",
    "NormalizeResponse",
    "BatchNormalizeRequest",
    "BatchNormalizeResponse",
    
    # Analyze schemas
    "AnalyzeRequest",
    "AnalyzeResponse",
    "GSEARequest",
    "GSEAResponse",
    "BatchAnalyzeRequest",
    "BatchAnalyzeResponse",
    
    # Compare schemas
    "CompareRequest",
    "CompareResponse",
    
    # Visualize schemas
    "VisualizeRequest",
    "VisualizeResponse",
    
    # Config schemas
    "ConfigUpdateRequest",
    "ConfigResponse"
]
