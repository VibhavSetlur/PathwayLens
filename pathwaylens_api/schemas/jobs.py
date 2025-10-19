"""
Job management schemas for PathwayLens API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    """Job creation request model."""
    job_type: str = Field(..., description="Type of job (normalize, analyze, compare)")
    parameters: Dict[str, Any] = Field(..., description="Job parameters")
    input_files: Optional[List[str]] = Field(None, description="Input file IDs")
    project_id: Optional[str] = Field(None, description="Project ID")


class JobResponse(BaseModel):
    """Job response model."""
    id: str
    user_id: Optional[str]
    job_type: str
    status: str
    parameters: Dict[str, Any]
    input_files: Optional[List[str]]
    output_files: Optional[List[str]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    updated_at: datetime
    progress: int = Field(0, ge=0, le=100)


class JobListResponse(BaseModel):
    """Job list response model."""
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class JobStatusUpdate(BaseModel):
    """Job status update model."""
    status: str
    progress: Optional[int] = None
    error_message: Optional[str] = None
    output_files: Optional[List[str]] = None
