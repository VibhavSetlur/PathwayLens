"""
Comparison schemas for PathwayLens API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ComparisonType(str, Enum):
    """Comparison type enumeration."""
    OVERLAP = "overlap"
    CORRELATION = "correlation"
    CLUSTERING = "clustering"


class ComparisonRequest(BaseModel):
    """Comparison request model."""
    comparison_type: str = Field(..., description="Type of comparison")
    datasets: List[Dict[str, Any]] = Field(..., description="Datasets to compare")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Comparison parameters")


class CompareRequest(BaseModel):
    """Comparison request model."""
    comparison_type: str = Field(..., description="Type of comparison")
    datasets: List[Dict[str, Any]] = Field(..., description="Datasets to compare")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Comparison parameters")


class ComparisonResponse(BaseModel):
    """Comparison response model."""
    job_id: str
    status: str
    comparison_type: str
    dataset_count: int
    results: Dict[str, Any]
    created_at: datetime


class ComparisonStatus(BaseModel):
    """Comparison status model."""
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ComparisonResult(BaseModel):
    """Comparison result model."""
    job_id: str
    status: str
    comparison_type: str
    dataset_count: int
    results: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    created_at: datetime
    completed_at: Optional[datetime] = None


class ComparisonParameters(BaseModel):
    """Comparison parameters model."""
    comparison_type: str
    significance_threshold: float = 0.05
    correction_method: str = "fdr_bh"
    min_overlap: int = 1
    include_network: bool = False


class CompareResponse(BaseModel):
    """Comparison response model."""
    job_id: str
    status: str
    comparison_type: str
    dataset_count: int
    results: Dict[str, Any]
    created_at: datetime
