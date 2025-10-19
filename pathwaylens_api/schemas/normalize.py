"""
Normalization schemas for PathwayLens API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class NormalizationType(str, Enum):
    """Normalization type enumeration."""
    GENE_ID = "gene_id"
    PATHWAY_ID = "pathway_id"
    OMICS_DATA = "omics_data"


class NormalizationRequest(BaseModel):
    """Normalization request model."""
    data: List[Dict[str, Any]] = Field(..., description="Gene data to normalize")
    species: str = Field(..., description="Species of input data")
    target_species: Optional[str] = Field(None, description="Target species for cross-species mapping")
    target_type: str = Field(..., description="Target identifier type")
    ambiguity_policy: str = Field("expand", description="How to handle ambiguous mappings")


class NormalizeRequest(BaseModel):
    """Normalization request model."""
    data: List[Dict[str, Any]] = Field(..., description="Gene data to normalize")
    species: str = Field(..., description="Species of input data")
    target_species: Optional[str] = Field(None, description="Target species for cross-species mapping")
    target_type: str = Field(..., description="Target identifier type")
    ambiguity_policy: str = Field("expand", description="How to handle ambiguous mappings")


class NormalizationResponse(BaseModel):
    """Normalization response model."""
    job_id: str
    status: str
    normalized_data: List[Dict[str, Any]]
    conversion_stats: Dict[str, Any]
    created_at: datetime


class NormalizeResponse(BaseModel):
    """Normalization response model."""
    job_id: str
    status: str
    normalized_data: List[Dict[str, Any]]
    conversion_stats: Dict[str, Any]
    created_at: datetime


class BatchNormalizeRequest(BaseModel):
    """Batch normalization request model."""
    datasets: List[Dict[str, Any]] = Field(..., description="Multiple datasets to normalize")
    species: str = Field(..., description="Species of input data")
    target_type: str = Field(..., description="Target identifier type")
    ambiguity_policy: str = Field("expand", description="How to handle ambiguous mappings")


class NormalizationStatus(BaseModel):
    """Normalization status model."""
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class NormalizationResult(BaseModel):
    """Normalization result model."""
    job_id: str
    status: str
    normalized_data: List[Dict[str, Any]]
    conversion_stats: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    created_at: datetime
    completed_at: Optional[datetime] = None


class NormalizationParameters(BaseModel):
    """Normalization parameters model."""
    species: str
    target_type: str
    target_species: Optional[str] = None
    ambiguity_policy: str = "expand"
    batch_size: int = 1000
    include_unmapped: bool = False


class BatchNormalizeResponse(BaseModel):
    """Batch normalization response model."""
    job_id: str
    status: str
    results: List[Dict[str, Any]]
    created_at: datetime
