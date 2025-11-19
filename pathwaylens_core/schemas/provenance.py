"""
Provenance tracking schemas for reproducible pathway analysis.

This module defines schemas for capturing complete analysis provenance including:
- Database versions and metadata
- Execution environment details
- Analysis manifests for reproducibility
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


class DatabaseVersion(BaseModel):
    """Version information for a pathway database."""
    
    database_name: str = Field(..., description="Database name (e.g., 'KEGG', 'Reactome')")
    version: str = Field(..., description="Database version identifier")
    download_date: Optional[str] = Field(None, description="Date database was downloaded")
    last_modified: Optional[str] = Field(None, description="Last modification date from source")
    num_pathways: int = Field(..., description="Number of pathways in database")
    num_genes: int = Field(..., description="Number of unique genes in database")
    checksum: Optional[str] = Field(None, description="MD5 checksum of database file")
    source_url: Optional[str] = Field(None, description="Source URL for database")
    
    model_config = ConfigDict(frozen=False)


class ExecutionEnvironment(BaseModel):
    """Execution environment snapshot for reproducibility."""
    
    # System information
    os_name: str = Field(..., description="Operating system name")
    os_version: str = Field(..., description="Operating system version")
    python_version: str = Field(..., description="Python version")
    
    # PathwayLens information
    pathwaylens_version: str = Field(..., description="PathwayLens version")
    
    # Dependencies
    dependencies: Dict[str, str] = Field(
        default_factory=dict,
        description="Package name to version mapping"
    )
    
    # Hardware (optional)
    cpu_count: Optional[int] = Field(None, description="Number of CPU cores")
    memory_gb: Optional[float] = Field(None, description="Total memory in GB")
    
    model_config = ConfigDict(frozen=False)


class InputFileMetadata(BaseModel):
    """Metadata for input files."""
    
    filename: str = Field(..., description="Input filename")
    filepath: str = Field(..., description="Full file path")
    checksum: str = Field(..., description="MD5 checksum of file")
    size_bytes: int = Field(..., description="File size in bytes")
    num_genes: Optional[int] = Field(None, description="Number of genes in file")
    file_format: Optional[str] = Field(None, description="Detected file format")
    
    model_config = ConfigDict(frozen=False)


class AnalysisManifest(BaseModel):
    """
    Complete analysis manifest for reproducibility.
    
    This manifest captures all information needed to reproduce an analysis,
    including software versions, database versions, parameters, and input data.
    """
    
    # Manifest metadata
    manifest_version: str = Field(default="1.0", description="Manifest schema version")
    analysis_id: str = Field(..., description="Unique analysis identifier")
    created_at: str = Field(..., description="Manifest creation timestamp (ISO 8601)")
    
    # Software environment
    pathwaylens_version: str = Field(..., description="PathwayLens version used")
    execution_environment: ExecutionEnvironment = Field(..., description="Execution environment")
    
    # Database versions
    database_versions: Dict[str, DatabaseVersion] = Field(
        ...,
        description="Database name to version info mapping"
    )
    
    # Analysis configuration
    analysis_type: str = Field(..., description="Type of analysis performed")
    parameters: Dict[str, Any] = Field(..., description="Analysis parameters")
    
    # Input data
    input_files: Dict[str, InputFileMetadata] = Field(
        ...,
        description="Input file metadata"
    )
    
    # Reproducibility
    random_seed: Optional[int] = Field(None, description="Random seed used (if applicable)")
    reproducibility_hash: str = Field(..., description="Hash of all reproducibility-critical data")
    
    # Execution details
    execution_time_seconds: Optional[float] = Field(None, description="Total execution time")
    completed_at: Optional[str] = Field(None, description="Analysis completion timestamp")
    
    # Quality metrics
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    
    model_config = ConfigDict(frozen=False)


class ProvenanceRecord(BaseModel):
    """
    Detailed provenance record for a specific analysis step.
    
    Used for tracking individual steps in complex analysis workflows.
    """
    
    step_id: str = Field(..., description="Unique step identifier")
    step_name: str = Field(..., description="Human-readable step name")
    step_type: str = Field(..., description="Type of step (e.g., 'normalization', 'enrichment')")
    
    # Timing
    started_at: str = Field(..., description="Step start timestamp")
    completed_at: Optional[str] = Field(None, description="Step completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Step duration")
    
    # Input/Output
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Step inputs")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Step outputs")
    
    # Parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step-specific parameters")
    
    # Status
    status: str = Field(..., description="Step status (e.g., 'completed', 'failed')")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    model_config = ConfigDict(frozen=False)


class AnalysisProvenance(BaseModel):
    """
    Complete provenance chain for an analysis.
    
    Tracks all steps and transformations applied to data.
    """
    
    analysis_id: str = Field(..., description="Analysis identifier")
    manifest: AnalysisManifest = Field(..., description="Analysis manifest")
    
    # Provenance chain
    provenance_records: List[ProvenanceRecord] = Field(
        default_factory=list,
        description="Ordered list of provenance records"
    )
    
    # Lineage
    parent_analysis_ids: List[str] = Field(
        default_factory=list,
        description="IDs of parent analyses (for derived analyses)"
    )
    
    model_config = ConfigDict(frozen=False)
