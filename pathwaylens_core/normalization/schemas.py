"""
Pydantic schemas for normalization module.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum
import pandas as pd


class SpeciesType(str, Enum):
    """Supported species types."""
    HUMAN = "human"
    MOUSE = "mouse"
    RAT = "rat"
    DROSOPHILA = "drosophila"
    ZEBRAFISH = "zebrafish"
    C_ELEGANS = "c_elegans"
    S_CEREVISIAE = "s_cerevisiae"


class IDType(str, Enum):
    """Supported identifier types."""
    SYMBOL = "symbol"
    ENSEMBL = "ensembl"
    ENTREZ = "entrez"
    UNIPROT = "uniprot"
    REFSEQ = "refseq"
    MGI = "mgi"
    FLYBASE = "flybase"


class InputData(BaseModel):
    """Input data model for normalization."""
    data: List[Dict[str, Any]] = Field(..., description="Input data")
    species: SpeciesType = Field(..., description="Species")
    id_type: IDType = Field(..., description="Identifier type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AmbiguityPolicy(str, Enum):
    """Ambiguity resolution policies."""
    EXPAND = "expand"  # Include all mappings
    COLLAPSE = "collapse"  # Use first mapping
    SKIP = "skip"  # Skip ambiguous entries
    ERROR = "error"  # Raise error on ambiguity


class NormalizedTable(BaseModel):
    """Schema for normalized data table."""
    
    data: pd.DataFrame = Field(..., description="Normalized data table")
    species: SpeciesType = Field(..., description="Species of the data")
    id_type: IDType = Field(..., description="Type of identifiers in the data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        arbitrary_types_allowed = True
        
    @validator('data')
    def validate_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if v.empty:
            raise ValueError("DataFrame cannot be empty")
        return v


class NormalizationResult(BaseModel):
    """Schema for normalization results."""
    
    job_id: str = Field(..., description="Unique job identifier")
    input_file: str = Field(..., description="Path to input file")
    output_file: Optional[str] = Field(None, description="Path to output file")
    species: SpeciesType = Field(..., description="Species of the data")
    input_id_type: IDType = Field(..., description="Input identifier type")
    output_id_type: IDType = Field(..., description="Output identifier type")
    total_input: int = Field(..., description="Total number of input entries")
    total_mapped: int = Field(..., description="Total number of mapped entries")
    total_unmapped: int = Field(..., description="Total number of unmapped entries")
    mapping_rate: float = Field(..., description="Mapping success rate")
    ambiguous_mappings: int = Field(default=0, description="Number of ambiguous mappings")
    duplicate_mappings: int = Field(default=0, description="Number of duplicate mappings")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    
    @validator('mapping_rate')
    def validate_mapping_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Mapping rate must be between 0 and 1")
        return v


class ConversionMapping(BaseModel):
    """Schema for individual conversion mappings."""
    
    input_id: str = Field(..., description="Input identifier")
    output_id: str = Field(..., description="Output identifier")
    confidence: float = Field(..., description="Mapping confidence score")
    source: str = Field(..., description="Source of the mapping")
    is_ambiguous: bool = Field(default=False, description="Whether mapping is ambiguous")
    alternative_mappings: List[str] = Field(default_factory=list, description="Alternative mappings")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class CrossSpeciesMapping(BaseModel):
    """Schema for cross-species ortholog mappings."""
    
    source_species: SpeciesType = Field(..., description="Source species")
    target_species: SpeciesType = Field(..., description="Target species")
    source_id: str = Field(..., description="Source identifier")
    target_id: str = Field(..., description="Target identifier")
    ortholog_type: str = Field(..., description="Type of ortholog relationship")
    confidence: float = Field(..., description="Ortholog confidence score")
    method: str = Field(..., description="Method used for ortholog detection")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class FormatDetectionResult(BaseModel):
    """Schema for format detection results."""
    
    format_type: str = Field(..., description="Detected format type")
    confidence: float = Field(..., description="Detection confidence")
    delimiter: Optional[str] = Field(None, description="Detected delimiter")
    encoding: str = Field(default="utf-8", description="File encoding")
    has_header: bool = Field(default=True, description="Whether file has header")
    column_mapping: Dict[str, str] = Field(default_factory=dict, description="Column mappings")
    sample_data: List[Dict[str, Any]] = Field(default_factory=list, description="Sample data rows")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class ValidationResult(BaseModel):
    """Schema for input validation results."""
    
    is_valid: bool = Field(..., description="Whether input is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    detected_species: Optional[SpeciesType] = Field(None, description="Detected species")
    detected_id_type: Optional[IDType] = Field(None, description="Detected ID type")
    file_info: Dict[str, Any] = Field(default_factory=dict, description="File information")
