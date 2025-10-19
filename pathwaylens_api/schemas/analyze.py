"""
Analysis schemas for PathwayLens API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """Analysis type enumeration."""
    ORA = "ora"
    GSEA = "gsea"
    GSVA = "gsva"


class DatabaseType(str, Enum):
    """Database type enumeration."""
    KEGG = "kegg"
    REACTOME = "reactome"
    GO = "go"
    WIKIPATHWAYS = "wikipathways"


class CorrectionMethod(str, Enum):
    """Multiple testing correction method enumeration."""
    BONFERRONI = "bonferroni"
    FDR_BH = "fdr_bh"
    FDR_BY = "fdr_by"
    NONE = "none"


class AnalysisRequest(BaseModel):
    """Analysis request model."""
    genes: List[str] = Field(..., description="List of gene identifiers")
    species: str = Field(..., description="Species of input data")
    analysis_type: str = Field("ora", description="Type of analysis (ora, gsea)")
    databases: List[str] = Field(..., description="Databases to use for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis parameters")


class AnalyzeRequest(BaseModel):
    """Analysis request model."""
    genes: List[str] = Field(..., description="List of gene identifiers")
    species: str = Field(..., description="Species of input data")
    analysis_type: str = Field("ora", description="Type of analysis (ora, gsea)")
    databases: List[str] = Field(..., description="Databases to use for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis parameters")


class AnalysisResponse(BaseModel):
    """Analysis response model."""
    job_id: str
    status: str
    analysis_type: str
    species: str
    input_gene_count: int
    total_pathways: int
    significant_pathways: int
    database_results: Dict[str, List[Dict[str, Any]]]
    consensus_results: List[Dict[str, Any]]
    created_at: datetime


class AnalyzeResponse(BaseModel):
    """Analysis response model."""
    job_id: str
    status: str
    analysis_type: str
    species: str
    input_gene_count: int
    total_pathways: int
    significant_pathways: int
    database_results: Dict[str, List[Dict[str, Any]]]
    consensus_results: List[Dict[str, Any]]
    created_at: datetime


class AnalysisStatus(BaseModel):
    """Analysis status model."""
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AnalysisResult(BaseModel):
    """Analysis result model."""
    job_id: str
    status: str
    analysis_type: str
    species: str
    input_gene_count: int
    total_pathways: int
    significant_pathways: int
    database_results: Dict[str, List[Dict[str, Any]]]
    consensus_results: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    created_at: datetime
    completed_at: Optional[datetime] = None


class GSEARequest(BaseModel):
    """GSEA analysis request model."""
    ranked_genes: List[Dict[str, Any]] = Field(..., description="Ranked gene list with scores")
    species: str = Field(..., description="Species of input data")
    databases: List[str] = Field(..., description="Databases to use for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis parameters")


class AnalysisParameters(BaseModel):
    """Analysis parameters model."""
    species: str
    analysis_type: str
    databases: List[str]
    significance_threshold: float = 0.05
    correction_method: str = "fdr_bh"
    min_pathway_size: int = 5
    max_pathway_size: int = 500


class GSEAResponse(BaseModel):
    """GSEA analysis response model."""
    job_id: str
    status: str
    analysis_type: str = "gsea"
    species: str
    input_gene_count: int
    total_pathways: int
    significant_pathways: int
    database_results: Dict[str, List[Dict[str, Any]]]
    consensus_results: List[Dict[str, Any]]
    created_at: datetime


class BatchAnalyzeRequest(BaseModel):
    """Batch analysis request model."""
    datasets: List[Dict[str, Any]] = Field(..., description="Multiple datasets to analyze")
    species: str = Field(..., description="Species of input data")
    analysis_type: str = Field("ora", description="Type of analysis")
    databases: List[str] = Field(..., description="Databases to use for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis parameters")


class BatchAnalyzeResponse(BaseModel):
    """Batch analysis response model."""
    job_id: str
    status: str
    results: List[Dict[str, Any]]
    created_at: datetime
