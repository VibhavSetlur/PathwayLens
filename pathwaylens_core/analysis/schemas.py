"""
Pydantic schemas for analysis module.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import pandas as pd
from datetime import datetime

from pathwaylens_core.types import OmicType, DataType


class AnalysisType(str, Enum):
    """Supported analysis types."""
    ORA = "ora"
    GSEA = "gsea"
    GSVA = "gsva"
    TOPOLOGY = "topology"
    MULTI_OMICS = "multi_omics"
    CONSENSUS = "consensus"


class DatabaseType(str, Enum):
    """Supported pathway databases."""
    KEGG = "kegg"
    REACTOME = "reactome"
    GO = "go"
    BIOCYC = "biocyc"
    PATHWAY_COMMONS = "pathway_commons"
    MSIGDB = "msigdb"
    PANTHER = "panther"
    WIKIPATHWAYS = "wikipathways"


class CorrectionMethod(str, Enum):
    """Multiple testing correction methods."""
    BONFERRONI = "bonferroni"
    FDR_BH = "fdr_bh"
    FDR_BY = "fdr_by"
    FDR_TSBH = "fdr_tsbh"
    FDR_TSBKY = "fdr_tsbky"
    HOLM = "holm"
    HOCHBERG = "hochberg"
    HOMMEL = "hommel"
    SIDAK = "sidak"
    SIDAK_SS = "sidak_ss"
    SIDAK_SD = "sidak_sd"


class ConsensusMethod(str, Enum):
    """Consensus analysis methods."""
    STOUFFER = "stouffer"
    FISHER = "fisher"
    BROWN = "brown"
    KOST = "kost"
    TIPPETT = "tippett"
    MUDHOLKAR_GEORGE = "mudholkar_george"
    WILKINSON = "wilkinson"
    PEARSON = "pearson"
    GEOMETRIC_MEAN = "geometric_mean"


class AnalysisParameters(BaseModel):
    """Parameters for pathway analysis."""
    
    # Basic parameters
    analysis_type: AnalysisType = Field(..., description="Type of analysis to perform")
    omic_type: OmicType = Field(..., description="Omic type (transcriptomics, proteomics, etc.)")
    data_type: DataType = Field(..., description="Specific data type (bulk, shotgun, etc.)")
    databases: List[DatabaseType] = Field(..., description="Databases to use for analysis")
    species: str = Field(..., description="Species for analysis")
    tool: str = Field(default="auto", description="Tool used to generate input")
    
    # Statistical parameters
    significance_threshold: float = Field(default=0.05, description="Significance threshold")
    lfc_threshold: float = Field(default=0.0, description="Log2 fold change threshold")
    correction_method: CorrectionMethod = Field(default=CorrectionMethod.FDR_BH, description="Multiple testing correction")
    min_pathway_size: int = Field(default=5, description="Minimum pathway size")
    max_pathway_size: int = Field(default=500, description="Maximum pathway size")
    
    # GSEA-specific parameters
    gsea_permutations: int = Field(default=1000, description="Number of permutations for GSEA")
    gsea_min_size: int = Field(default=15, description="Minimum gene set size for GSEA")
    gsea_max_size: int = Field(default=500, description="Maximum gene set size for GSEA")
    
    # GSVA-specific parameters (placeholders for surface)
    gsva_kernel: str = Field(default="gaussian", description="GSVA kernel method")
    gsva_min_size: int = Field(default=10, description="Minimum gene set size for GSVA")
    gsva_max_size: int = Field(default=500, description="Maximum gene set size for GSVA")
    
    # Topology/SPIA-like parameters
    topology_method: str = Field(default="spia", description="Topology analysis method")
    topology_use_direction: bool = Field(default=True, description="Use edge directionality if available")
    
    # Consensus parameters
    consensus_method: ConsensusMethod = Field(default=ConsensusMethod.STOUFFER, description="Consensus method")
    min_databases: int = Field(default=2, description="Minimum number of databases for consensus")
    
    # Background parameters
    background_source: str = Field(default="database", description="Source of background genes")
    custom_background: Optional[List[str]] = Field(default=None, description="Custom background gene list")
    background_size: Optional[int] = Field(default=None, description="Total number of background genes")
    
    # Additional parameters
    include_plots: bool = Field(default=True, description="Include plots in results")
    include_networks: bool = Field(default=True, description="Include network analysis")
    export_formats: List[str] = Field(default=["json", "csv"], description="Export formats")
    
    @field_validator('significance_threshold')
    @classmethod
    def validate_significance_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Significance threshold must be between 0 and 1")
        return v
    
    @field_validator('min_pathway_size')
    @classmethod
    def validate_min_pathway_size(cls, v):
        if v < 1:
            raise ValueError("Minimum pathway size must be at least 1")
        return v
    
    @field_validator('max_pathway_size')
    @classmethod
    def validate_max_pathway_size(cls, v):
        if v < 1:
            raise ValueError("Maximum pathway size must be at least 1")
        return v
    
    @field_validator('gsea_permutations')
    @classmethod
    def validate_gsea_permutations(cls, v):
        if v < 100:
            raise ValueError("GSEA permutations must be at least 100")
        return v


class PathwayResult(BaseModel):
    """Result for a single pathway."""
    
    pathway_id: str = Field(..., description="Pathway identifier")
    pathway_name: str = Field(..., description="Pathway name")
    database: DatabaseType = Field(..., description="Source database")
    
    # Statistical results
    p_value: float = Field(..., description="P-value")
    adjusted_p_value: float = Field(..., description="Adjusted p-value")
    enrichment_score: Optional[float] = Field(None, description="Enrichment score")
    normalized_enrichment_score: Optional[float] = Field(None, description="Normalized enrichment score")
    
    # Research-grade statistical metrics
    odds_ratio: Optional[float] = Field(None, description="Odds ratio for enrichment (ORA only)")
    odds_ratio_ci_lower: Optional[float] = Field(None, description="95% CI lower bound for odds ratio")
    odds_ratio_ci_upper: Optional[float] = Field(None, description="95% CI upper bound for odds ratio")
    fold_enrichment: Optional[float] = Field(None, description="Fold enrichment ratio")
    effect_size: Optional[float] = Field(None, description="Effect size (Cohen's h)")
    genes_expected: Optional[float] = Field(None, description="Expected genes by chance")
    statistical_power: Optional[float] = Field(None, description="Post-hoc statistical power")
    
    # Gene counts
    overlap_count: int = Field(..., description="Number of overlapping genes")
    pathway_count: int = Field(..., description="Total genes in pathway")
    input_count: int = Field(..., description="Total genes in input")
    
    # Gene lists
    overlapping_genes: List[str] = Field(..., description="List of overlapping genes")
    pathway_genes: List[str] = Field(default_factory=list, description="All genes in pathway")
    
    # Additional information
    pathway_url: Optional[str] = Field(None, description="URL to pathway information")
    pathway_description: Optional[str] = Field(None, description="Pathway description")
    pathway_category: Optional[str] = Field(None, description="Pathway category")
    
    # Metadata
    analysis_method: str = Field(..., description="Analysis method used")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    
    @field_validator('p_value')
    @classmethod
    def validate_p_value(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("P-value must be between 0 and 1")
        return v
    
    @field_validator('adjusted_p_value')
    @classmethod
    def validate_adjusted_p_value(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Adjusted p-value must be between 0 and 1")
        return v


class DatabaseResult(BaseModel):
    """Results from a single database."""
    
    database: DatabaseType = Field(..., description="Database name")
    total_pathways: int = Field(..., description="Total pathways analyzed")
    significant_pathways: int = Field(..., description="Number of significant pathways")
    pathways: List[PathwayResult] = Field(..., description="List of pathway results")
    
    # Database-specific metadata
    database_version: Optional[str] = Field(None, description="Database version")
    last_updated: Optional[str] = Field(None, description="Last update date")
    species: str = Field(..., description="Species")
    
    # Quality metrics
    coverage: float = Field(..., description="Coverage of input genes")
    redundancy: float = Field(default=0.0, description="Pathway redundancy")
    
    @field_validator('coverage')
    @classmethod
    def validate_coverage(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Coverage must be between 0 and 1")
        return v


class ConsensusResult(BaseModel):
    """Consensus results across databases."""
    
    consensus_method: ConsensusMethod = Field(..., description="Consensus method used")
    total_pathways: int = Field(..., description="Total pathways in consensus")
    significant_pathways: int = Field(..., description="Number of significant pathways")
    pathways: List[PathwayResult] = Field(..., description="List of consensus pathway results")
    
    # Consensus statistics
    database_agreement: Dict[str, float] = Field(..., description="Agreement between databases")
    consensus_score: float = Field(..., description="Overall consensus score")
    
    # Quality metrics
    reproducibility: float = Field(..., description="Reproducibility score")
    stability: float = Field(..., description="Stability score")
    
    @field_validator('consensus_score')
    @classmethod
    def validate_consensus_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Consensus score must be between 0 and 1")
        return v


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    
    # Job information
    job_id: str = Field(..., description="Unique job identifier")
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    parameters: AnalysisParameters = Field(..., description="Analysis parameters")
    
    # Input information
    input_file: str = Field(..., description="Path to input file")
    input_gene_count: int = Field(..., description="Number of input genes")
    input_species: str = Field(..., description="Species of input genes")
    
    # Results
    database_results: Dict[str, DatabaseResult] = Field(..., description="Results by database")
    consensus_results: Optional[ConsensusResult] = Field(None, description="Consensus results")
    
    # Summary statistics
    total_pathways: int = Field(..., description="Total pathways analyzed")
    significant_pathways: int = Field(..., description="Number of significant pathways")
    significant_databases: int = Field(..., description="Number of databases with significant results")
    
    # Quality metrics
    overall_quality: float = Field(..., description="Overall analysis quality")
    reproducibility: float = Field(..., description="Reproducibility score")
    
    # Metadata
    created_at: str = Field(..., description="Analysis creation time")
    completed_at: str = Field(..., description="Analysis completion time")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Output files
    output_files: Dict[str, str] = Field(default_factory=dict, description="Generated output files")
    visualization_files: Dict[str, str] = Field(default_factory=dict, description="Generated visualization files")
    
    # Warnings and errors
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    @field_validator('overall_quality')
    @classmethod
    def validate_overall_quality(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Overall quality must be between 0 and 1")
        return v
    
    @field_validator('reproducibility')
    @classmethod
    def validate_reproducibility(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Reproducibility must be between 0 and 1")
        return v


class AnalysisSummary(BaseModel):
    """Summary of analysis results."""
    
    job_id: str = Field(..., description="Job identifier")
    analysis_type: AnalysisType = Field(..., description="Analysis type")
    status: str = Field(..., description="Analysis status")
    
    # Quick statistics
    input_genes: int = Field(..., description="Number of input genes")
    significant_pathways: int = Field(..., description="Number of significant pathways")
    databases_used: int = Field(..., description="Number of databases used")
    
    # Top pathways
    top_pathways: List[Dict[str, Any]] = Field(..., description="Top significant pathways")
    
    # Quality indicators
    quality_score: float = Field(..., description="Overall quality score")
    completion_time: Optional[str] = Field(None, description="Completion time")
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Quality score must be between 0 and 1")
        return v


class AnalysisProgress(BaseModel):
    """Analysis progress information."""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., description="Progress percentage")
    current_step: str = Field(..., description="Current processing step")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    
    # Step details
    completed_steps: List[str] = Field(default_factory=list, description="Completed steps")
    remaining_steps: List[str] = Field(default_factory=list, description="Remaining steps")
    
    # Performance metrics
    processing_time: float = Field(default=0.0, description="Processing time so far")
    memory_usage: float = Field(default=0.0, description="Memory usage in MB")
    
    @field_validator('progress')
    @classmethod
    def validate_progress(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Progress must be between 0 and 100")
        return v
