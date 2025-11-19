"""
Pydantic schemas for comparison module.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
from datetime import datetime


class ComparisonType(str, Enum):
    """Types of comparisons that can be performed."""
    GENE_OVERLAP = "gene_overlap"
    PATHWAY_OVERLAP = "pathway_overlap"
    PATHWAY_CONCORDANCE = "pathway_concordance"
    ENRICHMENT_CORRELATION = "enrichment_correlation"
    DATASET_CLUSTERING = "dataset_clustering"
    COMPREHENSIVE = "comprehensive"


class ComparisonParameters(BaseModel):
    """Parameters for dataset comparison."""
    
    # Basic parameters
    comparison_type: ComparisonType = Field(..., description="Type of comparison to perform")
    species: str = Field(..., description="Species for comparison")
    
    # Statistical parameters
    significance_threshold: float = Field(default=0.05, description="Significance threshold")
    correlation_threshold: float = Field(default=0.3, description="Minimum correlation threshold")
    overlap_threshold: float = Field(default=0.1, description="Minimum overlap threshold")
    
    # Analysis parameters
    databases: List[str] = Field(default=["kegg", "reactome"], description="Databases to use")
    min_pathway_size: int = Field(default=5, description="Minimum pathway size")
    max_pathway_size: int = Field(default=500, description="Maximum pathway size")
    
    # Clustering parameters
    clustering_method: str = Field(default="hierarchical", description="Clustering method")
    distance_metric: str = Field(default="euclidean", description="Distance metric")
    linkage_method: str = Field(default="ward", description="Linkage method")
    
    # Output parameters
    include_plots: bool = Field(default=True, description="Include plots in results")
    include_networks: bool = Field(default=True, description="Include network analysis")
    export_formats: List[str] = Field(default=["json", "csv"], description="Export formats")
    
    @field_validator('significance_threshold')
    @classmethod
    def validate_significance_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Significance threshold must be between 0 and 1")
        return v
    
    @field_validator('correlation_threshold')
    @classmethod
    def validate_correlation_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Correlation threshold must be between 0 and 1")
        return v
    
    @field_validator('overlap_threshold')
    @classmethod
    def validate_overlap_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Overlap threshold must be between 0 and 1")
        return v


class OverlapStatistics(BaseModel):
    """Statistics for gene/pathway overlap analysis."""
    
    dataset1: str = Field(..., description="First dataset name")
    dataset2: str = Field(..., description="Second dataset name")
    
    # Gene overlap statistics
    total_genes_dataset1: int = Field(..., description="Total genes in dataset 1")
    total_genes_dataset2: int = Field(..., description="Total genes in dataset 2")
    overlapping_genes: int = Field(..., description="Number of overlapping genes")
    overlap_percentage: float = Field(..., description="Percentage of overlap")
    jaccard_index: float = Field(..., description="Jaccard similarity index")
    
    # Pathway overlap statistics
    total_pathways_dataset1: int = Field(..., description="Total pathways in dataset 1")
    total_pathways_dataset2: int = Field(..., description="Total pathways in dataset 2")
    overlapping_pathways: int = Field(..., description="Number of overlapping pathways")
    pathway_overlap_percentage: float = Field(..., description="Percentage of pathway overlap")
    pathway_jaccard_index: float = Field(..., description="Pathway Jaccard similarity index")
    
    # Gene lists
    genes_dataset1: List[str] = Field(..., description="Genes in dataset 1")
    genes_dataset2: List[str] = Field(..., description="Genes in dataset 2")
    overlapping_gene_list: List[str] = Field(..., description="List of overlapping genes")
    unique_genes_dataset1: List[str] = Field(..., description="Genes unique to dataset 1")
    unique_genes_dataset2: List[str] = Field(..., description="Genes unique to dataset 2")
    
    @field_validator('overlap_percentage')
    @classmethod
    def validate_overlap_percentage(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Overlap percentage must be between 0 and 1")
        return v
    
    @field_validator('jaccard_index')
    @classmethod
    def validate_jaccard_index(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Jaccard index must be between 0 and 1")
        return v


class CorrelationResult(BaseModel):
    """Result of correlation analysis."""
    
    dataset1: str = Field(..., description="First dataset name")
    dataset2: str = Field(..., description="Second dataset name")
    
    # Correlation statistics
    correlation: float = Field(..., description="Correlation coefficient")
    p_value: float = Field(..., description="P-value of correlation")
    confidence_interval: List[float] = Field(..., description="95% confidence interval")
    
    # Additional statistics
    sample_size: int = Field(..., description="Sample size for correlation")
    degrees_of_freedom: int = Field(..., description="Degrees of freedom")
    
    # Significance
    is_significant: bool = Field(..., description="Whether correlation is significant")
    significance_level: float = Field(default=0.05, description="Significance level used")
    
    @field_validator('correlation')
    @classmethod
    def validate_correlation(cls, v):
        if not -1 <= v <= 1:
            raise ValueError("Correlation must be between -1 and 1")
        return v
    
    @field_validator('p_value')
    @classmethod
    def validate_p_value(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("P-value must be between 0 and 1")
        return v


class ClusteringResult(BaseModel):
    """Result of clustering analysis."""
    
    method: str = Field(..., description="Clustering method used")
    distance_metric: str = Field(..., description="Distance metric used")
    linkage_method: str = Field(..., description="Linkage method used")
    
    # Clustering results
    num_clusters: int = Field(..., description="Number of clusters")
    cluster_labels: List[int] = Field(..., description="Cluster labels for each dataset")
    cluster_centers: List[List[float]] = Field(..., description="Cluster centers")
    
    # Quality metrics
    silhouette_score: float = Field(..., description="Silhouette score")
    inertia: float = Field(..., description="Inertia (within-cluster sum of squares)")
    calinski_harabasz_score: float = Field(..., description="Calinski-Harabasz score")
    
    # Cluster information
    cluster_sizes: List[int] = Field(..., description="Size of each cluster")
    cluster_datasets: Dict[int, List[str]] = Field(..., description="Datasets in each cluster")
    
    @field_validator('silhouette_score')
    @classmethod
    def validate_silhouette_score(cls, v):
        if not -1 <= v <= 1:
            raise ValueError("Silhouette score must be between -1 and 1")
        return v


class PathwayConcordance(BaseModel):
    """Pathway concordance analysis result."""
    
    pathway_id: str = Field(..., description="Pathway identifier")
    pathway_name: str = Field(..., description="Pathway name")
    database: str = Field(..., description="Source database")
    
    # Concordance statistics
    concordance_score: float = Field(..., description="Concordance score")
    num_datasets: int = Field(..., description="Number of datasets")
    num_significant: int = Field(..., description="Number of significant datasets")
    significance_rate: float = Field(..., description="Rate of significance across datasets")
    
    # P-values across datasets
    p_values: Dict[str, float] = Field(..., description="P-values for each dataset")
    adjusted_p_values: Dict[str, float] = Field(..., description="Adjusted P-values for each dataset")
    
    # Effect sizes
    effect_sizes: Dict[str, float] = Field(..., description="Effect sizes for each dataset")
    
    # Metadata
    pathway_size: int = Field(..., description="Number of genes in pathway")
    pathway_category: Optional[str] = Field(None, description="Pathway category")
    
    @field_validator('concordance_score')
    @classmethod
    def validate_concordance_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Concordance score must be between 0 and 1")
        return v
    
    @field_validator('significance_rate')
    @classmethod
    def validate_significance_rate(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Significance rate must be between 0 and 1")
        return v


class ComparisonResult(BaseModel):
    """Complete comparison result."""
    
    # Job information
    job_id: str = Field(..., description="Unique job identifier")
    comparison_type: ComparisonType = Field(..., description="Type of comparison performed")
    parameters: ComparisonParameters = Field(..., description="Comparison parameters")
    
    # Input information
    input_files: List[str] = Field(..., description="Input file paths")
    num_datasets: int = Field(..., description="Number of datasets compared")
    total_genes: int = Field(..., description="Total unique genes across datasets")
    unique_genes: int = Field(..., description="Number of unique genes")
    
    # Results
    overlap_statistics: Dict[str, OverlapStatistics] = Field(..., description="Overlap statistics between dataset pairs")
    correlation_results: Dict[str, CorrelationResult] = Field(..., description="Correlation results between dataset pairs")
    clustering_results: Optional[ClusteringResult] = Field(None, description="Clustering analysis results")
    pathway_concordance: List[PathwayConcordance] = Field(..., description="Pathway concordance analysis")
    
    # Summary statistics
    average_overlap: float = Field(..., description="Average overlap across all pairs")
    average_correlation: float = Field(..., description="Average correlation across all pairs")
    num_significant_pathways: int = Field(..., description="Number of significantly concordant pathways")
    
    # Quality metrics
    overall_quality: float = Field(..., description="Overall comparison quality")
    reproducibility: float = Field(..., description="Reproducibility score")
    
    # Metadata
    created_at: str = Field(..., description="Comparison creation time")
    completed_at: str = Field(..., description="Comparison completion time")
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


class ComparisonSummary(BaseModel):
    """Summary of comparison results."""
    
    job_id: str = Field(..., description="Job identifier")
    comparison_type: ComparisonType = Field(..., description="Comparison type")
    status: str = Field(..., description="Comparison status")
    
    # Quick statistics
    num_datasets: int = Field(..., description="Number of datasets")
    total_genes: int = Field(..., description="Total genes")
    average_overlap: float = Field(..., description="Average overlap")
    average_correlation: float = Field(..., description="Average correlation")
    
    # Top results
    top_overlaps: List[Dict[str, Any]] = Field(..., description="Top overlapping dataset pairs")
    top_correlations: List[Dict[str, Any]] = Field(..., description="Top correlated dataset pairs")
    top_concordant_pathways: List[Dict[str, Any]] = Field(..., description="Top concordant pathways")
    
    # Quality indicators
    quality_score: float = Field(..., description="Overall quality score")
    completion_time: Optional[str] = Field(None, description="Completion time")
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Quality score must be between 0 and 1")
        return v


class ComparisonProgress(BaseModel):
    """Comparison progress information."""
    
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
