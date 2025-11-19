"""
Pydantic schemas for visualization module.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class PlotType(str, Enum):
    """Types of plots that can be generated."""
    DOT_PLOT = "dot_plot"
    VOLCANO_PLOT = "volcano_plot"
    HEATMAP = "heatmap"
    NETWORK = "network"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    PCA_PLOT = "pca_plot"
    UMAP_PLOT = "umap_plot"
    MANHATTAN_PLOT = "manhattan_plot"
    ENRICHMENT_PLOT = "enrichment_plot"
    PATHWAY_MAP = "pathway_map"
    GENE_EXPRESSION = "gene_expression"
    TIME_SERIES = "time_series"
    COMPARISON_PLOT = "comparison_plot"
    CONSENSUS_PLOT = "consensus_plot"
    UPSET_PLOT = "upset_plot"
    MULTI_OMICS_HEATMAP = "multi_omics_heatmap"
    MULTI_OMICS_NETWORK = "multi_omics_network"
    MULTI_OMICS_SANKEY = "multi_omics_sankey"


class VisualizationParameters(BaseModel):
    """Parameters for visualization generation."""
    
    # Basic parameters
    plot_types: List[PlotType] = Field(..., description="Types of plots to generate")
    interactive: bool = Field(default=True, description="Generate interactive plots")
    output_format: str = Field(default="html", description="Output format (html, png, svg, pdf)")
    
    # Styling parameters
    theme: str = Field(default="light", description="Plot theme (light, dark, high_contrast)")
    color_scheme: str = Field(default="default", description="Color scheme")
    figure_size: List[int] = Field(default=[800, 600], description="Figure size [width, height]")
    dpi: int = Field(default=300, description="DPI for static plots")
    
    # Plot-specific parameters
    max_pathways: int = Field(default=50, description="Maximum number of pathways to show")
    significance_threshold: float = Field(default=0.05, description="Significance threshold for highlighting")
    fold_change_threshold: float = Field(default=1.5, description="Fold change threshold for highlighting")
    
    # Network parameters
    network_layout: str = Field(default="spring", description="Network layout algorithm")
    node_size_range: List[int] = Field(default=[10, 50], description="Node size range")
    edge_width_range: List[float] = Field(default=[1.0, 5.0], description="Edge width range")
    
    # Export parameters
    include_metadata: bool = Field(default=True, description="Include metadata in exports")
    include_data: bool = Field(default=True, description="Include raw data in exports")
    compress_output: bool = Field(default=False, description="Compress output files")
    
    @field_validator('significance_threshold')
    @classmethod
    def validate_significance_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Significance threshold must be between 0 and 1")
        return v
    
    @field_validator('fold_change_threshold')
    @classmethod
    def validate_fold_change_threshold(cls, v):
        if v <= 0:
            raise ValueError("Fold change threshold must be positive")
        return v
    
    @field_validator('figure_size')
    @classmethod
    def validate_figure_size(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("Figure size must be a list of two positive integers")
        return v
    
    @field_validator('dpi')
    @classmethod
    def validate_dpi(cls, v):
        if v <= 0:
            raise ValueError("DPI must be positive")
        return v


class PlotMetadata(BaseModel):
    """Metadata for a generated plot."""
    
    plot_type: PlotType = Field(..., description="Type of plot")
    title: str = Field(..., description="Plot title")
    description: Optional[str] = Field(None, description="Plot description")
    
    # Data information
    num_data_points: int = Field(..., description="Number of data points")
    num_pathways: int = Field(..., description="Number of pathways shown")
    num_genes: int = Field(..., description="Number of genes shown")
    
    # Statistical information
    significance_threshold: float = Field(..., description="Significance threshold used")
    num_significant: int = Field(..., description="Number of significant results")
    
    # Styling information
    theme: str = Field(..., description="Theme used")
    color_scheme: str = Field(..., description="Color scheme used")
    figure_size: List[int] = Field(..., description="Figure size")
    
    # Technical information
    interactive: bool = Field(..., description="Whether plot is interactive")
    output_format: str = Field(..., description="Output format")
    file_size: int = Field(..., description="File size in bytes")
    
    # Timestamps
    created_at: str = Field(..., description="Creation timestamp")
    processing_time: float = Field(..., description="Processing time in seconds")


class VisualizationResult(BaseModel):
    """Result of visualization generation."""
    
    # Job information
    job_id: str = Field(..., description="Unique job identifier")
    input_file: str = Field(..., description="Input file path")
    parameters: VisualizationParameters = Field(..., description="Visualization parameters")
    
    # Generated plots
    generated_plots: Dict[PlotType, str] = Field(..., description="Generated plot files")
    plot_metadata: Dict[PlotType, PlotMetadata] = Field(..., description="Plot metadata")
    
    # Summary statistics
    total_plots: int = Field(..., description="Total number of plots generated")
    total_size: int = Field(..., description="Total size of generated files in bytes")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    # Quality metrics
    plot_quality: float = Field(..., description="Overall plot quality score")
    data_coverage: float = Field(..., description="Data coverage score")
    
    # Metadata
    created_at: str = Field(..., description="Visualization creation time")
    completed_at: str = Field(..., description="Visualization completion time")
    
    # Output files
    output_files: Dict[str, str] = Field(default_factory=dict, description="Generated output files")
    dashboard_file: Optional[str] = Field(None, description="Generated dashboard file")
    
    # Warnings and errors
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    @field_validator('plot_quality')
    @classmethod
    def validate_plot_quality(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Plot quality must be between 0 and 1")
        return v
    
    @field_validator('data_coverage')
    @classmethod
    def validate_data_coverage(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Data coverage must be between 0 and 1")
        return v


class DashboardConfig(BaseModel):
    """Configuration for dashboard generation."""
    
    # Layout parameters
    layout: str = Field(default="grid", description="Dashboard layout (grid, tabbed, sidebar)")
    num_columns: int = Field(default=2, description="Number of columns in grid layout")
    plot_size: List[int] = Field(default=[400, 300], description="Default plot size")
    
    # Navigation parameters
    include_navigation: bool = Field(default=True, description="Include navigation menu")
    include_filters: bool = Field(default=True, description="Include filter controls")
    include_export: bool = Field(default=True, description="Include export controls")
    
    # Styling parameters
    theme: str = Field(default="light", description="Dashboard theme")
    color_scheme: str = Field(default="default", description="Color scheme")
    font_family: str = Field(default="Arial", description="Font family")
    font_size: int = Field(default=12, description="Font size")
    
    # Interactive parameters
    enable_zoom: bool = Field(default=True, description="Enable zoom functionality")
    enable_pan: bool = Field(default=True, description="Enable pan functionality")
    enable_hover: bool = Field(default=True, description="Enable hover tooltips")
    enable_selection: bool = Field(default=True, description="Enable selection functionality")
    
    # Export parameters
    export_formats: List[str] = Field(default=["html", "png", "pdf"], description="Export formats")
    include_metadata: bool = Field(default=True, description="Include metadata in exports")
    include_data: bool = Field(default=True, description="Include raw data in exports")
    
    @field_validator('num_columns')
    @classmethod
    def validate_num_columns(cls, v):
        if v <= 0:
            raise ValueError("Number of columns must be positive")
        return v
    
    @field_validator('plot_size')
    @classmethod
    def validate_plot_size(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("Plot size must be a list of two positive integers")
        return v
    
    @field_validator('font_size')
    @classmethod
    def validate_font_size(cls, v):
        if v <= 0:
            raise ValueError("Font size must be positive")
        return v


class ExportConfig(BaseModel):
    """Configuration for plot export."""
    
    # Export parameters
    formats: List[str] = Field(..., description="Export formats")
    quality: str = Field(default="high", description="Export quality (low, medium, high)")
    dpi: int = Field(default=300, description="DPI for raster formats")
    
    # File parameters
    filename_prefix: str = Field(default="plot", description="Filename prefix")
    include_timestamp: bool = Field(default=True, description="Include timestamp in filename")
    compress: bool = Field(default=False, description="Compress output files")
    
    # Metadata parameters
    include_metadata: bool = Field(default=True, description="Include metadata")
    include_data: bool = Field(default=True, description="Include raw data")
    include_parameters: bool = Field(default=True, description="Include visualization parameters")
    
    # Styling parameters
    background_color: str = Field(default="white", description="Background color")
    border: bool = Field(default=True, description="Include border")
    watermark: bool = Field(default=False, description="Include watermark")
    
    @field_validator('dpi')
    @classmethod
    def validate_dpi(cls, v):
        if v <= 0:
            raise ValueError("DPI must be positive")
        return v
    
    @field_validator('quality')
    @classmethod
    def validate_quality(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError("Quality must be one of: low, medium, high")
        return v


class VisualizationSummary(BaseModel):
    """Summary of visualization results."""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Visualization status")
    
    # Quick statistics
    num_plots: int = Field(..., description="Number of plots generated")
    total_size: int = Field(..., description="Total size in bytes")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Plot types
    plot_types: List[PlotType] = Field(..., description="Types of plots generated")
    interactive_plots: int = Field(..., description="Number of interactive plots")
    static_plots: int = Field(..., description="Number of static plots")
    
    # Quality indicators
    quality_score: float = Field(..., description="Overall quality score")
    completion_time: Optional[str] = Field(None, description="Completion time")
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Quality score must be between 0 and 1")
        return v


class VisualizationProgress(BaseModel):
    """Visualization progress information."""
    
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
