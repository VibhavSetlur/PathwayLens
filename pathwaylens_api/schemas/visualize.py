"""
Visualization schemas for PathwayLens API.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class PlotType(str, Enum):
    """Plot type enumeration."""
    DOT_PLOT = "dot_plot"
    VOLCANO_PLOT = "volcano_plot"
    HEATMAP = "heatmap"
    NETWORK = "network"
    PCA = "pca"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"


class VisualizationRequest(BaseModel):
    """Visualization request model."""
    analysis_id: str = Field(..., description="Analysis job ID")
    plot_types: List[str] = Field(..., description="Types of plots to generate")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Visualization parameters")


class VisualizeRequest(BaseModel):
    """Visualization request model."""
    analysis_id: str = Field(..., description="Analysis job ID")
    plot_types: List[str] = Field(..., description="Types of plots to generate")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Visualization parameters")


class VisualizationResponse(BaseModel):
    """Visualization response model."""
    job_id: str
    status: str
    visualization_id: str
    plots: List[Dict[str, Any]]
    created_at: datetime


class VisualizationStatus(BaseModel):
    """Visualization status model."""
    job_id: str
    status: str
    progress: float
    message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class VisualizationResult(BaseModel):
    """Visualization result model."""
    job_id: str
    status: str
    visualization_id: str
    plots: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    created_at: datetime
    completed_at: Optional[datetime] = None


class VisualizationParameters(BaseModel):
    """Visualization parameters model."""
    plot_types: List[str]
    theme: str = "default"
    width: int = 800
    height: int = 600
    format: str = "html"
    include_plotlyjs: bool = True


class VisualizeResponse(BaseModel):
    """Visualization response model."""
    job_id: str
    status: str
    visualization_id: str
    plots: List[Dict[str, Any]]
    created_at: datetime
