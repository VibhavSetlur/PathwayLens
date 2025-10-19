"""
PathwayLens Core - Next-generation computational biology platform.

This package provides the core functionality for PathwayLens 2.0, including
normalization, analysis, visualization, and data management capabilities.
"""

__version__ = "2.0.0"
__author__ = "PathwayLens Team"
__email__ = "pathwaylens@example.com"

from .normalization import Normalizer
from .analysis import AnalysisEngine
from .data import DatabaseManager
from .visualization import VisualizationEngine
from .comparison import ComparisonEngine

__all__ = [
    "Normalizer",
    "AnalysisEngine", 
    "DatabaseManager",
    "VisualizationEngine",
    "ComparisonEngine",
]
