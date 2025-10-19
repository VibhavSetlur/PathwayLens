"""
Comparison module for PathwayLens.

This module provides capabilities for comparing multiple datasets,
analyzing overlaps, correlations, and pathway concordance.
"""

from .engine import ComparisonEngine
from .overlap_analyzer import OverlapAnalyzer
from .correlation_analyzer import CorrelationAnalyzer
from .clustering_analyzer import ClusteringAnalyzer
from .visualization import ComparisonVisualizer
from .schemas import ComparisonResult, ComparisonParameters, ComparisonType

__all__ = [
    "ComparisonEngine",
    "OverlapAnalyzer",
    "CorrelationAnalyzer",
    "ClusteringAnalyzer",
    "ComparisonVisualizer",
    "ComparisonResult",
    "ComparisonParameters",
    "ComparisonType",
]
