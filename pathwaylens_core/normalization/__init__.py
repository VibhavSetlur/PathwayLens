"""
Normalization module for PathwayLens.

This module provides gene identifier conversion, format detection,
cross-species mapping, input validation, confidence scoring, and
orthology mapping capabilities.
"""

from .normalizer import Normalizer
from .format_detector import FormatDetector
from .id_converter import IDConverter
from .species_mapper import SpeciesMapper
from .validation import InputValidator
from .confidence_calculator import ConfidenceCalculator, ConfidenceFactors
from .orthology_engine import OrthologyEngine, OrthologMapping
from .schemas import NormalizedTable, NormalizationResult

__all__ = [
    "Normalizer",
    "FormatDetector",
    "IDConverter", 
    "SpeciesMapper",
    "InputValidator",
    "ConfidenceCalculator",
    "ConfidenceFactors",
    "OrthologyEngine",
    "OrthologMapping",
    "NormalizedTable",
    "NormalizationResult",
]
