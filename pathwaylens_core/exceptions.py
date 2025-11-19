"""
Custom exceptions for PathwayLens.

Provides specific exception types for better error handling and debugging.
"""


class PathwayLensError(Exception):
    """Base exception for all PathwayLens errors."""
    pass


class DatabaseError(PathwayLensError):
    """Errors related to database operations."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to pathway database."""
    pass


class DatabaseVersionError(DatabaseError):
    """Database version incompatibility."""
    pass


class InvalidParameterError(PathwayLensError):
    """Invalid analysis parameter provided."""
    pass


class InsufficientDataError(PathwayLensError):
    """Insufficient data for analysis."""
    pass


class AnalysisError(PathwayLensError):
    """Error during pathway analysis."""
    pass


class ORAError(AnalysisError):
    """Error during ORA analysis."""
    pass


class GSEAError(AnalysisError):
    """Error during GSEA analysis."""
    pass


class VisualizationError(PathwayLensError):
    """Error during visualization generation."""
    pass


class NormalizationError(PathwayLensError):
    """Error during data normalization."""
    pass


class IDConversionError(NormalizationError):
    """Error during ID conversion."""
    pass


class SpeciesMappingError(NormalizationError):
    """Error during species mapping."""
    pass


class ProvenanceError(PathwayLensError):
    """Error in provenance tracking."""
    pass


class ManifestError(ProvenanceError):
    """Error in manifest generation or validation."""
    pass


class OutputError(PathwayLensError):
    """Error in output generation."""
    pass


class FileFormatError(PathwayLensError):
    """Unsupported or invalid file format."""
    pass
