"""
Custom exceptions for PathwayLens CLI.
"""


class CLIException(Exception):
    """Base exception for CLI errors."""
    pass


class ConfigurationError(CLIException):
    """Configuration-related errors."""
    pass


class APIError(CLIException):
    """API-related errors."""
    pass


class AnalysisError(CLIException):
    """Analysis-related errors."""
    pass


class VisualizationError(CLIException):
    """Visualization-related errors."""
    pass


class FileError(CLIException):
    """File-related errors."""
    pass
