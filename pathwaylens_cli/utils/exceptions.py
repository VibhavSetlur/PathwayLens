"""
Custom exceptions for PathwayLens CLI.
"""


class CLIException(Exception):
    """Base exception for CLI errors."""
    def __init__(self, message: str, context: dict = None):
        self.message = message
        self.context = context or {}
        super().__init__(self.message)


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


class InputValidationError(CLIException):
    """Input validation errors."""
    pass
