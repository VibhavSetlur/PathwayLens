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


def log_diagnostic_error(exception: Exception, context: dict = None):
    """
    Log detailed error information to error_diagnostic.log.
    
    Args:
        exception: The exception that occurred.
        context: Additional context dictionary.
    """
    import sys
    import traceback
    from datetime import datetime
    from pathlib import Path
    
    log_file = Path("error_diagnostic.log")
    
    try:
        with open(log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR REPORT - {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Command: {' '.join(sys.argv)}\n")
            f.write(f"Exception Type: {type(exception).__name__}\n")
            f.write(f"Message: {str(exception)}\n")
            
            if context:
                f.write("\nContext:\n")
                for k, v in context.items():
                    f.write(f"  {k}: {v}\n")
            
            f.write("\nTraceback:\n")
            traceback.print_tb(exception.__traceback__, file=f)
            f.write(f"{'='*80}\n")
            
        return str(log_file.absolute())
    except Exception as e:
        # Fallback if logging fails
        print(f"Failed to write diagnostic log: {e}")
        return None
