"""
PathwayLens CLI - Standalone command-line interface.

This package provides the standalone CLI for PathwayLens 2.0, enabling
direct invocation as 'pathwaylens' command without the 'python3 -m' prefix.
"""

__version__ = "2.0.0"
__author__ = "PathwayLens Team"
__email__ = "pathwaylens@example.com"

from .main import app

__all__ = ["app"]
