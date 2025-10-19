"""
PathwayLens API routes.

This module provides all API route handlers for the PathwayLens 2.0 API.
"""

from . import auth, normalize, analyze, compare, visualize, jobs, config, info

__all__ = [
    "auth",
    "normalize", 
    "analyze",
    "compare",
    "visualize",
    "jobs",
    "config",
    "info"
]
