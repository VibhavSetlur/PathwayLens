"""
PathwayLens API - FastAPI web application for pathway analysis.

This module provides the RESTful API for PathwayLens 2.0, including:
- File upload and processing endpoints
- Pathway analysis endpoints
- Job management and status tracking
- Authentication and authorization
- Real-time progress updates via WebSocket
"""

__version__ = "2.0.0"
__author__ = "PathwayLens Team"
__email__ = "pathwaylens@example.com"

from .main import app

__all__ = ["app"]
