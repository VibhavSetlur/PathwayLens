"""
Configuration utilities for PathwayLens core.
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables and defaults.
    
    Returns:
        Configuration dictionary
    """
    return {
        "database": {
            "url": os.getenv("DATABASE_URL", "postgresql://localhost:5432/pathwaylens"),
            "echo": os.getenv("DATABASE_ECHO", "false").lower() == "true"
        },
        "redis": {
            "url": os.getenv("REDIS_URL", "redis://localhost:6379/0")
        },
        "api": {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("API_PORT", "8000")),
            "workers": int(os.getenv("API_WORKERS", "1"))
        },
        "celery": {
            "broker_url": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
            "result_backend": os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
        }
    }
