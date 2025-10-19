"""
Celery application for PathwayLens API.

This module configures Celery for asynchronous job processing.
"""

import os
from celery import Celery
from pathwaylens_core.utils.config import get_config

# Get configuration
config = get_config()

# Create Celery application
celery_app = Celery(
    "pathwaylens",
    broker=config.get("redis_url", "redis://localhost:6379/0"),
    backend=config.get("redis_url", "redis://localhost:6379/0"),
    include=[
        "pathwaylens_api.tasks.normalize",
        "pathwaylens_api.tasks.analyze",
        "pathwaylens_api.tasks.compare",
        "pathwaylens_api.tasks.visualize"
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1 hour
    task_routes={
        "pathwaylens_api.tasks.normalize.*": {"queue": "normalize"},
        "pathwaylens_api.tasks.analyze.*": {"queue": "analyze"},
        "pathwaylens_api.tasks.compare.*": {"queue": "compare"},
        "pathwaylens_api.tasks.visualize.*": {"queue": "visualize"},
    },
    task_annotations={
        "*": {"rate_limit": "10/s"},
        "pathwaylens_api.tasks.analyze.*": {"rate_limit": "5/s"},
        "pathwaylens_api.tasks.compare.*": {"rate_limit": "2/s"},
    }
)

# Periodic tasks
celery_app.conf.beat_schedule = {
    "cleanup-old-jobs": {
        "task": "pathwaylens_api.tasks.cleanup.cleanup_old_jobs",
        "schedule": 3600.0,  # Run every hour
    },
    "cleanup-storage": {
        "task": "pathwaylens_api.tasks.cleanup.cleanup_storage",
        "schedule": 86400.0,  # Run daily
    },
    "update-databases": {
        "task": "pathwaylens_api.tasks.maintenance.update_databases",
        "schedule": 604800.0,  # Run weekly
    },
}

if __name__ == "__main__":
    celery_app.start()
