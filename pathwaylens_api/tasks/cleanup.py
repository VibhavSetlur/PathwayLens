"""
Cleanup tasks for PathwayLens API.

This module provides Celery tasks for system cleanup and maintenance.
"""

from datetime import datetime, timedelta
from celery import current_task
from pathwaylens_api.celery_app import celery_app
from pathwaylens_api.utils.database import get_database_manager
from pathwaylens_api.utils.storage import get_storage_manager


@celery_app.task(bind=True, name="pathwaylens_api.tasks.cleanup.cleanup_old_jobs")
async def cleanup_old_jobs_task(self, days: int = 30):
    """Clean up old completed jobs task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting job cleanup"})
        
        # Get database manager
        db_manager = get_database_manager()
        
        # Calculate cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "message": "Querying old jobs"})
        
        # Query old completed jobs
        query = """
            SELECT id FROM jobs 
            WHERE status = 'completed' 
            AND completed_at < :cutoff_date
        """
        
        old_jobs = await db_manager.execute_query(query, {"cutoff_date": cutoff_date})
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 50, "message": f"Found {len(old_jobs)} old jobs"})
        
        # Delete old jobs (cascade will handle related records)
        deleted_count = 0
        for job in old_jobs:
            try:
                await db_manager.execute_query(
                    "DELETE FROM jobs WHERE id = :job_id",
                    {"job_id": job["id"]}
                )
                deleted_count += 1
            except Exception as e:
                # Log error but continue
                print(f"Failed to delete job {job['id']}: {e}")
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "status": "completed",
            "deleted_jobs": deleted_count,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "Job cleanup failed"}
        )
        raise


@celery_app.task(bind=True, name="pathwaylens_api.tasks.cleanup.cleanup_storage")
async def cleanup_storage_task(self, days: int = 30):
    """Clean up old files in storage task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting storage cleanup"})
        
        # Get storage manager
        storage_manager = get_storage_manager()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 50, "message": "Cleaning up old files"})
        
        # Clean up old files
        await storage_manager.cleanup_old_files(days)
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "status": "completed",
            "cleanup_days": days
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "Storage cleanup failed"}
        )
        raise
