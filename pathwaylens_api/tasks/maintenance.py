"""
Maintenance tasks for PathwayLens API.

This module provides Celery tasks for system maintenance.
"""

from celery import current_task
from pathwaylens_api.celery_app import celery_app
from pathwaylens_core.data.database_manager import DatabaseManager


@celery_app.task(bind=True, name="pathwaylens_api.tasks.maintenance.update_databases")
async def update_databases_task(self):
    """Update pathway databases task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting database update"})
        
        # Get database manager
        db_manager = DatabaseManager()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "message": "Updating KEGG database"})
        
        # Update KEGG database
        await db_manager.update_database("kegg")
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 50, "message": "Updating Reactome database"})
        
        # Update Reactome database
        await db_manager.update_database("reactome")
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 70, "message": "Updating GO database"})
        
        # Update GO database
        await db_manager.update_database("go")
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 90, "message": "Finalizing updates"})
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "status": "completed",
            "message": "Database update completed successfully"
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "Database update failed"}
        )
        raise
