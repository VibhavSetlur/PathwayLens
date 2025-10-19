"""
Comparison tasks for PathwayLens API.

This module provides Celery tasks for dataset comparison.
"""

from celery import current_task
from pathwaylens_api.celery_app import celery_app
from pathwaylens_core.comparison.engine import ComparisonEngine


@celery_app.task(bind=True, name="pathwaylens_api.tasks.compare.compare_datasets")
def compare_datasets_task(self, job_id: str, comparison_type: str, datasets: list, parameters: dict):
    """Compare multiple datasets task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting comparison"})
        
        # Initialize comparison engine
        engine = ComparisonEngine()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "message": "Processing datasets"})
        
        # Perform comparison
        result = engine.compare(
            comparison_type=comparison_type,
            datasets=datasets,
            parameters=parameters
        )
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 80, "message": "Finalizing results"})
        
        # Convert results to response format
        comparison_results = {
            "comparison_type": comparison_type,
            "dataset_count": len(datasets),
            "results": result.to_dict()
        }
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "job_id": job_id,
            "status": "completed",
            "comparison_type": comparison_type,
            "dataset_count": len(datasets),
            "results": comparison_results
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "Comparison failed"}
        )
        raise


def compare_analysis_results(job_id: str, comparison_type: str, datasets: list, parameters: dict):
    """Compare analysis results."""
    # For now, just call the datasets comparison function
    return compare_datasets_task(job_id, comparison_type, datasets, parameters)


def compare_multi_omics(job_id: str, comparison_type: str, datasets: list, parameters: dict):
    """Compare multi-omics datasets."""
    # For now, just call the datasets comparison function
    return compare_datasets_task(job_id, comparison_type, datasets, parameters)


def compare_pathways(job_id: str, comparison_type: str, datasets: list, parameters: dict):
    """Compare pathway datasets."""
    # For now, just call the datasets comparison function
    return compare_datasets_task(job_id, comparison_type, datasets, parameters)
