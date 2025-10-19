"""
Visualization tasks for PathwayLens API.

This module provides Celery tasks for visualization generation.
"""

from celery import current_task
from pathwaylens_api.celery_app import celery_app
from pathwaylens_core.visualization.engine import VisualizationEngine


@celery_app.task(bind=True, name="pathwaylens_api.tasks.visualize.create_visualizations")
def create_visualizations_task(self, job_id: str, analysis_id: str, plot_types: list, parameters: dict):
    """Create visualizations from analysis results task."""
    try:
        # Update task progress
        current_task.update_state(state="PROGRESS", meta={"progress": 10, "message": "Starting visualization generation"})
        
        # Initialize visualization engine
        engine = VisualizationEngine()
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 30, "message": "Loading analysis results"})
        
        # Get analysis results (this would need to be implemented to fetch from storage)
        # For now, we'll assume the results are passed in parameters
        analysis_results = parameters.get("analysis_results", {})
        
        if not analysis_results:
            raise ValueError("Analysis results not found")
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 50, "message": "Generating plots"})
        
        # Generate plots
        plots = []
        total_plots = len(plot_types)
        
        for i, plot_type in enumerate(plot_types):
            # Update progress
            progress = int(50 + (i / total_plots) * 40)
            current_task.update_state(
                state="PROGRESS", 
                meta={"progress": progress, "message": f"Generating {plot_type}"}
            )
            
            try:
                # Create plot
                plot_data = engine.create_plot(
                    plot_type=plot_type,
                    data=analysis_results,
                    parameters=parameters
                )
                
                plots.append({
                    "plot_type": plot_type,
                    "data": plot_data,
                    "format": "html"
                })
                
            except Exception as e:
                plots.append({
                    "plot_type": plot_type,
                    "error": str(e),
                    "format": "html"
                })
        
        # Update progress
        current_task.update_state(state="PROGRESS", meta={"progress": 100, "message": "Completed"})
        
        return {
            "job_id": job_id,
            "status": "completed",
            "visualization_id": analysis_id,
            "plots": plots
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e), "message": "Visualization generation failed"}
        )
        raise


def generate_visualization(job_id: str, analysis_id: str, plot_types: list, parameters: dict):
    """Generate visualization."""
    # For now, just call the visualizations task function
    return create_visualizations_task(job_id, analysis_id, plot_types, parameters)


def create_dashboard(job_id: str, analysis_id: str, plot_types: list, parameters: dict):
    """Create dashboard."""
    # For now, just call the visualizations task function
    return create_visualizations_task(job_id, analysis_id, plot_types, parameters)


def export_visualization(job_id: str, analysis_id: str, plot_types: list, parameters: dict):
    """Export visualization."""
    # For now, just call the visualizations task function
    return create_visualizations_task(job_id, analysis_id, plot_types, parameters)
