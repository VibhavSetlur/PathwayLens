"""
Visualization API routes for PathwayLens.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List, Any, Optional
from loguru import logger
import uuid
import os
from datetime import datetime

from ..schemas.visualize import (
    VisualizationRequest, VisualizationResponse, VisualizationStatus, VisualizationResult,
    VisualizationParameters, PlotType, ExportFormat
)
from ..utils.dependencies import get_current_user, get_database_session
from ..utils.exceptions import PathwayLensException
from ..utils.database import Job, JobResult
from ..tasks.visualize import create_visualizations_task, generate_report_task
from pathwaylens_core.visualization.report_generator import ReportGenerator
from pathwaylens_core.visualization.export_manager import ExportManager
from sqlalchemy import select


router = APIRouter(prefix="/visualize", tags=["visualization"])


@router.post("/create", response_model=VisualizationResponse)
async def create_visualizations_endpoint(
    request: VisualizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Create visualizations for analysis results.
    
    Args:
        request: Visualization request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        VisualizationResponse: Visualization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Visualization parameters are required")
        
        # Create job record using ORM
        job = Job(
            id=job_id,
            user_id=current_user["id"],
            job_type="create_visualizations",
            status="queued",
            parameters=request.parameters.model_dump(),
            input_files={"analysis_job_id": request.analysis_job_id},
            progress=0,
            created_at=datetime.utcnow()
        )
        
        # Store job record in database
        db_session.add(job)
        await db_session.commit()
        
        # Start background visualization task
        background_tasks.add_task(
            create_visualizations_task,
            job_id=job_id,
            analysis_job_id=request.analysis_job_id,
            plot_types=request.plot_types,
            parameters=request.parameters,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started visualization job {job_id} for user {current_user['id']}")
        
        return VisualizationResponse(
            job_id=job_id,
            status=VisualizationStatus.PENDING,
            message="Visualization job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in visualization creation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in visualization creation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/report", response_model=VisualizationResponse)
async def generate_report_endpoint(
    analysis_job_id: str,
    report_format: str = "html",
    include_interactive: bool = True,
    include_static: bool = True,
    theme: str = "light",
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Generate a comprehensive report for analysis results.
    
    Args:
        analysis_job_id: Analysis job ID to generate report for
        report_format: Report format ('html', 'pdf', 'zip')
        include_interactive: Whether to include interactive visualizations
        include_static: Whether to include static visualizations
        theme: Report theme ('light', 'dark', 'scientific')
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        VisualizationResponse: Visualization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job record using ORM
        job = Job(
            id=job_id,
            user_id=current_user["id"],
            job_type="generate_report",
            status="queued",
            parameters={
                "analysis_job_id": analysis_job_id,
                "report_format": report_format,
                "include_interactive": include_interactive,
                "include_static": include_static,
                "theme": theme
            },
            input_files={"analysis_job_id": analysis_job_id},
            progress=0,
            created_at=datetime.utcnow()
        )
        
        # Store job record in database
        db_session.add(job)
        await db_session.commit()
        
        # Start background report generation task
        background_tasks.add_task(
            generate_report_task,
            job_id=job_id,
            analysis_job_id=analysis_job_id,
            report_format=report_format,
            include_interactive=include_interactive,
            include_static=include_static,
            theme=theme,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started report generation job {job_id} for user {current_user['id']}")
        
        return VisualizationResponse(
            job_id=job_id,
            status=VisualizationStatus.PENDING,
            message="Report generation job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in report generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in report generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{job_id}", response_model=VisualizationStatus)
async def get_visualization_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Get visualization job status.
    
    Args:
        job_id: Visualization job ID
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        VisualizationStatus: Current visualization status
    """
    try:
        # Find job record using ORM
        result = await db_session.execute(
            select(Job).where(Job.id == job_id, Job.user_id == current_user["id"])
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Visualization job not found")
        
        return VisualizationStatus(
            job_id=job_id,
            status=job.status,
            progress=job.progress,
            message=getattr(job, 'message', ''),
            created_at=job.created_at,
            updated_at=job.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualization status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/result/{job_id}", response_model=VisualizationResult)
async def get_visualization_result(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Get visualization result.
    
    Args:
        job_id: Visualization job ID
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        VisualizationResult: Visualization result data
    """
    try:
        # Find job record using ORM
        result = await db_session.execute(
            select(Job).where(Job.id == job_id, Job.user_id == current_user["id"])
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Visualization job not found")
        
        if job.status != "completed":
            raise HTTPException(status_code=400, detail="Visualization job not completed")
        
        # Get job results
        results_query = await db_session.execute(
            select(JobResult).where(JobResult.job_id == job_id)
        )
        job_results = results_query.scalars().all()
        
        # Parse results
        visualizations = {}
        for result in job_results:
            if result.result_type == "visualization":
                visualizations[result.result_data.get("name", "unknown")] = result.result_data
        
        return VisualizationResult(
            job_id=job_id,
            visualization_type=job.job_type,
            parameters=job.parameters,
            visualizations=visualizations,
            output_files=job.output_files,
            metadata=job.parameters,
            created_at=job.created_at,
            completed_at=job.completed_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualization result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/download/{job_id}")
async def download_visualization(
    job_id: str,
    format: str = Query("html", description="Download format"),
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Download visualization files.
    
    Args:
        job_id: Visualization job ID
        format: Download format ('html', 'png', 'pdf', 'zip')
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        FileResponse: Downloaded file
    """
    try:
        # Find job record using ORM
        result = await db_session.execute(
            select(Job).where(Job.id == job_id, Job.user_id == current_user["id"])
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Visualization job not found")
        
        if job.status != "completed":
            raise HTTPException(status_code=400, detail="Visualization job not completed")
        
        # Get output files
        output_files = job.output_files or {}
        
        # Find the requested format file
        file_path = None
        if format in output_files:
            file_path = output_files[format]
        elif format == "zip" and "archive" in output_files:
            file_path = output_files["archive"]
        else:
            # Try to find any file with the requested format
            for file_type, path in output_files.items():
                if path.endswith(f".{format}"):
                    file_path = path
                    break
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found for format: {format}")
        
        # Return file
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading visualization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs", response_model=List[VisualizationStatus])
async def list_visualization_jobs(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    List visualization jobs for current user.
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        List[VisualizationStatus]: List of visualization job statuses
    """
    try:
        # Find job records using ORM
        result = await db_session.execute(
            select(Job)
            .where(Job.user_id == current_user["id"])
            .where(Job.job_type.in_(["create_visualizations", "generate_report"]))
            .order_by(Job.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        jobs = result.scalars().all()
        
        # Convert to VisualizationStatus objects
        visualization_statuses = []
        for job in jobs:
            visualization_statuses.append(VisualizationStatus(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                message=getattr(job, 'message', ''),
                created_at=job.created_at,
                updated_at=job.updated_at
            ))
        
        return visualization_statuses
        
    except Exception as e:
        logger.error(f"Error listing visualization jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/job/{job_id}")
async def delete_visualization_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Delete visualization job.
    
    Args:
        job_id: Visualization job ID
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        JSONResponse: Success message
    """
    try:
        # Find job record using ORM
        result = await db_session.execute(
            select(Job).where(Job.id == job_id, Job.user_id == current_user["id"])
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Visualization job not found")
        
        # Delete associated files
        if job.output_files:
            for file_path in job.output_files.values():
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_path}: {e}")
        
        # Delete job record
        await db_session.delete(job)
        await db_session.commit()
        
        logger.info(f"Deleted visualization job {job_id} for user {current_user['id']}")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Visualization job deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting visualization job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/parameters", response_model=Dict[str, Any])
async def get_visualization_parameters():
    """
    Get available visualization parameters.
    
    Returns:
        Dict[str, Any]: Available visualization parameters
    """
    try:
        # Return available visualization parameters
        parameters = {
            "plot_types": [t.value for t in PlotType],
            "export_formats": [f.value for f in ExportFormat],
            "themes": ["light", "dark", "scientific"],
            "default_parameters": {
                "figure_size": [800, 600],
                "dpi": 300,
                "interactive": True,
                "include_plotlyjs": True,
                "theme": "light"
            }
        }
        
        return parameters
        
    except Exception as e:
        logger.error(f"Error getting visualization parameters: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export/{analysis_job_id}")
async def export_analysis_visualizations(
    analysis_job_id: str,
    format: str = Query("html", description="Export format"),
    theme: str = Query("light", description="Report theme"),
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Export visualizations for an analysis job.
    
    Args:
        analysis_job_id: Analysis job ID
        format: Export format ('html', 'png', 'pdf', 'zip')
        theme: Report theme ('light', 'dark', 'scientific')
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        FileResponse: Exported file
    """
    try:
        # Find analysis job
        result = await db_session.execute(
            select(Job).where(Job.id == analysis_job_id, Job.user_id == current_user["id"])
        )
        analysis_job = result.scalar_one_or_none()
        
        if not analysis_job:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        if analysis_job.status != "completed":
            raise HTTPException(status_code=400, detail="Analysis job not completed")
        
        # Generate report
        report_generator = ReportGenerator()
        
        # Mock analysis result for now - in real implementation, this would come from the database
        from pathwaylens_core.analysis.schemas import AnalysisResult
        analysis_result = AnalysisResult(
            job_id=analysis_job_id,
            analysis_type=analysis_job.job_type,
            species="human",
            input_gene_count=1000,
            total_pathways=50,
            significant_pathways=10,
            database_results={},
            consensus_results=[],
            metadata={}
        )
        
        job_metadata = {
            "job_id": analysis_job_id,
            "created_at": analysis_job.created_at.isoformat(),
            "completed_at": analysis_job.completed_at.isoformat() if analysis_job.completed_at else None
        }
        
        # Generate report
        report_files = report_generator.generate_analysis_report(
            analysis_result=analysis_result,
            job_metadata=job_metadata,
            include_interactive=True,
            include_static=True,
            theme=theme
        )
        
        # Return the requested format
        if format == "html" and "html_report" in report_files:
            return FileResponse(
                path=report_files["html_report"],
                filename=f"analysis_report_{analysis_job_id}.html",
                media_type='text/html'
            )
        elif format == "zip" and "archive" in report_files:
            return FileResponse(
                path=report_files["archive"],
                filename=f"analysis_report_{analysis_job_id}.zip",
                media_type='application/zip'
            )
        else:
            raise HTTPException(status_code=400, detail=f"Format {format} not available")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting analysis visualizations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")