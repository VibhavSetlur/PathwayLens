"""
Visualization API routes for PathwayLens.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from loguru import logger
import uuid
from datetime import datetime

from ..schemas.visualize import (
    VisualizationRequest, VisualizationResponse, VisualizationStatus, VisualizationResult,
    VisualizationParameters, PlotType
)
from ..utils.dependencies import get_current_user, get_database
from ..utils.exceptions import PathwayLensException
from ..tasks.visualize import generate_visualization, create_dashboard, export_visualization


router = APIRouter(prefix="/visualize", tags=["visualization"])


@router.post("/generate", response_model=VisualizationResponse)
async def generate_visualization_endpoint(
    request: VisualizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Generate visualization.
    
    Args:
        request: Visualization request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        VisualizationResponse: Visualization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Visualization parameters are required")
        
        # Create visualization job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "visualization_type": "generate",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.visualization_jobs.insert_one(job_record)
        
        # Start background visualization task
        background_tasks.add_task(
            generate_visualization,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started visualization generation job {job_id} for user {current_user['id']}")
        
        return VisualizationResponse(
            job_id=job_id,
            status=VisualizationStatus.PENDING,
            message="Visualization job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in visualization generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in visualization generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/dashboard", response_model=VisualizationResponse)
async def create_dashboard_endpoint(
    request: VisualizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Create interactive dashboard.
    
    Args:
        request: Visualization request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        VisualizationResponse: Visualization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Visualization parameters are required")
        
        # Create visualization job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "visualization_type": "dashboard",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.visualization_jobs.insert_one(job_record)
        
        # Start background visualization task
        background_tasks.add_task(
            create_dashboard,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started dashboard creation job {job_id} for user {current_user['id']}")
        
        return VisualizationResponse(
            job_id=job_id,
            status=VisualizationStatus.PENDING,
            message="Dashboard creation job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in dashboard creation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in dashboard creation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/export", response_model=VisualizationResponse)
async def export_visualization_endpoint(
    request: VisualizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Export visualization.
    
    Args:
        request: Visualization request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        VisualizationResponse: Visualization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Visualization parameters are required")
        
        # Create visualization job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "visualization_type": "export",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.visualization_jobs.insert_one(job_record)
        
        # Start background visualization task
        background_tasks.add_task(
            export_visualization,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started visualization export job {job_id} for user {current_user['id']}")
        
        return VisualizationResponse(
            job_id=job_id,
            status=VisualizationStatus.PENDING,
            message="Visualization export job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in visualization export: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in visualization export: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{job_id}", response_model=VisualizationStatus)
async def get_visualization_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get visualization job status.
    
    Args:
        job_id: Visualization job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        VisualizationStatus: Current visualization status
    """
    try:
        # Find job record
        job_record = await db.visualization_jobs.find_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Visualization job not found")
        
        return VisualizationStatus(
            job_id=job_id,
            status=job_record["status"],
            progress=job_record.get("progress", 0),
            message=job_record.get("message", ""),
            created_at=job_record["created_at"],
            updated_at=job_record["updated_at"]
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
    db = Depends(get_database)
):
    """
    Get visualization result.
    
    Args:
        job_id: Visualization job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        VisualizationResult: Visualization result data
    """
    try:
        # Find job record
        job_record = await db.visualization_jobs.find_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Visualization job not found")
        
        if job_record["status"] != "completed":
            raise HTTPException(status_code=400, detail="Visualization job not completed")
        
        return VisualizationResult(
            job_id=job_id,
            visualization_type=job_record["visualization_type"],
            parameters=job_record["parameters"],
            results=job_record.get("results", {}),
            metadata=job_record.get("metadata", {}),
            created_at=job_record["created_at"],
            completed_at=job_record.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualization result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs", response_model=List[VisualizationStatus])
async def list_visualization_jobs(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    List visualization jobs for current user.
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        List[VisualizationStatus]: List of visualization job statuses
    """
    try:
        # Find job records
        job_records = await db.visualization_jobs.find({
            "user_id": current_user["id"]
        }).sort("created_at", -1).skip(offset).limit(limit).to_list(length=limit)
        
        # Convert to VisualizationStatus objects
        visualization_statuses = []
        for job_record in job_records:
            visualization_statuses.append(VisualizationStatus(
                job_id=job_record["job_id"],
                status=job_record["status"],
                progress=job_record.get("progress", 0),
                message=job_record.get("message", ""),
                created_at=job_record["created_at"],
                updated_at=job_record["updated_at"]
            ))
        
        return visualization_statuses
        
    except Exception as e:
        logger.error(f"Error listing visualization jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/job/{job_id}")
async def delete_visualization_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Delete visualization job.
    
    Args:
        job_id: Visualization job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        JSONResponse: Success message
    """
    try:
        # Find and delete job record
        result = await db.visualization_jobs.delete_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Visualization job not found")
        
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
            "default_parameters": {
                "width": 800,
                "height": 600,
                "title": "PathwayLens Visualization",
                "color_scheme": "viridis",
                "interactive": True,
                "export_format": "html"
            }
        }
        
        return parameters
        
    except Exception as e:
        logger.error(f"Error getting visualization parameters: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")