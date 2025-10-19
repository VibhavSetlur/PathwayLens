"""
Comparison API routes for PathwayLens.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from loguru import logger
import uuid
from datetime import datetime

from ..schemas.compare import (
    ComparisonRequest, ComparisonResponse, ComparisonStatus, ComparisonResult,
    ComparisonParameters, ComparisonType
)
from ..utils.dependencies import get_current_user, get_database
from ..utils.exceptions import PathwayLensException
from ..tasks.compare import compare_analysis_results, compare_multi_omics, compare_pathways


router = APIRouter(prefix="/compare", tags=["comparison"])


@router.post("/analysis-results", response_model=ComparisonResponse)
async def compare_analysis_results_endpoint(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Compare analysis results.
    
    Args:
        request: Comparison request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        ComparisonResponse: Comparison response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Comparison parameters are required")
        
        # Create comparison job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "comparison_type": "analysis_results",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.comparison_jobs.insert_one(job_record)
        
        # Start background comparison task
        background_tasks.add_task(
            compare_analysis_results,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started analysis results comparison job {job_id} for user {current_user['id']}")
        
        return ComparisonResponse(
            job_id=job_id,
            status=ComparisonStatus.PENDING,
            message="Comparison job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in analysis results comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in analysis results comparison: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/multi-omics", response_model=ComparisonResponse)
async def compare_multi_omics_endpoint(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Compare multi-omics data.
    
    Args:
        request: Comparison request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        ComparisonResponse: Comparison response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Comparison parameters are required")
        
        # Create comparison job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "comparison_type": "multi_omics",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.comparison_jobs.insert_one(job_record)
        
        # Start background comparison task
        background_tasks.add_task(
            compare_multi_omics,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started multi-omics comparison job {job_id} for user {current_user['id']}")
        
        return ComparisonResponse(
            job_id=job_id,
            status=ComparisonStatus.PENDING,
            message="Comparison job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in multi-omics comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in multi-omics comparison: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/pathways", response_model=ComparisonResponse)
async def compare_pathways_endpoint(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Compare pathways.
    
    Args:
        request: Comparison request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        ComparisonResponse: Comparison response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Comparison parameters are required")
        
        # Create comparison job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "comparison_type": "pathways",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.comparison_jobs.insert_one(job_record)
        
        # Start background comparison task
        background_tasks.add_task(
            compare_pathways,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started pathways comparison job {job_id} for user {current_user['id']}")
        
        return ComparisonResponse(
            job_id=job_id,
            status=ComparisonStatus.PENDING,
            message="Comparison job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in pathways comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in pathways comparison: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{job_id}", response_model=ComparisonStatus)
async def get_comparison_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get comparison job status.
    
    Args:
        job_id: Comparison job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        ComparisonStatus: Current comparison status
    """
    try:
        # Find job record
        job_record = await db.comparison_jobs.find_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Comparison job not found")
        
        return ComparisonStatus(
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
        logger.error(f"Error getting comparison status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/result/{job_id}", response_model=ComparisonResult)
async def get_comparison_result(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get comparison result.
    
    Args:
        job_id: Comparison job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        ComparisonResult: Comparison result data
    """
    try:
        # Find job record
        job_record = await db.comparison_jobs.find_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Comparison job not found")
        
        if job_record["status"] != "completed":
            raise HTTPException(status_code=400, detail="Comparison job not completed")
        
        return ComparisonResult(
            job_id=job_id,
            comparison_type=job_record["comparison_type"],
            parameters=job_record["parameters"],
            results=job_record.get("results", {}),
            metadata=job_record.get("metadata", {}),
            created_at=job_record["created_at"],
            completed_at=job_record.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs", response_model=List[ComparisonStatus])
async def list_comparison_jobs(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    List comparison jobs for current user.
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        List[ComparisonStatus]: List of comparison job statuses
    """
    try:
        # Find job records
        job_records = await db.comparison_jobs.find({
            "user_id": current_user["id"]
        }).sort("created_at", -1).skip(offset).limit(limit).to_list(length=limit)
        
        # Convert to ComparisonStatus objects
        comparison_statuses = []
        for job_record in job_records:
            comparison_statuses.append(ComparisonStatus(
                job_id=job_record["job_id"],
                status=job_record["status"],
                progress=job_record.get("progress", 0),
                message=job_record.get("message", ""),
                created_at=job_record["created_at"],
                updated_at=job_record["updated_at"]
            ))
        
        return comparison_statuses
        
    except Exception as e:
        logger.error(f"Error listing comparison jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/job/{job_id}")
async def delete_comparison_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Delete comparison job.
    
    Args:
        job_id: Comparison job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        JSONResponse: Success message
    """
    try:
        # Find and delete job record
        result = await db.comparison_jobs.delete_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Comparison job not found")
        
        logger.info(f"Deleted comparison job {job_id} for user {current_user['id']}")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Comparison job deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting comparison job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/parameters", response_model=Dict[str, Any])
async def get_comparison_parameters():
    """
    Get available comparison parameters.
    
    Returns:
        Dict[str, Any]: Available comparison parameters
    """
    try:
        # Return available comparison parameters
        parameters = {
            "comparison_types": [t.value for t in ComparisonType],
            "default_parameters": {
                "significance_threshold": 0.05,
                "correlation_threshold": 0.5,
                "overlap_threshold": 0.3,
                "clustering_method": "hierarchical",
                "distance_metric": "euclidean"
            }
        }
        
        return parameters
        
    except Exception as e:
        logger.error(f"Error getting comparison parameters: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")