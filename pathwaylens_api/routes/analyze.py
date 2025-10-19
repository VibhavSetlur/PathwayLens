"""
Analysis API routes for PathwayLens.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from loguru import logger
import uuid
from datetime import datetime

from ..schemas.analyze import (
    AnalysisRequest, AnalysisResponse, AnalysisStatus, AnalysisResult,
    AnalysisParameters, AnalysisType, DatabaseType, CorrectionMethod
)
from ..utils.dependencies import get_current_user, get_database_session
from ..utils.exceptions import PathwayLensException
from ..utils.database import Job, JobResult, AnalysisResult
from ..tasks.analyze import analyze_pathway_enrichment, analyze_multi_omics, analyze_statistical
from sqlalchemy import select


router = APIRouter(prefix="/analyze", tags=["analysis"])


@router.post("/pathway-enrichment", response_model=AnalysisResponse)
async def analyze_pathway_enrichment_endpoint(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Analyze pathway enrichment.
    
    Args:
        request: Analysis request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        AnalysisResponse: Analysis response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Analysis parameters are required")
        
        # Create job record using ORM
        job = Job(
            id=job_id,
            user_id=current_user["id"],
            job_type="analyze_pathway_enrichment",
            status="queued",
            parameters=request.parameters.model_dump(),
            input_files={"input_data": request.input_data},
            progress=0,
            created_at=datetime.utcnow()
        )
        
        # Store job record in database
        db_session.add(job)
        await db_session.commit()
        
        # Start background analysis task
        background_tasks.add_task(
            analyze_pathway_enrichment,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started pathway enrichment analysis job {job_id} for user {current_user['id']}")
        
        return AnalysisResponse(
            job_id=job_id,
            status=AnalysisStatus.PENDING,
            message="Analysis job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in pathway enrichment analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in pathway enrichment analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/multi-omics", response_model=AnalysisResponse)
async def analyze_multi_omics_endpoint(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Analyze multi-omics data.
    
    Args:
        request: Analysis request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        AnalysisResponse: Analysis response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Analysis parameters are required")
        
        # Create analysis job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "analysis_type": "multi_omics",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.analysis_jobs.insert_one(job_record)
        
        # Start background analysis task
        background_tasks.add_task(
            analyze_multi_omics,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started multi-omics analysis job {job_id} for user {current_user['id']}")
        
        return AnalysisResponse(
            job_id=job_id,
            status=AnalysisStatus.PENDING,
            message="Analysis job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in multi-omics analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in multi-omics analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/statistical", response_model=AnalysisResponse)
async def analyze_statistical_endpoint(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Perform statistical analysis.
    
    Args:
        request: Analysis request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        AnalysisResponse: Analysis response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Analysis parameters are required")
        
        # Create analysis job record
        job_record = {
            "job_id": job_id,
            "user_id": current_user["id"],
            "analysis_type": "statistical",
            "status": "pending",
            "parameters": request.parameters.model_dump(),
            "input_data": request.input_data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Store job record in database
        await db.analysis_jobs.insert_one(job_record)
        
        # Start background analysis task
        background_tasks.add_task(
            analyze_statistical,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started statistical analysis job {job_id} for user {current_user['id']}")
        
        return AnalysisResponse(
            job_id=job_id,
            status=AnalysisStatus.PENDING,
            message="Analysis job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in statistical analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in statistical analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{job_id}", response_model=AnalysisStatus)
async def get_analysis_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get analysis job status.
    
    Args:
        job_id: Analysis job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        AnalysisStatus: Current analysis status
    """
    try:
        # Find job record
        job_record = await db.analysis_jobs.find_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        return AnalysisStatus(
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
        logger.error(f"Error getting analysis status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/result/{job_id}", response_model=AnalysisResult)
async def get_analysis_result(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get analysis result.
    
    Args:
        job_id: Analysis job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        AnalysisResult: Analysis result data
    """
    try:
        # Find job record
        job_record = await db.analysis_jobs.find_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        if job_record["status"] != "completed":
            raise HTTPException(status_code=400, detail="Analysis job not completed")
        
        return AnalysisResult(
            job_id=job_id,
            analysis_type=job_record["analysis_type"],
            parameters=job_record["parameters"],
            results=job_record.get("results", {}),
            metadata=job_record.get("metadata", {}),
            created_at=job_record["created_at"],
            completed_at=job_record.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs", response_model=List[AnalysisStatus])
async def list_analysis_jobs(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    List analysis jobs for current user.
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        List[AnalysisStatus]: List of analysis job statuses
    """
    try:
        # Find job records
        job_records = await db.analysis_jobs.find({
            "user_id": current_user["id"]
        }).sort("created_at", -1).skip(offset).limit(limit).to_list(length=limit)
        
        # Convert to AnalysisStatus objects
        analysis_statuses = []
        for job_record in job_records:
            analysis_statuses.append(AnalysisStatus(
                job_id=job_record["job_id"],
                status=job_record["status"],
                progress=job_record.get("progress", 0),
                message=job_record.get("message", ""),
                created_at=job_record["created_at"],
                updated_at=job_record["updated_at"]
            ))
        
        return analysis_statuses
        
    except Exception as e:
        logger.error(f"Error listing analysis jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/job/{job_id}")
async def delete_analysis_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Delete analysis job.
    
    Args:
        job_id: Analysis job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        JSONResponse: Success message
    """
    try:
        # Find and delete job record
        result = await db.analysis_jobs.delete_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        logger.info(f"Deleted analysis job {job_id} for user {current_user['id']}")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Analysis job deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/parameters", response_model=Dict[str, Any])
async def get_analysis_parameters():
    """
    Get available analysis parameters.
    
    Returns:
        Dict[str, Any]: Available analysis parameters
    """
    try:
        # Return available analysis parameters
        parameters = {
            "analysis_types": [t.value for t in AnalysisType],
            "database_types": [t.value for t in DatabaseType],
            "correction_methods": [m.value for m in CorrectionMethod],
            "default_parameters": {
                "significance_threshold": 0.05,
                "min_pathway_size": 5,
                "max_pathway_size": 500,
                "permutations": 1000,
                "species": "human"
            }
        }
        
        return parameters
        
    except Exception as e:
        logger.error(f"Error getting analysis parameters: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")