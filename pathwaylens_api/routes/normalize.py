"""
Normalization API routes for PathwayLens.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from loguru import logger
import uuid
from datetime import datetime

from ..schemas.normalize import (
    NormalizationRequest, NormalizationResponse, NormalizationStatus, NormalizationResult,
    NormalizationParameters, NormalizationType
)
from ..utils.dependencies import get_current_user, get_database_session
from ..utils.exceptions import PathwayLensException
from ..utils.database import Job, JobResult
from ..tasks.normalize import normalize_gene_ids, normalize_pathway_ids, normalize_omics_data
from sqlalchemy import select


router = APIRouter(prefix="/normalize", tags=["normalization"])


@router.post("/gene-ids", response_model=NormalizationResponse)
async def normalize_gene_ids_endpoint(
    request: NormalizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Normalize gene IDs.
    
    Args:
        request: Normalization request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        NormalizationResponse: Normalization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Normalization parameters are required")
        
        # Create job record using ORM
        job = Job(
            id=job_id,
            user_id=current_user["id"],
            job_type="normalize_gene_ids",
            status="queued",
            parameters=request.parameters.model_dump(),
            input_files={"input_data": request.input_data},
            progress=0,
            created_at=datetime.utcnow()
        )
        
        # Store job record in database
        db_session.add(job)
        await db_session.commit()
        
        # Start background normalization task
        background_tasks.add_task(
            normalize_gene_ids,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started gene ID normalization job {job_id} for user {current_user['id']}")
        
        return NormalizationResponse(
            job_id=job_id,
            status=NormalizationStatus.PENDING,
            message="Normalization job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in gene ID normalization: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in gene ID normalization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/pathway-ids", response_model=NormalizationResponse)
async def normalize_pathway_ids_endpoint(
    request: NormalizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Normalize pathway IDs.
    
    Args:
        request: Normalization request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        NormalizationResponse: Normalization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Normalization parameters are required")
        
        # Create job record using ORM
        job = Job(
            id=job_id,
            user_id=current_user["id"],
            job_type="normalize_pathway_ids",
            status="queued",
            parameters=request.parameters.model_dump(),
            input_files={"input_data": request.input_data},
            progress=0,
            created_at=datetime.utcnow()
        )
        
        # Store job record in database
        db_session.add(job)
        await db_session.commit()
        
        # Start background normalization task
        background_tasks.add_task(
            normalize_pathway_ids,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started pathway ID normalization job {job_id} for user {current_user['id']}")
        
        return NormalizationResponse(
            job_id=job_id,
            status=NormalizationStatus.PENDING,
            message="Normalization job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in pathway ID normalization: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in pathway ID normalization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/omics-data", response_model=NormalizationResponse)
async def normalize_omics_data_endpoint(
    request: NormalizationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Normalize omics data.
    
    Args:
        request: Normalization request parameters
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        NormalizationResponse: Normalization response with job ID and status
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Validate request parameters
        if not request.parameters:
            raise HTTPException(status_code=400, detail="Normalization parameters are required")
        
        # Create job record using ORM
        job = Job(
            id=job_id,
            user_id=current_user["id"],
            job_type="normalize_omics_data",
            status="queued",
            parameters=request.parameters.model_dump(),
            input_files={"input_data": request.input_data},
            progress=0,
            created_at=datetime.utcnow()
        )
        
        # Store job record in database
        db_session.add(job)
        await db_session.commit()
        
        # Start background normalization task
        background_tasks.add_task(
            normalize_omics_data,
            job_id=job_id,
            parameters=request.parameters,
            input_data=request.input_data,
            user_id=current_user["id"]
        )
        
        logger.info(f"Started omics data normalization job {job_id} for user {current_user['id']}")
        
        return NormalizationResponse(
            job_id=job_id,
            status=NormalizationStatus.PENDING,
            message="Normalization job started successfully"
        )
        
    except PathwayLensException as e:
        logger.error(f"PathwayLens error in omics data normalization: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in omics data normalization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status/{job_id}", response_model=NormalizationStatus)
async def get_normalization_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db_session = Depends(get_database_session)
):
    """
    Get normalization job status.
    
    Args:
        job_id: Normalization job ID
        current_user: Current authenticated user
        db_session: Database session
        
    Returns:
        NormalizationStatus: Current normalization status
    """
    try:
        # Find job record using ORM
        result = await db_session.execute(
            select(Job).where(Job.id == job_id, Job.user_id == current_user["id"])
        )
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail="Normalization job not found")
        
        return NormalizationStatus(
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
        logger.error(f"Error getting normalization status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/result/{job_id}", response_model=NormalizationResult)
async def get_normalization_result(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get normalization result.
    
    Args:
        job_id: Normalization job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        NormalizationResult: Normalization result data
    """
    try:
        # Find job record
        job_record = await db.normalization_jobs.find_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if not job_record:
            raise HTTPException(status_code=404, detail="Normalization job not found")
        
        if job_record["status"] != "completed":
            raise HTTPException(status_code=400, detail="Normalization job not completed")
        
        return NormalizationResult(
            job_id=job_id,
            normalization_type=job_record["normalization_type"],
            parameters=job_record["parameters"],
            results=job_record.get("results", {}),
            metadata=job_record.get("metadata", {}),
            created_at=job_record["created_at"],
            completed_at=job_record.get("completed_at")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting normalization result: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/jobs", response_model=List[NormalizationStatus])
async def list_normalization_jobs(
    limit: int = 10,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    List normalization jobs for current user.
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        List[NormalizationStatus]: List of normalization job statuses
    """
    try:
        # Find job records
        job_records = await db.normalization_jobs.find({
            "user_id": current_user["id"]
        }).sort("created_at", -1).skip(offset).limit(limit).to_list(length=limit)
        
        # Convert to NormalizationStatus objects
        normalization_statuses = []
        for job_record in job_records:
            normalization_statuses.append(NormalizationStatus(
                job_id=job_record["job_id"],
                status=job_record["status"],
                progress=job_record.get("progress", 0),
                message=job_record.get("message", ""),
                created_at=job_record["created_at"],
                updated_at=job_record["updated_at"]
            ))
        
        return normalization_statuses
        
    except Exception as e:
        logger.error(f"Error listing normalization jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/job/{job_id}")
async def delete_normalization_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Delete normalization job.
    
    Args:
        job_id: Normalization job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        JSONResponse: Success message
    """
    try:
        # Find and delete job record
        result = await db.normalization_jobs.delete_one({
            "job_id": job_id,
            "user_id": current_user["id"]
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Normalization job not found")
        
        logger.info(f"Deleted normalization job {job_id} for user {current_user['id']}")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Normalization job deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting normalization job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/parameters", response_model=Dict[str, Any])
async def get_normalization_parameters():
    """
    Get available normalization parameters.
    
    Returns:
        Dict[str, Any]: Available normalization parameters
    """
    try:
        # Return available normalization parameters
        parameters = {
            "normalization_types": [t.value for t in NormalizationType],
            "default_parameters": {
                "input_format": "auto",
                "output_format": "entrezgene",
                "species": "human",
                "drop_unmapped": True,
                "batch_size": 1000
            }
        }
        
        return parameters
        
    except Exception as e:
        logger.error(f"Error getting normalization parameters: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")