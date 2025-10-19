"""
Job management API routes for PathwayLens.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime

from ..utils.dependencies import get_current_user, get_database
from ..utils.exceptions import PathwayLensException


router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=Dict[str, Any])
async def list_all_jobs(
    limit: int = 10,
    offset: int = 0,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    List all jobs for current user.
    
    Args:
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        job_type: Filter by job type (analysis, comparison, visualization, normalization)
        status: Filter by job status (pending, running, completed, failed)
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        Dict[str, Any]: List of jobs with metadata
    """
    try:
        # Build query filter
        query_filter = {"user_id": current_user["id"]}
        
        if job_type:
            query_filter["job_type"] = job_type
        
        if status:
            query_filter["status"] = status
        
        # Get jobs from all collections
        all_jobs = []
        
        # Analysis jobs
        analysis_jobs = await db.analysis_jobs.find(query_filter).sort("created_at", -1).to_list(length=None)
        for job in analysis_jobs:
            job["job_type"] = "analysis"
            all_jobs.append(job)
        
        # Comparison jobs
        comparison_jobs = await db.comparison_jobs.find(query_filter).sort("created_at", -1).to_list(length=None)
        for job in comparison_jobs:
            job["job_type"] = "comparison"
            all_jobs.append(job)
        
        # Visualization jobs
        visualization_jobs = await db.visualization_jobs.find(query_filter).sort("created_at", -1).to_list(length=None)
        for job in visualization_jobs:
            job["job_type"] = "visualization"
            all_jobs.append(job)
        
        # Normalization jobs
        normalization_jobs = await db.normalization_jobs.find(query_filter).sort("created_at", -1).to_list(length=None)
        for job in normalization_jobs:
            job["job_type"] = "normalization"
            all_jobs.append(job)
        
        # Sort all jobs by creation time
        all_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        total_jobs = len(all_jobs)
        paginated_jobs = all_jobs[offset:offset + limit]
        
        # Format response
        response = {
            "jobs": paginated_jobs,
            "total": total_jobs,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_jobs
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error listing all jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/status", response_model=Dict[str, Any])
async def get_job_status_summary(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get job status summary for current user.
    
    Args:
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        Dict[str, Any]: Job status summary
    """
    try:
        user_id = current_user["id"]
        
        # Get job counts by type and status
        job_summary = {
            "analysis": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
            "comparison": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
            "visualization": {"pending": 0, "running": 0, "completed": 0, "failed": 0},
            "normalization": {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        }
        
        # Count analysis jobs
        analysis_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        analysis_counts = await db.analysis_jobs.aggregate(analysis_pipeline).to_list(length=None)
        for count in analysis_counts:
            status = count["_id"]
            if status in job_summary["analysis"]:
                job_summary["analysis"][status] = count["count"]
        
        # Count comparison jobs
        comparison_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        comparison_counts = await db.comparison_jobs.aggregate(comparison_pipeline).to_list(length=None)
        for count in comparison_counts:
            status = count["_id"]
            if status in job_summary["comparison"]:
                job_summary["comparison"][status] = count["count"]
        
        # Count visualization jobs
        visualization_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        visualization_counts = await db.visualization_jobs.aggregate(visualization_pipeline).to_list(length=None)
        for count in visualization_counts:
            status = count["_id"]
            if status in job_summary["visualization"]:
                job_summary["visualization"][status] = count["count"]
        
        # Count normalization jobs
        normalization_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        normalization_counts = await db.normalization_jobs.aggregate(normalization_pipeline).to_list(length=None)
        for count in normalization_counts:
            status = count["_id"]
            if status in job_summary["normalization"]:
                job_summary["normalization"][status] = count["count"]
        
        # Calculate totals
        total_jobs = 0
        total_pending = 0
        total_running = 0
        total_completed = 0
        total_failed = 0
        
        for job_type in job_summary:
            for status in job_summary[job_type]:
                count = job_summary[job_type][status]
                total_jobs += count
                if status == "pending":
                    total_pending += count
                elif status == "running":
                    total_running += count
                elif status == "completed":
                    total_completed += count
                elif status == "failed":
                    total_failed += count
        
        response = {
            "job_summary": job_summary,
            "totals": {
                "total_jobs": total_jobs,
                "pending": total_pending,
                "running": total_running,
                "completed": total_completed,
                "failed": total_failed
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting job status summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Delete a job by ID.
    
    Args:
        job_id: Job ID to delete
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        JSONResponse: Success message
    """
    try:
        user_id = current_user["id"]
        deleted = False
        
        # Try to delete from each collection
        collections = [
            ("analysis_jobs", "analysis"),
            ("comparison_jobs", "comparison"),
            ("visualization_jobs", "visualization"),
            ("normalization_jobs", "normalization")
        ]
        
        for collection_name, job_type in collections:
            collection = getattr(db, collection_name)
            result = await collection.delete_one({
                "job_id": job_id,
                "user_id": user_id
            })
            
            if result.deleted_count > 0:
                deleted = True
                logger.info(f"Deleted {job_type} job {job_id} for user {user_id}")
                break
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JSONResponse(
            status_code=200,
            content={"message": "Job deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/cleanup")
async def cleanup_old_jobs(
    days_old: int = 30,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Cleanup old completed jobs.
    
    Args:
        days_old: Number of days old to consider for cleanup
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        JSONResponse: Cleanup summary
    """
    try:
        user_id = current_user["id"]
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        cleanup_summary = {
            "analysis_jobs": 0,
            "comparison_jobs": 0,
            "visualization_jobs": 0,
            "normalization_jobs": 0,
            "total_deleted": 0
        }
        
        # Cleanup analysis jobs
        result = await db.analysis_jobs.delete_many({
            "user_id": user_id,
            "status": "completed",
            "completed_at": {"$lt": cutoff_date}
        })
        cleanup_summary["analysis_jobs"] = result.deleted_count
        
        # Cleanup comparison jobs
        result = await db.comparison_jobs.delete_many({
            "user_id": user_id,
            "status": "completed",
            "completed_at": {"$lt": cutoff_date}
        })
        cleanup_summary["comparison_jobs"] = result.deleted_count
        
        # Cleanup visualization jobs
        result = await db.visualization_jobs.delete_many({
            "user_id": user_id,
            "status": "completed",
            "completed_at": {"$lt": cutoff_date}
        })
        cleanup_summary["visualization_jobs"] = result.deleted_count
        
        # Cleanup normalization jobs
        result = await db.normalization_jobs.delete_many({
            "user_id": user_id,
            "status": "completed",
            "completed_at": {"$lt": cutoff_date}
        })
        cleanup_summary["normalization_jobs"] = result.deleted_count
        
        # Calculate total
        cleanup_summary["total_deleted"] = (
            cleanup_summary["analysis_jobs"] +
            cleanup_summary["comparison_jobs"] +
            cleanup_summary["visualization_jobs"] +
            cleanup_summary["normalization_jobs"]
        )
        
        logger.info(f"Cleaned up {cleanup_summary['total_deleted']} old jobs for user {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Job cleanup completed successfully",
                "cleanup_summary": cleanup_summary
            }
        )
        
    except Exception as e:
        logger.error(f"Error cleaning up old jobs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """
    Get job logs.
    
    Args:
        job_id: Job ID
        current_user: Current authenticated user
        db: Database connection
        
    Returns:
        Dict[str, Any]: Job logs
    """
    try:
        user_id = current_user["id"]
        
        # Find job in any collection
        job = None
        job_type = None
        
        collections = [
            ("analysis_jobs", "analysis"),
            ("comparison_jobs", "comparison"),
            ("visualization_jobs", "visualization"),
            ("normalization_jobs", "normalization")
        ]
        
        for collection_name, job_type_name in collections:
            collection = getattr(db, collection_name)
            job = await collection.find_one({
                "job_id": job_id,
                "user_id": user_id
            })
            
            if job:
                job_type = job_type_name
                break
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get logs from job record
        logs = job.get("logs", [])
        
        return {
            "job_id": job_id,
            "job_type": job_type,
            "status": job["status"],
            "logs": logs
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job logs: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")