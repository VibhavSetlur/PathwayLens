"""
FastAPI dependencies for PathwayLens API.

This module provides dependency injection functions for the API.
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

from .database import get_database_session, get_database_manager
from .storage import get_storage_manager
from .exceptions import AuthenticationError, AuthorizationError
from pathwaylens_core.utils.config import get_config

# Security scheme
security = HTTPBearer()


class CurrentUser:
    """Current user context."""
    
    def __init__(self, user_id: str, email: str, name: str, role: str = "user"):
        self.user_id = user_id
        self.email = email
        self.name = name
        self.role = role
    
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role == "admin"
    
    def can_access_job(self, job_user_id: Optional[str]) -> bool:
        """Check if user can access a job."""
        return self.is_admin() or self.user_id == job_user_id


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> CurrentUser:
    """Get current authenticated user."""
    try:
        # Decode JWT token
        config = get_config()
        secret_key = config.get("auth", {}).get("secret_key")
        
        if not secret_key:
            raise AuthenticationError("Authentication not configured")
        
        payload = jwt.decode(
            credentials.credentials,
            secret_key,
            algorithms=["HS256"]
        )
        
        # Check token expiration
        exp = payload.get("exp")
        if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
            raise AuthenticationError("Token expired")
        
        # Extract user information
        user_id = payload.get("sub")
        email = payload.get("email")
        name = payload.get("name")
        role = payload.get("role", "user")
        
        if not user_id or not email:
            raise AuthenticationError("Invalid token")
        
        return CurrentUser(user_id, email, name, role)
        
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
    except Exception as e:
        raise AuthenticationError(f"Authentication failed: {e}")


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[CurrentUser]:
    """Get current user if authenticated, otherwise None."""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except AuthenticationError:
        return None


async def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Require admin role."""
    if not user.is_admin():
        raise AuthorizationError("Admin access required")
    return user


async def get_database():
    """Get database session dependency."""
    async with get_database_session() as session:
        yield session


async def get_storage():
    """Get storage manager dependency."""
    return get_storage_manager()


async def get_job_access(
    job_id: str,
    user: CurrentUser = Depends(get_current_user),
    db = Depends(get_database)
) -> Dict[str, Any]:
    """Get job with access control."""
    # Query job from database
    result = await db.execute(
        text("SELECT * FROM jobs WHERE id = :job_id"),
        {"job_id": job_id}
    )
    job = result.fetchone()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Check access permissions
    if not user.can_access_job(job.user_id):
        raise AuthorizationError("Access denied to this job")
    
    return dict(job._mapping)


async def get_project_access(
    project_id: str,
    user: CurrentUser = Depends(get_current_user),
    db = Depends(get_database)
) -> Dict[str, Any]:
    """Get project with access control."""
    # Query project from database
    result = await db.execute(
        text("SELECT * FROM projects WHERE id = :project_id"),
        {"project_id": project_id}
    )
    project = result.fetchone()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    
    # Check access permissions (owner or admin)
    if not (user.is_admin() or project.owner_id == user.user_id):
        raise AuthorizationError("Access denied to this project")
    
    return dict(project._mapping)


# Rate limiting dependencies
async def check_rate_limit(
    user: Optional[CurrentUser] = Depends(get_optional_user)
) -> None:
    """Check rate limit for user."""
    # This would integrate with the rate limiting middleware
    # For now, just a placeholder
    pass


# File upload dependencies
async def validate_file_upload(
    file_size: int,
    content_type: str,
    user: CurrentUser = Depends(get_current_user)
) -> None:
    """Validate file upload."""
    config = get_config()
    storage_config = config.get("storage", {})
    max_file_size = storage_config.get("max_file_size", 100 * 1024 * 1024)
    
    if file_size > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_file_size} bytes"
        )
    
    # Check content type
    allowed_types = [
        "text/csv",
        "text/plain",
        "application/json",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]
    
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {content_type}"
        )


# Analysis dependencies
async def validate_analysis_parameters(
    parameters: Dict[str, Any],
    user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Validate analysis parameters."""
    # Set defaults
    validated_params = {
        "significance_threshold": 0.05,
        "correction_method": "fdr_bh",
        "min_pathway_size": 5,
        "max_pathway_size": 500,
        "databases": ["kegg", "reactome", "go"]
    }
    
    # Update with provided parameters
    validated_params.update(parameters)
    
    # Validate values
    if not 0 < validated_params["significance_threshold"] < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Significance threshold must be between 0 and 1"
        )
    
    if validated_params["min_pathway_size"] < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Minimum pathway size must be at least 1"
        )
    
    if validated_params["max_pathway_size"] < validated_params["min_pathway_size"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum pathway size must be greater than minimum pathway size"
        )
    
    return validated_params


# Import required modules
from sqlalchemy import text
