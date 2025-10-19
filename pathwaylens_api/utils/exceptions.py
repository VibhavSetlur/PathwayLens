"""
PathwayLens API exceptions.

This module defines custom exceptions for the API layer.
"""

from typing import Any, Dict, Optional


class PathwayLensException(Exception):
    """Base exception for PathwayLens API."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "pathwaylens_error",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(PathwayLensException):
    """Exception raised for validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        if value is not None:
            error_details["value"] = value
            
        super().__init__(
            message=message,
            error_code="validation_error",
            status_code=400,
            details=error_details
        )


class AuthenticationError(PathwayLensException):
    """Exception raised for authentication errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="authentication_error",
            status_code=401,
            details=details
        )


class AuthorizationError(PathwayLensException):
    """Exception raised for authorization errors."""
    
    def __init__(
        self,
        message: str = "Access denied",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="authorization_error",
            status_code=403,
            details=details
        )


class NotFoundError(PathwayLensException):
    """Exception raised when a resource is not found."""
    
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if resource_type:
            error_details["resource_type"] = resource_type
        if resource_id:
            error_details["resource_id"] = resource_id
            
        super().__init__(
            message=message,
            error_code="not_found_error",
            status_code=404,
            details=error_details
        )


class DatabaseError(PathwayLensException):
    """Exception raised for database errors."""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
            
        super().__init__(
            message=message,
            error_code="database_error",
            status_code=500,
            details=error_details
        )


class JobError(PathwayLensException):
    """Exception raised for job-related errors."""
    
    def __init__(
        self,
        message: str = "Job operation failed",
        job_id: Optional[str] = None,
        job_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if job_id:
            error_details["job_id"] = job_id
        if job_type:
            error_details["job_type"] = job_type
            
        super().__init__(
            message=message,
            error_code="job_error",
            status_code=500,
            details=error_details
        )


class RateLimitError(PathwayLensException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if limit:
            error_details["limit"] = limit
        if window:
            error_details["window"] = window
            
        super().__init__(
            message=message,
            error_code="rate_limit_error",
            status_code=429,
            details=error_details
        )


class ExternalServiceError(PathwayLensException):
    """Exception raised for external service errors."""
    
    def __init__(
        self,
        message: str = "External service error",
        service: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if service:
            error_details["service"] = service
            
        super().__init__(
            message=message,
            error_code="external_service_error",
            status_code=502,
            details=error_details
        )


class FileProcessingError(PathwayLensException):
    """Exception raised for file processing errors."""
    
    def __init__(
        self,
        message: str = "File processing failed",
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if filename:
            error_details["filename"] = filename
        if file_type:
            error_details["file_type"] = file_type
            
        super().__init__(
            message=message,
            error_code="file_processing_error",
            status_code=400,
            details=error_details
        )


class StorageError(PathwayLensException):
    """Exception raised for storage errors."""
    
    def __init__(
        self,
        message: str = "Storage operation failed",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        error_details = details or {}
        if operation:
            error_details["operation"] = operation
            
        super().__init__(
            message=message,
            error_code="storage_error",
            status_code=500,
            details=error_details
        )
