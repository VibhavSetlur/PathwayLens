"""
Logging middleware for PathwayLens API.

This module provides request logging middleware for the API.
"""

import time
import logging
from typing import Dict, Any
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import json

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware for API requests."""
    
    def __init__(self, app, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = log_level.upper()
        self.logger = logging.getLogger("pathwaylens.api")
    
    async def dispatch(self, request: Request, call_next):
        """Process request through logging middleware."""
        # Start timing
        start_time = time.time()
        
        # Get request info
        request_info = self._get_request_info(request)
        
        # Log request
        self._log_request(request_info)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            self._log_response(request_info, response, process_time)
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            self._log_error(request_info, e, process_time)
            
            raise
    
    def _get_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract request information."""
        # Get client IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else request.client.host
        
        # Get user info if available
        user_info = {}
        if hasattr(request.state, 'user') and request.state.user:
            user_info = {
                "user_id": request.state.user.get("user_id"),
                "email": request.state.user.get("email"),
                "role": request.state.user.get("role")
            }
        
        return {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "content_type": request.headers.get("Content-Type", ""),
            "content_length": request.headers.get("Content-Length", "0"),
            "user": user_info,
            "timestamp": time.time()
        }
    
    def _log_request(self, request_info: Dict[str, Any]):
        """Log incoming request."""
        log_data = {
            "event": "request_start",
            "method": request_info["method"],
            "path": request_info["path"],
            "client_ip": request_info["client_ip"],
            "user_agent": request_info["user_agent"],
            "user": request_info["user"]
        }
        
        self.logger.info(f"Request started: {request_info['method']} {request_info['path']}", extra=log_data)
    
    def _log_response(self, request_info: Dict[str, Any], response: Response, process_time: float):
        """Log response."""
        log_data = {
            "event": "request_complete",
            "method": request_info["method"],
            "path": request_info["path"],
            "status_code": response.status_code,
            "process_time": process_time,
            "client_ip": request_info["client_ip"],
            "user": request_info["user"]
        }
        
        # Log level based on status code
        if response.status_code >= 500:
            self.logger.error(f"Request completed: {request_info['method']} {request_info['path']} - {response.status_code}", extra=log_data)
        elif response.status_code >= 400:
            self.logger.warning(f"Request completed: {request_info['method']} {request_info['path']} - {response.status_code}", extra=log_data)
        else:
            self.logger.info(f"Request completed: {request_info['method']} {request_info['path']} - {response.status_code}", extra=log_data)
    
    def _log_error(self, request_info: Dict[str, Any], error: Exception, process_time: float):
        """Log error."""
        log_data = {
            "event": "request_error",
            "method": request_info["method"],
            "path": request_info["path"],
            "error": str(error),
            "error_type": type(error).__name__,
            "process_time": process_time,
            "client_ip": request_info["client_ip"],
            "user": request_info["user"]
        }
        
        self.logger.error(f"Request error: {request_info['method']} {request_info['path']} - {error}", extra=log_data)
