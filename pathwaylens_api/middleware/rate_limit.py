"""
Rate limiting middleware for PathwayLens API.

This module provides rate limiting middleware for the API.
"""

import time
from typing import Dict, Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from collections import defaultdict, deque

from pathwaylens_api.utils.exceptions import RateLimitError
from pathwaylens_core.utils.config import get_config


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for API requests."""
    
    def __init__(self, app, default_limit: int = 100, default_window: int = 3600):
        super().__init__(app)
        self.default_limit = default_limit
        self.default_window = default_window
        self.requests = defaultdict(lambda: deque())
        self.config = get_config()
        
        # Get rate limit configuration
        self.rate_limits = self.config.get("rate_limits", {})
        self.authenticated_limit = self.rate_limits.get("authenticated", 1000)
        self.anonymous_limit = self.rate_limits.get("anonymous", 100)
        self.window_size = self.rate_limits.get("window_seconds", 3600)
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting middleware."""
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get rate limit for client
        limit = self._get_rate_limit(request, client_id)
        
        # Check rate limit
        if not self._check_rate_limit(client_id, limit):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_id, limit)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get user ID from request state (if authenticated)
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user['user_id']}"
        
        # Fall back to IP address
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _get_rate_limit(self, request: Request, client_id: str) -> int:
        """Get rate limit for client."""
        # Check if client is authenticated
        if client_id.startswith("user:"):
            return self.authenticated_limit
        
        # Check for specific endpoint limits
        endpoint = request.url.path
        endpoint_limits = self.rate_limits.get("endpoints", {})
        
        for pattern, limit in endpoint_limits.items():
            if endpoint.startswith(pattern):
                return limit
        
        # Default limit for anonymous users
        return self.anonymous_limit
    
    def _check_rate_limit(self, client_id: str, limit: int) -> bool:
        """Check if client has exceeded rate limit."""
        now = time.time()
        window_start = now - self.window_size
        
        # Clean old requests
        client_requests = self.requests[client_id]
        while client_requests and client_requests[0] < window_start:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= limit:
            return False
        
        # Add current request
        client_requests.append(now)
        return True
    
    def _add_rate_limit_headers(self, response: Response, client_id: str, limit: int):
        """Add rate limit headers to response."""
        now = time.time()
        window_start = now - self.window_size
        
        # Count current requests
        client_requests = self.requests[client_id]
        current_requests = len([req for req in client_requests if req >= window_start])
        
        # Add headers
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current_requests))
        response.headers["X-RateLimit-Reset"] = str(int(now + self.window_size))
        
        # Add retry-after header if rate limited
        if current_requests >= limit:
            response.headers["Retry-After"] = str(self.window_size)
    
    def _cleanup_old_requests(self):
        """Clean up old request records."""
        now = time.time()
        window_start = now - self.window_size
        
        for client_id in list(self.requests.keys()):
            client_requests = self.requests[client_id]
            while client_requests and client_requests[0] < window_start:
                client_requests.popleft()
            
            # Remove empty entries
            if not client_requests:
                del self.requests[client_id]
