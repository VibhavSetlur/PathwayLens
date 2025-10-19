"""
Authentication middleware for PathwayLens API.

This module provides authentication middleware for the API.
"""

from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import jwt
from datetime import datetime

from pathwaylens_api.utils.exceptions import AuthenticationError
from pathwaylens_core.utils.config import get_config

security = HTTPBearer(auto_error=False)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests."""
    
    def __init__(self, app, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
            "/api/v1/health",
            "/api/v1/info/version",
            "/api/v1/auth/login",
            "/api/v1/auth/register"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        # Skip authentication for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Check for authentication header
        authorization = request.headers.get("Authorization")
        if not authorization:
            # Allow anonymous access to public endpoints
            if self._is_public_endpoint(request.url.path):
                return await call_next(request)
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Validate token
        try:
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=authorization.replace("Bearer ", "")
            )
            
            # Verify token
            user_info = self._verify_token(credentials.credentials)
            
            # Add user info to request state
            request.state.user = user_info
            
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        
        return await call_next(request)
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (doesn't require authentication)."""
        public_paths = [
            "/api/v1/info/",
            "/api/v1/normalize/species",
            "/api/v1/normalize/identifier-types",
            "/api/v1/analyze/databases",
            "/api/v1/analyze/analysis-types",
            "/api/v1/compare/types",
            "/api/v1/visualize/plot-types"
        ]
        
        return any(path.startswith(public_path) for public_path in public_paths)
    
    def _verify_token(self, token: str) -> dict:
        """Verify JWT token and return user information."""
        try:
            config = get_config()
            secret_key = config.get("auth", {}).get("secret_key")
            
            if not secret_key:
                raise AuthenticationError("Authentication not configured")
            
            # Decode token
            payload = jwt.decode(
                token,
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
            
            return {
                "user_id": user_id,
                "email": email,
                "name": name,
                "role": role
            }
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
        except Exception as e:
            raise AuthenticationError(f"Token verification failed: {e}")
