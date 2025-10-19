"""
Authentication schemas for PathwayLens API.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Registration request model."""
    email: EmailStr
    password: str
    name: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    name: str
    role: str
    created_at: datetime
    updated_at: datetime


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str
