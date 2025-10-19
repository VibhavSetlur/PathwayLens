"""
Authentication routes for PathwayLens API.

This module provides authentication and authorization endpoints.
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import jwt
import hashlib
import secrets

from pathwaylens_api.utils.dependencies import get_current_user, get_database
from pathwaylens_api.utils.exceptions import AuthenticationError, ValidationError
from pathwaylens_core.utils.config import get_config

router = APIRouter()
security = HTTPBearer()


# Request/Response models
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


# Password hashing
def hash_password(password: str) -> str:
    """Hash password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}:{password_hash}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    try:
        salt, password_hash = hashed_password.split(":")
        return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
    except ValueError:
        return False


# JWT token functions
def create_access_token(user_id: str, email: str, name: str, role: str = "user") -> str:
    """Create JWT access token."""
    config = get_config()
    secret_key = config.get("auth", {}).get("secret_key")
    expires_delta = config.get("auth", {}).get("access_token_expire_minutes", 30)
    
    if not secret_key:
        raise AuthenticationError("Authentication not configured")
    
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    
    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    return jwt.encode(payload, secret_key, algorithm="HS256")


def create_refresh_token(user_id: str) -> str:
    """Create JWT refresh token."""
    config = get_config()
    secret_key = config.get("auth", {}).get("secret_key")
    expires_delta = config.get("auth", {}).get("refresh_token_expire_days", 7)
    
    if not secret_key:
        raise AuthenticationError("Authentication not configured")
    
    expire = datetime.utcnow() + timedelta(days=expires_delta)
    
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }
    
    return jwt.encode(payload, secret_key, algorithm="HS256")


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify JWT token."""
    config = get_config()
    secret_key = config.get("auth", {}).get("secret_key")
    
    if not secret_key:
        raise AuthenticationError("Authentication not configured")
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        
        if payload.get("type") != token_type:
            raise AuthenticationError("Invalid token type")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")


# Route handlers
@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    db = Depends(get_database)
):
    """Authenticate user and return access token."""
    try:
        # Query user from database
        result = await db.execute(
            text("SELECT * FROM users WHERE email = :email"),
            {"email": request.email}
        )
        user = result.fetchone()
        
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        # Verify password
        if not verify_password(request.password, user.password_hash):
            raise AuthenticationError("Invalid email or password")
        
        # Create tokens
        access_token = create_access_token(
            user.id, user.email, user.name, user.role
        )
        refresh_token = create_refresh_token(user.id)
        
        # Update last login
        await db.execute(
            text("UPDATE users SET last_login = :last_login WHERE id = :user_id"),
            {
                "last_login": datetime.utcnow(),
                "user_id": user.id
            }
        )
        
        config = get_config()
        expires_in = config.get("auth", {}).get("access_token_expire_minutes", 30) * 60
        
        return TokenResponse(
            access_token=access_token,
            expires_in=expires_in,
            refresh_token=refresh_token
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {e}"
        )


@router.post("/register", response_model=UserResponse)
async def register(
    request: RegisterRequest,
    db = Depends(get_database)
):
    """Register new user."""
    try:
        # Check if user already exists
        result = await db.execute(
            text("SELECT id FROM users WHERE email = :email"),
            {"email": request.email}
        )
        existing_user = result.fetchone()
        
        if existing_user:
            raise ValidationError("User with this email already exists")
        
        # Validate password strength
        if len(request.password) < 8:
            raise ValidationError("Password must be at least 8 characters long")
        
        # Create user
        user_id = str(uuid.uuid4())
        password_hash = hash_password(request.password)
        
        await db.execute(
            text("""
                INSERT INTO users (id, email, name, password_hash, role, created_at, updated_at)
                VALUES (:id, :email, :name, :password_hash, :role, :created_at, :updated_at)
            """),
            {
                "id": user_id,
                "email": request.email,
                "name": request.name,
                "password_hash": password_hash,
                "role": "user",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        )
        
        # Return user info
        return UserResponse(
            id=user_id,
            email=request.email,
            name=request.name,
            role="user",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
    except ValidationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {e}"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db = Depends(get_database)
):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = verify_token(request.refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise AuthenticationError("Invalid refresh token")
        
        # Get user info
        result = await db.execute(
            text("SELECT * FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        )
        user = result.fetchone()
        
        if not user:
            raise AuthenticationError("User not found")
        
        # Create new access token
        access_token = create_access_token(
            user.id, user.email, user.name, user.role
        )
        
        config = get_config()
        expires_in = config.get("auth", {}).get("access_token_expire_minutes", 30) * 60
        
        return TokenResponse(
            access_token=access_token,
            expires_in=expires_in
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {e}"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get current user information."""
    try:
        # Query user from database
        result = await db.execute(
            text("SELECT * FROM users WHERE id = :user_id"),
            {"user_id": user.user_id}
        )
        user_data = result.fetchone()
        
        if not user_data:
            raise AuthenticationError("User not found")
        
        return UserResponse(
            id=user_data.id,
            email=user_data.email,
            name=user_data.name,
            role=user_data.role,
            created_at=user_data.created_at,
            updated_at=user_data.updated_at
        )
        
    except AuthenticationError:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {e}"
        )


@router.post("/logout")
async def logout(
    user = Depends(get_current_user)
):
    """Logout user (client should discard tokens)."""
    # In a stateless JWT system, logout is handled client-side
    # by discarding the tokens. For enhanced security, you could
    # implement a token blacklist.
    return {"message": "Logged out successfully"}


@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    user = Depends(get_current_user),
    db = Depends(get_database)
):
    """Change user password."""
    try:
        # Get current user data
        result = await db.execute(
            text("SELECT password_hash FROM users WHERE id = :user_id"),
            {"user_id": user.user_id}
        )
        user_data = result.fetchone()
        
        if not user_data:
            raise AuthenticationError("User not found")
        
        # Verify current password
        if not verify_password(current_password, user_data.password_hash):
            raise AuthenticationError("Current password is incorrect")
        
        # Validate new password
        if len(new_password) < 8:
            raise ValidationError("New password must be at least 8 characters long")
        
        # Update password
        new_password_hash = hash_password(new_password)
        await db.execute(
            text("UPDATE users SET password_hash = :password_hash, updated_at = :updated_at WHERE id = :user_id"),
            {
                "password_hash": new_password_hash,
                "updated_at": datetime.utcnow(),
                "user_id": user.user_id
            }
        )
        
        return {"message": "Password changed successfully"}
        
    except (AuthenticationError, ValidationError):
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password change failed: {e}"
        )


# Import required modules
from sqlalchemy import text
