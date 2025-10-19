"""
Storage utilities for PathwayLens API.

This module provides file storage management for uploaded files and results.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO
from datetime import datetime, timedelta
import aiofiles
import aiofiles.os
from fastapi import UploadFile
import logging

from pathwaylens_core.utils.config import get_config
from .exceptions import FileProcessingError, StorageError

logger = logging.getLogger(__name__)


class StorageManager:
    """File storage manager."""
    
    def __init__(self, base_path: str, max_file_size: int = 100 * 1024 * 1024):  # 100MB default
        self.base_path = Path(base_path)
        self.max_file_size = max_file_size
        self.upload_path = self.base_path / "uploads"
        self.results_path = self.base_path / "results"
        self.temp_path = self.base_path / "temp"
        
        # Create directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.upload_path, self.results_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    async def save_upload(self, file: UploadFile, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Save uploaded file."""
        try:
            # Generate unique filename
            file_id = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix if file.filename else ""
            filename = f"{file_id}{file_extension}"
            
            # Create user-specific directory
            if user_id:
                user_dir = self.upload_path / user_id
                user_dir.mkdir(exist_ok=True)
                file_path = user_dir / filename
            else:
                file_path = self.upload_path / filename
            
            # Check file size
            file_size = 0
            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(8192):  # 8KB chunks
                    file_size += len(chunk)
                    if file_size > self.max_file_size:
                        raise FileProcessingError(
                            f"File too large. Maximum size: {self.max_file_size} bytes",
                            filename=file.filename,
                            details={"max_size": self.max_file_size, "actual_size": file_size}
                        )
                    await f.write(chunk)
            
            # Get file info
            file_info = {
                "file_id": file_id,
                "original_filename": file.filename,
                "filename": filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "content_type": file.content_type,
                "uploaded_at": datetime.utcnow().isoformat(),
                "user_id": user_id
            }
            
            logger.info(f"File uploaded successfully: {file_id}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save upload: {e}")
            raise FileProcessingError(
                f"Failed to save uploaded file: {e}",
                filename=file.filename,
                details={"error": str(e)}
            )
    
    async def get_file(self, file_id: str, user_id: Optional[str] = None) -> Optional[Path]:
        """Get file path by ID."""
        if user_id:
            user_dir = self.upload_path / user_id
            for file_path in user_dir.glob(f"{file_id}*"):
                return file_path
        else:
            for file_path in self.upload_path.glob(f"{file_id}*"):
                return file_path
        
        return None
    
    async def delete_file(self, file_id: str, user_id: Optional[str] = None) -> bool:
        """Delete file by ID."""
        try:
            file_path = await self.get_file(file_id, user_id)
            if file_path and file_path.exists():
                await aiofiles.os.remove(file_path)
                logger.info(f"File deleted: {file_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False
    
    async def save_result(self, job_id: str, result_data: Dict[str, Any], filename: str) -> str:
        """Save analysis result."""
        try:
            # Create job-specific directory
            job_dir = self.results_path / job_id
            job_dir.mkdir(exist_ok=True)
            
            file_path = job_dir / filename
            
            # Save result data
            async with aiofiles.open(file_path, 'w') as f:
                import json
                await f.write(json.dumps(result_data, indent=2))
            
            logger.info(f"Result saved: {job_id}/{filename}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            raise StorageError(f"Failed to save result: {e}")
    
    async def get_result(self, job_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Get analysis result."""
        try:
            file_path = self.results_path / job_id / filename
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'r') as f:
                import json
                content = await f.read()
                return json.loads(content)
                
        except Exception as e:
            logger.error(f"Failed to get result: {e}")
            return None
    
    async def list_results(self, job_id: str) -> List[str]:
        """List all result files for a job."""
        try:
            job_dir = self.results_path / job_id
            if not job_dir.exists():
                return []
            
            return [f.name for f in job_dir.iterdir() if f.is_file()]
            
        except Exception as e:
            logger.error(f"Failed to list results: {e}")
            return []
    
    async def cleanup_old_files(self, days: int = 30):
        """Clean up old files."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean up uploads
            for user_dir in self.upload_path.iterdir():
                if user_dir.is_dir():
                    for file_path in user_dir.iterdir():
                        if file_path.is_file():
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                await aiofiles.os.remove(file_path)
                                logger.info(f"Cleaned up old file: {file_path}")
            
            # Clean up results
            for job_dir in self.results_path.iterdir():
                if job_dir.is_dir():
                    for file_path in job_dir.iterdir():
                        if file_path.is_file():
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                await aiofiles.os.remove(file_path)
                                logger.info(f"Cleaned up old result: {file_path}")
            
            logger.info(f"Cleanup completed for files older than {days} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
    
    async def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information."""
        try:
            total_size = 0
            file_count = 0
            
            # Calculate uploads size
            for user_dir in self.upload_path.iterdir():
                if user_dir.is_dir():
                    for file_path in user_dir.iterdir():
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1
            
            # Calculate results size
            for job_dir in self.results_path.iterdir():
                if job_dir.is_dir():
                    for file_path in job_dir.iterdir():
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                            file_count += 1
            
            return {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "base_path": str(self.base_path),
                "max_file_size": self.max_file_size
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "file_count": 0,
                "base_path": str(self.base_path),
                "max_file_size": self.max_file_size
            }


class StorageError(Exception):
    """Storage-related error."""
    pass


# Global storage manager
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> StorageManager:
    """Get the global storage manager."""
    global _storage_manager
    
    if not _storage_manager:
        config = get_config()
        storage_config = config.get("storage", {})
        base_path = storage_config.get("base_path", "./storage")
        max_file_size = storage_config.get("max_file_size", 100 * 1024 * 1024)
        
        _storage_manager = StorageManager(base_path, max_file_size)
    
    return _storage_manager


async def initialize_storage():
    """Initialize storage system."""
    global _storage_manager
    
    config = get_config()
    storage_config = config.get("storage", {})
    base_path = storage_config.get("base_path", "./storage")
    max_file_size = storage_config.get("max_file_size", 100 * 1024 * 1024)
    
    _storage_manager = StorageManager(base_path, max_file_size)
    logger.info(f"Storage initialized: {base_path}")


async def cleanup_storage():
    """Cleanup storage system."""
    global _storage_manager
    
    if _storage_manager:
        await _storage_manager.cleanup_old_files()
        logger.info("Storage cleanup completed")
