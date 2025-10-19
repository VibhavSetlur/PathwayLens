"""
Cache manager for PathwayLens.

This module provides caching capabilities for database responses,
improving performance and reducing API calls.
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
from loguru import logger
import asyncio
import aiofiles


class CacheManager:
    """Manages caching of database responses and analysis results."""
    
    def __init__(self, cache_dir: str = ".pathwaylens/cache", max_size_mb: int = 1000, ttl_days: int = 90):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            ttl_days: Time to live in days
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.ttl_days = ttl_days
        self.ttl_seconds = ttl_days * 24 * 60 * 60
        
        self.logger = logger.bind(module="cache_manager")
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache metadata: {e}")
        
        return {
            "entries": {},
            "total_size": 0,
            "created_at": time.time()
        }
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache metadata: {e}")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key from input key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry["created_at"] > self.ttl_seconds
    
    def _cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = []
        
        for key, entry in self.metadata["entries"].items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            self.logger.info(f"Removed {len(expired_keys)} expired cache entries")
    
    def _remove_entry(self, key: str):
        """Remove a cache entry."""
        if key in self.metadata["entries"]:
            entry = self.metadata["entries"][key]
            cache_file = self._get_cache_file(key)
            
            # Remove file
            if cache_file.exists():
                cache_file.unlink()
            
            # Update metadata
            self.metadata["total_size"] -= entry["size"]
            del self.metadata["entries"][key]
    
    def _cleanup_size(self):
        """Remove oldest entries if cache size exceeds limit."""
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if self.metadata["total_size"] <= max_size_bytes:
            return
        
        # Sort entries by creation time (oldest first)
        sorted_entries = sorted(
            self.metadata["entries"].items(),
            key=lambda x: x[1]["created_at"]
        )
        
        # Remove oldest entries until size is under limit
        for key, entry in sorted_entries:
            if self.metadata["total_size"] <= max_size_bytes:
                break
            
            self._remove_entry(key)
        
        self.logger.info(f"Cache size reduced to {self.metadata['total_size'] / 1024 / 1024:.2f} MB")
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        cache_key = self._get_cache_key(key)
        
        if cache_key not in self.metadata["entries"]:
            return None
        
        entry = self.metadata["entries"][cache_key]
        
        # Check if expired
        if self._is_expired(entry):
            self._remove_entry(cache_key)
            self._save_metadata()
            return None
        
        # Load cached data
        cache_file = self._get_cache_file(cache_key)
        if not cache_file.exists():
            self._remove_entry(cache_key)
            self._save_metadata()
            return None
        
        try:
            async with aiofiles.open(cache_file, 'rb') as f:
                data = await f.read()
                return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Error loading cache entry {key}: {e}")
            self._remove_entry(cache_key)
            self._save_metadata()
            return None
    
    async def set(self, key: str, value: Any):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_key = self._get_cache_key(key)
        cache_file = self._get_cache_file(cache_key)
        
        try:
            # Serialize data
            data = pickle.dumps(value)
            data_size = len(data)
            
            # Save to file
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(data)
            
            # Update metadata
            if cache_key in self.metadata["entries"]:
                old_entry = self.metadata["entries"][cache_key]
                self.metadata["total_size"] -= old_entry["size"]
            
            self.metadata["entries"][cache_key] = {
                "key": key,
                "size": data_size,
                "created_at": time.time(),
                "file": str(cache_file)
            }
            
            self.metadata["total_size"] += data_size
            
            # Cleanup if needed
            self._cleanup_expired()
            self._cleanup_size()
            
            # Save metadata
            self._save_metadata()
            
        except Exception as e:
            self.logger.error(f"Error caching entry {key}: {e}")
    
    async def delete(self, key: str):
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        """
        cache_key = self._get_cache_key(key)
        self._remove_entry(cache_key)
        self._save_metadata()
    
    async def clear(self):
        """Clear all cache entries."""
        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        
        # Reset metadata
        self.metadata = {
            "entries": {},
            "total_size": 0,
            "created_at": time.time()
        }
        
        self._save_metadata()
        self.logger.info("Cache cleared")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        
        return {
            "total_entries": len(self.metadata["entries"]),
            "total_size_mb": self.metadata["total_size"] / 1024 / 1024,
            "max_size_mb": self.max_size_mb,
            "ttl_days": self.ttl_days,
            "cache_dir": str(self.cache_dir),
            "created_at": self.metadata["created_at"]
        }
    
    async def cleanup(self):
        """Clean up expired entries and manage cache size."""
        self._cleanup_expired()
        self._cleanup_size()
        self._save_metadata()
    
    def get_sync(self, method_name: str, *args, **kwargs):
        """
        Synchronous wrapper for async methods.
        
        Args:
            method_name: Name of the method to call
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result
        """
        method = getattr(self, method_name)
        return asyncio.run(method(*args, **kwargs))
    
    def __str__(self) -> str:
        """String representation of the cache manager."""
        return f"CacheManager(dir={self.cache_dir}, max_size={self.max_size_mb}MB, ttl={self.ttl_days}days)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the cache manager."""
        return f"CacheManager(cache_dir='{self.cache_dir}', max_size_mb={self.max_size_mb}, ttl_days={self.ttl_days})"
