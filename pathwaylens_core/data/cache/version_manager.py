"""
Version manager for PathwayLens.

This module provides version management for database updates,
ensuring data consistency and enabling rollback capabilities.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
from datetime import datetime


class VersionManager:
    """Manages versions of database data and analysis results."""
    
    def __init__(self, version_dir: str = ".pathwaylens/versions"):
        """
        Initialize version manager.
        
        Args:
            version_dir: Directory for version files
        """
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.bind(module="version_manager")
        
        # Version metadata
        self.metadata_file = self.version_dir / "versions.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load version metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading version metadata: {e}")
        
        return {
            "versions": {},
            "current_version": None,
            "created_at": time.time()
        }
    
    def _save_metadata(self):
        """Save version metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving version metadata: {e}")
    
    def create_version(self, version_id: str, description: str = "", data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new version.
        
        Args:
            version_id: Unique version identifier
            description: Version description
            data: Version data
            
        Returns:
            Version identifier
        """
        if version_id in self.metadata["versions"]:
            raise ValueError(f"Version {version_id} already exists")
        
        version_info = {
            "id": version_id,
            "description": description,
            "created_at": time.time(),
            "created_at_str": datetime.now().isoformat(),
            "data": data or {},
            "status": "active"
        }
        
        self.metadata["versions"][version_id] = version_info
        
        # Set as current version if it's the first one
        if self.metadata["current_version"] is None:
            self.metadata["current_version"] = version_id
        
        self._save_metadata()
        self.logger.info(f"Created version {version_id}: {description}")
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get version information.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Version information or None if not found
        """
        return self.metadata["versions"].get(version_id)
    
    def get_current_version(self) -> Optional[str]:
        """
        Get current version identifier.
        
        Returns:
            Current version identifier or None
        """
        return self.metadata["current_version"]
    
    def set_current_version(self, version_id: str):
        """
        Set current version.
        
        Args:
            version_id: Version identifier to set as current
        """
        if version_id not in self.metadata["versions"]:
            raise ValueError(f"Version {version_id} does not exist")
        
        self.metadata["current_version"] = version_id
        self._save_metadata()
        self.logger.info(f"Set current version to {version_id}")
    
    def list_versions(self) -> List[Dict[str, Any]]:
        """
        List all versions.
        
        Returns:
            List of version information
        """
        versions = []
        for version_id, version_info in self.metadata["versions"].items():
            versions.append({
                "id": version_id,
                "description": version_info["description"],
                "created_at": version_info["created_at_str"],
                "status": version_info["status"],
                "is_current": version_id == self.metadata["current_version"]
            })
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions
    
    def delete_version(self, version_id: str):
        """
        Delete a version.
        
        Args:
            version_id: Version identifier to delete
        """
        if version_id not in self.metadata["versions"]:
            raise ValueError(f"Version {version_id} does not exist")
        
        # Don't delete current version
        if version_id == self.metadata["current_version"]:
            raise ValueError("Cannot delete current version")
        
        del self.metadata["versions"][version_id]
        self._save_metadata()
        self.logger.info(f"Deleted version {version_id}")
    
    def update_version(self, version_id: str, description: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        """
        Update version information.
        
        Args:
            version_id: Version identifier
            description: New description (optional)
            data: New data (optional)
        """
        if version_id not in self.metadata["versions"]:
            raise ValueError(f"Version {version_id} does not exist")
        
        version_info = self.metadata["versions"][version_id]
        
        if description is not None:
            version_info["description"] = description
        
        if data is not None:
            version_info["data"] = data
        
        version_info["updated_at"] = time.time()
        version_info["updated_at_str"] = datetime.now().isoformat()
        
        self._save_metadata()
        self.logger.info(f"Updated version {version_id}")
    
    def get_version_data(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get version data.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Version data or None if not found
        """
        version_info = self.get_version(version_id)
        return version_info["data"] if version_info else None
    
    def get_current_version_data(self) -> Optional[Dict[str, Any]]:
        """
        Get current version data.
        
        Returns:
            Current version data or None
        """
        current_version = self.get_current_version()
        return self.get_version_data(current_version) if current_version else None
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """
        Compare two versions.
        
        Args:
            version1_id: First version identifier
            version2_id: Second version identifier
            
        Returns:
            Comparison results
        """
        version1 = self.get_version(version1_id)
        version2 = self.get_version(version2_id)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        # Compare data
        data1 = version1["data"]
        data2 = version2["data"]
        
        # Find differences
        differences = []
        
        # Check for new keys
        for key in data2:
            if key not in data1:
                differences.append(f"New key: {key}")
        
        # Check for removed keys
        for key in data1:
            if key not in data2:
                differences.append(f"Removed key: {key}")
        
        # Check for changed values
        for key in data1:
            if key in data2 and data1[key] != data2[key]:
                differences.append(f"Changed key: {key}")
        
        return {
            "version1": version1_id,
            "version2": version2_id,
            "differences": differences,
            "num_differences": len(differences),
            "version1_created": version1["created_at_str"],
            "version2_created": version2["created_at_str"]
        }
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get version history.
        
        Returns:
            List of version history entries
        """
        history = []
        for version_id, version_info in self.metadata["versions"].items():
            history.append({
                "id": version_id,
                "description": version_info["description"],
                "created_at": version_info["created_at_str"],
                "updated_at": version_info.get("updated_at_str"),
                "status": version_info["status"],
                "is_current": version_id == self.metadata["current_version"]
            })
        
        # Sort by creation time (newest first)
        history.sort(key=lambda x: x["created_at"], reverse=True)
        return history
    
    def cleanup_old_versions(self, keep_count: int = 10):
        """
        Clean up old versions, keeping only the most recent ones.
        
        Args:
            keep_count: Number of recent versions to keep
        """
        versions = self.list_versions()
        
        if len(versions) <= keep_count:
            return
        
        # Get versions to delete (excluding current version)
        versions_to_delete = []
        for version in versions[keep_count:]:
            if not version["is_current"]:
                versions_to_delete.append(version["id"])
        
        # Delete old versions
        for version_id in versions_to_delete:
            self.delete_version(version_id)
        
        self.logger.info(f"Cleaned up {len(versions_to_delete)} old versions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get version manager statistics."""
        versions = self.list_versions()
        
        return {
            "total_versions": len(versions),
            "current_version": self.metadata["current_version"],
            "oldest_version": versions[-1]["created_at"] if versions else None,
            "newest_version": versions[0]["created_at"] if versions else None,
            "version_dir": str(self.version_dir)
        }
    
    def __str__(self) -> str:
        """String representation of the version manager."""
        return f"VersionManager(dir={self.version_dir}, versions={len(self.metadata['versions'])})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the version manager."""
        return f"VersionManager(version_dir='{self.version_dir}', current_version='{self.metadata['current_version']}')"
