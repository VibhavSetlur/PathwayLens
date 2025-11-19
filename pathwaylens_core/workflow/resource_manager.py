"""
Resource management for workflows.

Tracks and limits resource usage.
"""

import psutil
import os
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ResourceLimits:
    """Resource limits for workflow execution."""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_disk_mb: Optional[int] = None
    timeout_seconds: Optional[int] = None


@dataclass
class ResourceUsage:
    """Current resource usage."""
    memory_mb: float
    cpu_percent: float
    disk_mb: float


class ResourceManager:
    """Manage workflow resources."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize resource manager.
        
        Args:
            limits: Resource limits
        """
        self.limits = limits or ResourceLimits()
        self.logger = logger.bind(module="resource_manager")
        self.process = psutil.Process(os.getpid())
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent(interval=0.1)
        
        # Get disk usage (simplified)
        disk_usage = psutil.disk_usage('/')
        
        return ResourceUsage(
            memory_mb=memory_info.rss / 1024 / 1024,
            cpu_percent=cpu_percent,
            disk_mb=disk_usage.used / 1024 / 1024
        )
    
    def check_limits(self) -> tuple[bool, Optional[str]]:
        """
        Check if current usage exceeds limits.
        
        Returns:
            Tuple of (within_limits, error_message)
        """
        usage = self.get_current_usage()
        
        if self.limits.max_memory_mb and usage.memory_mb > self.limits.max_memory_mb:
            return False, f"Memory limit exceeded: {usage.memory_mb:.1f}MB > {self.limits.max_memory_mb}MB"
        
        if self.limits.max_cpu_percent and usage.cpu_percent > self.limits.max_cpu_percent:
            return False, f"CPU limit exceeded: {usage.cpu_percent:.1f}% > {self.limits.max_cpu_percent}%"
        
        if self.limits.max_disk_mb and usage.disk_mb > self.limits.max_disk_mb:
            return False, f"Disk limit exceeded: {usage.disk_mb:.1f}MB > {self.limits.max_disk_mb}MB"
        
        return True, None
    
    def set_limits(self, limits: ResourceLimits):
        """Update resource limits."""
        self.limits = limits
        self.logger.info(f"Updated resource limits: {limits}")



