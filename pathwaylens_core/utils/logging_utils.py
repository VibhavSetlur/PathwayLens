"""
Enhanced logging utilities for PathwayLens.

Provides comprehensive logging configuration and utilities
for better debugging and monitoring.
"""

import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from functools import wraps
import asyncio
from datetime import datetime

from loguru import logger


class LoggingConfig:
    """Logging configuration manager."""
    
    def __init__(
        self,
        level: str = "INFO",
        log_file: Optional[Path] = None,
        console: bool = True,
        file_rotation: str = "10 MB",
        file_retention: str = "30 days",
        file_compression: str = "gz",
        backtrace: bool = True,
        diagnose: bool = True
    ):
        """
        Initialize logging configuration.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file
            console: Enable console logging
            file_rotation: File rotation size
            file_retention: Log retention period
            file_compression: Compression format
            backtrace: Enable backtrace in logs
            diagnose: Enable diagnosis in logs
        """
        self.level = level
        self.log_file = log_file
        self.console = console
        self.file_rotation = file_rotation
        self.file_retention = file_retention
        self.file_compression = file_compression
        self.backtrace = backtrace
        self.diagnose = diagnose
    
    def setup(self):
        """Setup logging configuration."""
        # Remove default handler
        logger.remove()
        
        # Console handler
        if self.console:
            logger.add(
                sys.stderr,
                level=self.level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True,
                backtrace=self.backtrace,
                diagnose=self.diagnose
            )
        
        # File handler
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                str(log_path),
                level="DEBUG",  # Always log DEBUG to file
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation=self.file_rotation,
                retention=self.file_retention,
                compression=self.file_compression,
                backtrace=True,
                diagnose=True
            )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    **kwargs
):
    """
    Setup logging for PathwayLens.
    
    Args:
        level: Log level
        log_file: Path to log file
        console: Enable console logging
        **kwargs: Additional configuration options
    """
    config = LoggingConfig(level=level, log_file=log_file, console=console, **kwargs)
    config.setup()
    return logger


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
    """
    func_logger = logger.bind(function=func.__name__)
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        func_logger.debug(f"Calling {func.__name__} with args={args[:3]}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            func_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            func_logger.error(
                f"{func.__name__} failed: {str(e)}\n{traceback.format_exc()}"
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        func_logger.debug(f"Calling {func.__name__} with args={args[:3]}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            func_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            func_logger.error(
                f"{func.__name__} failed: {str(e)}\n{traceback.format_exc()}"
            )
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log execution time.
    
    Args:
        func: Function to decorate
    """
    func_logger = logger.bind(function=func.__name__)
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            func_logger.info(f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            func_logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}"
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            func_logger.info(f"{func.__name__} executed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            func_logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}"
            )
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class ContextLogger:
    """Logger with context management."""
    
    def __init__(self, module: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize context logger.
        
        Args:
            module: Module name
            context: Additional context
        """
        self.module = module
        self.context = context or {}
        self.logger = logger.bind(module=module, **self.context)
    
    def add_context(self, **kwargs):
        """Add context to logger."""
        self.context.update(kwargs)
        self.logger = logger.bind(module=self.module, **self.context)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)


def get_module_logger(module: str, context: Optional[Dict[str, Any]] = None) -> ContextLogger:
    """
    Get logger for a module.
    
    Args:
        module: Module name
        context: Additional context
        
    Returns:
        ContextLogger instance
    """
    return ContextLogger(module, context)



