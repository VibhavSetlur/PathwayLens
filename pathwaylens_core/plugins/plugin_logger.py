"""
Plugin logger for PathwayLens.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from .base_plugin import BasePlugin


class PluginLogger:
    """Logger for PathwayLens plugins."""
    
    def __init__(self, logging_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin logger.
        
        Args:
            logging_config: Logging configuration dictionary
        """
        self.logger = logger.bind(module="plugin_logger")
        
        # Default logging configuration
        self.default_logging_config = {
            'enabled': True,
            'level': 'INFO',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}',
            'file': 'logs/plugins/plugin.log',
            'max_size': '10MB',
            'retention': '7 days',
            'rotation': '1 day',
            'compression': 'gz',
            'backtrace': True,
            'diagnose': True,
            'enqueue': True,
            'catch': True
        }
        
        # Current logging configuration
        self.logging_config = logging_config or self.default_logging_config.copy()
        
        # Plugin loggers
        self.plugin_loggers: Dict[str, Any] = {}
        
        # Log handlers
        self.log_handlers: List[Any] = []
        
        # Initialize logging
        self._initialize_logging()
    
    def _initialize_logging(self) -> None:
        """Initialize logging system."""
        try:
            if not self.logging_config.get('enabled', True):
                self.logger.info("Plugin logging is disabled")
                return
            
            # Create log directory
            log_file = self.logging_config.get('file', 'logs/plugins/plugin.log')
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure loguru logger
            logger.remove()  # Remove default handler
            
            # Add file handler
            logger.add(
                log_file,
                level=self.logging_config.get('level', 'INFO'),
                format=self.logging_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}'),
                rotation=self.logging_config.get('rotation', '1 day'),
                retention=self.logging_config.get('retention', '7 days'),
                compression=self.logging_config.get('compression', 'gz'),
                backtrace=self.logging_config.get('backtrace', True),
                diagnose=self.logging_config.get('diagnose', True),
                enqueue=self.logging_config.get('enqueue', True),
                catch=self.logging_config.get('catch', True)
            )
            
            # Add console handler
            logger.add(
                sys.stderr,
                level=self.logging_config.get('level', 'INFO'),
                format=self.logging_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}'),
                backtrace=self.logging_config.get('backtrace', True),
                diagnose=self.logging_config.get('diagnose', True),
                enqueue=self.logging_config.get('enqueue', True),
                catch=self.logging_config.get('catch', True)
            )
            
            self.logger.info("Plugin logging initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin logging: {e}")
    
    def get_plugin_logger(self, plugin_name: str) -> Any:
        """
        Get logger for a specific plugin.
        
        Args:
            plugin_name: Name of plugin to get logger for
            
        Returns:
            Plugin logger instance
        """
        try:
            if plugin_name not in self.plugin_loggers:
                # Create plugin-specific logger
                plugin_logger = logger.bind(plugin=plugin_name)
                self.plugin_loggers[plugin_name] = plugin_logger
            
            return self.plugin_loggers[plugin_name]
            
        except Exception as e:
            self.logger.error(f"Failed to get logger for plugin {plugin_name}: {e}")
            return logger
    
    def log_plugin_event(
        self, 
        plugin_name: str, 
        event_type: str, 
        message: str, 
        level: str = 'INFO',
        **kwargs
    ) -> None:
        """
        Log a plugin event.
        
        Args:
            plugin_name: Name of plugin
            event_type: Type of event
            message: Log message
            level: Log level
            **kwargs: Additional log data
        """
        try:
            plugin_logger = self.get_plugin_logger(plugin_name)
            
            # Add event data
            log_data = {
                'plugin': plugin_name,
                'event_type': event_type,
                **kwargs
            }
            
            # Log the event
            if level.upper() == 'DEBUG':
                plugin_logger.debug(message, **log_data)
            elif level.upper() == 'INFO':
                plugin_logger.info(message, **log_data)
            elif level.upper() == 'WARNING':
                plugin_logger.warning(message, **log_data)
            elif level.upper() == 'ERROR':
                plugin_logger.error(message, **log_data)
            elif level.upper() == 'CRITICAL':
                plugin_logger.critical(message, **log_data)
            else:
                plugin_logger.info(message, **log_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log event for plugin {plugin_name}: {e}")
    
    def log_plugin_initialization(self, plugin_name: str, success: bool, message: str = "") -> None:
        """
        Log plugin initialization.
        
        Args:
            plugin_name: Name of plugin
            success: Whether initialization was successful
            message: Additional message
        """
        event_type = 'initialization'
        level = 'INFO' if success else 'ERROR'
        message = f"Plugin {plugin_name} {'initialized successfully' if success else 'initialization failed'}: {message}"
        
        self.log_plugin_event(plugin_name, event_type, message, level, success=success)
    
    def log_plugin_execution(self, plugin_name: str, success: bool, execution_time: float, message: str = "") -> None:
        """
        Log plugin execution.
        
        Args:
            plugin_name: Name of plugin
            success: Whether execution was successful
            execution_time: Execution time in seconds
            message: Additional message
        """
        event_type = 'execution'
        level = 'INFO' if success else 'ERROR'
        message = f"Plugin {plugin_name} {'executed successfully' if success else 'execution failed'} in {execution_time:.2f}s: {message}"
        
        self.log_plugin_event(plugin_name, event_type, message, level, success=success, execution_time=execution_time)
    
    def log_plugin_cleanup(self, plugin_name: str, success: bool, message: str = "") -> None:
        """
        Log plugin cleanup.
        
        Args:
            plugin_name: Name of plugin
            success: Whether cleanup was successful
            message: Additional message
        """
        event_type = 'cleanup'
        level = 'INFO' if success else 'ERROR'
        message = f"Plugin {plugin_name} {'cleaned up successfully' if success else 'cleanup failed'}: {message}"
        
        self.log_plugin_event(plugin_name, event_type, message, level, success=success)
    
    def log_plugin_error(self, plugin_name: str, error: Exception, message: str = "") -> None:
        """
        Log plugin error.
        
        Args:
            plugin_name: Name of plugin
            error: Exception that occurred
            message: Additional message
        """
        event_type = 'error'
        level = 'ERROR'
        message = f"Plugin {plugin_name} error: {message}: {str(error)}"
        
        self.log_plugin_event(plugin_name, event_type, message, level, error=str(error), error_type=type(error).__name__)
    
    def log_plugin_warning(self, plugin_name: str, message: str, **kwargs) -> None:
        """
        Log plugin warning.
        
        Args:
            plugin_name: Name of plugin
            message: Warning message
            **kwargs: Additional log data
        """
        event_type = 'warning'
        level = 'WARNING'
        message = f"Plugin {plugin_name} warning: {message}"
        
        self.log_plugin_event(plugin_name, event_type, message, level, **kwargs)
    
    def log_plugin_info(self, plugin_name: str, message: str, **kwargs) -> None:
        """
        Log plugin info.
        
        Args:
            plugin_name: Name of plugin
            message: Info message
            **kwargs: Additional log data
        """
        event_type = 'info'
        level = 'INFO'
        message = f"Plugin {plugin_name} info: {message}"
        
        self.log_plugin_event(plugin_name, event_type, message, level, **kwargs)
    
    def log_plugin_debug(self, plugin_name: str, message: str, **kwargs) -> None:
        """
        Log plugin debug.
        
        Args:
            plugin_name: Name of plugin
            message: Debug message
            **kwargs: Additional log data
        """
        event_type = 'debug'
        level = 'DEBUG'
        message = f"Plugin {plugin_name} debug: {message}"
        
        self.log_plugin_event(plugin_name, event_type, message, level, **kwargs)
    
    def get_log_file(self, plugin_name: Optional[str] = None) -> str:
        """
        Get log file path.
        
        Args:
            plugin_name: Name of plugin to get log file for (None for main log)
            
        Returns:
            Log file path
        """
        if plugin_name:
            # Plugin-specific log file
            log_file = self.logging_config.get('file', 'logs/plugins/plugin.log')
            log_path = Path(log_file)
            plugin_log_file = log_path.parent / f"{plugin_name}.log"
            return str(plugin_log_file)
        else:
            # Main log file
            return self.logging_config.get('file', 'logs/plugins/plugin.log')
    
    def get_log_level(self) -> str:
        """
        Get current log level.
        
        Returns:
            Current log level
        """
        return self.logging_config.get('level', 'INFO')
    
    def set_log_level(self, level: str) -> None:
        """
        Set log level.
        
        Args:
            level: New log level
        """
        self.logging_config['level'] = level
        self._initialize_logging()
        self.logger.info(f"Log level set to {level}")
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return self.logging_config.copy()
    
    def update_logging_config(self, config: Dict[str, Any]) -> None:
        """
        Update logging configuration.
        
        Args:
            config: New logging configuration
        """
        self.logging_config.update(config)
        self._initialize_logging()
        self.logger.info("Logging configuration updated")
    
    def is_logging_enabled(self) -> bool:
        """
        Check if logging is enabled.
        
        Returns:
            True if logging is enabled, False otherwise
        """
        return self.logging_config.get('enabled', True)
    
    def enable_logging(self) -> None:
        """Enable logging."""
        self.logging_config['enabled'] = True
        self._initialize_logging()
        self.logger.info("Plugin logging enabled")
    
    def disable_logging(self) -> None:
        """Disable logging."""
        self.logging_config['enabled'] = False
        logger.remove()
        self.logger.info("Plugin logging disabled")
    
    def clear_logs(self, plugin_name: Optional[str] = None) -> bool:
        """
        Clear logs.
        
        Args:
            plugin_name: Name of plugin to clear logs for (None for all)
            
        Returns:
            True if clearing successful, False otherwise
        """
        try:
            if plugin_name:
                # Clear plugin-specific log
                plugin_log_file = self.get_log_file(plugin_name)
                if Path(plugin_log_file).exists():
                    Path(plugin_log_file).unlink()
                self.logger.info(f"Cleared logs for plugin: {plugin_name}")
            else:
                # Clear main log
                main_log_file = self.get_log_file()
                if Path(main_log_file).exists():
                    Path(main_log_file).unlink()
                self.logger.info("Cleared all plugin logs")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear logs: {e}")
            return False
    
    def get_log_size(self, plugin_name: Optional[str] = None) -> int:
        """
        Get log file size.
        
        Args:
            plugin_name: Name of plugin to get log size for (None for main log)
            
        Returns:
            Log file size in bytes
        """
        try:
            log_file = self.get_log_file(plugin_name)
            if Path(log_file).exists():
                return Path(log_file).stat().st_size
            return 0
        except Exception as e:
            self.logger.error(f"Failed to get log size: {e}")
            return 0
    
    def get_log_lines(self, plugin_name: Optional[str] = None, lines: int = 100) -> List[str]:
        """
        Get last N lines from log file.
        
        Args:
            plugin_name: Name of plugin to get log lines for (None for main log)
            lines: Number of lines to get
            
        Returns:
            List of log lines
        """
        try:
            log_file = self.get_log_file(plugin_name)
            if not Path(log_file).exists():
                return []
            
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
                return log_lines[-lines:] if len(log_lines) > lines else log_lines
                
        except Exception as e:
            self.logger.error(f"Failed to get log lines: {e}")
            return []
    
    def cleanup(self) -> None:
        """Cleanup logging resources."""
        try:
            # Remove all log handlers
            logger.remove()
            
            # Clear plugin loggers
            self.plugin_loggers.clear()
            
            self.logger.info("Plugin logger cleaned up")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin logger: {e}")
