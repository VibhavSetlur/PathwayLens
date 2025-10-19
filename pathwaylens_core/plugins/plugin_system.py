"""
Plugin system for PathwayLens.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from .base_plugin import BasePlugin
from .plugin_manager import PluginManager
from .plugin_loader import PluginLoader
from .plugin_registry import PluginRegistry
from .plugin_validator import PluginValidator
from .plugin_config import PluginConfig
from .plugin_security import PluginSecurity
from .plugin_monitor import PluginMonitor
from .plugin_logger import PluginLogger
from .plugin_factory import PluginFactory


class PluginSystem:
    """Main plugin system for PathwayLens."""
    
    def __init__(self, plugin_directory: Optional[str] = None):
        """
        Initialize the plugin system.
        
        Args:
            plugin_directory: Directory containing plugins
        """
        self.logger = logger.bind(module="plugin_system")
        self.plugin_directory = plugin_directory or "plugins"
        
        # Initialize plugin system components
        self.plugin_manager = PluginManager(self.plugin_directory)
        self.plugin_loader = PluginLoader(self.plugin_directory)
        self.plugin_registry = PluginRegistry()
        self.plugin_validator = PluginValidator()
        self.plugin_config = PluginConfig()
        self.plugin_security = PluginSecurity()
        self.plugin_monitor = PluginMonitor()
        self.plugin_logger = PluginLogger()
        self.plugin_factory = PluginFactory(self.plugin_directory)
        
        # System state
        self.initialized = False
        self.plugins_loaded = False
    
    async def initialize(self) -> bool:
        """
        Initialize the plugin system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if self.initialized:
                self.logger.warning("Plugin system already initialized")
                return True
            
            self.logger.info("Initializing plugin system")
            
            # Initialize plugin components
            await self._initialize_components()
            
            # Load plugins
            await self._load_plugins()
            
            # Validate plugins
            await self._validate_plugins()
            
            # Setup security
            await self._setup_security()
            
            # Setup monitoring
            await self._setup_monitoring()
            
            # Setup logging
            await self._setup_logging()
            
            self.initialized = True
            self.logger.info("Plugin system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin system: {e}")
            return False
    
    async def _initialize_components(self) -> None:
        """Initialize plugin system components."""
        try:
            # Initialize plugin manager
            await self.plugin_manager.load_plugins()
            
            # Initialize plugin loader
            await self.plugin_loader.load_plugins()
            
            # Initialize plugin factory
            await self.plugin_factory.discover_plugins()
            
            self.logger.info("Plugin system components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin system components: {e}")
            raise
    
    async def _load_plugins(self) -> None:
        """Load plugins into the system."""
        try:
            # Load plugins using the plugin manager
            for plugin_name, plugin in self.plugin_manager.plugins.items():
                # Register plugin in registry
                await self.plugin_registry.register_plugin(plugin)
                
                # Activate plugin if enabled
                if self.plugin_config.is_plugin_enabled(plugin_name):
                    await self.plugin_registry.activate_plugin(plugin_name)
            
            self.plugins_loaded = True
            self.logger.info(f"Loaded {len(self.plugin_manager)} plugins")
            
        except Exception as e:
            self.logger.error(f"Failed to load plugins: {e}")
            raise
    
    async def _validate_plugins(self) -> None:
        """Validate loaded plugins."""
        try:
            for plugin_name, plugin in self.plugin_manager.plugins.items():
                # Validate plugin
                validation_result = await self.plugin_validator.validate_plugin(plugin)
                
                if not validation_result['valid']:
                    self.logger.warning(f"Plugin {plugin_name} validation failed: {validation_result['errors']}")
                
                # Validate plugin security
                security_result = await self.plugin_security.validate_plugin_security(plugin)
                
                if not security_result['secure']:
                    self.logger.warning(f"Plugin {plugin_name} security validation failed: {security_result['violations']}")
            
            self.logger.info("Plugin validation completed")
            
        except Exception as e:
            self.logger.error(f"Failed to validate plugins: {e}")
            raise
    
    async def _setup_security(self) -> None:
        """Setup plugin security."""
        try:
            # Get security configuration
            security_config = self.plugin_config.get_security_config()
            
            # Update security configuration
            self.plugin_security.update_security_config(security_config)
            
            self.logger.info("Plugin security setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup plugin security: {e}")
            raise
    
    async def _setup_monitoring(self) -> None:
        """Setup plugin monitoring."""
        try:
            # Get monitoring configuration
            monitoring_config = self.plugin_config.get_config('monitoring')
            
            if monitoring_config:
                # Update monitoring configuration
                self.plugin_monitor.update_monitoring_config(monitoring_config)
            
            self.logger.info("Plugin monitoring setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup plugin monitoring: {e}")
            raise
    
    async def _setup_logging(self) -> None:
        """Setup plugin logging."""
        try:
            # Get logging configuration
            logging_config = self.plugin_config.get_logging_config()
            
            if logging_config:
                # Update logging configuration
                self.plugin_logger.update_logging_config(logging_config)
            
            self.logger.info("Plugin logging setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup plugin logging: {e}")
            raise
    
    async def execute_plugin(
        self, 
        plugin_name: str, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a plugin.
        
        Args:
            plugin_name: Name of plugin to execute
            input_data: Input data for the plugin
            parameters: Optional parameters for execution
            
        Returns:
            Plugin execution result
        """
        try:
            if not self.initialized:
                raise RuntimeError("Plugin system not initialized")
            
            # Check if plugin is enabled
            if not self.plugin_config.is_plugin_enabled(plugin_name):
                raise ValueError(f"Plugin {plugin_name} is disabled")
            
            # Check if plugin is secure
            security_status = self.plugin_security.get_security_status(plugin_name)
            if security_status and not security_status.get('secure', False):
                raise ValueError(f"Plugin {plugin_name} failed security validation")
            
            # Start monitoring if enabled
            if self.plugin_monitor.is_monitoring_enabled():
                await self.plugin_monitor.start_monitoring(self.plugin_manager.get_plugin(plugin_name))
            
            # Execute plugin
            result = await self.plugin_manager.execute_plugin(plugin_name, input_data, parameters)
            
            # Stop monitoring
            if self.plugin_monitor.is_monitoring_enabled():
                await self.plugin_monitor.stop_monitoring(plugin_name)
            
            # Log plugin execution
            self.plugin_logger.log_plugin_execution(plugin_name, True, 0.0, "Plugin executed successfully")
            
            return result
            
        except Exception as e:
            # Log plugin error
            self.plugin_logger.log_plugin_error(plugin_name, e, "Plugin execution failed")
            
            # Stop monitoring
            if self.plugin_monitor.is_monitoring_enabled():
                await self.plugin_monitor.stop_monitoring(plugin_name)
            
            raise
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of plugin to get
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugin_manager.get_plugin(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """
        List all plugin names.
        
        Returns:
            List of plugin names
        """
        return self.plugin_manager.list_plugins()
    
    def list_plugins_by_category(self, category: str) -> List[str]:
        """
        List plugins by category.
        
        Args:
            category: Plugin category
            
        Returns:
            List of plugin names in category
        """
        return self.plugin_manager.list_plugins_by_category(category)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get plugin information.
        
        Args:
            plugin_name: Name of plugin to get info for
            
        Returns:
            Plugin information or None if not found
        """
        return self.plugin_manager.get_plugin_info(plugin_name)
    
    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all plugins.
        
        Returns:
            Dictionary with plugin information
        """
        return self.plugin_manager.get_all_plugin_info()
    
    def get_plugin_categories(self) -> Dict[str, List[str]]:
        """
        Get plugin categories.
        
        Returns:
            Dictionary with plugin categories
        """
        return self.plugin_manager.get_categories()
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """
        Get plugin statistics.
        
        Returns:
            Plugin statistics dictionary
        """
        return self.plugin_factory.get_plugin_statistics()
    
    def get_validation_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get plugin validation results.
        
        Returns:
            Dictionary with validation results
        """
        return self.plugin_validator.get_validation_results()
    
    def get_security_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get plugin security status.
        
        Returns:
            Dictionary with security status
        """
        return self.plugin_security.get_all_security_status()
    
    def get_monitoring_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get plugin monitoring data.
        
        Returns:
            Dictionary with monitoring data
        """
        return self.plugin_monitor.get_monitoring_data()
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get plugin alerts.
        
        Returns:
            List of alerts
        """
        return self.plugin_monitor.get_alerts()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get plugin system status.
        
        Returns:
            System status dictionary
        """
        return {
            'initialized': self.initialized,
            'plugins_loaded': self.plugins_loaded,
            'total_plugins': len(self.plugin_manager),
            'active_plugins': len(self.plugin_registry.get_active_plugins()),
            'inactive_plugins': len(self.plugin_registry.get_inactive_plugins()),
            'secure_plugins': len(self.plugin_security.get_secure_plugins()),
            'insecure_plugins': len(self.plugin_security.get_insecure_plugins()),
            'monitoring_active': self.plugin_monitor.monitoring_active,
            'total_alerts': len(self.plugin_monitor.get_alerts()),
            'unacknowledged_alerts': len([a for a in self.plugin_monitor.get_alerts() if not a['acknowledged']])
        }
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if reload successful, False otherwise
        """
        try:
            if not self.initialized:
                raise RuntimeError("Plugin system not initialized")
            
            # Reload plugin using the plugin manager
            success = await self.plugin_manager.unregister_plugin(plugin_name)
            if not success:
                return False
            
            # Reload plugin using the plugin loader
            success = await self.plugin_loader.reload_plugin(plugin_name)
            if not success:
                return False
            
            # Re-register plugin in registry
            plugin = self.plugin_manager.get_plugin(plugin_name)
            if plugin:
                await self.plugin_registry.register_plugin(plugin)
                
                # Activate plugin if enabled
                if self.plugin_config.is_plugin_enabled(plugin_name):
                    await self.plugin_registry.activate_plugin(plugin_name)
            
            self.logger.info(f"Reloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload plugin {plugin_name}: {e}")
            return False
    
    async def reload_all_plugins(self) -> bool:
        """
        Reload all plugins.
        
        Returns:
            True if reload successful, False otherwise
        """
        try:
            if not self.initialized:
                raise RuntimeError("Plugin system not initialized")
            
            # Reload all plugins using the plugin manager
            success = await self.plugin_manager.cleanup_all_plugins()
            if not success:
                return False
            
            # Reload all plugins using the plugin loader
            success = await self.plugin_loader.reload_all_plugins()
            if not success:
                return False
            
            # Re-register all plugins in registry
            for plugin_name, plugin in self.plugin_manager.plugins.items():
                await self.plugin_registry.register_plugin(plugin)
                
                # Activate plugin if enabled
                if self.plugin_config.is_plugin_enabled(plugin_name):
                    await self.plugin_registry.activate_plugin(plugin_name)
            
            self.logger.info("Reloaded all plugins")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload all plugins: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """
        Cleanup plugin system.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if not self.initialized:
                self.logger.warning("Plugin system not initialized")
                return True
            
            self.logger.info("Cleaning up plugin system")
            
            # Cleanup plugin components
            await self.plugin_manager.cleanup_all_plugins()
            await self.plugin_loader.cleanup()
            await self.plugin_registry.cleanup()
            await self.plugin_monitor.cleanup()
            self.plugin_logger.cleanup()
            
            # Reset system state
            self.initialized = False
            self.plugins_loaded = False
            
            self.logger.info("Plugin system cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin system: {e}")
            return False
    
    def __len__(self) -> int:
        """Get number of plugins."""
        return len(self.plugin_manager)
    
    def __contains__(self, plugin_name: str) -> bool:
        """Check if plugin exists."""
        return plugin_name in self.plugin_manager
    
    def __iter__(self):
        """Iterate over plugins."""
        return iter(self.plugin_manager)
