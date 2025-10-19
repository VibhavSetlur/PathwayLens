"""
Plugin factory for PathwayLens.
"""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Callable
from loguru import logger

from .base_plugin import BasePlugin
from .plugin_utils import PluginUtils


class PluginFactory:
    """Factory for creating and managing PathwayLens plugins."""
    
    def __init__(self, plugin_directory: Optional[str] = None):
        """
        Initialize the plugin factory.
        
        Args:
            plugin_directory: Directory containing plugins
        """
        self.logger = logger.bind(module="plugin_factory")
        self.plugin_directory = plugin_directory or "plugins"
        
        # Plugin registry
        self.plugin_registry: Dict[str, Type[BasePlugin]] = {}
        
        # Plugin instances
        self.plugin_instances: Dict[str, BasePlugin] = {}
        
        # Plugin categories
        self.plugin_categories: Dict[str, List[str]] = {
            'analysis': [],
            'visualization': [],
            'data_processing': [],
            'export': [],
            'import': [],
            'custom': []
        }
    
    async def discover_plugins(self, plugin_directory: Optional[str] = None) -> bool:
        """
        Discover plugins in the plugin directory.
        
        Args:
            plugin_directory: Directory to discover plugins in
            
        Returns:
            True if discovery successful, False otherwise
        """
        if plugin_directory:
            self.plugin_directory = plugin_directory
        
        self.logger.info(f"Discovering plugins in {self.plugin_directory}")
        
        try:
            # Load plugin classes
            plugin_classes = PluginUtils.load_plugin_from_directory(self.plugin_directory)
            
            # Register plugin classes
            for plugin_class in plugin_classes:
                await self._register_plugin_class(plugin_class)
            
            self.logger.info(f"Discovered {len(self.plugin_registry)} plugins")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to discover plugins: {e}")
            return False
    
    async def _register_plugin_class(self, plugin_class: Type[BasePlugin]) -> bool:
        """Register a plugin class."""
        try:
            # Validate plugin class
            validation_result = PluginUtils.validate_plugin_class(plugin_class)
            if not validation_result['valid']:
                self.logger.error(f"Plugin class validation failed: {validation_result['errors']}")
                return False
            
            # Create plugin instance to get metadata
            plugin_instance = plugin_class()
            plugin_name = plugin_instance.name
            
            # Check if plugin is already registered
            if plugin_name in self.plugin_registry:
                self.logger.warning(f"Plugin {plugin_name} is already registered")
                return False
            
            # Register plugin class
            self.plugin_registry[plugin_name] = plugin_class
            
            # Categorize plugin
            await self._categorize_plugin(plugin_instance)
            
            self.logger.info(f"Registered plugin class: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin class: {e}")
            return False
    
    async def _categorize_plugin(self, plugin_instance: BasePlugin) -> None:
        """Categorize a plugin based on its tags and metadata."""
        try:
            plugin_name = plugin_instance.name
            tags = plugin_instance.get_tags()
            
            # Categorize based on tags
            categorized = False
            for tag in tags:
                if tag in self.plugin_categories:
                    self.plugin_categories[tag].append(plugin_name)
                    categorized = True
                    break
            
            # Default to custom category if not categorized
            if not categorized:
                self.plugin_categories['custom'].append(plugin_name)
            
        except Exception as e:
            self.logger.error(f"Failed to categorize plugin {plugin_name}: {e}")
    
    async def create_plugin(self, plugin_name: str, **kwargs) -> Optional[BasePlugin]:
        """
        Create a plugin instance.
        
        Args:
            plugin_name: Name of plugin to create
            **kwargs: Arguments for plugin constructor
            
        Returns:
            Plugin instance or None if creation failed
        """
        try:
            if plugin_name not in self.plugin_registry:
                self.logger.error(f"Plugin {plugin_name} not found in registry")
                return None
            
            # Get plugin class
            plugin_class = self.plugin_registry[plugin_name]
            
            # Create plugin instance
            plugin_instance = PluginUtils.create_plugin_instance(plugin_class, **kwargs)
            if not plugin_instance:
                self.logger.error(f"Failed to create plugin instance: {plugin_name}")
                return None
            
            # Store plugin instance
            self.plugin_instances[plugin_name] = plugin_instance
            
            self.logger.info(f"Created plugin instance: {plugin_name}")
            return plugin_instance
            
        except Exception as e:
            self.logger.error(f"Failed to create plugin {plugin_name}: {e}")
            return None
    
    async def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Get a plugin instance.
        
        Args:
            plugin_name: Name of plugin to get
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugin_instances.get(plugin_name)
    
    async def remove_plugin(self, plugin_name: str) -> bool:
        """
        Remove a plugin instance.
        
        Args:
            plugin_name: Name of plugin to remove
            
        Returns:
            True if removal successful, False otherwise
        """
        try:
            if plugin_name not in self.plugin_instances:
                self.logger.warning(f"Plugin {plugin_name} not found in instances")
                return False
            
            # Get plugin instance
            plugin_instance = self.plugin_instances[plugin_name]
            
            # Cleanup plugin if initialized
            if plugin_instance.initialized:
                await plugin_instance.cleanup()
            
            # Remove from instances
            del self.plugin_instances[plugin_name]
            
            self.logger.info(f"Removed plugin instance: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove plugin {plugin_name}: {e}")
            return False
    
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
            # Get or create plugin instance
            plugin_instance = await self.get_plugin(plugin_name)
            if not plugin_instance:
                plugin_instance = await self.create_plugin(plugin_name)
                if not plugin_instance:
                    raise ValueError(f"Failed to create plugin instance: {plugin_name}")
            
            # Initialize plugin if not already initialized
            if not plugin_instance.initialized:
                success = await plugin_instance.initialize()
                if not success:
                    raise RuntimeError(f"Failed to initialize plugin: {plugin_name}")
            
            # Execute plugin
            result = await plugin_instance.execute(input_data, parameters)
            
            self.logger.info(f"Executed plugin: {plugin_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute plugin {plugin_name}: {e}")
            raise
    
    def list_plugins(self) -> List[str]:
        """
        List all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self.plugin_registry.keys())
    
    def list_plugin_instances(self) -> List[str]:
        """
        List all plugin instance names.
        
        Returns:
            List of plugin instance names
        """
        return list(self.plugin_instances.keys())
    
    def list_plugins_by_category(self, category: str) -> List[str]:
        """
        List plugins by category.
        
        Args:
            category: Plugin category
            
        Returns:
            List of plugin names in category
        """
        return self.plugin_categories.get(category, [])
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get plugin information.
        
        Args:
            plugin_name: Name of plugin to get info for
            
        Returns:
            Plugin information or None if not found
        """
        if plugin_name in self.plugin_registry:
            plugin_class = self.plugin_registry[plugin_name]
            return PluginUtils.get_plugin_info(plugin_class)
        return None
    
    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all plugins.
        
        Returns:
            Dictionary with plugin information
        """
        return {name: self.get_plugin_info(name) for name in self.plugin_registry.keys()}
    
    def get_plugin_categories(self) -> Dict[str, List[str]]:
        """
        Get plugin categories.
        
        Returns:
            Dictionary with plugin categories
        """
        return self.plugin_categories.copy()
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """
        Get plugin statistics.
        
        Returns:
            Plugin statistics dictionary
        """
        try:
            plugin_classes = list(self.plugin_registry.values())
            return PluginUtils.get_plugin_statistics(plugin_classes)
        except Exception as e:
            self.logger.error(f"Failed to get plugin statistics: {e}")
            return {}
    
    async def cleanup_all_plugins(self) -> bool:
        """
        Cleanup all plugin instances.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            for plugin_name, plugin_instance in self.plugin_instances.items():
                try:
                    if plugin_instance.initialized:
                        await plugin_instance.cleanup()
                except Exception as e:
                    self.logger.error(f"Failed to cleanup plugin {plugin_name}: {e}")
            
            self.plugin_instances.clear()
            
            self.logger.info("Cleaned up all plugin instances")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup all plugins: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if reload successful, False otherwise
        """
        try:
            if plugin_name not in self.plugin_registry:
                self.logger.error(f"Plugin {plugin_name} not found in registry")
                return False
            
            # Remove existing plugin instance
            await self.remove_plugin(plugin_name)
            
            # Reload plugin class
            plugin_class = self.plugin_registry[plugin_name]
            
            # Re-import the module
            module = inspect.getmodule(plugin_class)
            if module:
                importlib.reload(module)
            
            # Re-register plugin class
            await self._register_plugin_class(plugin_class)
            
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
            # Cleanup all plugin instances
            await self.cleanup_all_plugins()
            
            # Clear plugin registry
            self.plugin_registry.clear()
            
            # Clear plugin categories
            for category_plugins in self.plugin_categories.values():
                category_plugins.clear()
            
            # Rediscover plugins
            success = await self.discover_plugins()
            
            if success:
                self.logger.info("Reloaded all plugins")
                return True
            else:
                self.logger.error("Failed to reload all plugins")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to reload all plugins: {e}")
            return False
    
    def export_plugin_info(self, output_file: str) -> bool:
        """
        Export plugin information to file.
        
        Args:
            output_file: Path to output file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            plugin_classes = list(self.plugin_registry.values())
            return PluginUtils.export_plugin_info(plugin_classes, output_file)
        except Exception as e:
            self.logger.error(f"Failed to export plugin info: {e}")
            return False
    
    def __len__(self) -> int:
        """Get number of registered plugins."""
        return len(self.plugin_registry)
    
    def __contains__(self, plugin_name: str) -> bool:
        """Check if plugin is registered."""
        return plugin_name in self.plugin_registry
    
    def __iter__(self):
        """Iterate over registered plugins."""
        return iter(self.plugin_registry.keys())
