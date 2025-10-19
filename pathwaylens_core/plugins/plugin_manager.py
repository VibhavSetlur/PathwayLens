"""
Plugin manager for PathwayLens.
"""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union
from loguru import logger

from .base_plugin import BasePlugin


class PluginManager:
    """Manages PathwayLens plugins."""
    
    def __init__(self, plugin_directory: Optional[str] = None):
        """
        Initialize the plugin manager.
        
        Args:
            plugin_directory: Directory containing plugins
        """
        self.logger = logger.bind(module="plugin_manager")
        self.plugin_directory = plugin_directory or "plugins"
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_classes: Dict[str, Type[BasePlugin]] = {}
        
        # Plugin categories
        self.categories = {
            'analysis': [],
            'visualization': [],
            'data_processing': [],
            'export': [],
            'import': [],
            'custom': []
        }
    
    async def load_plugins(self, plugin_directory: Optional[str] = None) -> bool:
        """
        Load all plugins from the plugin directory.
        
        Args:
            plugin_directory: Directory containing plugins
            
        Returns:
            True if loading successful, False otherwise
        """
        if plugin_directory:
            self.plugin_directory = plugin_directory
        
        self.logger.info(f"Loading plugins from {self.plugin_directory}")
        
        try:
            plugin_path = Path(self.plugin_directory)
            if not plugin_path.exists():
                self.logger.warning(f"Plugin directory {self.plugin_directory} does not exist")
                return False
            
            # Load plugins from Python files
            for plugin_file in plugin_path.glob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue
                
                await self._load_plugin_from_file(plugin_file)
            
            # Load plugins from subdirectories
            for plugin_dir in plugin_path.iterdir():
                if plugin_dir.is_dir() and not plugin_dir.name.startswith("__"):
                    await self._load_plugin_from_directory(plugin_dir)
            
            self.logger.info(f"Loaded {len(self.plugins)} plugins")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugins: {e}")
            return False
    
    async def _load_plugin_from_file(self, plugin_file: Path) -> bool:
        """Load plugin from a Python file."""
        try:
            # Import the module
            module_name = plugin_file.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    
                    await self._register_plugin_class(name, obj)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_file}: {e}")
            return False
    
    async def _load_plugin_from_directory(self, plugin_dir: Path) -> bool:
        """Load plugin from a directory."""
        try:
            # Look for __init__.py or main plugin file
            init_file = plugin_dir / "__init__.py"
            main_file = plugin_dir / f"{plugin_dir.name}.py"
            
            if init_file.exists():
                await self._load_plugin_from_file(init_file)
            elif main_file.exists():
                await self._load_plugin_from_file(main_file)
            else:
                self.logger.warning(f"No plugin file found in {plugin_dir}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_dir}: {e}")
            return False
    
    async def _register_plugin_class(self, name: str, plugin_class: Type[BasePlugin]) -> bool:
        """Register a plugin class."""
        try:
            # Create plugin instance
            plugin_instance = plugin_class()
            
            # Initialize plugin
            if await plugin_instance.initialize():
                self.plugins[plugin_instance.name] = plugin_instance
                self.plugin_classes[name] = plugin_class
                
                # Categorize plugin
                await self._categorize_plugin(plugin_instance)
                
                self.logger.info(f"Registered plugin: {plugin_instance.name}")
                return True
            else:
                self.logger.warning(f"Failed to initialize plugin: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to register plugin {name}: {e}")
            return False
    
    async def _categorize_plugin(self, plugin: BasePlugin) -> None:
        """Categorize a plugin based on its tags and metadata."""
        tags = plugin.get_tags()
        
        # Categorize based on tags
        for tag in tags:
            if tag in self.categories:
                self.categories[tag].append(plugin.name)
                break
        else:
            # Default to custom category
            self.categories['custom'].append(plugin.name)
    
    async def register_plugin(self, plugin: BasePlugin) -> bool:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if plugin.name in self.plugins:
                self.logger.warning(f"Plugin {plugin.name} already registered")
                return False
            
            # Initialize plugin
            if await plugin.initialize():
                self.plugins[plugin.name] = plugin
                await self._categorize_plugin(plugin)
                
                self.logger.info(f"Registered plugin: {plugin.name}")
                return True
            else:
                self.logger.warning(f"Failed to initialize plugin: {plugin.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin.name}: {e}")
            return False
    
    async def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if plugin_name not in self.plugins:
                self.logger.warning(f"Plugin {plugin_name} not registered")
                return False
            
            plugin = self.plugins[plugin_name]
            
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from plugins
            del self.plugins[plugin_name]
            
            # Remove from categories
            for category_plugins in self.categories.values():
                if plugin_name in category_plugins:
                    category_plugins.remove(plugin_name)
            
            self.logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
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
            if plugin_name not in self.plugins:
                raise ValueError(f"Plugin {plugin_name} not found")
            
            plugin = self.plugins[plugin_name]
            
            # Validate parameters
            if parameters and not plugin.validate_parameters(parameters):
                raise ValueError(f"Invalid parameters for plugin {plugin_name}")
            
            # Execute plugin
            result = await plugin.execute(input_data, parameters)
            
            self.logger.info(f"Executed plugin: {plugin_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute plugin {plugin_name}: {e}")
            raise
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of plugin to get
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """
        List all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
    
    def list_plugins_by_category(self, category: str) -> List[str]:
        """
        List plugins by category.
        
        Args:
            category: Plugin category
            
        Returns:
            List of plugin names in category
        """
        return self.categories.get(category, [])
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get plugin information.
        
        Args:
            plugin_name: Name of plugin to get info for
            
        Returns:
            Plugin information or None if not found
        """
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.get_info()
        return None
    
    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all plugins.
        
        Returns:
            Dictionary with plugin information
        """
        return {name: plugin.get_info() for name, plugin in self.plugins.items()}
    
    def get_categories(self) -> Dict[str, List[str]]:
        """
        Get plugin categories.
        
        Returns:
            Dictionary with plugin categories
        """
        return self.categories.copy()
    
    async def cleanup_all_plugins(self) -> bool:
        """
        Cleanup all plugins.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            for plugin_name, plugin in self.plugins.items():
                try:
                    await plugin.cleanup()
                except Exception as e:
                    self.logger.error(f"Failed to cleanup plugin {plugin_name}: {e}")
            
            self.plugins.clear()
            self.plugin_classes.clear()
            
            # Clear categories
            for category_plugins in self.categories.values():
                category_plugins.clear()
            
            self.logger.info("Cleaned up all plugins")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugins: {e}")
            return False
    
    def __len__(self) -> int:
        """Get number of registered plugins."""
        return len(self.plugins)
    
    def __contains__(self, plugin_name: str) -> bool:
        """Check if plugin is registered."""
        return plugin_name in self.plugins
    
    def __iter__(self):
        """Iterate over registered plugins."""
        return iter(self.plugins.values())
