"""
Plugin loader for PathwayLens.
"""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union
from loguru import logger

from .base_plugin import BasePlugin
from .plugin_manager import PluginManager


class PluginLoader:
    """Loads and manages PathwayLens plugins."""
    
    def __init__(self, plugin_directory: Optional[str] = None):
        """
        Initialize the plugin loader.
        
        Args:
            plugin_directory: Directory containing plugins
        """
        self.logger = logger.bind(module="plugin_loader")
        self.plugin_directory = plugin_directory or "plugins"
        self.plugin_manager = PluginManager(plugin_directory)
        
        # Plugin loading options
        self.loading_options = {
            'auto_load': True,
            'load_subdirectories': True,
            'validate_plugins': True,
            'initialize_plugins': True
        }
    
    async def load_plugins(
        self, 
        plugin_directory: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Load all plugins from the plugin directory.
        
        Args:
            plugin_directory: Directory containing plugins
            options: Loading options
            
        Returns:
            True if loading successful, False otherwise
        """
        if plugin_directory:
            self.plugin_directory = plugin_directory
        
        if options:
            self.loading_options.update(options)
        
        self.logger.info(f"Loading plugins from {self.plugin_directory}")
        
        try:
            # Load plugins using the plugin manager
            success = await self.plugin_manager.load_plugins(self.plugin_directory)
            
            if success:
                self.logger.info(f"Successfully loaded {len(self.plugin_manager)} plugins")
                
                # Validate plugins if requested
                if self.loading_options.get('validate_plugins', True):
                    await self._validate_loaded_plugins()
                
                # Initialize plugins if requested
                if self.loading_options.get('initialize_plugins', True):
                    await self._initialize_loaded_plugins()
                
                return True
            else:
                self.logger.error("Failed to load plugins")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load plugins: {e}")
            return False
    
    async def _validate_loaded_plugins(self) -> None:
        """Validate loaded plugins."""
        self.logger.info("Validating loaded plugins")
        
        for plugin_name, plugin in self.plugin_manager.plugins.items():
            try:
                # Check plugin compatibility
                if not plugin.is_compatible("2.0.0"):  # Example version
                    self.logger.warning(f"Plugin {plugin_name} may not be compatible")
                
                # Check plugin metadata
                metadata = plugin.get_metadata()
                if not metadata.get('name') or not metadata.get('version'):
                    self.logger.warning(f"Plugin {plugin_name} has incomplete metadata")
                
                # Check plugin dependencies
                dependencies = plugin.get_dependencies()
                if dependencies:
                    self.logger.info(f"Plugin {plugin_name} has dependencies: {dependencies}")
                
            except Exception as e:
                self.logger.error(f"Failed to validate plugin {plugin_name}: {e}")
    
    async def _initialize_loaded_plugins(self) -> None:
        """Initialize loaded plugins."""
        self.logger.info("Initializing loaded plugins")
        
        for plugin_name, plugin in self.plugin_manager.plugins.items():
            try:
                if not plugin.initialized:
                    success = await plugin.initialize()
                    if success:
                        self.logger.info(f"Successfully initialized plugin: {plugin_name}")
                    else:
                        self.logger.warning(f"Failed to initialize plugin: {plugin_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
    
    async def load_plugin_from_file(self, plugin_file: str) -> bool:
        """
        Load a plugin from a specific file.
        
        Args:
            plugin_file: Path to plugin file
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            plugin_path = Path(plugin_file)
            if not plugin_path.exists():
                self.logger.error(f"Plugin file {plugin_file} does not exist")
                return False
            
            # Load plugin using the plugin manager
            success = await self.plugin_manager._load_plugin_from_file(plugin_path)
            
            if success:
                self.logger.info(f"Successfully loaded plugin from {plugin_file}")
                return True
            else:
                self.logger.error(f"Failed to load plugin from {plugin_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_file}: {e}")
            return False
    
    async def load_plugin_from_directory(self, plugin_directory: str) -> bool:
        """
        Load a plugin from a specific directory.
        
        Args:
            plugin_directory: Path to plugin directory
            
        Returns:
            True if loading successful, False otherwise
        """
        try:
            plugin_path = Path(plugin_directory)
            if not plugin_path.exists() or not plugin_path.is_dir():
                self.logger.error(f"Plugin directory {plugin_directory} does not exist or is not a directory")
                return False
            
            # Load plugin using the plugin manager
            success = await self.plugin_manager._load_plugin_from_directory(plugin_path)
            
            if success:
                self.logger.info(f"Successfully loaded plugin from {plugin_directory}")
                return True
            else:
                self.logger.error(f"Failed to load plugin from {plugin_directory}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {plugin_directory}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a specific plugin.
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if reload successful, False otherwise
        """
        try:
            if plugin_name not in self.plugin_manager.plugins:
                self.logger.error(f"Plugin {plugin_name} not found")
                return False
            
            # Unregister and re-register plugin
            await self.plugin_manager.unregister_plugin(plugin_name)
            
            # Find and reload plugin file
            plugin_path = Path(self.plugin_directory)
            for plugin_file in plugin_path.glob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue
                
                # Import and check if this file contains the plugin
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Check if this module contains the plugin
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BasePlugin) and 
                        obj != BasePlugin and
                        obj().name == plugin_name):
                        
                        # Re-register plugin
                        await self.plugin_manager._register_plugin_class(name, obj)
                        self.logger.info(f"Successfully reloaded plugin: {plugin_name}")
                        return True
            
            self.logger.error(f"Could not find plugin file for {plugin_name}")
            return False
            
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
            self.logger.info("Reloading all plugins")
            
            # Get list of plugin names
            plugin_names = list(self.plugin_manager.plugins.keys())
            
            # Unregister all plugins
            await self.plugin_manager.cleanup_all_plugins()
            
            # Reload all plugins
            success = await self.load_plugins()
            
            if success:
                self.logger.info(f"Successfully reloaded {len(plugin_names)} plugins")
                return True
            else:
                self.logger.error("Failed to reload plugins")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to reload all plugins: {e}")
            return False
    
    def get_plugin_manager(self) -> PluginManager:
        """
        Get the plugin manager instance.
        
        Returns:
            PluginManager instance
        """
        return self.plugin_manager
    
    def get_loaded_plugins(self) -> Dict[str, BasePlugin]:
        """
        Get all loaded plugins.
        
        Returns:
            Dictionary of loaded plugins
        """
        return self.plugin_manager.plugins.copy()
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific plugin.
        
        Args:
            plugin_name: Name of plugin to get info for
            
        Returns:
            Plugin information or None if not found
        """
        return self.plugin_manager.get_plugin_info(plugin_name)
    
    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all loaded plugins.
        
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
    
    async def cleanup(self) -> bool:
        """
        Cleanup plugin loader and all plugins.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info("Cleaning up plugin loader")
            
            # Cleanup all plugins
            await self.plugin_manager.cleanup_all_plugins()
            
            self.logger.info("Plugin loader cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin loader: {e}")
            return False
    
    def __len__(self) -> int:
        """Get number of loaded plugins."""
        return len(self.plugin_manager)
    
    def __contains__(self, plugin_name: str) -> bool:
        """Check if plugin is loaded."""
        return plugin_name in self.plugin_manager
    
    def __iter__(self):
        """Iterate over loaded plugins."""
        return iter(self.plugin_manager)
