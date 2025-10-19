"""
Plugin registry for PathwayLens.
"""

import asyncio
from typing import Dict, List, Any, Optional, Type, Union
from loguru import logger

from .base_plugin import BasePlugin
from .plugin_manager import PluginManager


class PluginRegistry:
    """Registry for managing PathwayLens plugins."""
    
    def __init__(self):
        """Initialize the plugin registry."""
        self.logger = logger.bind(module="plugin_registry")
        
        # Plugin registry
        self.registry: Dict[str, Dict[str, Any]] = {}
        
        # Plugin categories
        self.categories = {
            'analysis': [],
            'visualization': [],
            'data_processing': [],
            'export': [],
            'import': [],
            'custom': []
        }
        
        # Plugin metadata
        self.metadata = {
            'total_plugins': 0,
            'registered_plugins': 0,
            'active_plugins': 0,
            'last_updated': None
        }
    
    async def register_plugin(
        self, 
        plugin: BasePlugin, 
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a plugin in the registry.
        
        Args:
            plugin: Plugin instance to register
            category: Plugin category
            metadata: Additional plugin metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            plugin_name = plugin.name
            
            if plugin_name in self.registry:
                self.logger.warning(f"Plugin {plugin_name} already registered")
                return False
            
            # Determine category
            if not category:
                category = self._determine_category(plugin)
            
            # Create registry entry
            registry_entry = {
                'plugin': plugin,
                'category': category,
                'metadata': metadata or {},
                'status': 'registered',
                'registered_at': self._get_current_timestamp()
            }
            
            # Add to registry
            self.registry[plugin_name] = registry_entry
            
            # Add to category
            if category in self.categories:
                self.categories[category].append(plugin_name)
            else:
                self.categories['custom'].append(plugin_name)
            
            # Update metadata
            self.metadata['total_plugins'] += 1
            self.metadata['registered_plugins'] += 1
            self.metadata['last_updated'] = self._get_current_timestamp()
            
            self.logger.info(f"Registered plugin: {plugin_name} in category: {category}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin_name}: {e}")
            return False
    
    async def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin from the registry.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if plugin_name not in self.registry:
                self.logger.warning(f"Plugin {plugin_name} not registered")
                return False
            
            # Get plugin info
            plugin_info = self.registry[plugin_name]
            category = plugin_info['category']
            
            # Remove from category
            if category in self.categories and plugin_name in self.categories[category]:
                self.categories[category].remove(plugin_name)
            
            # Remove from registry
            del self.registry[plugin_name]
            
            # Update metadata
            self.metadata['total_plugins'] -= 1
            self.metadata['registered_plugins'] -= 1
            self.metadata['last_updated'] = self._get_current_timestamp()
            
            self.logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    async def activate_plugin(self, plugin_name: str) -> bool:
        """
        Activate a plugin.
        
        Args:
            plugin_name: Name of plugin to activate
            
        Returns:
            True if activation successful, False otherwise
        """
        try:
            if plugin_name not in self.registry:
                self.logger.error(f"Plugin {plugin_name} not registered")
                return False
            
            plugin_info = self.registry[plugin_name]
            plugin = plugin_info['plugin']
            
            # Initialize plugin if not already initialized
            if not plugin.initialized:
                success = await plugin.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize plugin {plugin_name}")
                    return False
            
            # Update status
            plugin_info['status'] = 'active'
            plugin_info['activated_at'] = self._get_current_timestamp()
            
            # Update metadata
            self.metadata['active_plugins'] += 1
            self.metadata['last_updated'] = self._get_current_timestamp()
            
            self.logger.info(f"Activated plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to activate plugin {plugin_name}: {e}")
            return False
    
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """
        Deactivate a plugin.
        
        Args:
            plugin_name: Name of plugin to deactivate
            
        Returns:
            True if deactivation successful, False otherwise
        """
        try:
            if plugin_name not in self.registry:
                self.logger.error(f"Plugin {plugin_name} not registered")
                return False
            
            plugin_info = self.registry[plugin_name]
            plugin = plugin_info['plugin']
            
            # Cleanup plugin if initialized
            if plugin.initialized:
                await plugin.cleanup()
            
            # Update status
            plugin_info['status'] = 'inactive'
            plugin_info['deactivated_at'] = self._get_current_timestamp()
            
            # Update metadata
            self.metadata['active_plugins'] -= 1
            self.metadata['last_updated'] = self._get_current_timestamp()
            
            self.logger.info(f"Deactivated plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Get a plugin by name.
        
        Args:
            plugin_name: Name of plugin to get
            
        Returns:
            Plugin instance or None if not found
        """
        if plugin_name in self.registry:
            return self.registry[plugin_name]['plugin']
        return None
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get plugin information.
        
        Args:
            plugin_name: Name of plugin to get info for
            
        Returns:
            Plugin information or None if not found
        """
        if plugin_name in self.registry:
            plugin_info = self.registry[plugin_name]
            return {
                'name': plugin_name,
                'category': plugin_info['category'],
                'status': plugin_info['status'],
                'metadata': plugin_info['metadata'],
                'registered_at': plugin_info['registered_at'],
                'activated_at': plugin_info.get('activated_at'),
                'deactivated_at': plugin_info.get('deactivated_at')
            }
        return None
    
    def get_all_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered plugins.
        
        Returns:
            Dictionary with plugin information
        """
        return {name: self.get_plugin_info(name) for name in self.registry.keys()}
    
    def get_plugins_by_category(self, category: str) -> List[str]:
        """
        Get plugins by category.
        
        Args:
            category: Plugin category
            
        Returns:
            List of plugin names in category
        """
        return self.categories.get(category, [])
    
    def get_active_plugins(self) -> List[str]:
        """
        Get list of active plugins.
        
        Returns:
            List of active plugin names
        """
        return [
            name for name, info in self.registry.items() 
            if info['status'] == 'active'
        ]
    
    def get_inactive_plugins(self) -> List[str]:
        """
        Get list of inactive plugins.
        
        Returns:
            List of inactive plugin names
        """
        return [
            name for name, info in self.registry.items() 
            if info['status'] == 'inactive'
        ]
    
    def get_registry_metadata(self) -> Dict[str, Any]:
        """
        Get registry metadata.
        
        Returns:
            Registry metadata dictionary
        """
        return self.metadata.copy()
    
    def get_categories(self) -> Dict[str, List[str]]:
        """
        Get plugin categories.
        
        Returns:
            Dictionary with plugin categories
        """
        return self.categories.copy()
    
    def _determine_category(self, plugin: BasePlugin) -> str:
        """Determine plugin category based on tags and metadata."""
        tags = plugin.get_tags()
        
        # Check tags for category
        for tag in tags:
            if tag in self.categories:
                return tag
        
        # Default to custom category
        return 'custom'
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    async def cleanup(self) -> bool:
        """
        Cleanup plugin registry and all plugins.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            self.logger.info("Cleaning up plugin registry")
            
            # Deactivate all active plugins
            for plugin_name in self.get_active_plugins():
                await self.deactivate_plugin(plugin_name)
            
            # Clear registry
            self.registry.clear()
            
            # Clear categories
            for category_plugins in self.categories.values():
                category_plugins.clear()
            
            # Reset metadata
            self.metadata = {
                'total_plugins': 0,
                'registered_plugins': 0,
                'active_plugins': 0,
                'last_updated': None
            }
            
            self.logger.info("Plugin registry cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin registry: {e}")
            return False
    
    def __len__(self) -> int:
        """Get number of registered plugins."""
        return len(self.registry)
    
    def __contains__(self, plugin_name: str) -> bool:
        """Check if plugin is registered."""
        return plugin_name in self.registry
    
    def __iter__(self):
        """Iterate over registered plugins."""
        return iter(self.registry.values())
