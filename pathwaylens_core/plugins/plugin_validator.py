"""
Plugin validator for PathwayLens.
"""

import asyncio
import inspect
from typing import Dict, List, Any, Optional, Type, Union
from loguru import logger

from .base_plugin import BasePlugin


class PluginValidator:
    """Validates PathwayLens plugins."""
    
    def __init__(self):
        """Initialize the plugin validator."""
        self.logger = logger.bind(module="plugin_validator")
        
        # Validation rules
        self.validation_rules = {
            'required_methods': ['initialize', 'execute', 'cleanup'],
            'required_attributes': ['name', 'version', 'description'],
            'required_metadata': ['name', 'version', 'description', 'author', 'license'],
            'valid_categories': ['analysis', 'visualization', 'data_processing', 'export', 'import', 'custom']
        }
        
        # Validation results
        self.validation_results: Dict[str, Dict[str, Any]] = {}
    
    async def validate_plugin(self, plugin: BasePlugin) -> Dict[str, Any]:
        """
        Validate a plugin.
        
        Args:
            plugin: Plugin instance to validate
            
        Returns:
            Validation results dictionary
        """
        try:
            plugin_name = plugin.name
            self.logger.info(f"Validating plugin: {plugin_name}")
            
            validation_result = {
                'plugin_name': plugin_name,
                'valid': True,
                'errors': [],
                'warnings': [],
                'info': [],
                'validation_timestamp': self._get_current_timestamp()
            }
            
            # Validate plugin class
            await self._validate_plugin_class(plugin, validation_result)
            
            # Validate plugin methods
            await self._validate_plugin_methods(plugin, validation_result)
            
            # Validate plugin attributes
            await self._validate_plugin_attributes(plugin, validation_result)
            
            # Validate plugin metadata
            await self._validate_plugin_metadata(plugin, validation_result)
            
            # Validate plugin compatibility
            await self._validate_plugin_compatibility(plugin, validation_result)
            
            # Validate plugin dependencies
            await self._validate_plugin_dependencies(plugin, validation_result)
            
            # Store validation results
            self.validation_results[plugin_name] = validation_result
            
            # Determine overall validity
            validation_result['valid'] = len(validation_result['errors']) == 0
            
            self.logger.info(f"Plugin validation completed: {plugin_name} - {'Valid' if validation_result['valid'] else 'Invalid'}")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate plugin {plugin_name}: {e}")
            return {
                'plugin_name': plugin_name,
                'valid': False,
                'errors': [f"Validation failed: {e}"],
                'warnings': [],
                'info': [],
                'validation_timestamp': self._get_current_timestamp()
            }
    
    async def _validate_plugin_class(self, plugin: BasePlugin, validation_result: Dict[str, Any]) -> None:
        """Validate plugin class."""
        try:
            # Check if plugin is instance of BasePlugin
            if not isinstance(plugin, BasePlugin):
                validation_result['errors'].append("Plugin must be an instance of BasePlugin")
                return
            
            # Check if plugin class is properly defined
            plugin_class = plugin.__class__
            if not inspect.isclass(plugin_class):
                validation_result['errors'].append("Plugin class is not properly defined")
                return
            
            # Check if plugin class has required methods
            for method_name in self.validation_rules['required_methods']:
                if not hasattr(plugin_class, method_name):
                    validation_result['errors'].append(f"Plugin class missing required method: {method_name}")
                elif not callable(getattr(plugin_class, method_name)):
                    validation_result['errors'].append(f"Plugin class method {method_name} is not callable")
            
            validation_result['info'].append("Plugin class validation passed")
            
        except Exception as e:
            validation_result['errors'].append(f"Plugin class validation failed: {e}")
    
    async def _validate_plugin_methods(self, plugin: BasePlugin, validation_result: Dict[str, Any]) -> None:
        """Validate plugin methods."""
        try:
            # Check required methods
            for method_name in self.validation_rules['required_methods']:
                if not hasattr(plugin, method_name):
                    validation_result['errors'].append(f"Plugin missing required method: {method_name}")
                    continue
                
                method = getattr(plugin, method_name)
                if not callable(method):
                    validation_result['errors'].append(f"Plugin method {method_name} is not callable")
                    continue
                
                # Check method signature
                try:
                    sig = inspect.signature(method)
                    if method_name == 'initialize':
                        if len(sig.parameters) != 1:  # self parameter
                            validation_result['warnings'].append(f"Method {method_name} should not take parameters")
                    elif method_name == 'execute':
                        if len(sig.parameters) < 2:  # self and input_data parameters
                            validation_result['errors'].append(f"Method {method_name} must take at least input_data parameter")
                    elif method_name == 'cleanup':
                        if len(sig.parameters) != 1:  # self parameter
                            validation_result['warnings'].append(f"Method {method_name} should not take parameters")
                except Exception as e:
                    validation_result['warnings'].append(f"Could not validate method signature for {method_name}: {e}")
            
            validation_result['info'].append("Plugin methods validation passed")
            
        except Exception as e:
            validation_result['errors'].append(f"Plugin methods validation failed: {e}")
    
    async def _validate_plugin_attributes(self, plugin: BasePlugin, validation_result: Dict[str, Any]) -> None:
        """Validate plugin attributes."""
        try:
            # Check required attributes
            for attr_name in self.validation_rules['required_attributes']:
                if not hasattr(plugin, attr_name):
                    validation_result['errors'].append(f"Plugin missing required attribute: {attr_name}")
                    continue
                
                attr_value = getattr(plugin, attr_name)
                if not attr_value:
                    validation_result['warnings'].append(f"Plugin attribute {attr_name} is empty")
                    continue
                
                # Validate attribute types
                if attr_name == 'name' and not isinstance(attr_value, str):
                    validation_result['errors'].append(f"Plugin attribute {attr_name} must be a string")
                elif attr_name == 'version' and not isinstance(attr_value, str):
                    validation_result['errors'].append(f"Plugin attribute {attr_name} must be a string")
                elif attr_name == 'description' and not isinstance(attr_value, str):
                    validation_result['errors'].append(f"Plugin attribute {attr_name} must be a string")
            
            validation_result['info'].append("Plugin attributes validation passed")
            
        except Exception as e:
            validation_result['errors'].append(f"Plugin attributes validation failed: {e}")
    
    async def _validate_plugin_metadata(self, plugin: BasePlugin, validation_result: Dict[str, Any]) -> None:
        """Validate plugin metadata."""
        try:
            metadata = plugin.get_metadata()
            
            # Check required metadata
            for metadata_key in self.validation_rules['required_metadata']:
                if metadata_key not in metadata:
                    validation_result['warnings'].append(f"Plugin metadata missing: {metadata_key}")
                    continue
                
                metadata_value = metadata[metadata_key]
                if not metadata_value:
                    validation_result['warnings'].append(f"Plugin metadata {metadata_key} is empty")
                    continue
            
            # Validate metadata types
            if 'name' in metadata and not isinstance(metadata['name'], str):
                validation_result['errors'].append("Plugin metadata 'name' must be a string")
            if 'version' in metadata and not isinstance(metadata['version'], str):
                validation_result['errors'].append("Plugin metadata 'version' must be a string")
            if 'description' in metadata and not isinstance(metadata['description'], str):
                validation_result['errors'].append("Plugin metadata 'description' must be a string")
            if 'author' in metadata and not isinstance(metadata['author'], str):
                validation_result['warnings'].append("Plugin metadata 'author' should be a string")
            if 'license' in metadata and not isinstance(metadata['license'], str):
                validation_result['warnings'].append("Plugin metadata 'license' should be a string")
            
            # Validate tags
            if 'tags' in metadata:
                tags = metadata['tags']
                if not isinstance(tags, list):
                    validation_result['warnings'].append("Plugin metadata 'tags' should be a list")
                else:
                    for tag in tags:
                        if not isinstance(tag, str):
                            validation_result['warnings'].append(f"Plugin metadata tag should be a string: {tag}")
            
            # Validate dependencies
            if 'dependencies' in metadata:
                dependencies = metadata['dependencies']
                if not isinstance(dependencies, list):
                    validation_result['warnings'].append("Plugin metadata 'dependencies' should be a list")
                else:
                    for dep in dependencies:
                        if not isinstance(dep, str):
                            validation_result['warnings'].append(f"Plugin metadata dependency should be a string: {dep}")
            
            validation_result['info'].append("Plugin metadata validation passed")
            
        except Exception as e:
            validation_result['errors'].append(f"Plugin metadata validation failed: {e}")
    
    async def _validate_plugin_compatibility(self, plugin: BasePlugin, validation_result: Dict[str, Any]) -> None:
        """Validate plugin compatibility."""
        try:
            # Check compatibility with PathwayLens version
            pathwaylens_version = "2.0.0"  # Example version
            if not plugin.is_compatible(pathwaylens_version):
                validation_result['warnings'].append(f"Plugin may not be compatible with PathwayLens {pathwaylens_version}")
            
            validation_result['info'].append("Plugin compatibility validation passed")
            
        except Exception as e:
            validation_result['warnings'].append(f"Plugin compatibility validation failed: {e}")
    
    async def _validate_plugin_dependencies(self, plugin: BasePlugin, validation_result: Dict[str, Any]) -> None:
        """Validate plugin dependencies."""
        try:
            dependencies = plugin.get_dependencies()
            
            if dependencies:
                # Check if dependencies are properly formatted
                for dep in dependencies:
                    if not isinstance(dep, str):
                        validation_result['warnings'].append(f"Plugin dependency should be a string: {dep}")
                        continue
                    
                    # Check dependency format (basic check)
                    if not dep or dep.isspace():
                        validation_result['warnings'].append(f"Plugin dependency is empty or whitespace: {dep}")
                        continue
                
                validation_result['info'].append(f"Plugin has {len(dependencies)} dependencies")
            else:
                validation_result['info'].append("Plugin has no dependencies")
            
        except Exception as e:
            validation_result['warnings'].append(f"Plugin dependencies validation failed: {e}")
    
    def get_validation_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all validation results.
        
        Returns:
            Dictionary with validation results
        """
        return self.validation_results.copy()
    
    def get_validation_result(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get validation result for a specific plugin.
        
        Args:
            plugin_name: Name of plugin to get validation result for
            
        Returns:
            Validation result or None if not found
        """
        return self.validation_results.get(plugin_name)
    
    def get_valid_plugins(self) -> List[str]:
        """
        Get list of valid plugins.
        
        Returns:
            List of valid plugin names
        """
        return [
            name for name, result in self.validation_results.items()
            if result.get('valid', False)
        ]
    
    def get_invalid_plugins(self) -> List[str]:
        """
        Get list of invalid plugins.
        
        Returns:
            List of invalid plugin names
        """
        return [
            name for name, result in self.validation_results.items()
            if not result.get('valid', False)
        ]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get validation summary.
        
        Returns:
            Validation summary dictionary
        """
        total_plugins = len(self.validation_results)
        valid_plugins = len(self.get_valid_plugins())
        invalid_plugins = len(self.get_invalid_plugins())
        
        return {
            'total_plugins': total_plugins,
            'valid_plugins': valid_plugins,
            'invalid_plugins': invalid_plugins,
            'validation_timestamp': self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def clear_validation_results(self) -> None:
        """Clear all validation results."""
        self.validation_results.clear()
        self.logger.info("Validation results cleared")
