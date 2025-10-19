"""
Plugin utilities for PathwayLens.
"""

import asyncio
import importlib
import inspect
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union, Callable
from loguru import logger

from .base_plugin import BasePlugin


class PluginUtils:
    """Utility functions for PathwayLens plugins."""
    
    def __init__(self):
        """Initialize the plugin utilities."""
        self.logger = logger.bind(module="plugin_utils")
    
    @staticmethod
    def load_plugin_from_file(plugin_file: str) -> Optional[Type[BasePlugin]]:
        """
        Load a plugin class from a file.
        
        Args:
            plugin_file: Path to plugin file
            
        Returns:
            Plugin class or None if not found
        """
        try:
            plugin_path = Path(plugin_file)
            if not plugin_path.exists():
                logger.error(f"Plugin file {plugin_file} does not exist")
                return None
            
            # Import the module
            module_name = plugin_path.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    return obj
            
            logger.warning(f"No plugin class found in {plugin_file}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load plugin from {plugin_file}: {e}")
            return None
    
    @staticmethod
    def load_plugin_from_directory(plugin_directory: str) -> List[Type[BasePlugin]]:
        """
        Load plugin classes from a directory.
        
        Args:
            plugin_directory: Path to plugin directory
            
        Returns:
            List of plugin classes
        """
        plugin_classes = []
        
        try:
            plugin_path = Path(plugin_directory)
            if not plugin_path.exists() or not plugin_path.is_dir():
                logger.error(f"Plugin directory {plugin_directory} does not exist or is not a directory")
                return plugin_classes
            
            # Load plugins from Python files
            for plugin_file in plugin_path.glob("*.py"):
                if plugin_file.name.startswith("__"):
                    continue
                
                plugin_class = PluginUtils.load_plugin_from_file(str(plugin_file))
                if plugin_class:
                    plugin_classes.append(plugin_class)
            
            # Load plugins from subdirectories
            for plugin_dir in plugin_path.iterdir():
                if plugin_dir.is_dir() and not plugin_dir.name.startswith("__"):
                    subdir_plugins = PluginUtils.load_plugin_from_directory(str(plugin_dir))
                    plugin_classes.extend(subdir_plugins)
            
            logger.info(f"Loaded {len(plugin_classes)} plugin classes from {plugin_directory}")
            return plugin_classes
            
        except Exception as e:
            logger.error(f"Failed to load plugins from {plugin_directory}: {e}")
            return plugin_classes
    
    @staticmethod
    def create_plugin_instance(plugin_class: Type[BasePlugin], **kwargs) -> Optional[BasePlugin]:
        """
        Create a plugin instance.
        
        Args:
            plugin_class: Plugin class to instantiate
            **kwargs: Arguments for plugin constructor
            
        Returns:
            Plugin instance or None if creation failed
        """
        try:
            plugin_instance = plugin_class(**kwargs)
            logger.info(f"Created plugin instance: {plugin_instance.name}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to create plugin instance: {e}")
            return None
    
    @staticmethod
    def validate_plugin_class(plugin_class: Type[BasePlugin]) -> Dict[str, Any]:
        """
        Validate a plugin class.
        
        Args:
            plugin_class: Plugin class to validate
            
        Returns:
            Validation results dictionary
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        try:
            # Check if class is a subclass of BasePlugin
            if not issubclass(plugin_class, BasePlugin):
                validation_result['valid'] = False
                validation_result['errors'].append("Plugin class must be a subclass of BasePlugin")
                return validation_result
            
            # Check required methods
            required_methods = ['initialize', 'execute', 'cleanup']
            for method_name in required_methods:
                if not hasattr(plugin_class, method_name):
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Plugin class missing required method: {method_name}")
                elif not callable(getattr(plugin_class, method_name)):
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Plugin class method {method_name} is not callable")
            
            # Check if class can be instantiated
            try:
                plugin_instance = plugin_class()
                validation_result['info'].append(f"Plugin class can be instantiated: {plugin_instance.name}")
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Plugin class cannot be instantiated: {e}")
            
            validation_result['info'].append("Plugin class validation completed")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Plugin class validation failed: {e}")
        
        return validation_result
    
    @staticmethod
    def get_plugin_metadata(plugin_class: Type[BasePlugin]) -> Dict[str, Any]:
        """
        Get metadata from a plugin class.
        
        Args:
            plugin_class: Plugin class to get metadata from
            
        Returns:
            Plugin metadata dictionary
        """
        try:
            plugin_instance = plugin_class()
            return plugin_instance.get_metadata()
        except Exception as e:
            logger.error(f"Failed to get plugin metadata: {e}")
            return {}
    
    @staticmethod
    def get_plugin_info(plugin_class: Type[BasePlugin]) -> Dict[str, Any]:
        """
        Get information from a plugin class.
        
        Args:
            plugin_class: Plugin class to get info from
            
        Returns:
            Plugin information dictionary
        """
        try:
            plugin_instance = plugin_class()
            return plugin_instance.get_info()
        except Exception as e:
            logger.error(f"Failed to get plugin info: {e}")
            return {}
    
    @staticmethod
    def get_plugin_dependencies(plugin_class: Type[BasePlugin]) -> List[str]:
        """
        Get dependencies from a plugin class.
        
        Args:
            plugin_class: Plugin class to get dependencies from
            
        Returns:
            List of plugin dependencies
        """
        try:
            plugin_instance = plugin_class()
            return plugin_instance.get_dependencies()
        except Exception as e:
            logger.error(f"Failed to get plugin dependencies: {e}")
            return []
    
    @staticmethod
    def get_plugin_tags(plugin_class: Type[BasePlugin]) -> List[str]:
        """
        Get tags from a plugin class.
        
        Args:
            plugin_class: Plugin class to get tags from
            
        Returns:
            List of plugin tags
        """
        try:
            plugin_instance = plugin_class()
            return plugin_instance.get_tags()
        except Exception as e:
            logger.error(f"Failed to get plugin tags: {e}")
            return []
    
    @staticmethod
    def check_plugin_compatibility(plugin_class: Type[BasePlugin], pathwaylens_version: str) -> bool:
        """
        Check if a plugin is compatible with PathwayLens version.
        
        Args:
            plugin_class: Plugin class to check compatibility for
            pathwaylens_version: PathwayLens version
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            plugin_instance = plugin_class()
            return plugin_instance.is_compatible(pathwaylens_version)
        except Exception as e:
            logger.error(f"Failed to check plugin compatibility: {e}")
            return False
    
    @staticmethod
    def save_plugin_config(plugin_name: str, config: Dict[str, Any], config_file: str) -> bool:
        """
        Save plugin configuration to file.
        
        Args:
            plugin_name: Name of plugin
            config: Configuration dictionary
            config_file: Path to configuration file
            
        Returns:
            True if saving successful, False otherwise
        """
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration based on file extension
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            logger.info(f"Saved configuration for plugin {plugin_name} to {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration for plugin {plugin_name}: {e}")
            return False
    
    @staticmethod
    def load_plugin_config(plugin_name: str, config_file: str) -> Optional[Dict[str, Any]]:
        """
        Load plugin configuration from file.
        
        Args:
            plugin_name: Name of plugin
            config_file: Path to configuration file
            
        Returns:
            Configuration dictionary or None if loading failed
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                logger.warning(f"Configuration file {config_file} does not exist for plugin {plugin_name}")
                return None
            
            # Load configuration based on file extension
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return None
            
            logger.info(f"Loaded configuration for plugin {plugin_name} from {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration for plugin {plugin_name}: {e}")
            return None
    
    @staticmethod
    def create_plugin_directory(plugin_name: str, base_directory: str = "plugins") -> str:
        """
        Create a directory for a plugin.
        
        Args:
            plugin_name: Name of plugin
            base_directory: Base directory for plugins
            
        Returns:
            Path to created plugin directory
        """
        try:
            plugin_dir = Path(base_directory) / plugin_name
            plugin_dir.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py file
            init_file = plugin_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
            
            # Create plugin.py file
            plugin_file = plugin_dir / f"{plugin_name}.py"
            if not plugin_file.exists():
                plugin_template = PluginUtils._get_plugin_template(plugin_name)
                plugin_file.write_text(plugin_template)
            
            # Create config.yml file
            config_file = plugin_dir / "config.yml"
            if not config_file.exists():
                config_template = PluginUtils._get_config_template(plugin_name)
                config_file.write_text(config_template)
            
            logger.info(f"Created plugin directory: {plugin_dir}")
            return str(plugin_dir)
            
        except Exception as e:
            logger.error(f"Failed to create plugin directory for {plugin_name}: {e}")
            return ""
    
    @staticmethod
    def _get_plugin_template(plugin_name: str) -> str:
        """Get plugin template code."""
        return f'''"""
{plugin_name} plugin for PathwayLens.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .base_plugin import BasePlugin


class {plugin_name.title().replace('_', '')}Plugin(BasePlugin):
    """Plugin for {plugin_name} functionality."""
    
    def __init__(self):
        super().__init__(
            name="{plugin_name}",
            version="1.0.0",
            description="Plugin for {plugin_name} functionality"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = []
        self.tags = ["{plugin_name}", "custom"]
        
        # Plugin state
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing {plugin_name} plugin")
            
            # Perform initialization tasks
            # e.g., load configuration, setup resources, etc.
            
            self.initialized = True
            self.logger.info("{plugin_name} plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {plugin_name} plugin: {{e}}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing {plugin_name} plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Process input data
            result = await self._process_data(input_data, parameters)
            
            self.logger.info(f"{plugin_name} plugin executed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute {plugin_name} plugin: {{e}}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up {plugin_name} plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("{plugin_name} plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup {plugin_name} plugin: {{e}}")
            return False
    
    async def _process_data(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data."""
        # Example processing logic
        result = {{
            'plugin_name': self.name,
            'input_data': input_data,
            'parameters': parameters or {{}},
            'processed_at': self._get_current_timestamp()
        }}
        
        return result
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        # Override in subclasses for specific validation
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return []
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return []
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {{}}
    
    def is_compatible(self, pathwaylens_version: str) -> bool:
        """Check if plugin is compatible with PathwayLens version."""
        # Plugin requires PathwayLens >= 2.0.0
        try:
            from packaging import version
            min_version = "2.0.0"
            return version.parse(pathwaylens_version) >= version.parse(min_version)
        except Exception:
            return True
'''
    
    @staticmethod
    def _get_config_template(plugin_name: str) -> str:
        """Get configuration template."""
        return f'''# Configuration for {plugin_name} plugin

plugin:
  name: {plugin_name}
  version: "1.0.0"
  description: "Plugin for {plugin_name} functionality"
  author: "PathwayLens Team"
  license: "MIT"
  dependencies: []
  tags: ["{plugin_name}", "custom"]

settings:
  enabled: true
  auto_load: true
  priority: 0

parameters:
  # Add plugin-specific parameters here
  example_parameter:
    type: string
    required: false
    description: "Example parameter"
    default: "default_value"

logging:
  enabled: true
  level: "INFO"
  file: "logs/plugins/{plugin_name}.log"

security:
  enabled: true
  sandbox: true
  allowed_imports: []
  blocked_imports: []
'''
    
    @staticmethod
    def get_plugin_statistics(plugin_classes: List[Type[BasePlugin]]) -> Dict[str, Any]:
        """
        Get statistics about plugin classes.
        
        Args:
            plugin_classes: List of plugin classes
            
        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                'total_plugins': len(plugin_classes),
                'by_category': {},
                'by_author': {},
                'by_license': {},
                'by_version': {},
                'with_dependencies': 0,
                'without_dependencies': 0,
                'compatible_plugins': 0,
                'incompatible_plugins': 0
            }
            
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class()
                    
                    # Category statistics
                    tags = plugin_instance.get_tags()
                    for tag in tags:
                        if tag in stats['by_category']:
                            stats['by_category'][tag] += 1
                        else:
                            stats['by_category'][tag] = 1
                    
                    # Author statistics
                    author = plugin_instance.metadata.get('author', 'Unknown')
                    if author in stats['by_author']:
                        stats['by_author'][author] += 1
                    else:
                        stats['by_author'][author] = 1
                    
                    # License statistics
                    license_info = plugin_instance.metadata.get('license', 'Unknown')
                    if license_info in stats['by_license']:
                        stats['by_license'][license_info] += 1
                    else:
                        stats['by_license'][license_info] = 1
                    
                    # Version statistics
                    version = plugin_instance.metadata.get('version', 'Unknown')
                    if version in stats['by_version']:
                        stats['by_version'][version] += 1
                    else:
                        stats['by_version'][version] = 1
                    
                    # Dependencies statistics
                    dependencies = plugin_instance.get_dependencies()
                    if dependencies:
                        stats['with_dependencies'] += 1
                    else:
                        stats['without_dependencies'] += 1
                    
                    # Compatibility statistics
                    if plugin_instance.is_compatible("2.0.0"):
                        stats['compatible_plugins'] += 1
                    else:
                        stats['incompatible_plugins'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to get statistics for plugin class: {e}")
                    continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get plugin statistics: {e}")
            return {}
    
    @staticmethod
    def export_plugin_info(plugin_classes: List[Type[BasePlugin]], output_file: str) -> bool:
        """
        Export plugin information to file.
        
        Args:
            plugin_classes: List of plugin classes
            output_file: Path to output file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            plugin_info = []
            
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class()
                    info = plugin_instance.get_info()
                    plugin_info.append(info)
                except Exception as e:
                    logger.warning(f"Failed to get info for plugin class: {e}")
                    continue
            
            # Export based on file extension
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(plugin_info, f, indent=2)
            elif output_path.suffix.lower() in ['.yml', '.yaml']:
                with open(output_path, 'w') as f:
                    yaml.dump(plugin_info, f, default_flow_style=False, indent=2)
            else:
                logger.error(f"Unsupported output file format: {output_path.suffix}")
                return False
            
            logger.info(f"Exported plugin information to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export plugin information: {e}")
            return False
