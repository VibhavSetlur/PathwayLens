"""
Plugin configuration for PathwayLens.
"""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from .base_plugin import BasePlugin


class PluginConfig:
    """Configuration manager for PathwayLens plugins."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the plugin configuration.
        
        Args:
            config_file: Path to configuration file
        """
        self.logger = logger.bind(module="plugin_config")
        self.config_file = config_file or "config/plugins.yml"
        
        # Default configuration
        self.default_config = {
            'plugins': {
                'enabled': True,
                'auto_load': True,
                'load_subdirectories': True,
                'validate_plugins': True,
                'initialize_plugins': True,
                'plugin_directory': 'plugins',
                'config_directory': 'config/plugins',
                'log_directory': 'logs/plugins'
            },
            'categories': {
                'analysis': {
                    'enabled': True,
                    'priority': 1,
                    'auto_load': True
                },
                'visualization': {
                    'enabled': True,
                    'priority': 2,
                    'auto_load': True
                },
                'data_processing': {
                    'enabled': True,
                    'priority': 3,
                    'auto_load': True
                },
                'export': {
                    'enabled': True,
                    'priority': 4,
                    'auto_load': True
                },
                'import': {
                    'enabled': True,
                    'priority': 5,
                    'auto_load': True
                },
                'custom': {
                    'enabled': True,
                    'priority': 6,
                    'auto_load': False
                }
            },
            'logging': {
                'enabled': True,
                'level': 'INFO',
                'format': '{time} | {level} | {module} | {message}',
                'file': 'logs/plugins/plugin.log',
                'max_size': '10MB',
                'retention': '7 days'
            },
            'security': {
                'enabled': True,
                'sandbox_plugins': True,
                'allowed_imports': [
                    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
                    'plotly', 'requests', 'json', 'yaml', 'csv', 'datetime'
                ],
                'blocked_imports': [
                    'os', 'sys', 'subprocess', 'shutil', 'glob', 'pathlib'
                ]
            }
        }
        
        # Current configuration
        self.config = self.default_config.copy()
        
        # Load configuration if file exists
        if Path(self.config_file).exists():
            self.load_config()
    
    def load_config(self, config_file: Optional[str] = None) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if loading successful, False otherwise
        """
        if config_file:
            self.config_file = config_file
        
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                self.logger.warning(f"Configuration file {self.config_file} does not exist")
                return False
            
            # Load configuration based on file extension
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            # Merge with default configuration
            self.config = self._merge_config(self.default_config, loaded_config)
            
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {self.config_file}: {e}")
            return False
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if saving successful, False otherwise
        """
        if config_file:
            self.config_file = config_file
        
        try:
            config_path = Path(self.config_file)
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration based on file extension
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            else:
                self.logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {self.config_file}: {e}")
            return False
    
    def get_config(self, key: Optional[str] = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            
        Returns:
            Configuration value or entire config if key is None
        """
        if key is None:
            return self.config.copy()
        
        # Navigate nested keys
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        
        return value
    
    def set_config(self, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            value: Configuration value
            
        Returns:
            True if setting successful, False otherwise
        """
        try:
            # Navigate nested keys
            keys = key.split('.')
            config = self.config
            
            # Navigate to parent of target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            self.logger.info(f"Configuration updated: {key} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set configuration {key}: {e}")
            return False
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific plugin.
        
        Args:
            plugin_name: Name of plugin to get config for
            
        Returns:
            Plugin configuration dictionary
        """
        plugin_config = self.get_config(f'plugins.{plugin_name}')
        if plugin_config is None:
            # Return default plugin configuration
            return {
                'enabled': True,
                'auto_load': True,
                'priority': 0,
                'parameters': {}
            }
        return plugin_config
    
    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """
        Set configuration for a specific plugin.
        
        Args:
            plugin_name: Name of plugin to set config for
            config: Plugin configuration dictionary
            
        Returns:
            True if setting successful, False otherwise
        """
        return self.set_config(f'plugins.{plugin_name}', config)
    
    def get_category_config(self, category: str) -> Dict[str, Any]:
        """
        Get configuration for a specific category.
        
        Args:
            category: Name of category to get config for
            
        Returns:
            Category configuration dictionary
        """
        category_config = self.get_config(f'categories.{category}')
        if category_config is None:
            # Return default category configuration
            return {
                'enabled': True,
                'priority': 0,
                'auto_load': True
            }
        return category_config
    
    def set_category_config(self, category: str, config: Dict[str, Any]) -> bool:
        """
        Set configuration for a specific category.
        
        Args:
            category: Name of category to set config for
            config: Category configuration dictionary
            
        Returns:
            True if setting successful, False otherwise
        """
        return self.set_config(f'categories.{category}', config)
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """
        Check if a plugin is enabled.
        
        Args:
            plugin_name: Name of plugin to check
            
        Returns:
            True if plugin is enabled, False otherwise
        """
        plugin_config = self.get_plugin_config(plugin_name)
        return plugin_config.get('enabled', True)
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a plugin.
        
        Args:
            plugin_name: Name of plugin to enable
            
        Returns:
            True if enabling successful, False otherwise
        """
        return self.set_config(f'plugins.{plugin_name}.enabled', True)
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable a plugin.
        
        Args:
            plugin_name: Name of plugin to disable
            
        Returns:
            True if disabling successful, False otherwise
        """
        return self.set_config(f'plugins.{plugin_name}.enabled', False)
    
    def is_category_enabled(self, category: str) -> bool:
        """
        Check if a category is enabled.
        
        Args:
            category: Name of category to check
            
        Returns:
            True if category is enabled, False otherwise
        """
        category_config = self.get_category_config(category)
        return category_config.get('enabled', True)
    
    def enable_category(self, category: str) -> bool:
        """
        Enable a category.
        
        Args:
            category: Name of category to enable
            
        Returns:
            True if enabling successful, False otherwise
        """
        return self.set_config(f'categories.{category}.enabled', True)
    
    def disable_category(self, category: str) -> bool:
        """
        Disable a category.
        
        Args:
            category: Name of category to disable
            
        Returns:
            True if disabling successful, False otherwise
        """
        return self.set_config(f'categories.{category}.enabled', False)
    
    def get_plugin_priority(self, plugin_name: str) -> int:
        """
        Get plugin priority.
        
        Args:
            plugin_name: Name of plugin to get priority for
            
        Returns:
            Plugin priority
        """
        plugin_config = self.get_plugin_config(plugin_name)
        return plugin_config.get('priority', 0)
    
    def set_plugin_priority(self, plugin_name: str, priority: int) -> bool:
        """
        Set plugin priority.
        
        Args:
            plugin_name: Name of plugin to set priority for
            priority: Plugin priority
            
        Returns:
            True if setting successful, False otherwise
        """
        return self.set_config(f'plugins.{plugin_name}.priority', priority)
    
    def get_plugin_parameters(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get plugin parameters.
        
        Args:
            plugin_name: Name of plugin to get parameters for
            
        Returns:
            Plugin parameters dictionary
        """
        plugin_config = self.get_plugin_config(plugin_name)
        return plugin_config.get('parameters', {})
    
    def set_plugin_parameters(self, plugin_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Set plugin parameters.
        
        Args:
            plugin_name: Name of plugin to set parameters for
            parameters: Plugin parameters dictionary
            
        Returns:
            True if setting successful, False otherwise
        """
        return self.set_config(f'plugins.{plugin_name}.parameters', parameters)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return self.get_config('logging') or {}
    
    def get_security_config(self) -> Dict[str, Any]:
        """
        Get security configuration.
        
        Returns:
            Security configuration dictionary
        """
        return self.get_config('security') or {}
    
    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Merge loaded configuration with default configuration."""
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_config(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to defaults.
        
        Returns:
            True if reset successful, False otherwise
        """
        try:
            self.config = self.default_config.copy()
            self.logger.info("Configuration reset to defaults")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required sections
            required_sections = ['plugins', 'categories', 'logging', 'security']
            for section in required_sections:
                if section not in self.config:
                    self.logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate plugins section
            plugins_config = self.config['plugins']
            if not isinstance(plugins_config, dict):
                self.logger.error("Plugins configuration must be a dictionary")
                return False
            
            # Validate categories section
            categories_config = self.config['categories']
            if not isinstance(categories_config, dict):
                self.logger.error("Categories configuration must be a dictionary")
                return False
            
            # Validate logging section
            logging_config = self.config['logging']
            if not isinstance(logging_config, dict):
                self.logger.error("Logging configuration must be a dictionary")
                return False
            
            # Validate security section
            security_config = self.config['security']
            if not isinstance(security_config, dict):
                self.logger.error("Security configuration must be a dictionary")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
