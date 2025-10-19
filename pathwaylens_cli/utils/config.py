"""
Configuration management for PathwayLens CLI.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class Config:
    """Configuration manager for PathwayLens CLI."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.logger = logger.bind(module="config")
        self.config_file = config_file or self._get_default_config_file()
        self.config = self._load_config()
    
    def _get_default_config_file(self) -> str:
        """Get the default configuration file path."""
        # Try to find config file in current directory, then home directory
        current_dir_config = Path.cwd() / "pathwaylens_config.yaml"
        home_dir_config = Path.home() / ".pathwaylens" / "config.yaml"
        
        if current_dir_config.exists():
            return str(current_dir_config)
        elif home_dir_config.exists():
            return str(home_dir_config)
        else:
            # Create default config in home directory
            home_dir_config.parent.mkdir(exist_ok=True)
            return str(home_dir_config)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if not os.path.exists(self.config_file):
                # Create default configuration
                default_config = self._get_default_config()
                self._save_config(default_config)
                return default_config
            
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif self.config_file.endswith('.json'):
                    config = json.load(f)
                else:
                    # Try YAML first, then JSON
                    try:
                        config = yaml.safe_load(f)
                    except:
                        f.seek(0)
                        config = json.load(f)
                
                self.logger.info(f"Configuration loaded from: {self.config_file}")
                return config or {}
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "api": {
                "base_url": "http://localhost:8000",
                "api_key": None,
                "timeout": 300,
                "retry_attempts": 3
            },
            "analysis": {
                "default_databases": ["kegg", "reactome"],
                "default_species": "human",
                "significance_threshold": 0.05,
                "correction_method": "fdr_bh",
                "min_pathway_size": 5,
                "max_pathway_size": 500
            },
            "visualization": {
                "default_theme": "default",
                "output_format": "html",
                "include_plotlyjs": True,
                "width": 800,
                "height": 600
            },
            "output": {
                "directory": "pathwaylens_outputs",
                "create_subdirectories": True,
                "include_timestamp": True,
                "overwrite_existing": False
            },
            "logging": {
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
                "file": None,
                "console": True
            }
        }
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif self.config_file.endswith('.json'):
                    json.dump(config, f, indent=2)
                else:
                    # Default to YAML
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            keys = key.split('.')
            value = self.config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration value: {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        try:
            keys = key.split('.')
            config = self.config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            # Save configuration
            self._save_config(self.config)
            
            self.logger.info(f"Configuration value set: {key} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to set configuration value: {e}")
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get("api", {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.get("analysis", {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get("visualization", {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get("output", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get("logging", {})
    
    def update_api_config(self, **kwargs) -> None:
        """Update API configuration."""
        for key, value in kwargs.items():
            self.set(f"api.{key}", value)
    
    def update_analysis_config(self, **kwargs) -> None:
        """Update analysis configuration."""
        for key, value in kwargs.items():
            self.set(f"analysis.{key}", value)
    
    def update_visualization_config(self, **kwargs) -> None:
        """Update visualization configuration."""
        for key, value in kwargs.items():
            self.set(f"visualization.{key}", value)
    
    def update_output_config(self, **kwargs) -> None:
        """Update output configuration."""
        for key, value in kwargs.items():
            self.set(f"output.{key}", value)
    
    def update_logging_config(self, **kwargs) -> None:
        """Update logging configuration."""
        for key, value in kwargs.items():
            self.set(f"logging.{key}", value)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = self._get_default_config()
        self._save_config(self.config)
        self.logger.info("Configuration reset to defaults")
    
    def export_config(self, output_file: str) -> None:
        """
        Export configuration to a file.
        
        Args:
            output_file: Output file path
        """
        try:
            with open(output_file, 'w') as f:
                if output_file.endswith('.yaml') or output_file.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif output_file.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    # Default to YAML
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration exported to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
    
    def import_config(self, input_file: str) -> None:
        """
        Import configuration from a file.
        
        Args:
            input_file: Input file path
        """
        try:
            with open(input_file, 'r') as f:
                if input_file.endswith('.yaml') or input_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif input_file.endswith('.json'):
                    config = json.load(f)
                else:
                    # Try YAML first, then JSON
                    try:
                        config = yaml.safe_load(f)
                    except:
                        f.seek(0)
                        config = json.load(f)
            
            if config:
                self.config = config
                self._save_config(self.config)
                self.logger.info(f"Configuration imported from: {input_file}")
            else:
                self.logger.warning("No configuration data found in file")
                
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
    
    def get_config_file_path(self) -> str:
        """Get the configuration file path."""
        return self.config_file
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self.logger.info("Configuration reloaded")
