"""
Configuration loader for PathwayLens CLI.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from loguru import logger

from .exceptions import ConfigurationError

class ConfigLoader:
    """Loads and validates configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dictionary containing configuration values.
            
        Raises:
            ConfigurationError: If file cannot be read or parsed.
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
            
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {path.suffix}")
                    
            if not isinstance(config, dict):
                raise ConfigurationError("Configuration file must contain a dictionary")
                
            return config
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")

    @staticmethod
    def merge_configs(cli_args: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge CLI arguments with file configuration.
        CLI arguments take precedence over file configuration.
        
        Args:
            cli_args: Dictionary of CLI arguments (filtered for None values).
            file_config: Dictionary of configuration from file.
            
        Returns:
            Merged configuration dictionary.
        """
        # Start with file config
        merged = file_config.copy()
        
        # Update with non-None CLI args
        for key, value in cli_args.items():
            if value is not None:
                merged[key] = value
                
        return merged
