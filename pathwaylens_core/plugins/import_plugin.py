"""
Import plugin for PathwayLens.
"""

from typing import Dict, Any, Optional, List
from loguru import logger
import pandas as pd
import json
import csv

from .base_plugin import BasePlugin


class ImportPlugin(BasePlugin):
    """Plugin for custom import methods."""
    
    def __init__(self):
        super().__init__(
            name="import_plugin",
            version="1.0.0",
            description="Custom import methods for PathwayLens"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = ["pandas", "openpyxl", "xlrd"]
        self.tags = ["import", "custom", "data"]
        
        # Plugin state
        self.initialized = False
        self.import_methods = {
            'custom_json': 'Custom JSON Import',
            'custom_csv': 'Custom CSV Import',
            'custom_excel': 'Custom Excel Import',
            'custom_tsv': 'Custom TSV Import'
        }
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing import plugin")
            
            # Perform initialization tasks
            # e.g., load import methods, setup resources, etc.
            
            self.initialized = True
            self.logger.info("Import plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize import plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing import plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Get import method
            method = parameters.get('method', 'custom_json') if parameters else 'custom_json'
            
            # Execute import
            result = await self._execute_import(input_data, method, parameters)
            
            self.logger.info(f"Import plugin executed successfully with method: {method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute import plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up import plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("Import plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup import plugin: {e}")
            return False
    
    async def _execute_import(
        self, 
        input_data: Any, 
        method: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute import method."""
        if method == 'custom_json':
            return await self._custom_json_import(input_data, parameters)
        elif method == 'custom_csv':
            return await self._custom_csv_import(input_data, parameters)
        elif method == 'custom_excel':
            return await self._custom_excel_import(input_data, parameters)
        elif method == 'custom_tsv':
            return await self._custom_tsv_import(input_data, parameters)
        else:
            raise ValueError(f"Unknown import method: {method}")
    
    async def _custom_json_import(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom JSON import."""
        self.logger.info("Executing custom JSON import")
        
        # Example custom JSON import logic
        result = {
            'method': 'custom_json',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'import': {
                'format': 'json',
                'data': input_data,
                'metadata': {
                    'imported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['import']['metadata']['unique_items'] = len(set(input_data))
            result['import']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['import']['metadata']['total_keys'] = len(input_data)
            result['import']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
        return result
    
    async def _custom_csv_import(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom CSV import."""
        self.logger.info("Executing custom CSV import")
        
        # Example custom CSV import logic
        result = {
            'method': 'custom_csv',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'import': {
                'format': 'csv',
                'data': input_data,
                'metadata': {
                    'imported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1,
                    'delimiter': ','
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['import']['metadata']['unique_items'] = len(set(input_data))
            result['import']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['import']['metadata']['total_keys'] = len(input_data)
            result['import']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
        return result
    
    async def _custom_excel_import(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom Excel import."""
        self.logger.info("Executing custom Excel import")
        
        # Example custom Excel import logic
        result = {
            'method': 'custom_excel',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'import': {
                'format': 'excel',
                'data': input_data,
                'metadata': {
                    'imported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1,
                    'sheets': ['Sheet1']
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['import']['metadata']['unique_items'] = len(set(input_data))
            result['import']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['import']['metadata']['total_keys'] = len(input_data)
            result['import']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
        return result
    
    async def _custom_tsv_import(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom TSV import."""
        self.logger.info("Executing custom TSV import")
        
        # Example custom TSV import logic
        result = {
            'method': 'custom_tsv',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'import': {
                'format': 'tsv',
                'data': input_data,
                'metadata': {
                    'imported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1,
                    'delimiter': '\t'
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['import']['metadata']['unique_items'] = len(set(input_data))
            result['import']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['import']['metadata']['total_keys'] = len(input_data)
            result['import']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
        return result
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        # Validate method parameter
        if 'method' in parameters:
            method = parameters['method']
            if method not in self.import_methods:
                self.logger.error(f"Invalid import method: {method}")
                return False
        
        # Validate input_format parameter
        if 'input_format' in parameters:
            input_format = parameters['input_format']
            valid_formats = ['json', 'csv', 'excel', 'tsv']
            if input_format not in valid_formats:
                self.logger.error(f"Invalid input format: {input_format}")
                return False
        
        # Validate filename parameter
        if 'filename' in parameters:
            filename = parameters['filename']
            if not isinstance(filename, str) or not filename:
                self.logger.error("Invalid filename")
                return False
        
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return ['method']
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return ['input_format', 'filename', 'include_metadata', 'compression']
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {
            'method': {
                'type': 'string',
                'required': True,
                'description': 'Import method to use',
                'options': list(self.import_methods.keys())
            },
            'input_format': {
                'type': 'string',
                'required': False,
                'description': 'Input format for import',
                'options': ['json', 'csv', 'excel', 'tsv'],
                'default': 'json'
            },
            'filename': {
                'type': 'string',
                'required': False,
                'description': 'Input filename',
                'default': 'imported_data'
            },
            'include_metadata': {
                'type': 'boolean',
                'required': False,
                'description': 'Include metadata in import',
                'default': True
            },
            'compression': {
                'type': 'string',
                'required': False,
                'description': 'Compression method',
                'options': ['none', 'gzip', 'zip'],
                'default': 'none'
            }
        }
    
    def is_compatible(self, pathwaylens_version: str) -> bool:
        """Check if plugin is compatible with PathwayLens version."""
        # Plugin requires PathwayLens >= 2.0.0
        try:
            from packaging import version
            min_version = "2.0.0"
            return version.parse(pathwaylens_version) >= version.parse(min_version)
        except Exception:
            return True
