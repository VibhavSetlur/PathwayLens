"""
Export plugin for PathwayLens.
"""

from typing import Dict, Any, Optional, List
from loguru import logger
import pandas as pd
import json
import csv

from .base_plugin import BasePlugin


class ExportPlugin(BasePlugin):
    """Plugin for custom export methods."""
    
    def __init__(self):
        super().__init__(
            name="export_plugin",
            version="1.0.0",
            description="Custom export methods for PathwayLens"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = ["pandas", "openpyxl", "xlsxwriter"]
        self.tags = ["export", "custom", "data"]
        
        # Plugin state
        self.initialized = False
        self.export_methods = {
            'custom_json': 'Custom JSON Export',
            'custom_csv': 'Custom CSV Export',
            'custom_excel': 'Custom Excel Export',
            'custom_tsv': 'Custom TSV Export'
        }
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing export plugin")
            
            # Perform initialization tasks
            # e.g., load export methods, setup resources, etc.
            
            self.initialized = True
            self.logger.info("Export plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize export plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing export plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Get export method
            method = parameters.get('method', 'custom_json') if parameters else 'custom_json'
            
            # Execute export
            result = await self._execute_export(input_data, method, parameters)
            
            self.logger.info(f"Export plugin executed successfully with method: {method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute export plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up export plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("Export plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup export plugin: {e}")
            return False
    
    async def _execute_export(
        self, 
        input_data: Any, 
        method: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute export method."""
        if method == 'custom_json':
            return await self._custom_json_export(input_data, parameters)
        elif method == 'custom_csv':
            return await self._custom_csv_export(input_data, parameters)
        elif method == 'custom_excel':
            return await self._custom_excel_export(input_data, parameters)
        elif method == 'custom_tsv':
            return await self._custom_tsv_export(input_data, parameters)
        else:
            raise ValueError(f"Unknown export method: {method}")
    
    async def _custom_json_export(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom JSON export."""
        self.logger.info("Executing custom JSON export")
        
        # Example custom JSON export logic
        result = {
            'method': 'custom_json',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'export': {
                'format': 'json',
                'data': input_data,
                'metadata': {
                    'exported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['export']['metadata']['unique_items'] = len(set(input_data))
            result['export']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['export']['metadata']['total_keys'] = len(input_data)
            result['export']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
        return result
    
    async def _custom_csv_export(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom CSV export."""
        self.logger.info("Executing custom CSV export")
        
        # Example custom CSV export logic
        result = {
            'method': 'custom_csv',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'export': {
                'format': 'csv',
                'data': input_data,
                'metadata': {
                    'exported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1,
                    'delimiter': ','
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['export']['metadata']['unique_items'] = len(set(input_data))
            result['export']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['export']['metadata']['total_keys'] = len(input_data)
            result['export']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
        return result
    
    async def _custom_excel_export(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom Excel export."""
        self.logger.info("Executing custom Excel export")
        
        # Example custom Excel export logic
        result = {
            'method': 'custom_excel',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'export': {
                'format': 'excel',
                'data': input_data,
                'metadata': {
                    'exported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1,
                    'sheets': ['Sheet1']
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['export']['metadata']['unique_items'] = len(set(input_data))
            result['export']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['export']['metadata']['total_keys'] = len(input_data)
            result['export']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
        return result
    
    async def _custom_tsv_export(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom TSV export."""
        self.logger.info("Executing custom TSV export")
        
        # Example custom TSV export logic
        result = {
            'method': 'custom_tsv',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'export': {
                'format': 'tsv',
                'data': input_data,
                'metadata': {
                    'exported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1,
                    'delimiter': '\t'
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            result['export']['metadata']['unique_items'] = len(set(input_data))
            result['export']['metadata']['sample_items'] = input_data[:5] if len(input_data) > 5 else input_data
        elif isinstance(input_data, dict):
            result['export']['metadata']['total_keys'] = len(input_data)
            result['export']['metadata']['key_types'] = {k: type(v).__name__ for k, v in input_data.items()}
        
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
            if method not in self.export_methods:
                self.logger.error(f"Invalid export method: {method}")
                return False
        
        # Validate output_format parameter
        if 'output_format' in parameters:
            output_format = parameters['output_format']
            valid_formats = ['json', 'csv', 'excel', 'tsv']
            if output_format not in valid_formats:
                self.logger.error(f"Invalid output format: {output_format}")
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
        return ['output_format', 'filename', 'include_metadata', 'compression']
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {
            'method': {
                'type': 'string',
                'required': True,
                'description': 'Export method to use',
                'options': list(self.export_methods.keys())
            },
            'output_format': {
                'type': 'string',
                'required': False,
                'description': 'Output format for export',
                'options': ['json', 'csv', 'excel', 'tsv'],
                'default': 'json'
            },
            'filename': {
                'type': 'string',
                'required': False,
                'description': 'Output filename',
                'default': 'exported_data'
            },
            'include_metadata': {
                'type': 'boolean',
                'required': False,
                'description': 'Include metadata in export',
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
