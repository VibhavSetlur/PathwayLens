"""
Custom plugin for PathwayLens.
"""

from typing import Dict, Any, Optional, List
from loguru import logger
import pandas as pd
import numpy as np

from .base_plugin import BasePlugin


class CustomPlugin(BasePlugin):
    """Plugin for custom user-defined methods."""
    
    def __init__(self):
        super().__init__(
            name="custom_plugin",
            version="1.0.0",
            description="Custom user-defined methods for PathwayLens"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = ["pandas", "numpy", "scipy"]
        self.tags = ["custom", "user_defined", "extensible"]
        
        # Plugin state
        self.initialized = False
        self.custom_methods = {
            'custom_analysis': 'Custom Analysis Method',
            'custom_visualization': 'Custom Visualization Method',
            'custom_export': 'Custom Export Method',
            'custom_import': 'Custom Import Method'
        }
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing custom plugin")
            
            # Perform initialization tasks
            # e.g., load custom methods, setup resources, etc.
            
            self.initialized = True
            self.logger.info("Custom plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize custom plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing custom plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Get custom method
            method = parameters.get('method', 'custom_analysis') if parameters else 'custom_analysis'
            
            # Execute custom method
            result = await self._execute_custom_method(input_data, method, parameters)
            
            self.logger.info(f"Custom plugin executed successfully with method: {method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute custom plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up custom plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("Custom plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup custom plugin: {e}")
            return False
    
    async def _execute_custom_method(
        self, 
        input_data: Any, 
        method: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute custom method."""
        if method == 'custom_analysis':
            return await self._custom_analysis_method(input_data, parameters)
        elif method == 'custom_visualization':
            return await self._custom_visualization_method(input_data, parameters)
        elif method == 'custom_export':
            return await self._custom_export_method(input_data, parameters)
        elif method == 'custom_import':
            return await self._custom_import_method(input_data, parameters)
        else:
            raise ValueError(f"Unknown custom method: {method}")
    
    async def _custom_analysis_method(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom analysis method."""
        self.logger.info("Executing custom analysis method")
        
        # Example custom analysis logic
        result = {
            'method': 'custom_analysis',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'analysis': {
                'type': 'custom',
                'data': input_data,
                'metadata': {
                    'analyzed_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example analysis
        if isinstance(input_data, list):
            # Example custom analysis: calculate statistics
            if len(input_data) > 0:
                try:
                    # Try to convert to numeric for statistical analysis
                    numeric_data = [float(x) for x in input_data if str(x).replace('.', '').replace('-', '').isdigit()]
                    if numeric_data:
                        result['analysis']['statistics'] = {
                            'mean': np.mean(numeric_data),
                            'median': np.median(numeric_data),
                            'std': np.std(numeric_data),
                            'min': min(numeric_data),
                            'max': max(numeric_data)
                        }
                except Exception:
                    # Fallback to basic analysis
                    result['analysis']['statistics'] = {
                        'total_items': len(input_data),
                        'unique_items': len(set(input_data)),
                        'sample_items': input_data[:5] if len(input_data) > 5 else input_data
                    }
        
        return result
    
    async def _custom_visualization_method(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom visualization method."""
        self.logger.info("Executing custom visualization method")
        
        # Example custom visualization logic
        result = {
            'method': 'custom_visualization',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'visualization': {
                'type': 'custom',
                'data': input_data,
                'metadata': {
                    'visualized_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example visualization data
        if isinstance(input_data, list):
            # Example custom visualization: create a simple plot
            plot_data = []
            for i, item in enumerate(input_data[:10]):  # Limit to first 10 items
                plot_data.append({
                    'x': i,
                    'y': hash(str(item)) % 100,  # Simple hash-based y value
                    'label': str(item)
                })
            
            result['visualization']['plot_data'] = plot_data
            result['visualization']['metadata']['plot_type'] = 'scatter'
            result['visualization']['metadata']['data_points'] = len(plot_data)
        
        return result
    
    async def _custom_export_method(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom export method."""
        self.logger.info("Executing custom export method")
        
        # Example custom export logic
        result = {
            'method': 'custom_export',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'export': {
                'type': 'custom',
                'data': input_data,
                'metadata': {
                    'exported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example export processing
        if isinstance(input_data, list):
            # Example custom export: create summary
            result['export']['summary'] = {
                'total_items': len(input_data),
                'unique_items': len(set(input_data)),
                'sample_items': input_data[:5] if len(input_data) > 5 else input_data,
                'export_format': 'custom_json'
            }
        elif isinstance(input_data, dict):
            result['export']['summary'] = {
                'total_keys': len(input_data),
                'key_types': {k: type(v).__name__ for k, v in input_data.items()},
                'export_format': 'custom_json'
            }
        
        return result
    
    async def _custom_import_method(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom import method."""
        self.logger.info("Executing custom import method")
        
        # Example custom import logic
        result = {
            'method': 'custom_import',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'import': {
                'type': 'custom',
                'data': input_data,
                'metadata': {
                    'imported_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example import processing
        if isinstance(input_data, list):
            # Example custom import: validate and clean data
            cleaned_data = []
            for item in input_data:
                if item is not None and str(item).strip():
                    cleaned_data.append(str(item).strip())
            
            result['import']['cleaned_data'] = cleaned_data
            result['import']['metadata']['cleaning_stats'] = {
                'original_count': len(input_data),
                'cleaned_count': len(cleaned_data),
                'removed_count': len(input_data) - len(cleaned_data)
            }
        elif isinstance(input_data, dict):
            # Example custom import: validate keys
            valid_keys = []
            for key, value in input_data.items():
                if key and str(key).strip() and value is not None:
                    valid_keys.append(key)
            
            result['import']['valid_keys'] = valid_keys
            result['import']['metadata']['validation_stats'] = {
                'total_keys': len(input_data),
                'valid_keys': len(valid_keys),
                'invalid_keys': len(input_data) - len(valid_keys)
            }
        
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
            if method not in self.custom_methods:
                self.logger.error(f"Invalid custom method: {method}")
                return False
        
        # Validate custom_parameter parameter
        if 'custom_parameter' in parameters:
            custom_param = parameters['custom_parameter']
            if not isinstance(custom_param, (str, int, float, bool)):
                self.logger.error("Invalid custom parameter type")
                return False
        
        # Validate output_format parameter
        if 'output_format' in parameters:
            output_format = parameters['output_format']
            valid_formats = ['json', 'csv', 'tsv']
            if output_format not in valid_formats:
                self.logger.error(f"Invalid output format: {output_format}")
                return False
        
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return ['method']
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return ['custom_parameter', 'output_format', 'include_metadata', 'verbose']
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {
            'method': {
                'type': 'string',
                'required': True,
                'description': 'Custom method to use',
                'options': list(self.custom_methods.keys())
            },
            'custom_parameter': {
                'type': 'string',
                'required': False,
                'description': 'Custom parameter for the method',
                'default': 'default_value'
            },
            'output_format': {
                'type': 'string',
                'required': False,
                'description': 'Output format for the method',
                'options': ['json', 'csv', 'tsv'],
                'default': 'json'
            },
            'include_metadata': {
                'type': 'boolean',
                'required': False,
                'description': 'Include metadata in output',
                'default': True
            },
            'verbose': {
                'type': 'boolean',
                'required': False,
                'description': 'Enable verbose output',
                'default': False
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
