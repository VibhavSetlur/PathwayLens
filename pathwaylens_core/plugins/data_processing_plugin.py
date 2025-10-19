"""
Data processing plugin for PathwayLens.
"""

from typing import Dict, Any, Optional, List
from loguru import logger
import pandas as pd
import numpy as np

from .base_plugin import BasePlugin


class DataProcessingPlugin(BasePlugin):
    """Plugin for custom data processing methods."""
    
    def __init__(self):
        super().__init__(
            name="data_processing_plugin",
            version="1.0.0",
            description="Custom data processing methods for PathwayLens"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = ["pandas", "numpy", "scipy"]
        self.tags = ["data_processing", "custom", "statistics"]
        
        # Plugin state
        self.initialized = False
        self.processing_methods = {
            'custom_normalization': 'Custom Data Normalization',
            'custom_filtering': 'Custom Data Filtering',
            'custom_aggregation': 'Custom Data Aggregation',
            'custom_transformation': 'Custom Data Transformation'
        }
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing data processing plugin")
            
            # Perform initialization tasks
            # e.g., load processing methods, setup resources, etc.
            
            self.initialized = True
            self.logger.info("Data processing plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data processing plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing data processing plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Get processing method
            method = parameters.get('method', 'custom_normalization') if parameters else 'custom_normalization'
            
            # Execute processing
            result = await self._execute_processing(input_data, method, parameters)
            
            self.logger.info(f"Data processing plugin executed successfully with method: {method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute data processing plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up data processing plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("Data processing plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup data processing plugin: {e}")
            return False
    
    async def _execute_processing(
        self, 
        input_data: Any, 
        method: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute processing method."""
        if method == 'custom_normalization':
            return await self._custom_normalization_processing(input_data, parameters)
        elif method == 'custom_filtering':
            return await self._custom_filtering_processing(input_data, parameters)
        elif method == 'custom_aggregation':
            return await self._custom_aggregation_processing(input_data, parameters)
        elif method == 'custom_transformation':
            return await self._custom_transformation_processing(input_data, parameters)
        else:
            raise ValueError(f"Unknown processing method: {method}")
    
    async def _custom_normalization_processing(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom normalization processing."""
        self.logger.info("Executing custom normalization processing")
        
        # Example custom normalization logic
        result = {
            'method': 'custom_normalization',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'processing': {
                'type': 'normalization',
                'data': input_data,
                'metadata': {
                    'processed_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example processing
        if isinstance(input_data, list):
            # Example normalization: z-score
            if len(input_data) > 1:
                mean_val = np.mean(input_data)
                std_val = np.std(input_data)
                if std_val > 0:
                    normalized_data = [(x - mean_val) / std_val for x in input_data]
                    result['processing']['data'] = normalized_data
                    result['processing']['metadata']['normalization_stats'] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min(normalized_data),
                        'max': max(normalized_data)
                    }
        
        return result
    
    async def _custom_filtering_processing(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom filtering processing."""
        self.logger.info("Executing custom filtering processing")
        
        # Example custom filtering logic
        result = {
            'method': 'custom_filtering',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'processing': {
                'type': 'filtering',
                'data': input_data,
                'metadata': {
                    'processed_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example filtering
        if isinstance(input_data, list):
            # Example filtering: remove duplicates
            filtered_data = list(set(input_data))
            result['processing']['data'] = filtered_data
            result['processing']['metadata']['filtering_stats'] = {
                'original_count': len(input_data),
                'filtered_count': len(filtered_data),
                'removed_count': len(input_data) - len(filtered_data)
            }
        
        return result
    
    async def _custom_aggregation_processing(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom aggregation processing."""
        self.logger.info("Executing custom aggregation processing")
        
        # Example custom aggregation logic
        result = {
            'method': 'custom_aggregation',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'processing': {
                'type': 'aggregation',
                'data': input_data,
                'metadata': {
                    'processed_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example aggregation
        if isinstance(input_data, list):
            # Example aggregation: group by value and count
            from collections import Counter
            aggregated_data = dict(Counter(input_data))
            result['processing']['data'] = aggregated_data
            result['processing']['metadata']['aggregation_stats'] = {
                'original_count': len(input_data),
                'unique_values': len(aggregated_data),
                'most_common': max(aggregated_data.items(), key=lambda x: x[1]) if aggregated_data else None
            }
        
        return result
    
    async def _custom_transformation_processing(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Custom transformation processing."""
        self.logger.info("Executing custom transformation processing")
        
        # Example custom transformation logic
        result = {
            'method': 'custom_transformation',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'processing': {
                'type': 'transformation',
                'data': input_data,
                'metadata': {
                    'processed_at': self._get_current_timestamp(),
                    'total_records': len(input_data) if hasattr(input_data, '__len__') else 1
                }
            }
        }
        
        # Add some example transformation
        if isinstance(input_data, list):
            # Example transformation: log transformation
            try:
                transformed_data = [np.log(x) if x > 0 else 0 for x in input_data]
                result['processing']['data'] = transformed_data
                result['processing']['metadata']['transformation_stats'] = {
                    'original_min': min(input_data),
                    'original_max': max(input_data),
                    'transformed_min': min(transformed_data),
                    'transformed_max': max(transformed_data)
                }
            except Exception as e:
                self.logger.warning(f"Log transformation failed: {e}")
                # Fallback to simple transformation
                transformed_data = [x * 2 for x in input_data]
                result['processing']['data'] = transformed_data
                result['processing']['metadata']['transformation_stats'] = {
                    'transformation_type': 'multiply_by_2',
                    'original_min': min(input_data),
                    'original_max': max(input_data),
                    'transformed_min': min(transformed_data),
                    'transformed_max': max(transformed_data)
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
            if method not in self.processing_methods:
                self.logger.error(f"Invalid processing method: {method}")
                return False
        
        # Validate threshold parameter
        if 'threshold' in parameters:
            threshold = parameters['threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0:
                self.logger.error("Invalid threshold value")
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
        return ['threshold', 'output_format', 'include_metadata', 'verbose']
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {
            'method': {
                'type': 'string',
                'required': True,
                'description': 'Processing method to use',
                'options': list(self.processing_methods.keys())
            },
            'threshold': {
                'type': 'float',
                'required': False,
                'description': 'Processing threshold',
                'default': 0.05,
                'min': 0.0
            },
            'output_format': {
                'type': 'string',
                'required': False,
                'description': 'Output format for processing',
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
