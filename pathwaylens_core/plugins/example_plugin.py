"""
Example plugin for PathwayLens.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .base_plugin import BasePlugin


class ExamplePlugin(BasePlugin):
    """Example plugin demonstrating plugin functionality."""
    
    def __init__(self):
        super().__init__(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin for PathwayLens"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = ["pandas", "numpy"]
        self.tags = ["example", "analysis", "custom"]
        
        # Plugin state
        self.initialized = False
        self.execution_count = 0
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing example plugin")
            
            # Perform initialization tasks
            # e.g., load configuration, setup resources, etc.
            
            self.initialized = True
            self.logger.info("Example plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize example plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing example plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Process input data
            result = await self._process_data(input_data, parameters)
            
            # Update execution count
            self.execution_count += 1
            
            self.logger.info(f"Example plugin executed successfully (count: {self.execution_count})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute example plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up example plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            self.execution_count = 0
            
            self.logger.info("Example plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup example plugin: {e}")
            return False
    
    async def _process_data(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data."""
        # Example processing logic
        result = {
            'input_type': type(input_data).__name__,
            'input_size': len(input_data) if hasattr(input_data, '__len__') else 1,
            'parameters': parameters or {},
            'execution_count': self.execution_count + 1,
            'processed_at': self._get_current_timestamp()
        }
        
        # Add some example analysis
        if isinstance(input_data, list):
            result['analysis'] = {
                'total_items': len(input_data),
                'unique_items': len(set(input_data)),
                'sample_items': input_data[:5] if len(input_data) > 5 else input_data
            }
        elif isinstance(input_data, dict):
            result['analysis'] = {
                'total_keys': len(input_data),
                'key_types': {k: type(v).__name__ for k, v in input_data.items()},
                'sample_keys': list(input_data.keys())[:5]
            }
        else:
            result['analysis'] = {
                'data_type': type(input_data).__name__,
                'data_value': str(input_data)[:100]  # Truncate for logging
            }
        
        return result
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        # Example validation logic
        required_params = self.get_required_parameters()
        
        for param in required_params:
            if param not in parameters:
                self.logger.error(f"Required parameter {param} not provided")
                return False
        
        # Validate parameter types and values
        if 'threshold' in parameters:
            threshold = parameters['threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                self.logger.error("Invalid threshold value")
                return False
        
        if 'method' in parameters:
            method = parameters['method']
            valid_methods = ['method1', 'method2', 'method3']
            if method not in valid_methods:
                self.logger.error(f"Invalid method: {method}")
                return False
        
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return ['method']
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return ['threshold', 'output_format', 'verbose']
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {
            'method': {
                'type': 'string',
                'required': True,
                'description': 'Analysis method to use',
                'options': ['method1', 'method2', 'method3']
            },
            'threshold': {
                'type': 'float',
                'required': False,
                'description': 'Significance threshold',
                'default': 0.05,
                'min': 0.0,
                'max': 1.0
            },
            'output_format': {
                'type': 'string',
                'required': False,
                'description': 'Output format',
                'options': ['json', 'csv', 'tsv'],
                'default': 'json'
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
        # Example compatibility check
        try:
            from packaging import version
            
            # Plugin requires PathwayLens >= 2.0.0
            min_version = "2.0.0"
            return version.parse(pathwaylens_version) >= version.parse(min_version)
            
        except Exception:
            # If version parsing fails, assume compatible
            return True
