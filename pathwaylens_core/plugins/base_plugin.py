"""
Base plugin class for PathwayLens.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from loguru import logger


class BasePlugin(ABC):
    """Base class for all PathwayLens plugins."""
    
    def __init__(self, name: str, version: str, description: str = ""):
        """
        Initialize the plugin.
        
        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
        """
        self.name = name
        self.version = version
        self.description = description
        self.logger = logger.bind(module=f"plugin_{name}")
        
        # Plugin metadata
        self.metadata = {
            'name': name,
            'version': version,
            'description': description,
            'author': getattr(self, 'author', 'Unknown'),
            'license': getattr(self, 'license', 'Unknown'),
            'dependencies': getattr(self, 'dependencies', []),
            'tags': getattr(self, 'tags', [])
        }
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the plugin.
        
        Args:
            input_data: Input data for the plugin
            parameters: Optional parameters for execution
            
        Returns:
            Plugin execution result
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get plugin metadata.
        
        Returns:
            Plugin metadata dictionary
        """
        return self.metadata.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information.
        
        Returns:
            Plugin information dictionary
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'metadata': self.metadata
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate plugin parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid, False otherwise
        """
        # Override in subclasses for specific validation
        return True
    
    def get_required_parameters(self) -> List[str]:
        """
        Get list of required parameters.
        
        Returns:
            List of required parameter names
        """
        # Override in subclasses for specific requirements
        return []
    
    def get_optional_parameters(self) -> List[str]:
        """
        Get list of optional parameters.
        
        Returns:
            List of optional parameter names
        """
        # Override in subclasses for specific requirements
        return []
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter information.
        
        Returns:
            Dictionary with parameter information
        """
        # Override in subclasses for specific parameter info
        return {}
    
    def is_compatible(self, pathwaylens_version: str) -> bool:
        """
        Check if plugin is compatible with PathwayLens version.
        
        Args:
            pathwaylens_version: PathwayLens version
            
        Returns:
            True if compatible, False otherwise
        """
        # Override in subclasses for specific compatibility checks
        return True
    
    def get_dependencies(self) -> List[str]:
        """
        Get plugin dependencies.
        
        Returns:
            List of dependency names
        """
        return self.metadata.get('dependencies', [])
    
    def get_tags(self) -> List[str]:
        """
        Get plugin tags.
        
        Returns:
            List of plugin tags
        """
        return self.metadata.get('tags', [])
    
    def __str__(self) -> str:
        """String representation of the plugin."""
        return f"{self.name} v{self.version}"
    
    def __repr__(self) -> str:
        """Representation of the plugin."""
        return f"<{self.__class__.__name__}(name='{self.name}', version='{self.version}')>"
