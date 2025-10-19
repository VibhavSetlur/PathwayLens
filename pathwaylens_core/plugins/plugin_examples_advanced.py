"""
Advanced plugin examples for PathwayLens.
"""

import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
import pandas as pd
import numpy as np

from .base_plugin import BasePlugin


class AdvancedAnalysisPlugin(BasePlugin):
    """Advanced analysis plugin for PathwayLens."""
    
    def __init__(self):
        super().__init__(
            name="advanced_analysis_plugin",
            version="1.0.0",
            description="Advanced analysis plugin for PathwayLens"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = ["pandas", "numpy", "scipy", "scikit-learn"]
        self.tags = ["analysis", "advanced", "machine_learning"]
        
        # Plugin state
        self.initialized = False
        self.analysis_methods = {
            'pca': 'Principal Component Analysis',
            'clustering': 'Clustering Analysis',
            'classification': 'Classification Analysis',
            'regression': 'Regression Analysis',
            'feature_selection': 'Feature Selection'
        }
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing advanced analysis plugin")
            
            # Perform initialization tasks
            # e.g., load analysis methods, setup resources, etc.
            
            self.initialized = True
            self.logger.info("Advanced analysis plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced analysis plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing advanced analysis plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Get analysis method
            method = parameters.get('method', 'pca') if parameters else 'pca'
            
            # Execute analysis
            result = await self._execute_analysis(input_data, method, parameters)
            
            self.logger.info(f"Advanced analysis plugin executed successfully with method: {method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute advanced analysis plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up advanced analysis plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("Advanced analysis plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup advanced analysis plugin: {e}")
            return False
    
    async def _execute_analysis(
        self, 
        input_data: Any, 
        method: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute analysis method."""
        if method == 'pca':
            return await self._pca_analysis(input_data, parameters)
        elif method == 'clustering':
            return await self._clustering_analysis(input_data, parameters)
        elif method == 'classification':
            return await self._classification_analysis(input_data, parameters)
        elif method == 'regression':
            return await self._regression_analysis(input_data, parameters)
        elif method == 'feature_selection':
            return await self._feature_selection_analysis(input_data, parameters)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
    
    async def _pca_analysis(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Principal Component Analysis."""
        self.logger.info("Executing PCA analysis")
        
        # Example PCA analysis logic
        result = {
            'method': 'pca',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'analysis_results': {
                'pca': {},
                'summary': {}
            }
        }
        
        # Add some example PCA results
        if isinstance(input_data, list):
            if len(input_data) > 1:
                # Simple PCA example
                result['analysis_results']['pca'] = {
                    'components': 2,  # Example number of components
                    'explained_variance_ratio': [0.6, 0.3],  # Example explained variance
                    'cumulative_variance': [0.6, 0.9],  # Example cumulative variance
                    'eigenvalues': [2.4, 1.2],  # Example eigenvalues
                    'loadings': {
                        'PC1': [0.8, 0.6, 0.4],  # Example loadings for PC1
                        'PC2': [0.4, 0.8, 0.6]   # Example loadings for PC2
                    }
                }
                
                result['analysis_results']['summary'] = {
                    'total_variance_explained': 0.9,
                    'components_retained': 2,
                    'dimensionality_reduction': True,
                    'data_quality': 'good'
                }
        
        return result
    
    async def _clustering_analysis(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Clustering analysis."""
        self.logger.info("Executing clustering analysis")
        
        # Example clustering analysis logic
        result = {
            'method': 'clustering',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'analysis_results': {
                'clustering': {},
                'summary': {}
            }
        }
        
        # Add some example clustering results
        if isinstance(input_data, list):
            if len(input_data) > 1:
                # Simple clustering example
                result['analysis_results']['clustering'] = {
                    'algorithm': 'kmeans',
                    'n_clusters': 3,
                    'cluster_labels': [0, 1, 2, 0, 1, 2] * (len(input_data) // 6 + 1),
                    'centroids': [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]
                    ],
                    'inertia': 15.5,  # Example inertia
                    'silhouette_score': 0.75  # Example silhouette score
                }
                
                result['analysis_results']['summary'] = {
                    'optimal_clusters': 3,
                    'clustering_quality': 'good',
                    'separation': 'clear',
                    'stability': 'high'
                }
        
        return result
    
    async def _classification_analysis(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Classification analysis."""
        self.logger.info("Executing classification analysis")
        
        # Example classification analysis logic
        result = {
            'method': 'classification',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'analysis_results': {
                'classification': {},
                'summary': {}
            }
        }
        
        # Add some example classification results
        if isinstance(input_data, list):
            if len(input_data) > 1:
                # Simple classification example
                result['analysis_results']['classification'] = {
                    'algorithm': 'random_forest',
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85,
                    'confusion_matrix': [
                        [20, 3],
                        [2, 25]
                    ],
                    'feature_importance': {
                        'feature1': 0.4,
                        'feature2': 0.3,
                        'feature3': 0.2,
                        'feature4': 0.1
                    }
                }
                
                result['analysis_results']['summary'] = {
                    'model_performance': 'good',
                    'overfitting': 'low',
                    'generalization': 'high',
                    'feature_selection': 'effective'
                }
        
        return result
    
    async def _regression_analysis(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Regression analysis."""
        self.logger.info("Executing regression analysis")
        
        # Example regression analysis logic
        result = {
            'method': 'regression',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'analysis_results': {
                'regression': {},
                'summary': {}
            }
        }
        
        # Add some example regression results
        if isinstance(input_data, list):
            if len(input_data) > 1:
                # Simple regression example
                result['analysis_results']['regression'] = {
                    'algorithm': 'linear_regression',
                    'r_squared': 0.75,
                    'adjusted_r_squared': 0.72,
                    'mse': 0.15,
                    'rmse': 0.39,
                    'mae': 0.32,
                    'coefficients': {
                        'intercept': 1.0,
                        'slope': 0.5
                    },
                    'p_values': {
                        'intercept': 0.001,
                        'slope': 0.005
                    }
                }
                
                result['analysis_results']['summary'] = {
                    'model_fit': 'good',
                    'significance': 'high',
                    'prediction_accuracy': 'moderate',
                    'residuals': 'normal'
                }
        
        return result
    
    async def _feature_selection_analysis(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Feature selection analysis."""
        self.logger.info("Executing feature selection analysis")
        
        # Example feature selection analysis logic
        result = {
            'method': 'feature_selection',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'analysis_results': {
                'feature_selection': {},
                'summary': {}
            }
        }
        
        # Add some example feature selection results
        if isinstance(input_data, list):
            if len(input_data) > 1:
                # Simple feature selection example
                result['analysis_results']['feature_selection'] = {
                    'method': 'mutual_information',
                    'selected_features': ['feature1', 'feature2', 'feature3'],
                    'feature_scores': {
                        'feature1': 0.8,
                        'feature2': 0.6,
                        'feature3': 0.4,
                        'feature4': 0.2
                    },
                    'threshold': 0.3,
                    'n_features_selected': 3
                }
                
                result['analysis_results']['summary'] = {
                    'feature_quality': 'high',
                    'redundancy': 'low',
                    'relevance': 'high',
                    'dimensionality': 'reduced'
                }
        
        return result
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        # Validate method parameter
        if 'method' in parameters:
            method = parameters['method']
            if method not in self.analysis_methods:
                self.logger.error(f"Invalid analysis method: {method}")
                return False
        
        # Validate threshold parameter
        if 'threshold' in parameters:
            threshold = parameters['threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                self.logger.error("Invalid threshold value")
                return False
        
        # Validate n_components parameter
        if 'n_components' in parameters:
            n_components = parameters['n_components']
            if not isinstance(n_components, int) or n_components < 1:
                self.logger.error("Invalid n_components value")
                return False
        
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return ['method']
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return ['threshold', 'n_components', 'output_format', 'verbose']
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {
            'method': {
                'type': 'string',
                'required': True,
                'description': 'Analysis method to use',
                'options': list(self.analysis_methods.keys())
            },
            'threshold': {
                'type': 'float',
                'required': False,
                'description': 'Significance threshold',
                'default': 0.05,
                'min': 0.0,
                'max': 1.0
            },
            'n_components': {
                'type': 'integer',
                'required': False,
                'description': 'Number of components for PCA',
                'default': 2,
                'min': 1,
                'max': 10
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
        # Plugin requires PathwayLens >= 2.0.0
        try:
            from packaging import version
            min_version = "2.0.0"
            return version.parse(pathwaylens_version) >= version.parse(min_version)
        except Exception:
            return True


class AdvancedVisualizationPlugin(BasePlugin):
    """Advanced visualization plugin for PathwayLens."""
    
    def __init__(self):
        super().__init__(
            name="advanced_visualization_plugin",
            version="1.0.0",
            description="Advanced visualization plugin for PathwayLens"
        )
        
        # Plugin-specific attributes
        self.author = "PathwayLens Team"
        self.license = "MIT"
        self.dependencies = ["plotly", "matplotlib", "seaborn", "bokeh"]
        self.tags = ["visualization", "advanced", "interactive"]
        
        # Plugin state
        self.initialized = False
        self.visualization_methods = {
            'heatmap': 'Heatmap Visualization',
            'network': 'Network Visualization',
            '3d_scatter': '3D Scatter Plot',
            'dashboard': 'Interactive Dashboard',
            'animation': 'Animated Visualization'
        }
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing advanced visualization plugin")
            
            # Perform initialization tasks
            # e.g., load visualization methods, setup resources, etc.
            
            self.initialized = True
            self.logger.info("Advanced visualization plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced visualization plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing advanced visualization plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Get visualization method
            method = parameters.get('method', 'heatmap') if parameters else 'heatmap'
            
            # Execute visualization
            result = await self._execute_visualization(input_data, method, parameters)
            
            self.logger.info(f"Advanced visualization plugin executed successfully with method: {method}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute advanced visualization plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up advanced visualization plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("Advanced visualization plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup advanced visualization plugin: {e}")
            return False
    
    async def _execute_visualization(
        self, 
        input_data: Any, 
        method: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute visualization method."""
        if method == 'heatmap':
            return await self._heatmap_visualization(input_data, parameters)
        elif method == 'network':
            return await self._network_visualization(input_data, parameters)
        elif method == '3d_scatter':
            return await self._3d_scatter_visualization(input_data, parameters)
        elif method == 'dashboard':
            return await self._dashboard_visualization(input_data, parameters)
        elif method == 'animation':
            return await self._animation_visualization(input_data, parameters)
        else:
            raise ValueError(f"Unknown visualization method: {method}")
    
    async def _heatmap_visualization(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Heatmap visualization."""
        self.logger.info("Executing heatmap visualization")
        
        # Example heatmap visualization logic
        result = {
            'method': 'heatmap',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'visualization': {
                'type': 'heatmap',
                'data': [],
                'metadata': {
                    'title': 'Advanced Heatmap',
                    'x_axis': 'Samples',
                    'y_axis': 'Features',
                    'color_scale': 'viridis',
                    'interactive': True
                }
            }
        }
        
        # Add some example heatmap data
        if isinstance(input_data, list):
            # Create example heatmap data
            heatmap_data = []
            for i, item in enumerate(input_data[:10]):  # First 10 items
                row = {
                    'feature': f'Feature_{i}',
                    'samples': {}
                }
                for j in range(5):  # 5 samples
                    # Example expression values
                    import random
                    row['samples'][f'Sample_{j}'] = random.uniform(-2, 2)
                heatmap_data.append(row)
            
            result['visualization']['data'] = heatmap_data
        
        return result
    
    async def _network_visualization(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Network visualization."""
        self.logger.info("Executing network visualization")
        
        # Example network visualization logic
        result = {
            'method': 'network',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'visualization': {
                'type': 'network',
                'nodes': [],
                'edges': [],
                'metadata': {
                    'title': 'Advanced Network',
                    'layout': 'force',
                    'node_size': 'degree',
                    'edge_width': 'weight',
                    'interactive': True
                }
            }
        }
        
        # Add some example network data
        if isinstance(input_data, list):
            # Create example network nodes
            nodes = []
            for i, item in enumerate(input_data[:15]):  # First 15 items
                nodes.append({
                    'id': f'node_{i}',
                    'label': str(item),
                    'size': 10 + i,
                    'color': f'color_{i % 5}',
                    'degree': i + 1
                })
            
            # Create example network edges
            edges = []
            for i in range(len(nodes) - 1):
                edges.append({
                    'source': nodes[i]['id'],
                    'target': nodes[i + 1]['id'],
                    'weight': 0.5 + (i * 0.1),
                    'type': 'interaction'
                })
            
            result['visualization']['nodes'] = nodes
            result['visualization']['edges'] = edges
        
        return result
    
    async def _3d_scatter_visualization(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """3D scatter plot visualization."""
        self.logger.info("Executing 3D scatter plot visualization")
        
        # Example 3D scatter plot visualization logic
        result = {
            'method': '3d_scatter',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'visualization': {
                'type': '3d_scatter',
                'data': [],
                'metadata': {
                    'title': '3D Scatter Plot',
                    'x_axis': 'X Values',
                    'y_axis': 'Y Values',
                    'z_axis': 'Z Values',
                    'interactive': True
                }
            }
        }
        
        # Add some example 3D scatter plot data
        if isinstance(input_data, list):
            # Create example 3D scatter plot data
            scatter_3d_data = []
            for i, item in enumerate(input_data[:20]):  # First 20 items
                scatter_3d_data.append({
                    'x': i,
                    'y': hash(str(item)) % 100,  # Simple hash-based y value
                    'z': hash(str(item)) % 50,   # Simple hash-based z value
                    'label': str(item),
                    'color': f'color_{i % 5}',
                    'size': 5 + (i % 10)
                })
            
            result['visualization']['data'] = scatter_3d_data
        
        return result
    
    async def _dashboard_visualization(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Interactive dashboard visualization."""
        self.logger.info("Executing dashboard visualization")
        
        # Example dashboard visualization logic
        result = {
            'method': 'dashboard',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'visualization': {
                'type': 'dashboard',
                'components': [],
                'metadata': {
                    'title': 'Interactive Dashboard',
                    'layout': 'grid',
                    'interactive': True,
                    'responsive': True
                }
            }
        }
        
        # Add some example dashboard components
        if isinstance(input_data, list):
            # Create example dashboard components
            components = [
                {
                    'type': 'summary_statistics',
                    'title': 'Summary Statistics',
                    'data': {
                        'total_items': len(input_data),
                        'unique_items': len(set(input_data)),
                        'sample_items': input_data[:5] if len(input_data) > 5 else input_data
                    }
                },
                {
                    'type': 'distribution_plot',
                    'title': 'Distribution',
                    'data': {
                        'values': input_data[:10] if len(input_data) > 10 else input_data
                    }
                },
                {
                    'type': 'correlation_matrix',
                    'title': 'Correlations',
                    'data': {
                        'correlations': {
                            'feature1': {'feature2': 0.8, 'feature3': 0.6},
                            'feature2': {'feature1': 0.8, 'feature3': 0.7},
                            'feature3': {'feature1': 0.6, 'feature2': 0.7}
                        }
                    }
                }
            ]
            
            result['visualization']['components'] = components
        
        return result
    
    async def _animation_visualization(
        self, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Animated visualization."""
        self.logger.info("Executing animation visualization")
        
        # Example animation visualization logic
        result = {
            'method': 'animation',
            'input_data': {
                'type': type(input_data).__name__,
                'size': len(input_data) if hasattr(input_data, '__len__') else 1
            },
            'parameters': parameters or {},
            'visualization': {
                'type': 'animation',
                'frames': [],
                'metadata': {
                    'title': 'Animated Visualization',
                    'duration': 5.0,
                    'fps': 30,
                    'loop': True,
                    'interactive': True
                }
            }
        }
        
        # Add some example animation frames
        if isinstance(input_data, list):
            # Create example animation frames
            frames = []
            for frame in range(10):  # 10 frames
                frame_data = {
                    'frame': frame,
                    'time': frame * 0.5,  # 0.5 seconds per frame
                    'data': []
                }
                
                for i, item in enumerate(input_data[:10]):  # First 10 items
                    frame_data['data'].append({
                        'x': i + frame * 0.1,  # Animate x position
                        'y': hash(str(item)) % 100,  # Simple hash-based y value
                        'label': str(item),
                        'color': f'color_{i % 5}',
                        'size': 5 + (frame % 10)
                    })
                
                frames.append(frame_data)
            
            result['visualization']['frames'] = frames
        
        return result
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        # Validate method parameter
        if 'method' in parameters:
            method = parameters['method']
            if method not in self.visualization_methods:
                self.logger.error(f"Invalid visualization method: {method}")
                return False
        
        # Validate output_format parameter
        if 'output_format' in parameters:
            output_format = parameters['output_format']
            valid_formats = ['html', 'png', 'svg', 'pdf', 'json']
            if output_format not in valid_formats:
                self.logger.error(f"Invalid output format: {output_format}")
                return False
        
        # Validate width and height parameters
        if 'width' in parameters:
            width = parameters['width']
            if not isinstance(width, int) or width < 100 or width > 2000:
                self.logger.error("Invalid width value")
                return False
        
        if 'height' in parameters:
            height = parameters['height']
            if not isinstance(height, int) or height < 100 or height > 2000:
                self.logger.error("Invalid height value")
                return False
        
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return ['method']
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return ['output_format', 'width', 'height', 'title', 'color_scheme', 'interactive']
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {
            'method': {
                'type': 'string',
                'required': True,
                'description': 'Visualization method to use',
                'options': list(self.visualization_methods.keys())
            },
            'output_format': {
                'type': 'string',
                'required': False,
                'description': 'Output format for visualization',
                'options': ['html', 'png', 'svg', 'pdf', 'json'],
                'default': 'html'
            },
            'width': {
                'type': 'integer',
                'required': False,
                'description': 'Width of visualization',
                'default': 800,
                'min': 100,
                'max': 2000
            },
            'height': {
                'type': 'integer',
                'required': False,
                'description': 'Height of visualization',
                'default': 600,
                'min': 100,
                'max': 2000
            },
            'title': {
                'type': 'string',
                'required': False,
                'description': 'Title for visualization',
                'default': 'Advanced Visualization'
            },
            'color_scheme': {
                'type': 'string',
                'required': False,
                'description': 'Color scheme for visualization',
                'options': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
                'default': 'viridis'
            },
            'interactive': {
                'type': 'boolean',
                'required': False,
                'description': 'Enable interactive features',
                'default': True
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
