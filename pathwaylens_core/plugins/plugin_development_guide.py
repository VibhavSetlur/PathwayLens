"""
Plugin development guide for PathwayLens.
"""

import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger

from .base_plugin import BasePlugin
from .plugin_system import PluginSystem


class PluginDevelopmentGuide:
    """Guide for developing PathwayLens plugins."""
    
    def __init__(self):
        """Initialize the plugin development guide."""
        self.logger = logger.bind(module="plugin_development_guide")
        
        # Development templates
        self.templates = {
            'plugin_template': self._get_plugin_template(),
            'config_template': self._get_config_template(),
            'test_template': self._get_test_template(),
            'documentation_template': self._get_documentation_template()
        }
    
    def get_plugin_template(self) -> str:
        """Get plugin template code."""
        return '''"""
{plugin_name} plugin for PathwayLens.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .base_plugin import BasePlugin


class {plugin_class_name}Plugin(BasePlugin):
    """Plugin for {plugin_name} functionality."""
    
    def __init__(self):
        super().__init__(
            name="{plugin_name}",
            version="1.0.0",
            description="Plugin for {plugin_name} functionality"
        )
        
        # Plugin-specific attributes
        self.author = "Your Name"
        self.license = "MIT"
        self.dependencies = []
        self.tags = ["{plugin_name}", "custom"]
        
        # Plugin state
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing {plugin_name} plugin")
            
            # Perform initialization tasks
            # e.g., load configuration, setup resources, etc.
            
            self.initialized = True
            self.logger.info("{plugin_name} plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {plugin_name} plugin: {{e}}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing {plugin_name} plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Process input data
            result = await self._process_data(input_data, parameters)
            
            self.logger.info(f"{plugin_name} plugin executed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute {plugin_name} plugin: {{e}}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up {plugin_name} plugin")
            
            # Perform cleanup tasks
            # e.g., close files, release resources, etc.
            
            self.initialized = False
            
            self.logger.info("{plugin_name} plugin cleaned up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup {plugin_name} plugin: {{e}}")
            return False
    
    async def _process_data(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data."""
        # Example processing logic
        result = {{
            'plugin_name': self.name,
            'input_data': input_data,
            'parameters': parameters or {{}},
            'processed_at': self._get_current_timestamp()
        }}
        
        return result
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        # Override in subclasses for specific validation
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return []
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return []
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {{}}
    
    def is_compatible(self, pathwaylens_version: str) -> bool:
        """Check if plugin is compatible with PathwayLens version."""
        # Plugin requires PathwayLens >= 2.0.0
        try:
            from packaging import version
            min_version = "2.0.0"
            return version.parse(pathwaylens_version) >= version.parse(min_version)
        except Exception:
            return True
'''
    
    def get_config_template(self) -> str:
        """Get configuration template."""
        return '''# Configuration for {plugin_name} plugin

plugin:
  name: {plugin_name}
  version: "1.0.0"
  description: "Plugin for {plugin_name} functionality"
  author: "Your Name"
  license: "MIT"
  dependencies: []
  tags: ["{plugin_name}", "custom"]

settings:
  enabled: true
  auto_load: true
  priority: 0

parameters:
  # Add plugin-specific parameters here
  example_parameter:
    type: string
    required: false
    description: "Example parameter"
    default: "default_value"

logging:
  enabled: true
  level: "INFO"
  file: "logs/plugins/{plugin_name}.log"

security:
  enabled: true
  sandbox: true
  allowed_imports: []
  blocked_imports: []
'''
    
    def get_test_template(self) -> str:
        """Get test template."""
        return '''"""
Tests for {plugin_name} plugin.
"""

import pytest
import asyncio
from typing import Dict, Any, Optional

from .{plugin_name} import {plugin_class_name}Plugin


class Test{plugin_class_name}Plugin:
    """Test class for {plugin_name} plugin."""
    
    def setup_method(self):
        """Setup test method."""
        self.plugin = {plugin_class_name}Plugin()
    
    def teardown_method(self):
        """Teardown test method."""
        if self.plugin.initialized:
            asyncio.run(self.plugin.cleanup())
    
    @pytest.mark.asyncio
    async def test_plugin_initialization(self):
        """Test plugin initialization."""
        success = await self.plugin.initialize()
        assert success, "Plugin initialization should succeed"
        assert self.plugin.initialized, "Plugin should be initialized"
    
    @pytest.mark.asyncio
    async def test_plugin_execution(self):
        """Test plugin execution."""
        # Initialize plugin
        await self.plugin.initialize()
        
        # Execute plugin
        result = await self.plugin.execute("test_data")
        assert result is not None, "Plugin execution should return result"
        assert result['plugin_name'] == "{plugin_name}", "Plugin name should match"
    
    @pytest.mark.asyncio
    async def test_plugin_cleanup(self):
        """Test plugin cleanup."""
        # Initialize plugin
        await self.plugin.initialize()
        
        # Cleanup plugin
        success = await self.plugin.cleanup()
        assert success, "Plugin cleanup should succeed"
        assert not self.plugin.initialized, "Plugin should not be initialized"
    
    def test_plugin_metadata(self):
        """Test plugin metadata."""
        metadata = self.plugin.get_metadata()
        assert metadata['name'] == "{plugin_name}", "Plugin name should match"
        assert metadata['version'] == "1.0.0", "Plugin version should match"
        assert metadata['description'] == "Plugin for {plugin_name} functionality", "Plugin description should match"
    
    def test_plugin_parameters(self):
        """Test plugin parameters."""
        required_params = self.plugin.get_required_parameters()
        optional_params = self.plugin.get_optional_parameters()
        param_info = self.plugin.get_parameter_info()
        
        assert isinstance(required_params, list), "Required parameters should be a list"
        assert isinstance(optional_params, list), "Optional parameters should be a list"
        assert isinstance(param_info, dict), "Parameter info should be a dictionary"
    
    def test_plugin_compatibility(self):
        """Test plugin compatibility."""
        assert self.plugin.is_compatible("2.0.0"), "Plugin should be compatible with PathwayLens 2.0.0"
        assert self.plugin.is_compatible("2.1.0"), "Plugin should be compatible with PathwayLens 2.1.0"
        assert not self.plugin.is_compatible("1.9.0"), "Plugin should not be compatible with PathwayLens 1.9.0"
'''
    
    def get_documentation_template(self) -> str:
        """Get documentation template."""
        return '''# {plugin_name} Plugin

## Overview

This plugin provides functionality for {plugin_name} in PathwayLens.

## Installation

The plugin is automatically loaded when PathwayLens starts. No additional installation is required.

## Usage

### Basic Usage

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()

# Execute plugin
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data="your_data_here",
    parameters={{"method": "default_method"}}
)
```

### Parameters

| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|----------|
| method | string | Yes | Method to use | default_method |
| threshold | float | No | Significance threshold | 0.05 |
| output_format | string | No | Output format | json |

## Examples

### Basic Example

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()

# Execute plugin with basic data
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data=["gene1", "gene2", "gene3"]
)

print(result)
```

### Advanced Example

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()

# Execute plugin with custom parameters
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data={{"genes": ["gene1", "gene2"], "expression": [1.5, 2.3]}},
    parameters={{
        "method": "advanced_method",
        "threshold": 0.01,
        "output_format": "csv"
    }}
)

print(result)
```

## API Reference

### Methods

#### `initialize() -> bool`

Initialize the plugin.

**Returns:**
- `bool`: True if initialization successful, False otherwise

#### `execute(input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any`

Execute the plugin.

**Parameters:**
- `input_data`: Input data for the plugin
- `parameters`: Optional parameters for execution

**Returns:**
- `Any`: Plugin execution result

#### `cleanup() -> bool`

Cleanup plugin resources.

**Returns:**
- `bool`: True if cleanup successful, False otherwise

## Configuration

The plugin can be configured through the PathwayLens configuration system.

## Dependencies

No dependencies required.

## License

MIT

## Author

Your Name

## Version

1.0.0

## Tags

{plugin_name}, custom

## Compatibility

- PathwayLens: >= 2.0.0
- Python: >= 3.8

## Support

For support and questions, please refer to the PathwayLens documentation or contact the plugin author.
'''
    
    def create_plugin_project(self, plugin_name: str, output_dir: str = "plugins") -> bool:
        """
        Create a new plugin project.
        
        Args:
            plugin_name: Name of the plugin
            output_dir: Output directory for the plugin project
            
        Returns:
            True if project creation successful, False otherwise
        """
        try:
            import os
            
            # Create plugin directory
            plugin_dir = os.path.join(output_dir, plugin_name)
            os.makedirs(plugin_dir, exist_ok=True)
            
            # Create plugin file
            plugin_file = os.path.join(plugin_dir, f"{plugin_name}.py")
            plugin_class_name = ''.join(word.capitalize() for word in plugin_name.split('_'))
            plugin_content = self.get_plugin_template().format(
                plugin_name=plugin_name,
                plugin_class_name=plugin_class_name
            )
            
            with open(plugin_file, 'w') as f:
                f.write(plugin_content)
            
            # Create __init__.py file
            init_file = os.path.join(plugin_dir, "__init__.py")
            with open(init_file, 'w') as f:
                f.write(f'"""Plugin package for {plugin_name}."""\n')
            
            # Create config file
            config_file = os.path.join(plugin_dir, "config.yml")
            config_content = self.get_config_template().format(plugin_name=plugin_name)
            
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Create test file
            test_file = os.path.join(plugin_dir, f"test_{plugin_name}.py")
            test_content = self.get_test_template().format(
                plugin_name=plugin_name,
                plugin_class_name=plugin_class_name
            )
            
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            # Create documentation file
            doc_file = os.path.join(plugin_dir, "README.md")
            doc_content = self.get_documentation_template().format(plugin_name=plugin_name)
            
            with open(doc_file, 'w') as f:
                f.write(doc_content)
            
            self.logger.info(f"Created plugin project: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create plugin project {plugin_name}: {e}")
            return False
    
    def get_development_guide(self) -> str:
        """Get plugin development guide."""
        return """# PathwayLens Plugin Development Guide

## Introduction

This guide will help you develop plugins for PathwayLens. Plugins allow you to extend PathwayLens functionality with custom analysis methods, visualizations, and data processing tools.

## Prerequisites

- PathwayLens 2.0.0 or higher
- Python 3.8 or higher
- Basic knowledge of Python and bioinformatics
- Understanding of PathwayLens architecture

## Plugin Architecture

### Base Plugin Class

All plugins must inherit from the `BasePlugin` class and implement the following methods:

- `initialize()`: Initialize the plugin
- `execute()`: Execute the plugin functionality
- `cleanup()`: Cleanup plugin resources

### Plugin Structure

```
plugins/
├── your_plugin/
│   ├── __init__.py
│   ├── your_plugin.py
│   ├── config.yml
│   ├── test_your_plugin.py
│   └── README.md
```

## Creating a New Plugin

### Step 1: Create Plugin Project

Use the plugin development guide to create a new plugin project:

```python
from pathwaylens_core.plugins import PluginDevelopmentGuide

guide = PluginDevelopmentGuide()
guide.create_plugin_project("your_plugin_name")
```

### Step 2: Implement Plugin Logic

Edit the generated plugin file to implement your functionality:

```python
class YourPluginPlugin(BasePlugin):
    def __init__(self):
        super().__init__(
            name="your_plugin_name",
            version="1.0.0",
            description="Your plugin description"
        )
        
        # Plugin-specific attributes
        self.author = "Your Name"
        self.license = "MIT"
        self.dependencies = ["pandas", "numpy"]
        self.tags = ["analysis", "custom"]
    
    async def initialize(self) -> bool:
        # Initialize your plugin
        return True
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        # Implement your plugin logic
        return {"result": "your_result"}
    
    async def cleanup(self) -> bool:
        # Cleanup resources
        return True
```

### Step 3: Configure Plugin

Edit the configuration file to set up your plugin:

```yaml
plugin:
  name: your_plugin_name
  version: "1.0.0"
  description: "Your plugin description"
  author: "Your Name"
  license: "MIT"
  dependencies: ["pandas", "numpy"]
  tags: ["analysis", "custom"]

settings:
  enabled: true
  auto_load: true
  priority: 0

parameters:
  your_parameter:
    type: string
    required: true
    description: "Your parameter description"
    default: "default_value"
```

### Step 4: Test Plugin

Write tests for your plugin:

```python
import pytest
from your_plugin import YourPluginPlugin

class TestYourPluginPlugin:
    @pytest.mark.asyncio
    async def test_plugin_initialization(self):
        plugin = YourPluginPlugin()
        success = await plugin.initialize()
        assert success
    
    @pytest.mark.asyncio
    async def test_plugin_execution(self):
        plugin = YourPluginPlugin()
        await plugin.initialize()
        result = await plugin.execute("test_data")
        assert result is not None
```

### Step 5: Document Plugin

Create documentation for your plugin:

```markdown
# Your Plugin

## Overview

Your plugin description.

## Usage

```python
from pathwaylens_core.plugins import PluginSystem

plugin_system = PluginSystem()
await plugin_system.initialize()

result = await plugin_system.execute_plugin(
    "your_plugin_name",
    input_data="your_data",
    parameters={"your_parameter": "value"}
)
```

## Parameters

| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|----------|
| your_parameter | string | Yes | Your parameter description | default_value |
```

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
    try:
        # Your plugin logic
        return result
    except ValueError as e:
        self.logger.error(f"Parameter error: {e}")
        raise
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
        raise
```

### 2. Parameter Validation

Validate input parameters:

```python
def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
    if 'required_param' not in parameters:
        self.logger.error("Required parameter 'required_param' not provided")
        return False
    
    if not isinstance(parameters['required_param'], str):
        self.logger.error("Parameter 'required_param' must be a string")
        return False
    
    return True
```

### 3. Resource Management

Properly manage resources:

```python
async def initialize(self) -> bool:
    try:
        # Initialize resources
        self.resource = open("file.txt", "r")
        return True
    except Exception as e:
        self.logger.error(f"Failed to initialize resources: {e}")
        return False

async def cleanup(self) -> bool:
    try:
        # Cleanup resources
        if hasattr(self, 'resource'):
            self.resource.close()
        return True
    except Exception as e:
        self.logger.error(f"Failed to cleanup resources: {e}")
        return False
```

### 4. Logging

Use proper logging:

```python
import logging

class YourPluginPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"plugin_{self.name}")
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        self.logger.info("Starting plugin execution")
        
        try:
            # Your logic
            result = self._process_data(input_data)
            self.logger.info("Plugin execution completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Plugin execution failed: {e}")
            raise
```

## Plugin Categories

### Analysis Plugins

Analysis plugins perform data analysis:

```python
class AnalysisPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["analysis", "statistics"]
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        # Perform analysis
        return analysis_result
```

### Visualization Plugins

Visualization plugins create visualizations:

```python
class VisualizationPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["visualization", "plotting"]
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        # Create visualization
        return visualization_result
```

### Data Processing Plugins

Data processing plugins process data:

```python
class DataProcessingPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["data_processing", "transformation"]
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        # Process data
        return processed_data
```

## Testing

### Unit Tests

Write unit tests for your plugin:

```python
import pytest
from your_plugin import YourPluginPlugin

class TestYourPluginPlugin:
    @pytest.mark.asyncio
    async def test_plugin_initialization(self):
        plugin = YourPluginPlugin()
        success = await plugin.initialize()
        assert success
    
    @pytest.mark.asyncio
    async def test_plugin_execution(self):
        plugin = YourPluginPlugin()
        await plugin.initialize()
        result = await plugin.execute("test_data")
        assert result is not None
    
    def test_plugin_metadata(self):
        plugin = YourPluginPlugin()
        metadata = plugin.get_metadata()
        assert metadata['name'] == "your_plugin_name"
```

### Integration Tests

Write integration tests:

```python
import pytest
from pathwaylens_core.plugins import PluginSystem

class TestYourPluginIntegration:
    @pytest.mark.asyncio
    async def test_plugin_integration(self):
        plugin_system = PluginSystem()
        await plugin_system.initialize()
        
        result = await plugin_system.execute_plugin(
            "your_plugin_name",
            input_data="test_data"
        )
        
        assert result is not None
```

## Deployment

### Local Development

For local development, place your plugin in the `plugins` directory:

```
plugins/
├── your_plugin/
│   ├── __init__.py
│   ├── your_plugin.py
│   ├── config.yml
│   ├── test_your_plugin.py
│   └── README.md
```

### Production Deployment

For production deployment, package your plugin:

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="pathwaylens-your-plugin",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pathwaylens>=2.0.0",
        "pandas",
        "numpy"
    ],
    entry_points={
        "pathwaylens.plugins": [
            "your_plugin = your_plugin:YourPluginPlugin"
        ]
    }
)
```

## Troubleshooting

### Common Issues

1. **Plugin not loading**: Check plugin directory structure and imports
2. **Parameter errors**: Validate parameter names and types
3. **Execution errors**: Check input data format and plugin logic
4. **Resource errors**: Ensure proper resource management

### Debug Mode

Enable debug mode for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your plugin code
```

## Resources

- [PathwayLens Documentation](https://pathwaylens.readthedocs.io/)
- [Plugin API Reference](https://pathwaylens.readthedocs.io/en/latest/plugins/api/)
- [Community Forum](https://github.com/pathwaylens/pathwaylens/discussions)
- [Plugin Examples](https://github.com/pathwaylens/pathwaylens/tree/main/plugins/examples)
"""
    
    def get_plugin_categories_guide(self) -> str:
        """Get plugin categories guide."""
        return """# PathwayLens Plugin Categories

## Analysis Plugins

Analysis plugins perform data analysis and statistical computations.

### Characteristics

- Process biological data
- Perform statistical analysis
- Generate analysis results
- Support multiple analysis methods

### Example

```python
class AnalysisPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["analysis", "statistics"]
        self.analysis_methods = {
            'method1': 'Method 1 Description',
            'method2': 'Method 2 Description'
        }
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        method = parameters.get('method', 'method1')
        
        if method == 'method1':
            return await self._method1_analysis(input_data)
        elif method == 'method2':
            return await self._method2_analysis(input_data)
        else:
            raise ValueError(f"Unknown analysis method: {method}")
```

## Visualization Plugins

Visualization plugins create visual representations of data.

### Characteristics

- Generate plots and charts
- Support multiple output formats
- Interactive visualizations
- Customizable appearance

### Example

```python
class VisualizationPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["visualization", "plotting"]
        self.plot_types = {
            'scatter': 'Scatter Plot',
            'line': 'Line Plot',
            'bar': 'Bar Chart'
        }
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        plot_type = parameters.get('plot_type', 'scatter')
        
        if plot_type == 'scatter':
            return await self._create_scatter_plot(input_data)
        elif plot_type == 'line':
            return await self._create_line_plot(input_data)
        elif plot_type == 'bar':
            return await self._create_bar_chart(input_data)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
```

## Data Processing Plugins

Data processing plugins transform and manipulate data.

### Characteristics

- Data transformation
- Format conversion
- Data cleaning
- Data validation

### Example

```python
class DataProcessingPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["data_processing", "transformation"]
        self.processing_methods = {
            'normalize': 'Data Normalization',
            'filter': 'Data Filtering',
            'aggregate': 'Data Aggregation'
        }
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        method = parameters.get('method', 'normalize')
        
        if method == 'normalize':
            return await self._normalize_data(input_data)
        elif method == 'filter':
            return await self._filter_data(input_data)
        elif method == 'aggregate':
            return await self._aggregate_data(input_data)
        else:
            raise ValueError(f"Unknown processing method: {method}")
```

## Export Plugins

Export plugins save data in various formats.

### Characteristics

- Multiple output formats
- Data serialization
- File generation
- Format validation

### Example

```python
class ExportPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["export", "data"]
        self.export_formats = {
            'json': 'JSON Format',
            'csv': 'CSV Format',
            'excel': 'Excel Format'
        }
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        format_type = parameters.get('format', 'json')
        
        if format_type == 'json':
            return await self._export_json(input_data)
        elif format_type == 'csv':
            return await self._export_csv(input_data)
        elif format_type == 'excel':
            return await self._export_excel(input_data)
        else:
            raise ValueError(f"Unknown export format: {format_type}")
```

## Import Plugins

Import plugins load data from various sources.

### Characteristics

- Multiple input formats
- Data parsing
- Format detection
- Data validation

### Example

```python
class ImportPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["import", "data"]
        self.import_formats = {
            'json': 'JSON Format',
            'csv': 'CSV Format',
            'excel': 'Excel Format'
        }
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        format_type = parameters.get('format', 'json')
        
        if format_type == 'json':
            return await self._import_json(input_data)
        elif format_type == 'csv':
            return await self._import_csv(input_data)
        elif format_type == 'excel':
            return await self._import_excel(input_data)
        else:
            raise ValueError(f"Unknown import format: {format_type}")
```

## Custom Plugins

Custom plugins provide specialized functionality.

### Characteristics

- Unique functionality
- Domain-specific features
- Custom algorithms
- Specialized tools

### Example

```python
class CustomPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.tags = ["custom", "specialized"]
        self.custom_methods = {
            'method1': 'Custom Method 1',
            'method2': 'Custom Method 2'
        }
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        method = parameters.get('method', 'method1')
        
        if method == 'method1':
            return await self._custom_method1(input_data)
        elif method == 'method2':
            return await self._custom_method2(input_data)
        else:
            raise ValueError(f"Unknown custom method: {method}")
```

## Choosing a Category

When developing a plugin, choose the appropriate category based on its primary function:

- **Analysis**: Statistical analysis, data mining, pattern recognition
- **Visualization**: Plotting, charting, visual representation
- **Data Processing**: Data transformation, cleaning, validation
- **Export**: Saving data in various formats
- **Import**: Loading data from various sources
- **Custom**: Specialized functionality not covered by other categories

## Category Tags

Use appropriate tags to categorize your plugin:

```python
class YourPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        # Primary category
        self.tags = ["analysis"]
        
        # Additional tags
        self.tags.extend(["statistics", "bioinformatics", "custom"])
```

## Best Practices

1. **Choose the right category**: Select the category that best describes your plugin's primary function
2. **Use appropriate tags**: Tag your plugin with relevant keywords
3. **Follow category conventions**: Adhere to the patterns and conventions of your chosen category
4. **Document category-specific features**: Explain how your plugin fits within its category
5. **Test category-specific functionality**: Ensure your plugin works as expected within its category
"""
