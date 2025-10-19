"""
Plugin documentation for PathwayLens.
"""

import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger

from .base_plugin import BasePlugin
from .plugin_system import PluginSystem


class PluginDocumentation:
    """Documentation generator for PathwayLens plugins."""
    
    def __init__(self, plugin_system: Optional[PluginSystem] = None):
        """
        Initialize the plugin documentation generator.
        
        Args:
            plugin_system: Plugin system instance
        """
        self.logger = logger.bind(module="plugin_documentation")
        self.plugin_system = plugin_system
        
        # Documentation templates
        self.templates = {
            'plugin_readme': self._get_plugin_readme_template(),
            'plugin_api': self._get_plugin_api_template(),
            'plugin_examples': self._get_plugin_examples_template(),
            'plugin_tutorial': self._get_plugin_tutorial_template()
        }
    
    async def generate_plugin_documentation(self, plugin_name: str, output_dir: str = "docs/plugins") -> bool:
        """
        Generate documentation for a specific plugin.
        
        Args:
            plugin_name: Name of plugin to generate documentation for
            output_dir: Output directory for documentation
            
        Returns:
            True if documentation generation successful, False otherwise
        """
        try:
            if not self.plugin_system:
                self.logger.error("Plugin system not available")
                return False
            
            # Get plugin information
            plugin_info = self.plugin_system.get_plugin_info(plugin_name)
            if not plugin_info:
                self.logger.error(f"Plugin {plugin_name} not found")
                return False
            
            # Create output directory
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate documentation files
            success = True
            
            # Generate README
            readme_success = await self._generate_plugin_readme(plugin_name, plugin_info, output_dir)
            success = success and readme_success
            
            # Generate API documentation
            api_success = await self._generate_plugin_api(plugin_name, plugin_info, output_dir)
            success = success and api_success
            
            # Generate examples
            examples_success = await self._generate_plugin_examples(plugin_name, plugin_info, output_dir)
            success = success and examples_success
            
            # Generate tutorial
            tutorial_success = await self._generate_plugin_tutorial(plugin_name, plugin_info, output_dir)
            success = success and tutorial_success
            
            if success:
                self.logger.info(f"Generated documentation for plugin: {plugin_name}")
            else:
                self.logger.error(f"Failed to generate documentation for plugin: {plugin_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to generate plugin documentation: {e}")
            return False
    
    async def _generate_plugin_readme(self, plugin_name: str, plugin_info: Dict[str, Any], output_dir: str) -> bool:
        """Generate plugin README documentation."""
        try:
            # Get plugin instance
            plugin = self.plugin_system.get_plugin(plugin_name)
            if not plugin:
                return False
            
            # Generate README content
            readme_content = self._generate_readme_content(plugin_name, plugin_info, plugin)
            
            # Write README file
            readme_file = f"{output_dir}/{plugin_name}/README.md"
            import os
            os.makedirs(os.path.dirname(readme_file), exist_ok=True)
            
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            
            self.logger.info(f"Generated README for plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate README for plugin {plugin_name}: {e}")
            return False
    
    async def _generate_plugin_api(self, plugin_name: str, plugin_info: Dict[str, Any], output_dir: str) -> bool:
        """Generate plugin API documentation."""
        try:
            # Get plugin instance
            plugin = self.plugin_system.get_plugin(plugin_name)
            if not plugin:
                return False
            
            # Generate API content
            api_content = self._generate_api_content(plugin_name, plugin_info, plugin)
            
            # Write API file
            api_file = f"{output_dir}/{plugin_name}/API.md"
            import os
            os.makedirs(os.path.dirname(api_file), exist_ok=True)
            
            with open(api_file, 'w') as f:
                f.write(api_content)
            
            self.logger.info(f"Generated API documentation for plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate API documentation for plugin {plugin_name}: {e}")
            return False
    
    async def _generate_plugin_examples(self, plugin_name: str, plugin_info: Dict[str, Any], output_dir: str) -> bool:
        """Generate plugin examples documentation."""
        try:
            # Get plugin instance
            plugin = self.plugin_system.get_plugin(plugin_name)
            if not plugin:
                return False
            
            # Generate examples content
            examples_content = self._generate_examples_content(plugin_name, plugin_info, plugin)
            
            # Write examples file
            examples_file = f"{output_dir}/{plugin_name}/EXAMPLES.md"
            import os
            os.makedirs(os.path.dirname(examples_file), exist_ok=True)
            
            with open(examples_file, 'w') as f:
                f.write(examples_content)
            
            self.logger.info(f"Generated examples for plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate examples for plugin {plugin_name}: {e}")
            return False
    
    async def _generate_plugin_tutorial(self, plugin_name: str, plugin_info: Dict[str, Any], output_dir: str) -> bool:
        """Generate plugin tutorial documentation."""
        try:
            # Get plugin instance
            plugin = self.plugin_system.get_plugin(plugin_name)
            if not plugin:
                return False
            
            # Generate tutorial content
            tutorial_content = self._generate_tutorial_content(plugin_name, plugin_info, plugin)
            
            # Write tutorial file
            tutorial_file = f"{output_dir}/{plugin_name}/TUTORIAL.md"
            import os
            os.makedirs(os.path.dirname(tutorial_file), exist_ok=True)
            
            with open(tutorial_file, 'w') as f:
                f.write(tutorial_content)
            
            self.logger.info(f"Generated tutorial for plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate tutorial for plugin {plugin_name}: {e}")
            return False
    
    def _generate_readme_content(self, plugin_name: str, plugin_info: Dict[str, Any], plugin: BasePlugin) -> str:
        """Generate README content for a plugin."""
        return f"""# {plugin_name}

{plugin_info.get('description', 'No description available')}

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

{self._generate_parameters_documentation(plugin)}

## Examples

See [EXAMPLES.md](EXAMPLES.md) for detailed usage examples.

## API Reference

See [API.md](API.md) for complete API documentation.

## Tutorial

See [TUTORIAL.md](TUTORIAL.md) for a step-by-step tutorial.

## Configuration

The plugin can be configured through the PathwayLens configuration system.

## Dependencies

{self._generate_dependencies_documentation(plugin)}

## License

{plugin_info.get('license', 'Unknown')}

## Author

{plugin_info.get('author', 'Unknown')}

## Version

{plugin_info.get('version', 'Unknown')}

## Tags

{', '.join(plugin.get_tags())}

## Compatibility

- PathwayLens: >= 2.0.0
- Python: >= 3.8

## Support

For support and questions, please refer to the PathwayLens documentation or contact the plugin author.
"""
    
    def _generate_api_content(self, plugin_name: str, plugin_info: Dict[str, Any], plugin: BasePlugin) -> str:
        """Generate API content for a plugin."""
        return f"""# {plugin_name} API Reference

## Plugin Class

### `{plugin.__class__.__name__}`

{plugin_info.get('description', 'No description available')}

## Methods

### `initialize() -> bool`

Initialize the plugin.

**Returns:**
- `bool`: True if initialization successful, False otherwise

### `execute(input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any`

Execute the plugin.

**Parameters:**
- `input_data`: Input data for the plugin
- `parameters`: Optional parameters for execution

**Returns:**
- `Any`: Plugin execution result

### `cleanup() -> bool`

Cleanup plugin resources.

**Returns:**
- `bool`: True if cleanup successful, False otherwise

## Parameters

{self._generate_parameters_documentation(plugin)}

## Metadata

{self._generate_metadata_documentation(plugin_info)}

## Dependencies

{self._generate_dependencies_documentation(plugin)}

## Tags

{self._generate_tags_documentation(plugin)}
"""
    
    def _generate_examples_content(self, plugin_name: str, plugin_info: Dict[str, Any], plugin: BasePlugin) -> str:
        """Generate examples content for a plugin."""
        return f"""# {plugin_name} Examples

## Basic Example

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()

# Execute plugin with basic data
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data=["gene1", "gene2", "gene3"],
    parameters={{"method": "default_method"}}
)

print(result)
```

## Advanced Example

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
        "threshold": 0.05,
        "output_format": "json"
    }}
)

print(result)
```

## Error Handling Example

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()

try:
    # Execute plugin
    result = await plugin_system.execute_plugin(
        "{plugin_name}",
        input_data="invalid_data",
        parameters={{"method": "invalid_method"}}
    )
except ValueError as e:
    print(f"Parameter error: {{e}}")
except RuntimeError as e:
    print(f"Execution error: {{e}}")
except Exception as e:
    print(f"Unexpected error: {{e}}")
```

## Batch Processing Example

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()

# Process multiple datasets
datasets = [
    ["gene1", "gene2", "gene3"],
    ["gene4", "gene5", "gene6"],
    ["gene7", "gene8", "gene9"]
]

results = []
for dataset in datasets:
    result = await plugin_system.execute_plugin(
        "{plugin_name}",
        input_data=dataset,
        parameters={{"method": "batch_method"}}
    )
    results.append(result)

print(f"Processed {{len(results)}} datasets")
```
"""
    
    def _generate_tutorial_content(self, plugin_name: str, plugin_info: Dict[str, Any], plugin: BasePlugin) -> str:
        """Generate tutorial content for a plugin."""
        return f"""# {plugin_name} Tutorial

## Introduction

This tutorial will guide you through using the {plugin_name} plugin in PathwayLens.

## Prerequisites

- PathwayLens 2.0.0 or higher
- Python 3.8 or higher
- Basic knowledge of Python and bioinformatics

## Step 1: Installation

The plugin is automatically loaded when PathwayLens starts. No additional installation is required.

## Step 2: Basic Usage

### 2.1 Initialize the Plugin System

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()
```

### 2.2 Execute the Plugin

```python
# Execute plugin with basic data
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data=["gene1", "gene2", "gene3"]
)

print(result)
```

## Step 3: Advanced Usage

### 3.1 Using Custom Parameters

```python
# Execute plugin with custom parameters
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data=["gene1", "gene2", "gene3"],
    parameters={{
        "method": "advanced_method",
        "threshold": 0.05,
        "output_format": "json"
    }}
)
```

### 3.2 Error Handling

```python
try:
    result = await plugin_system.execute_plugin(
        "{plugin_name}",
        input_data="invalid_data"
    )
except ValueError as e:
    print(f"Parameter error: {{e}}")
except RuntimeError as e:
    print(f"Execution error: {{e}}")
```

## Step 4: Best Practices

### 4.1 Data Preparation

- Ensure your input data is properly formatted
- Validate data types and ranges
- Handle missing or invalid data

### 4.2 Parameter Selection

- Choose appropriate methods for your data
- Set reasonable thresholds
- Consider output format requirements

### 4.3 Performance Optimization

- Use batch processing for large datasets
- Monitor memory usage
- Consider parallel processing

## Step 5: Troubleshooting

### Common Issues

1. **Plugin not found**: Ensure the plugin is properly installed and loaded
2. **Parameter errors**: Check parameter names and types
3. **Execution errors**: Verify input data format and content

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Execute plugin with debug information
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data="your_data",
    parameters={{"verbose": True}}
)
```

## Step 6: Next Steps

- Explore other plugins in PathwayLens
- Create custom plugins for your specific needs
- Contribute to the PathwayLens community

## Additional Resources

- [PathwayLens Documentation](https://pathwaylens.readthedocs.io/)
- [Plugin Development Guide](https://pathwaylens.readthedocs.io/en/latest/plugins/)
- [Community Forum](https://github.com/pathwaylens/pathwaylens/discussions)
"""
    
    def _generate_parameters_documentation(self, plugin: BasePlugin) -> str:
        """Generate parameters documentation for a plugin."""
        try:
            parameter_info = plugin.get_parameter_info()
            if not parameter_info:
                return "No parameters available."
            
            doc = "| Parameter | Type | Required | Description | Default |\n"
            doc += "|-----------|------|----------|-------------|----------|\n"
            
            for param_name, param_info in parameter_info.items():
                param_type = param_info.get('type', 'string')
                required = param_info.get('required', False)
                description = param_info.get('description', 'No description')
                default = param_info.get('default', 'None')
                
                doc += f"| `{param_name}` | {param_type} | {'Yes' if required else 'No'} | {description} | {default} |\n"
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to generate parameters documentation: {e}")
            return "Failed to generate parameters documentation."
    
    def _generate_dependencies_documentation(self, plugin: BasePlugin) -> str:
        """Generate dependencies documentation for a plugin."""
        try:
            dependencies = plugin.get_dependencies()
            if not dependencies:
                return "No dependencies required."
            
            doc = "The following packages are required:\n\n"
            for dep in dependencies:
                doc += f"- {dep}\n"
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to generate dependencies documentation: {e}")
            return "Failed to generate dependencies documentation."
    
    def _generate_tags_documentation(self, plugin: BasePlugin) -> str:
        """Generate tags documentation for a plugin."""
        try:
            tags = plugin.get_tags()
            if not tags:
                return "No tags available."
            
            return ", ".join(tags)
            
        except Exception as e:
            self.logger.error(f"Failed to generate tags documentation: {e}")
            return "Failed to generate tags documentation."
    
    def _generate_metadata_documentation(self, plugin_info: Dict[str, Any]) -> str:
        """Generate metadata documentation for a plugin."""
        try:
            doc = "| Property | Value |\n"
            doc += "|----------|-------|\n"
            
            for key, value in plugin_info.items():
                doc += f"| {key} | {value} |\n"
            
            return doc
            
        except Exception as e:
            self.logger.error(f"Failed to generate metadata documentation: {e}")
            return "Failed to generate metadata documentation."
    
    def _get_plugin_readme_template(self) -> str:
        """Get plugin README template."""
        return """# {plugin_name}

{description}

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

## Examples

See [EXAMPLES.md](EXAMPLES.md) for detailed usage examples.

## API Reference

See [API.md](API.md) for complete API documentation.

## Tutorial

See [TUTORIAL.md](TUTORIAL.md) for a step-by-step tutorial.

## License

{license}

## Author

{author}

## Version

{version}
"""
    
    def _get_plugin_api_template(self) -> str:
        """Get plugin API template."""
        return """# {plugin_name} API Reference

## Plugin Class

### `{plugin_class}`

{description}

## Methods

### `initialize() -> bool`

Initialize the plugin.

**Returns:**
- `bool`: True if initialization successful, False otherwise

### `execute(input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any`

Execute the plugin.

**Parameters:**
- `input_data`: Input data for the plugin
- `parameters`: Optional parameters for execution

**Returns:**
- `Any`: Plugin execution result

### `cleanup() -> bool`

Cleanup plugin resources.

**Returns:**
- `bool`: True if cleanup successful, False otherwise

## Parameters

{parameters}

## Dependencies

{dependencies}

## Tags

{tags}
"""
    
    def _get_plugin_examples_template(self) -> str:
        """Get plugin examples template."""
        return """# {plugin_name} Examples

## Basic Example

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()

# Execute plugin with basic data
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data=["gene1", "gene2", "gene3"],
    parameters={{"method": "default_method"}}
)

print(result)
```

## Advanced Example

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
        "threshold": 0.05,
        "output_format": "json"
    }}
)

print(result)
```
"""
    
    def _get_plugin_tutorial_template(self) -> str:
        """Get plugin tutorial template."""
        return """# {plugin_name} Tutorial

## Introduction

This tutorial will guide you through using the {plugin_name} plugin in PathwayLens.

## Prerequisites

- PathwayLens 2.0.0 or higher
- Python 3.8 or higher
- Basic knowledge of Python and bioinformatics

## Step 1: Installation

The plugin is automatically loaded when PathwayLens starts. No additional installation is required.

## Step 2: Basic Usage

### 2.1 Initialize the Plugin System

```python
from pathwaylens_core.plugins import PluginSystem

# Initialize plugin system
plugin_system = PluginSystem()
await plugin_system.initialize()
```

### 2.2 Execute the Plugin

```python
# Execute plugin with basic data
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data=["gene1", "gene2", "gene3"]
)

print(result)
```

## Step 3: Advanced Usage

### 3.1 Using Custom Parameters

```python
# Execute plugin with custom parameters
result = await plugin_system.execute_plugin(
    "{plugin_name}",
    input_data=["gene1", "gene2", "gene3"],
    parameters={{
        "method": "advanced_method",
        "threshold": 0.05,
        "output_format": "json"
    }}
)
```

## Step 4: Best Practices

### 4.1 Data Preparation

- Ensure your input data is properly formatted
- Validate data types and ranges
- Handle missing or invalid data

### 4.2 Parameter Selection

- Choose appropriate methods for your data
- Set reasonable thresholds
- Consider output format requirements

## Step 5: Troubleshooting

### Common Issues

1. **Plugin not found**: Ensure the plugin is properly installed and loaded
2. **Parameter errors**: Check parameter names and types
3. **Execution errors**: Verify input data format and content

## Additional Resources

- [PathwayLens Documentation](https://pathwaylens.readthedocs.io/)
- [Plugin Development Guide](https://pathwaylens.readthedocs.io/en/latest/plugins/)
- [Community Forum](https://github.com/pathwaylens/pathwaylens/discussions)
"""
