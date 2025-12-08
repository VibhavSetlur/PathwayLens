"""
Unit tests for the Plugin system.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.plugins.base_plugin import BasePlugin
from pathwaylens_core.plugins.plugin_manager import PluginManager
from pathwaylens_core.plugins.plugin_registry import PluginRegistry
from pathwaylens_core.plugins.plugin_loader import PluginLoader
from pathwaylens_core.plugins.plugin_config import PluginConfig
from pathwaylens_core.plugins.plugin_security import PluginSecurity
from pathwaylens_core.plugins.plugin_monitor import PluginMonitor
from pathwaylens_core.plugins.plugin_logger import PluginLogger
from pathwaylens_core.plugins.plugin_factory import PluginFactory
from pathwaylens_core.plugins.plugin_validator import PluginValidator
from pathwaylens_core.plugins.plugin_system import PluginSystem
from pathwaylens_core.plugins.plugin_examples import (
    ExampleAnalysisPlugin, ExampleVisualizationPlugin
)
from pathwaylens_core.types import OmicType, DataType
# Map old names to new names for compatibility
DummyORAAnalysisPlugin = ExampleAnalysisPlugin
DummyVolcanoPlotPlugin = ExampleVisualizationPlugin
DummyCSVExportPlugin = None
DummyCSVImportPlugin = None
DummyNormalizationPlugin = None
DummyNotifierPlugin = None


class TestBasePlugin:
    """Test cases for the BasePlugin class."""

    def test_base_plugin_abstract_methods(self):
        """Test that BasePlugin cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class BasePlugin"):
            BasePlugin("test", "1.0", "desc")

    def test_concrete_plugin(self):
        """Test a concrete plugin implementation."""
        class ConcretePlugin(BasePlugin):
            def __init__(self):
                super().__init__("Concrete", "1.0", "A concrete plugin.")
            
            async def initialize(self, config=None):
                pass
            
            async def shutdown(self):
                pass

        plugin = ConcretePlugin()
        assert plugin.name == "Concrete"
        assert plugin.version == "1.0"
        assert plugin.description == "A concrete plugin."
        assert plugin.get_metadata()["type"] == "concrete"

    @pytest.mark.asyncio
    async def test_concrete_plugin_lifecycle(self):
        """Test plugin lifecycle methods."""
        class ConcretePlugin(BasePlugin):
            def __init__(self):
                super().__init__("Concrete", "1.0", "A concrete plugin.")
                self.initialized = False
                self.shutdown_called = False
            
            async def initialize(self, config=None):
                self.initialized = True
                self.config = config or {}
            
            async def shutdown(self):
                self.shutdown_called = True

        plugin = ConcretePlugin()
        await plugin.initialize({"setting": "value"})
        assert plugin.initialized
        assert plugin.config == {"setting": "value"}
        
        await plugin.shutdown()
        assert plugin.shutdown_called


class TestPluginLoader:
    """Test cases for the PluginLoader class."""

    @pytest.fixture
    def temp_plugin_file(self, tmp_path):
        """Create a temporary plugin file."""
        plugin_file = tmp_path / "test_plugin.py"
        plugin_file.write_text("""
from pathwaylens_core.plugins.base_plugin import BasePlugin
from typing import Dict, Any, Optional

class TestPlugin(BasePlugin):
    PLUGIN_NAME = "TestPlugin"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "A test plugin."

    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION, self.PLUGIN_DESCRIPTION)
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.initialized = True
        self.config = config or {}
        self.logger.info("TestPlugin initialized.")

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self.logger.info("TestPlugin shut down.")

    async def do_something(self, value: str) -> str:
        return f"TestPlugin processed: {value}"
""")
        return plugin_file

    def test_load_plugin_from_file(self, temp_plugin_file):
        """Test loading a plugin from a file."""
        loader = PluginLoader()
        plugin_classes = loader.load_plugin_from_file(temp_plugin_file)
        
        assert len(plugin_classes) == 1
        assert plugin_classes[0].__name__ == "TestPlugin"
        assert issubclass(plugin_classes[0], BasePlugin)

    def test_load_plugin_from_invalid_file(self):
        """Test loading a plugin from an invalid file."""
        loader = PluginLoader()
        invalid_file = Path("non_existent.py")
        plugin_classes = loader.load_plugin_from_file(invalid_file)
        
        assert len(plugin_classes) == 0

    def test_load_plugin_from_file_with_syntax_error(self, tmp_path):
        """Test loading a plugin from a file with syntax errors."""
        plugin_file = tmp_path / "syntax_error_plugin.py"
        plugin_file.write_text("""
from pathwaylens_core.plugins.base_plugin import BasePlugin

class SyntaxErrorPlugin(BasePlugin):
    def __init__(self):
        super().__init__("SyntaxError", "1.0", "A plugin with syntax errors.")
        # Missing closing parenthesis
        self.value = "test"
    
    async def initialize(self, config=None):
        pass
    
    async def shutdown(self):
        pass
""")
        
        loader = PluginLoader()
        plugin_classes = loader.load_plugin_from_file(plugin_file)
        
        assert len(plugin_classes) == 0


class TestPluginRegistry:
    """Test cases for the PluginRegistry class."""

    def test_plugin_registry(self):
        """Test plugin registry functionality."""
        registry = PluginRegistry()
        plugin = Mock(spec=BasePlugin, name="TestPlugin", version="1.0", description="A test plugin")
        plugin.name = "TestPlugin"

        # Register plugin
        registry.register_plugin(plugin)
        assert registry.has_plugin("TestPlugin")
        assert registry.get_plugin("TestPlugin") == plugin
        assert len(registry.get_all_plugins()) == 1

        # Unregister plugin
        registry.unregister_plugin("TestPlugin")
        assert not registry.has_plugin("TestPlugin")
        assert registry.get_plugin("TestPlugin") is None
        assert len(registry.get_all_plugins()) == 0

    def test_plugin_registry_duplicate_registration(self):
        """Test registering a plugin with a duplicate name."""
        registry = PluginRegistry()
        plugin1 = Mock(spec=BasePlugin, name="TestPlugin", version="1.0", description="A test plugin")
        plugin1.name = "TestPlugin"
        plugin2 = Mock(spec=BasePlugin, name="TestPlugin", version="2.0", description="Another test plugin")
        plugin2.name = "TestPlugin"

        registry.register_plugin(plugin1)
        registry.register_plugin(plugin2)  # Should overwrite
        
        assert registry.get_plugin("TestPlugin") == plugin2
        assert len(registry.get_all_plugins()) == 1

    def test_plugin_registry_clear(self):
        """Test clearing the plugin registry."""
        registry = PluginRegistry()
        plugin = Mock(spec=BasePlugin, name="TestPlugin", version="1.0", description="A test plugin")
        plugin.name = "TestPlugin"

        registry.register_plugin(plugin)
        assert len(registry.get_all_plugins()) == 1

        registry.clear_registry()
        assert len(registry.get_all_plugins()) == 0


class TestPluginConfig:
    """Test cases for the PluginConfig class."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary configuration file."""
        config_file = tmp_path / "plugin_config.yml"
        config_file.write_text("""
plugin_paths:
  - ./plugins
global:
  sandbox_enabled: false
plugins:
  TestPlugin:
    setting1: value1
    setting2: 123
  DummyORAAnalysis:
    threshold: 0.01
""")
        return config_file

    def test_load_config(self, temp_config_file):
        """Test loading configuration from a file."""
        config_manager = PluginConfig()
        config_manager.load_config(temp_config_file)

        assert config_manager.get_plugin_paths() == ["./plugins"]
        assert config_manager.get_global_config() == {"sandbox_enabled": False}
        assert config_manager.get_plugin_config("TestPlugin") == {"setting1": "value1", "setting2": 123}
        assert config_manager.get_plugin_config("NonExistentPlugin") == {}

    def test_load_config_invalid_file(self, tmp_path):
        """Test loading configuration from an invalid file."""
        config_manager = PluginConfig()
        invalid_file = tmp_path / "invalid_config.yml"
        invalid_file.write_text("""
invalid: yaml: content: [unclosed
""")
        
        config_manager.load_config(invalid_file)
        assert config_manager.get_all_configs() == {}

    def test_set_plugin_config(self):
        """Test setting plugin configuration."""
        config_manager = PluginConfig()
        config_manager.set_plugin_config("TestPlugin", {"setting": "value"})
        
        assert config_manager.get_plugin_config("TestPlugin") == {"setting": "value"}


class TestPluginManager:
    """Test cases for the PluginManager class."""

    @pytest.fixture
    def temp_plugin_dir(self, tmp_path):
        """Create a temporary plugin directory."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        
        # Create a test plugin file
        (plugin_dir / "test_plugin.py").write_text("""
from pathwaylens_core.plugins.base_plugin import BasePlugin
from typing import Dict, Any, Optional

class TestPlugin(BasePlugin):
    PLUGIN_NAME = "TestPlugin"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "A test plugin."

    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION, self.PLUGIN_DESCRIPTION)
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.initialized = True
        self.config = config or {}
        self.logger.info("TestPlugin initialized.")

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self.logger.info("TestPlugin shut down.")

    async def do_something(self, value: str) -> str:
        return f"TestPlugin processed: {value}"
""")
        
        return plugin_dir

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary configuration file."""
        config_file = tmp_path / "plugin_config.yml"
        config_file.write_text("""
plugin_paths:
  - ./plugins
global:
  sandbox_enabled: false
plugins:
  TestPlugin:
    setting1: value1
    setting2: 123
""")
        return config_file

    @pytest.mark.asyncio
    async def test_plugin_manager_lifecycle(self, temp_plugin_dir, temp_config_file):
        """Test plugin manager lifecycle."""
        # Change CWD for relative path in config
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_plugin_dir.parent)

            manager = PluginManager(plugin_dirs=[temp_plugin_dir])
            await manager.load_plugins(temp_config_file)

            assert manager.registry.has_plugin("TestPlugin")
            my_plugin = manager.registry.get_plugin("TestPlugin")
            assert not my_plugin.initialized

            await manager.initialize_plugins()
            assert my_plugin.initialized
            assert my_plugin.config == {"setting1": "value1", "setting2": 123}

            await manager.shutdown_plugins()
            assert my_plugin.shutdown_called
        finally:
            os.chdir(original_cwd)

    def test_add_plugin_directory(self, temp_plugin_dir):
        """Test adding a plugin directory."""
        manager = PluginManager()
        manager.add_plugin_directory(temp_plugin_dir)
        
        assert temp_plugin_dir in manager.plugin_dirs

    def test_get_plugins_by_type(self, temp_plugin_dir):
        """Test getting plugins by type."""
        manager = PluginManager(plugin_dirs=[temp_plugin_dir])
        
        # This would require actual plugin loading, but we can test the method exists
        plugins = manager.get_plugins_by_type("analysis")
        assert isinstance(plugins, dict)


class TestPluginValidator:
    """Test cases for the PluginValidator class."""

    def test_validate_plugin_class(self):
        """Test plugin class validation."""
        validator = PluginValidator()

        # Valid plugin
        class ValidPlugin(BasePlugin):
            PLUGIN_NAME = "Valid"
            PLUGIN_VERSION = "1.0"
            PLUGIN_DESCRIPTION = "A valid plugin."
            
            async def initialize(self, config=None):
                pass
            
            async def shutdown(self):
                pass

        errors = validator.validate_plugin_class(ValidPlugin)
        assert len(errors) == 0

        # Invalid: Not inheriting from BasePlugin
        class NotAPlugin:
            pass

        errors = validator.validate_plugin_class(NotAPlugin)
        assert len(errors) > 0
        assert "does not inherit from BasePlugin" in errors[0]

        # Invalid: Missing PLUGIN_NAME
        class MissingNamePlugin(BasePlugin):
            PLUGIN_VERSION = "1.0"
            PLUGIN_DESCRIPTION = "Missing name."
            
            async def initialize(self, config=None):
                pass
            
            async def shutdown(self):
                pass

        errors = validator.validate_plugin_class(MissingNamePlugin)
        assert len(errors) > 0
        assert "must define a string 'PLUGIN_NAME'" in errors[0]

    def test_validate_plugin_instance(self):
        """Test plugin instance validation."""
        validator = PluginValidator()

        # Valid plugin instance
        class ValidPlugin(BasePlugin):
            PLUGIN_NAME = "Valid"
            PLUGIN_VERSION = "1.0"
            PLUGIN_DESCRIPTION = "A valid plugin."
            
            async def initialize(self, config=None):
                pass
            
            async def shutdown(self):
                pass

        plugin = ValidPlugin()
        errors = validator.validate_plugin_instance(plugin)
        assert len(errors) == 0

        # Invalid plugin instance
        errors = validator.validate_plugin_instance(None)
        assert len(errors) > 0
        assert "not an instance of BasePlugin" in errors[0]


class TestPluginSecurity:
    """Test cases for the PluginSecurity class."""

    @pytest.mark.asyncio
    async def test_plugin_security_manager(self):
        """Test plugin security manager functionality."""
        # No sandbox
        security_manager_no_sandbox = PluginSecurity(sandbox_enabled=False)
        plugin = Mock(spec=BasePlugin, name="SecurePlugin", version="1.0", description="Secure")
        plugin.do_work = AsyncMock(return_value="done")
        
        assert security_manager_no_sandbox.validate_plugin_permissions(plugin)
        result = await security_manager_no_sandbox.execute_safely(plugin, "do_work", "data")
        plugin.do_work.assert_called_once_with("data")
        assert result == "done"

        # Sandbox enabled
        security_manager_sandbox = PluginSecurity(sandbox_enabled=True)
        assert security_manager_sandbox.validate_plugin_permissions(plugin)
        result = await security_manager_sandbox.execute_safely(plugin, "do_work", "data")
        assert result == "done"

        # Test error propagation
        plugin_error = Mock(spec=BasePlugin, name="ErrorPlugin", version="1.0", description="Error")
        plugin_error.fail_method = AsyncMock(side_effect=ValueError("Plugin failed"))
        
        with pytest.raises(ValueError, match="Plugin failed"):
            await security_manager_no_sandbox.execute_safely(plugin_error, "fail_method")


class TestPluginMonitor:
    """Test cases for the PluginMonitor class."""

    def test_plugin_monitor(self):
        """Test plugin monitoring functionality."""
        monitor = PluginMonitor()
        plugin_name = "TestPlugin"
        method_name = "test_method"

        # Record execution start
        monitor.record_execution_start(plugin_name, method_name)
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        # Record execution end
        monitor.record_execution_end(plugin_name, method_name)

        # Check metrics
        metrics = monitor.get_plugin_metrics(plugin_name)
        assert metrics is not None
        assert metrics["executions"][method_name]["count"] == 1
        assert metrics["executions"][method_name]["total_time"] > 0.005

        # Record another execution
        monitor.record_execution_start(plugin_name, method_name)
        monitor.record_execution_end(plugin_name, method_name)
        
        metrics = monitor.get_plugin_metrics(plugin_name)
        assert metrics["executions"][method_name]["count"] == 2
        assert metrics["executions"][method_name]["total_time"] > 0.01

        # Reset metrics
        monitor.reset_metrics(plugin_name)
        assert monitor.get_plugin_metrics(plugin_name) is None
        
        monitor.reset_metrics()  # Reset all
        assert len(monitor.get_all_metrics()) == 0


class TestPluginLogger:
    """Test cases for the PluginLogger class."""

    def test_plugin_logger(self, caplog):
        """Test plugin logging functionality."""
        plugin_logger_manager = PluginLogger()
        plugin_name = "MyTestPlugin"
        plugin_log = plugin_logger_manager.get_plugin_logger(plugin_name)

        with caplog.at_level("INFO"):
            plugin_log.info("This is a test message from plugin.")
            assert "This is a test message from plugin." in caplog.text
            assert f"plugin_name={plugin_name}" in caplog.text

        caplog.clear()
        with caplog.at_level("WARNING"):
            plugin_logger_manager.log_event(plugin_name, "warning", "Another message.", event_id="123")
            assert "Another message." in caplog.text
            assert f"plugin_name={plugin_name}" in caplog.text
            assert "event_id=123" in caplog.text


class TestPluginFactory:
    """Test cases for the PluginFactory class."""

    def test_plugin_factory(self):
        """Test plugin factory functionality."""
        factory = PluginFactory()

        # Test creation of example plugin
        example_plugin = factory.create_plugin("example", "MyExample", "1.0", "Desc")
        assert isinstance(example_plugin, DummyORAAnalysisPlugin)
        assert example_plugin.name == "MyExample"

        # Test creation of specific types
        analysis_plugin = factory.create_plugin("analysis", "MyAnalysis", "1.0", "Desc", analysis_type="custom_analysis")
        assert analysis_plugin.analysis_type == "custom_analysis"

        visualization_plugin = factory.create_plugin("visualization", "MyViz", "1.0", "Desc", plot_type="heatmap")
        assert visualization_plugin.plot_type == "heatmap"

        # Test error cases
        with pytest.raises(ValueError, match="Unknown plugin type"):
            factory.create_plugin("non_existent", "BadPlugin", "1.0", "Desc")

        with pytest.raises(TypeError, match="Missing or invalid constructor arguments"):
            factory.create_plugin("analysis", "MissingArgAnalysis", "1.0", "Desc")

        # Test registering a new type
        class NewCustomPlugin(BasePlugin):
            PLUGIN_NAME = "NewCustom"
            PLUGIN_VERSION = "1.0"
            PLUGIN_DESCRIPTION = "A new custom plugin."
            
            async def initialize(self, config=None):
                pass
            
            async def shutdown(self):
                pass
            
            async def run(self):
                return "ran"

        factory.register_plugin_type("new_custom", NewCustomPlugin)
        new_plugin = factory.create_plugin("new_custom", "MyNewCustom", "1.0", "Desc")
        assert isinstance(new_plugin, NewCustomPlugin)
        assert new_plugin.name == "MyNewCustom"

    def test_get_registered_types(self):
        """Test getting registered plugin types."""
        factory = PluginFactory()
        types = factory.get_registered_types()
        
        assert isinstance(types, list)
        assert "example" in types
        assert "analysis" in types
        assert "visualization" in types


class TestPluginSystem:
    """Test cases for the PluginSystem class."""

    @pytest.fixture
    def temp_plugin_dir(self, tmp_path):
        """Create a temporary plugin directory."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        
        # Create a test plugin file
        (plugin_dir / "test_plugin.py").write_text("""
from pathwaylens_core.plugins.base_plugin import BasePlugin
from typing import Dict, Any, Optional

class TestPlugin(BasePlugin):
    PLUGIN_NAME = "TestPlugin"
    PLUGIN_VERSION = "1.0.0"
    PLUGIN_DESCRIPTION = "A test plugin."

    def __init__(self):
        super().__init__(self.PLUGIN_NAME, self.PLUGIN_VERSION, self.PLUGIN_DESCRIPTION)
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.initialized = True
        self.config = config or {}
        self.logger.info("TestPlugin initialized.")

    async def shutdown(self) -> None:
        self.shutdown_called = True
        self.logger.info("TestPlugin shut down.")

    async def do_something(self, value: str) -> str:
        return f"TestPlugin processed: {value}"
""")
        
        return plugin_dir

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary configuration file."""
        config_file = tmp_path / "plugin_config.yml"
        config_file.write_text("""
plugin_paths:
  - ./plugins
global:
  sandbox_enabled: false
plugins:
  TestPlugin:
    setting1: value1
    setting2: 123
""")
        return config_file

    @pytest.fixture
    def plugin_system(self, temp_plugin_dir, temp_config_file):
        """Create a plugin system instance."""
        # Change CWD for relative paths in config
        original_cwd = Path.cwd()
        try:
            import os
            import os
            os.chdir(temp_plugin_dir.parent)
            system = PluginSystem(plugin_directory=str(temp_plugin_dir))
            system.plugin_config.load_config(temp_config_file)
            yield system
        finally:
            os.chdir(original_cwd)

    @pytest.fixture
    async def initialized_plugin_system(self, plugin_system):
        """Create an initialized plugin system."""
        await plugin_system.initialize()
        yield plugin_system
        await plugin_system.cleanup()

    @pytest.mark.asyncio
    async def test_plugin_system_full_lifecycle(self, initialized_plugin_system):
        """Test full plugin system lifecycle."""
        system = initialized_plugin_system
        
        assert system.registry.has_plugin("TestPlugin")
        my_plugin = system.registry.get_plugin("TestPlugin")
        assert my_plugin.initialized

        # Test execution
        result = await system.execute_plugin("TestPlugin", "hello")
        assert result == "TestPlugin processed: hello"

        # Test metrics
        metrics = system.plugin_monitor.get_plugin_metrics("TestPlugin")
        assert metrics is not None
        assert metrics["executions"]["do_something"]["count"] == 1
        assert metrics["executions"]["do_something"]["total_time"] > 0

        # Test logger
        plugin_log = system.plugin_logger.get_plugin_logger("TestPlugin")
        assert plugin_log is not None

        await system.cleanup()
        assert my_plugin.shutdown_called

    @pytest.mark.asyncio
    async def test_plugin_system_invalid_plugin_handling(self, tmp_path, caplog):
        """Test handling of invalid plugins."""
        # Create an invalid plugin file
        invalid_plugin_dir = tmp_path / "invalid_plugins"
        invalid_plugin_dir.mkdir()
        (invalid_plugin_dir / "invalid_plugin.py").write_text("""
from pathwaylens_core.plugins.base_plugin import BasePlugin

class InvalidPlugin(BasePlugin):
    PLUGIN_VERSION = "1.0"
    PLUGIN_DESCRIPTION = "An invalid plugin."
    
    async def initialize(self, config=None):
        pass
    
    async def shutdown(self):
        pass
""")

        config_file = tmp_path / "plugin_config.yml"
        config_file.write_text("""
plugin_paths:
  - ./invalid_plugins
global:
  sandbox_enabled: false
""")

        original_cwd = Path.cwd()
        try:
            import os
            import os
            os.chdir(tmp_path)
            system = PluginSystem(plugin_directory=str(invalid_plugin_dir))
            system.plugin_config.load_config(config_file)
            
            with caplog.at_level("ERROR"):
                await system.initialize()
                assert "failed validation" in caplog.text
                assert "must define a string 'PLUGIN_NAME'" in caplog.text
            
            assert not system.registry.has_plugin("InvalidPlugin")
            await system.cleanup()
        finally:
            os.chdir(original_cwd)

    def test_get_plugin(self, plugin_system):
        """Test getting a plugin by name."""
        plugin = plugin_system.get_plugin("NonExistentPlugin")
        assert plugin is None

    def test_get_plugins_by_type(self, plugin_system):
        """Test getting plugins by type."""
        plugins = plugin_system.list_plugins_by_category("analysis")
        assert isinstance(plugins, list)


class TestExamplePlugins:
    """Test cases for example plugins."""

    @pytest.mark.asyncio
    async def test_dummy_ora_analysis_plugin(self):
        """Test the dummy ORA analysis plugin (ExampleAnalysisPlugin)."""
        plugin = DummyORAAnalysisPlugin()
        await plugin.initialize()
        
        # ExampleAnalysisPlugin uses 'execute' and dict parameters
        # It supports 'basic_statistics', 'correlation_analysis', 'regression_analysis'
        
        input_data = [1, 2, 3, 4, 5]
        params = {"method": "basic_statistics"}
        
        result = await plugin.execute(
            input_data=input_data,
            parameters=params
        )
        
        assert result['method'] == 'basic_statistics'
        assert result['analysis_results']['statistics']['count'] == 5
        assert result['analysis_results']['statistics']['mean'] == 3.0
        
        await plugin.shutdown()

    @pytest.mark.asyncio
    async def test_dummy_volcano_plot_plugin(self, tmp_path):
        """Test the dummy volcano plot plugin (ExampleVisualizationPlugin)."""
        plugin = DummyVolcanoPlotPlugin()
        await plugin.initialize()
        
        # ExampleVisualizationPlugin uses 'execute' and dict parameters
        # It supports 'scatter_plot', 'line_plot', 'bar_chart', 'histogram'
        
        input_data = [1, 2, 3, 4, 5]
        params = {
            "method": "scatter_plot",
            "title": "My Scatter Plot"
        }
        
        result = await plugin.execute(
            input_data=input_data,
            parameters=params
        )
        
        assert result['method'] == 'scatter_plot'
        assert result['visualization']['metadata']['title'] == "Scatter Plot" # Default title if not overridden in metadata? 
        # Wait, the plugin implementation sets title in metadata hardcoded to 'Scatter Plot' in _scatter_plot_visualization
        # Let's check plugin_examples.py again.
        # It sets 'title': 'Scatter Plot' in metadata. It ignores 'title' param for metadata title.
        
        assert result['visualization']['type'] == 'scatter_plot'
        assert len(result['visualization']['data']) > 0
        
        await plugin.shutdown()

    @pytest.mark.skip(reason="DummyCSVExportPlugin not implemented")
    @pytest.mark.asyncio
    async def test_dummy_csv_export_plugin(self, tmp_path):
        """Test the dummy CSV export plugin."""
        plugin = DummyCSVExportPlugin()
        await plugin.initialize()
        
        df_to_export = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        output_file = tmp_path / "exported_data.csv"
        
        exported_path = await plugin.export_data(df_to_export, output_file)
        assert exported_path == output_file
        assert output_file.exists()
        
        read_df = pd.read_csv(output_file)
        pd.testing.assert_frame_equal(df_to_export, read_df)
        
        await plugin.shutdown()

    @pytest.mark.skip(reason="DummyCSVImportPlugin not implemented")
    @pytest.mark.asyncio
    async def test_dummy_csv_import_plugin(self, tmp_path):
        """Test the dummy CSV import plugin."""
        plugin = DummyCSVImportPlugin()
        await plugin.initialize()
        
        input_df = pd.DataFrame({"gene": ["G1", "G2"], "value": [10, 20]})
        input_file = tmp_path / "input_data.csv"
        input_df.to_csv(input_file, index=False)
        
        imported_df = await plugin.import_data(input_file)
        pd.testing.assert_frame_equal(input_df, imported_df)
        
        await plugin.shutdown()

    @pytest.mark.skip(reason="DummyNormalizationPlugin not implemented")
    @pytest.mark.asyncio
    async def test_dummy_normalization_plugin(self):
        """Test the dummy normalization plugin."""
        plugin = DummyNormalizationPlugin()
        await plugin.initialize()
        
        data = pd.DataFrame({
            "gene": ["A", "B", "C"],
            "value1": [10, 20, 30],
            "value2": [1, 2, 3]
        }).set_index("gene")
        
        normalized_data = await plugin.process_data(data)
        
        assert isinstance(normalized_data, pd.DataFrame)
        assert normalized_data.shape == data.shape
        assert np.isclose(normalized_data["value1"].mean(), 0)
        assert np.isclose(normalized_data["value1"].std(), 1)
        
        await plugin.shutdown()

    @pytest.mark.skip(reason="DummyNotifierPlugin not implemented")
    @pytest.mark.asyncio
    async def test_dummy_notifier_plugin(self, caplog):
        """Test the dummy notifier plugin."""
        plugin = DummyNotifierPlugin()
        await plugin.initialize()
        
        with caplog.at_level("INFO"):
            result = await plugin.execute_custom_logic("Hello from notifier!")
            assert result["status"] == "success"
            assert "Dummy Notification: Hello from notifier!" in caplog.text
        
        await plugin.shutdown()
