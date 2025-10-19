"""
Plugin tests for PathwayLens.
"""

import asyncio
import pytest
from typing import Dict, List, Any, Optional
from loguru import logger

from .base_plugin import BasePlugin
from .plugin_manager import PluginManager
from .plugin_loader import PluginLoader
from .plugin_registry import PluginRegistry
from .plugin_validator import PluginValidator
from .plugin_config import PluginConfig
from .plugin_security import PluginSecurity
from .plugin_monitor import PluginMonitor
from .plugin_logger import PluginLogger
from .plugin_factory import PluginFactory
from .plugin_system import PluginSystem
from .plugin_examples import ExampleAnalysisPlugin, ExampleVisualizationPlugin


class TestPlugin(BasePlugin):
    """Test plugin for testing purposes."""
    
    def __init__(self):
        super().__init__(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin for testing purposes"
        )
        
        # Plugin-specific attributes
        self.author = "Test Author"
        self.license = "MIT"
        self.dependencies = []
        self.tags = ["test", "custom"]
        
        # Plugin state
        self.initialized = False
        self.execution_count = 0
    
    async def initialize(self) -> bool:
        """Initialize the plugin."""
        try:
            self.logger.info("Initializing test plugin")
            self.initialized = True
            self.logger.info("Test plugin initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize test plugin: {e}")
            return False
    
    async def execute(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the plugin."""
        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")
            
            self.logger.info("Executing test plugin")
            
            # Validate parameters
            if parameters and not self.validate_parameters(parameters):
                raise ValueError("Invalid parameters")
            
            # Process input data
            result = await self._process_data(input_data, parameters)
            
            # Update execution count
            self.execution_count += 1
            
            self.logger.info(f"Test plugin executed successfully (count: {self.execution_count})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute test plugin: {e}")
            raise
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.logger.info("Cleaning up test plugin")
            self.initialized = False
            self.execution_count = 0
            self.logger.info("Test plugin cleaned up successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cleanup test plugin: {e}")
            return False
    
    async def _process_data(self, input_data: Any, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input data."""
        result = {
            'plugin_name': self.name,
            'input_data': input_data,
            'parameters': parameters or {},
            'execution_count': self.execution_count + 1,
            'processed_at': self._get_current_timestamp()
        }
        
        return result
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate plugin parameters."""
        return True
    
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        return []
    
    def get_optional_parameters(self) -> List[str]:
        """Get list of optional parameters."""
        return []
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter information."""
        return {}
    
    def is_compatible(self, pathwaylens_version: str) -> bool:
        """Check if plugin is compatible with PathwayLens version."""
        return True


class TestPluginSystem:
    """Test class for plugin system."""
    
    def __init__(self):
        """Initialize the test plugin system."""
        self.logger = logger.bind(module="test_plugin_system")
        self.plugin_system = None
    
    async def setup(self) -> None:
        """Setup test plugin system."""
        try:
            self.plugin_system = PluginSystem("test_plugins")
            await self.plugin_system.initialize()
            self.logger.info("Test plugin system setup completed")
        except Exception as e:
            self.logger.error(f"Failed to setup test plugin system: {e}")
            raise
    
    async def teardown(self) -> None:
        """Teardown test plugin system."""
        try:
            if self.plugin_system:
                await self.plugin_system.cleanup()
            self.logger.info("Test plugin system teardown completed")
        except Exception as e:
            self.logger.error(f"Failed to teardown test plugin system: {e}")
            raise
    
    async def test_plugin_creation(self) -> bool:
        """Test plugin creation."""
        try:
            # Create test plugin
            test_plugin = TestPlugin()
            
            # Test plugin initialization
            success = await test_plugin.initialize()
            assert success, "Plugin initialization should succeed"
            
            # Test plugin execution
            result = await test_plugin.execute("test_data")
            assert result is not None, "Plugin execution should return result"
            assert result['plugin_name'] == "test_plugin", "Plugin name should match"
            
            # Test plugin cleanup
            success = await test_plugin.cleanup()
            assert success, "Plugin cleanup should succeed"
            
            self.logger.info("Plugin creation test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin creation test failed: {e}")
            return False
    
    async def test_plugin_manager(self) -> bool:
        """Test plugin manager."""
        try:
            # Create plugin manager
            plugin_manager = PluginManager("test_plugins")
            
            # Test plugin registration
            test_plugin = TestPlugin()
            success = await plugin_manager.register_plugin(test_plugin)
            assert success, "Plugin registration should succeed"
            
            # Test plugin execution
            result = await plugin_manager.execute_plugin("test_plugin", "test_data")
            assert result is not None, "Plugin execution should return result"
            
            # Test plugin unregistration
            success = await plugin_manager.unregister_plugin("test_plugin")
            assert success, "Plugin unregistration should succeed"
            
            # Test cleanup
            await plugin_manager.cleanup_all_plugins()
            
            self.logger.info("Plugin manager test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin manager test failed: {e}")
            return False
    
    async def test_plugin_loader(self) -> bool:
        """Test plugin loader."""
        try:
            # Create plugin loader
            plugin_loader = PluginLoader("test_plugins")
            
            # Test plugin loading
            success = await plugin_loader.load_plugins()
            assert success, "Plugin loading should succeed"
            
            # Test plugin execution
            result = await plugin_loader.execute_plugin("test_plugin", "test_data")
            assert result is not None, "Plugin execution should return result"
            
            # Test cleanup
            await plugin_loader.cleanup()
            
            self.logger.info("Plugin loader test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin loader test failed: {e}")
            return False
    
    async def test_plugin_registry(self) -> bool:
        """Test plugin registry."""
        try:
            # Create plugin registry
            plugin_registry = PluginRegistry()
            
            # Test plugin registration
            test_plugin = TestPlugin()
            success = await plugin_registry.register_plugin(test_plugin)
            assert success, "Plugin registration should succeed"
            
            # Test plugin activation
            success = await plugin_registry.activate_plugin("test_plugin")
            assert success, "Plugin activation should succeed"
            
            # Test plugin deactivation
            success = await plugin_registry.deactivate_plugin("test_plugin")
            assert success, "Plugin deactivation should succeed"
            
            # Test plugin unregistration
            success = await plugin_registry.unregister_plugin("test_plugin")
            assert success, "Plugin unregistration should succeed"
            
            # Test cleanup
            await plugin_registry.cleanup()
            
            self.logger.info("Plugin registry test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin registry test failed: {e}")
            return False
    
    async def test_plugin_validator(self) -> bool:
        """Test plugin validator."""
        try:
            # Create plugin validator
            plugin_validator = PluginValidator()
            
            # Test plugin validation
            test_plugin = TestPlugin()
            validation_result = await plugin_validator.validate_plugin(test_plugin)
            assert validation_result['valid'], "Plugin validation should succeed"
            
            # Test validation results
            assert len(validation_result['errors']) == 0, "Plugin should have no validation errors"
            
            self.logger.info("Plugin validator test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin validator test failed: {e}")
            return False
    
    async def test_plugin_config(self) -> bool:
        """Test plugin configuration."""
        try:
            # Create plugin config
            plugin_config = PluginConfig()
            
            # Test configuration loading
            success = plugin_config.load_config()
            assert success, "Configuration loading should succeed"
            
            # Test configuration validation
            success = plugin_config.validate_config()
            assert success, "Configuration validation should succeed"
            
            # Test configuration saving
            success = plugin_config.save_config()
            assert success, "Configuration saving should succeed"
            
            self.logger.info("Plugin config test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin config test failed: {e}")
            return False
    
    async def test_plugin_security(self) -> bool:
        """Test plugin security."""
        try:
            # Create plugin security
            plugin_security = PluginSecurity()
            
            # Test plugin security validation
            test_plugin = TestPlugin()
            security_result = await plugin_security.validate_plugin_security(test_plugin)
            assert security_result['secure'], "Plugin security validation should succeed"
            
            # Test security status
            assert len(security_result['violations']) == 0, "Plugin should have no security violations"
            
            self.logger.info("Plugin security test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin security test failed: {e}")
            return False
    
    async def test_plugin_monitor(self) -> bool:
        """Test plugin monitoring."""
        try:
            # Create plugin monitor
            plugin_monitor = PluginMonitor()
            
            # Test plugin monitoring
            test_plugin = TestPlugin()
            success = await plugin_monitor.start_monitoring(test_plugin)
            assert success, "Plugin monitoring should start successfully"
            
            # Test monitoring data
            monitoring_data = plugin_monitor.get_monitoring_data("test_plugin")
            assert monitoring_data is not None, "Monitoring data should be available"
            
            # Test monitoring stop
            success = await plugin_monitor.stop_monitoring("test_plugin")
            assert success, "Plugin monitoring should stop successfully"
            
            # Test cleanup
            await plugin_monitor.cleanup()
            
            self.logger.info("Plugin monitor test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin monitor test failed: {e}")
            return False
    
    async def test_plugin_logger(self) -> bool:
        """Test plugin logging."""
        try:
            # Create plugin logger
            plugin_logger = PluginLogger()
            
            # Test plugin logging
            plugin_logger.log_plugin_event("test_plugin", "test_event", "Test message", "INFO")
            
            # Test plugin initialization logging
            plugin_logger.log_plugin_initialization("test_plugin", True, "Test initialization")
            
            # Test plugin execution logging
            plugin_logger.log_plugin_execution("test_plugin", True, 1.0, "Test execution")
            
            # Test plugin cleanup logging
            plugin_logger.log_plugin_cleanup("test_plugin", True, "Test cleanup")
            
            # Test cleanup
            plugin_logger.cleanup()
            
            self.logger.info("Plugin logger test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin logger test failed: {e}")
            return False
    
    async def test_plugin_factory(self) -> bool:
        """Test plugin factory."""
        try:
            # Create plugin factory
            plugin_factory = PluginFactory("test_plugins")
            
            # Test plugin discovery
            success = await plugin_factory.discover_plugins()
            assert success, "Plugin discovery should succeed"
            
            # Test plugin creation
            test_plugin = await plugin_factory.create_plugin("test_plugin")
            assert test_plugin is not None, "Plugin creation should succeed"
            
            # Test plugin execution
            result = await plugin_factory.execute_plugin("test_plugin", "test_data")
            assert result is not None, "Plugin execution should return result"
            
            # Test cleanup
            await plugin_factory.cleanup_all_plugins()
            
            self.logger.info("Plugin factory test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin factory test failed: {e}")
            return False
    
    async def test_plugin_system(self) -> bool:
        """Test plugin system."""
        try:
            # Create plugin system
            plugin_system = PluginSystem("test_plugins")
            
            # Test plugin system initialization
            success = await plugin_system.initialize()
            assert success, "Plugin system initialization should succeed"
            
            # Test plugin execution
            result = await plugin_system.execute_plugin("test_plugin", "test_data")
            assert result is not None, "Plugin execution should return result"
            
            # Test system status
            status = plugin_system.get_system_status()
            assert status is not None, "System status should be available"
            
            # Test cleanup
            await plugin_system.cleanup()
            
            self.logger.info("Plugin system test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plugin system test failed: {e}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run all plugin tests."""
        try:
            self.logger.info("Running all plugin tests")
            
            # Setup test environment
            await self.setup()
            
            # Run individual tests
            tests = [
                self.test_plugin_creation,
                self.test_plugin_manager,
                self.test_plugin_loader,
                self.test_plugin_registry,
                self.test_plugin_validator,
                self.test_plugin_config,
                self.test_plugin_security,
                self.test_plugin_monitor,
                self.test_plugin_logger,
                self.test_plugin_factory,
                self.test_plugin_system
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test in tests:
                try:
                    success = await test()
                    if success:
                        passed_tests += 1
                except Exception as e:
                    self.logger.error(f"Test {test.__name__} failed: {e}")
            
            # Teardown test environment
            await self.teardown()
            
            # Report results
            self.logger.info(f"Plugin tests completed: {passed_tests}/{total_tests} passed")
            return passed_tests == total_tests
            
        except Exception as e:
            self.logger.error(f"Failed to run plugin tests: {e}")
            return False


async def main():
    """Main function to run plugin tests."""
    try:
        # Create test plugin system
        test_system = TestPluginSystem()
        
        # Run all tests
        success = await test_system.run_all_tests()
        
        if success:
            print("All plugin tests passed!")
        else:
            print("Some plugin tests failed!")
        
        return success
        
    except Exception as e:
        print(f"Failed to run plugin tests: {e}")
        return False


if __name__ == "__main__":
    # Run tests
    asyncio.run(main())
