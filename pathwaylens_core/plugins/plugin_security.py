"""
Plugin security for PathwayLens.
"""

import asyncio
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type, Union
from loguru import logger

from .base_plugin import BasePlugin


class PluginSecurity:
    """Security manager for PathwayLens plugins."""
    
    def __init__(self, security_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin security manager.
        
        Args:
            security_config: Security configuration dictionary
        """
        self.logger = logger.bind(module="plugin_security")
        
        # Default security configuration
        self.default_security_config = {
            'enabled': True,
            'sandbox_plugins': True,
            'allowed_imports': [
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
                'plotly', 'requests', 'json', 'yaml', 'csv', 'datetime',
                'collections', 'itertools', 'functools', 'operator',
                'math', 'statistics', 'random', 'uuid', 'hashlib',
                'base64', 'urllib', 'http', 'email', 'mimetypes'
            ],
            'blocked_imports': [
                'os', 'sys', 'subprocess', 'shutil', 'glob', 'pathlib',
                'socket', 'ssl', 'urllib.request', 'urllib.parse',
                'http.client', 'ftplib', 'smtplib', 'poplib', 'imaplib',
                'telnetlib', 'xmlrpc', 'pickle', 'marshal', 'shelve',
                'dbm', 'sqlite3', 'pymongo', 'psycopg2', 'mysql'
            ],
            'allowed_functions': [
                'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
                'set', 'frozenset', 'range', 'enumerate', 'zip', 'map', 'filter',
                'sorted', 'reversed', 'sum', 'min', 'max', 'abs', 'round',
                'pow', 'divmod', 'bin', 'hex', 'oct', 'chr', 'ord', 'ascii',
                'repr', 'format', 'hash', 'id', 'type', 'isinstance', 'issubclass'
            ],
            'blocked_functions': [
                'eval', 'exec', 'compile', 'open', 'file', 'input', 'raw_input',
                'exit', 'quit', 'help', 'dir', 'vars', 'locals', 'globals',
                'getattr', 'setattr', 'delattr', 'hasattr', 'callable',
                'reload', 'importlib.reload', '__import__'
            ],
            'max_execution_time': 300,  # 5 minutes
            'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'max_network_requests': 10,
            'allowed_network_hosts': [
                'localhost', '127.0.0.1', '::1'
            ],
            'blocked_network_hosts': [
                '0.0.0.0', '255.255.255.255'
            ]
        }
        
        # Current security configuration
        self.security_config = security_config or self.default_security_config.copy()
        
        # Security violations
        self.violations: List[Dict[str, Any]] = []
        
        # Plugin security status
        self.plugin_security_status: Dict[str, Dict[str, Any]] = {}
    
    async def validate_plugin_security(self, plugin: BasePlugin) -> Dict[str, Any]:
        """
        Validate plugin security.
        
        Args:
            plugin: Plugin instance to validate
            
        Returns:
            Security validation results
        """
        try:
            plugin_name = plugin.name
            self.logger.info(f"Validating security for plugin: {plugin_name}")
            
            security_result = {
                'plugin_name': plugin_name,
                'secure': True,
                'violations': [],
                'warnings': [],
                'info': [],
                'validation_timestamp': self._get_current_timestamp()
            }
            
            # Check plugin imports
            await self._validate_plugin_imports(plugin, security_result)
            
            # Check plugin functions
            await self._validate_plugin_functions(plugin, security_result)
            
            # Check plugin file access
            await self._validate_plugin_file_access(plugin, security_result)
            
            # Check plugin network access
            await self._validate_plugin_network_access(plugin, security_result)
            
            # Check plugin resource usage
            await self._validate_plugin_resource_usage(plugin, security_result)
            
            # Store security status
            self.plugin_security_status[plugin_name] = security_result
            
            # Determine overall security
            security_result['secure'] = len(security_result['violations']) == 0
            
            self.logger.info(f"Plugin security validation completed: {plugin_name} - {'Secure' if security_result['secure'] else 'Insecure'}")
            return security_result
            
        except Exception as e:
            self.logger.error(f"Failed to validate plugin security {plugin_name}: {e}")
            return {
                'plugin_name': plugin_name,
                'secure': False,
                'violations': [f"Security validation failed: {e}"],
                'warnings': [],
                'info': [],
                'validation_timestamp': self._get_current_timestamp()
            }
    
    async def _validate_plugin_imports(self, plugin: BasePlugin, security_result: Dict[str, Any]) -> None:
        """Validate plugin imports."""
        try:
            # Get plugin source code
            plugin_source = inspect.getsource(plugin.__class__)
            
            # Check for blocked imports
            blocked_imports = self.security_config.get('blocked_imports', [])
            for blocked_import in blocked_imports:
                if f'import {blocked_import}' in plugin_source or f'from {blocked_import}' in plugin_source:
                    security_result['violations'].append(f"Blocked import detected: {blocked_import}")
            
            # Check for allowed imports
            allowed_imports = self.security_config.get('allowed_imports', [])
            import_lines = [line.strip() for line in plugin_source.split('\n') if line.strip().startswith(('import ', 'from '))]
            
            for import_line in import_lines:
                # Extract module name
                if import_line.startswith('import '):
                    module_name = import_line.split()[1].split('.')[0]
                elif import_line.startswith('from '):
                    module_name = import_line.split()[1].split('.')[0]
                else:
                    continue
                
                if module_name not in allowed_imports and module_name not in blocked_imports:
                    security_result['warnings'].append(f"Unknown import: {module_name}")
            
            security_result['info'].append("Plugin imports validation completed")
            
        except Exception as e:
            security_result['violations'].append(f"Import validation failed: {e}")
    
    async def _validate_plugin_functions(self, plugin: BasePlugin, security_result: Dict[str, Any]) -> None:
        """Validate plugin functions."""
        try:
            # Get plugin source code
            plugin_source = inspect.getsource(plugin.__class__)
            
            # Check for blocked functions
            blocked_functions = self.security_config.get('blocked_functions', [])
            for blocked_function in blocked_functions:
                if blocked_function in plugin_source:
                    security_result['violations'].append(f"Blocked function detected: {blocked_function}")
            
            # Check for allowed functions
            allowed_functions = self.security_config.get('allowed_functions', [])
            function_calls = []
            
            # Simple function call detection
            for line in plugin_source.split('\n'):
                for func in allowed_functions:
                    if f'{func}(' in line:
                        function_calls.append(func)
            
            if function_calls:
                security_result['info'].append(f"Plugin uses {len(function_calls)} allowed functions")
            
            security_result['info'].append("Plugin functions validation completed")
            
        except Exception as e:
            security_result['violations'].append(f"Function validation failed: {e}")
    
    async def _validate_plugin_file_access(self, plugin: BasePlugin, security_result: Dict[str, Any]) -> None:
        """Validate plugin file access."""
        try:
            # Get plugin source code
            plugin_source = inspect.getsource(plugin.__class__)
            
            # Check for file access functions
            file_access_functions = ['open', 'file', 'read', 'write', 'append']
            for func in file_access_functions:
                if func in plugin_source:
                    security_result['warnings'].append(f"File access function detected: {func}")
            
            # Check for path manipulation
            path_functions = ['os.path', 'pathlib', 'glob', 'shutil']
            for func in path_functions:
                if func in plugin_source:
                    security_result['warnings'].append(f"Path manipulation detected: {func}")
            
            security_result['info'].append("Plugin file access validation completed")
            
        except Exception as e:
            security_result['violations'].append(f"File access validation failed: {e}")
    
    async def _validate_plugin_network_access(self, plugin: BasePlugin, security_result: Dict[str, Any]) -> None:
        """Validate plugin network access."""
        try:
            # Get plugin source code
            plugin_source = inspect.getsource(plugin.__class__)
            
            # Check for network access functions
            network_functions = ['requests', 'urllib', 'http', 'socket', 'ssl']
            for func in network_functions:
                if func in plugin_source:
                    security_result['warnings'].append(f"Network access function detected: {func}")
            
            # Check for blocked network hosts
            blocked_hosts = self.security_config.get('blocked_network_hosts', [])
            for host in blocked_hosts:
                if host in plugin_source:
                    security_result['violations'].append(f"Blocked network host detected: {host}")
            
            security_result['info'].append("Plugin network access validation completed")
            
        except Exception as e:
            security_result['violations'].append(f"Network access validation failed: {e}")
    
    async def _validate_plugin_resource_usage(self, plugin: BasePlugin, security_result: Dict[str, Any]) -> None:
        """Validate plugin resource usage."""
        try:
            # Get plugin source code
            plugin_source = inspect.getsource(plugin.__class__)
            
            # Check for resource-intensive operations
            resource_operations = ['while True', 'for i in range(1000000)', 'time.sleep']
            for op in resource_operations:
                if op in plugin_source:
                    security_result['warnings'].append(f"Resource-intensive operation detected: {op}")
            
            # Check for memory-intensive operations
            memory_operations = ['numpy.zeros', 'numpy.ones', 'pandas.DataFrame']
            for op in memory_operations:
                if op in plugin_source:
                    security_result['warnings'].append(f"Memory-intensive operation detected: {op}")
            
            security_result['info'].append("Plugin resource usage validation completed")
            
        except Exception as e:
            security_result['violations'].append(f"Resource usage validation failed: {e}")
    
    def get_security_status(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get security status for a specific plugin.
        
        Args:
            plugin_name: Name of plugin to get security status for
            
        Returns:
            Security status or None if not found
        """
        return self.plugin_security_status.get(plugin_name)
    
    def get_all_security_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get security status for all plugins.
        
        Returns:
            Dictionary with security status for all plugins
        """
        return self.plugin_security_status.copy()
    
    def get_secure_plugins(self) -> List[str]:
        """
        Get list of secure plugins.
        
        Returns:
            List of secure plugin names
        """
        return [
            name for name, status in self.plugin_security_status.items()
            if status.get('secure', False)
        ]
    
    def get_insecure_plugins(self) -> List[str]:
        """
        Get list of insecure plugins.
        
        Returns:
            List of insecure plugin names
        """
        return [
            name for name, status in self.plugin_security_status.items()
            if not status.get('secure', False)
        ]
    
    def get_security_summary(self) -> Dict[str, Any]:
        """
        Get security summary.
        
        Returns:
            Security summary dictionary
        """
        total_plugins = len(self.plugin_security_status)
        secure_plugins = len(self.get_secure_plugins())
        insecure_plugins = len(self.get_insecure_plugins())
        
        return {
            'total_plugins': total_plugins,
            'secure_plugins': secure_plugins,
            'insecure_plugins': insecure_plugins,
            'security_timestamp': self._get_current_timestamp()
        }
    
    def add_violation(self, plugin_name: str, violation: str, severity: str = 'warning') -> None:
        """
        Add a security violation.
        
        Args:
            plugin_name: Name of plugin with violation
            violation: Description of violation
            severity: Severity level (warning, error, critical)
        """
        violation_record = {
            'plugin_name': plugin_name,
            'violation': violation,
            'severity': severity,
            'timestamp': self._get_current_timestamp()
        }
        
        self.violations.append(violation_record)
        self.logger.warning(f"Security violation in {plugin_name}: {violation}")
    
    def get_violations(self, plugin_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get security violations.
        
        Args:
            plugin_name: Name of plugin to get violations for (None for all)
            
        Returns:
            List of security violations
        """
        if plugin_name:
            return [v for v in self.violations if v['plugin_name'] == plugin_name]
        return self.violations.copy()
    
    def clear_violations(self, plugin_name: Optional[str] = None) -> None:
        """
        Clear security violations.
        
        Args:
            plugin_name: Name of plugin to clear violations for (None for all)
        """
        if plugin_name:
            self.violations = [v for v in self.violations if v['plugin_name'] != plugin_name]
        else:
            self.violations.clear()
        
        self.logger.info(f"Security violations cleared for {plugin_name or 'all plugins'}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def update_security_config(self, config: Dict[str, Any]) -> None:
        """
        Update security configuration.
        
        Args:
            config: New security configuration
        """
        self.security_config.update(config)
        self.logger.info("Security configuration updated")
    
    def get_security_config(self) -> Dict[str, Any]:
        """
        Get security configuration.
        
        Returns:
            Security configuration dictionary
        """
        return self.security_config.copy()
    
    def is_security_enabled(self) -> bool:
        """
        Check if security is enabled.
        
        Returns:
            True if security is enabled, False otherwise
        """
        return self.security_config.get('enabled', True)
    
    def enable_security(self) -> None:
        """Enable security."""
        self.security_config['enabled'] = True
        self.logger.info("Security enabled")
    
    def disable_security(self) -> None:
        """Disable security."""
        self.security_config['enabled'] = False
        self.logger.warning("Security disabled")
    
    def is_sandbox_enabled(self) -> bool:
        """
        Check if sandbox is enabled.
        
        Returns:
            True if sandbox is enabled, False otherwise
        """
        return self.security_config.get('sandbox_plugins', True)
    
    def enable_sandbox(self) -> None:
        """Enable sandbox."""
        self.security_config['sandbox_plugins'] = True
        self.logger.info("Sandbox enabled")
    
    def disable_sandbox(self) -> None:
        """Disable sandbox."""
        self.security_config['sandbox_plugins'] = False
        self.logger.warning("Sandbox disabled")
