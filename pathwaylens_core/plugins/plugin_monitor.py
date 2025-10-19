"""
Plugin monitor for PathwayLens.
"""

import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from .base_plugin import BasePlugin


class PluginMonitor:
    """Monitor for PathwayLens plugins."""
    
    def __init__(self, monitoring_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin monitor.
        
        Args:
            monitoring_config: Monitoring configuration dictionary
        """
        self.logger = logger.bind(module="plugin_monitor")
        
        # Default monitoring configuration
        self.default_monitoring_config = {
            'enabled': True,
            'monitor_execution_time': True,
            'monitor_memory_usage': True,
            'monitor_cpu_usage': True,
            'monitor_network_usage': True,
            'monitor_file_operations': True,
            'max_execution_time': 300,  # 5 minutes
            'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'max_cpu_usage': 80,  # 80%
            'max_network_requests': 10,
            'monitoring_interval': 1,  # 1 second
            'alert_thresholds': {
                'execution_time': 0.8,  # 80% of max
                'memory_usage': 0.8,    # 80% of max
                'cpu_usage': 0.8,       # 80% of max
                'network_requests': 0.8  # 80% of max
            }
        }
        
        # Current monitoring configuration
        self.monitoring_config = monitoring_config or self.default_monitoring_config.copy()
        
        # Monitoring data
        self.monitoring_data: Dict[str, Dict[str, Any]] = {}
        
        # Alerts
        self.alerts: List[Dict[str, Any]] = []
        
        # Monitoring status
        self.monitoring_active = False
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_monitoring(self, plugin: BasePlugin) -> bool:
        """
        Start monitoring a plugin.
        
        Args:
            plugin: Plugin instance to monitor
            
        Returns:
            True if monitoring started successfully, False otherwise
        """
        try:
            if not self.monitoring_config.get('enabled', True):
                self.logger.info("Monitoring is disabled")
                return False
            
            plugin_name = plugin.name
            
            if plugin_name in self.monitoring_tasks:
                self.logger.warning(f"Plugin {plugin_name} is already being monitored")
                return False
            
            # Initialize monitoring data
            self.monitoring_data[plugin_name] = {
                'start_time': time.time(),
                'execution_time': 0,
                'memory_usage': 0,
                'cpu_usage': 0,
                'network_requests': 0,
                'file_operations': 0,
                'status': 'monitoring',
                'last_update': time.time()
            }
            
            # Start monitoring task
            monitoring_task = asyncio.create_task(
                self._monitor_plugin(plugin_name)
            )
            self.monitoring_tasks[plugin_name] = monitoring_task
            
            self.monitoring_active = True
            self.logger.info(f"Started monitoring plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring plugin {plugin_name}: {e}")
            return False
    
    async def stop_monitoring(self, plugin_name: str) -> bool:
        """
        Stop monitoring a plugin.
        
        Args:
            plugin_name: Name of plugin to stop monitoring
            
        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        try:
            if plugin_name not in self.monitoring_tasks:
                self.logger.warning(f"Plugin {plugin_name} is not being monitored")
                return False
            
            # Cancel monitoring task
            monitoring_task = self.monitoring_tasks[plugin_name]
            monitoring_task.cancel()
            
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
            
            # Remove from monitoring tasks
            del self.monitoring_tasks[plugin_name]
            
            # Update monitoring data
            if plugin_name in self.monitoring_data:
                self.monitoring_data[plugin_name]['status'] = 'stopped'
                self.monitoring_data[plugin_name]['end_time'] = time.time()
            
            # Check if any plugins are still being monitored
            if not self.monitoring_tasks:
                self.monitoring_active = False
            
            self.logger.info(f"Stopped monitoring plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring plugin {plugin_name}: {e}")
            return False
    
    async def _monitor_plugin(self, plugin_name: str) -> None:
        """Monitor a plugin."""
        try:
            while plugin_name in self.monitoring_tasks:
                # Update monitoring data
                await self._update_monitoring_data(plugin_name)
                
                # Check for alerts
                await self._check_alerts(plugin_name)
                
                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_config.get('monitoring_interval', 1))
                
        except asyncio.CancelledError:
            self.logger.info(f"Monitoring task cancelled for plugin: {plugin_name}")
        except Exception as e:
            self.logger.error(f"Monitoring task failed for plugin {plugin_name}: {e}")
    
    async def _update_monitoring_data(self, plugin_name: str) -> None:
        """Update monitoring data for a plugin."""
        try:
            if plugin_name not in self.monitoring_data:
                return
            
            monitoring_data = self.monitoring_data[plugin_name]
            current_time = time.time()
            
            # Update execution time
            if self.monitoring_config.get('monitor_execution_time', True):
                monitoring_data['execution_time'] = current_time - monitoring_data['start_time']
            
            # Update memory usage
            if self.monitoring_config.get('monitor_memory_usage', True):
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    monitoring_data['memory_usage'] = memory_info.rss
                except Exception as e:
                    self.logger.warning(f"Failed to get memory usage for {plugin_name}: {e}")
            
            # Update CPU usage
            if self.monitoring_config.get('monitor_cpu_usage', True):
                try:
                    process = psutil.Process()
                    cpu_percent = process.cpu_percent()
                    monitoring_data['cpu_usage'] = cpu_percent
                except Exception as e:
                    self.logger.warning(f"Failed to get CPU usage for {plugin_name}: {e}")
            
            # Update network requests (simplified)
            if self.monitoring_config.get('monitor_network_usage', True):
                # This is a simplified implementation
                # In practice, would need to track network requests more accurately
                monitoring_data['network_requests'] = 0
            
            # Update file operations (simplified)
            if self.monitoring_config.get('monitor_file_operations', True):
                # This is a simplified implementation
                # In practice, would need to track file operations more accurately
                monitoring_data['file_operations'] = 0
            
            # Update last update time
            monitoring_data['last_update'] = current_time
            
        except Exception as e:
            self.logger.error(f"Failed to update monitoring data for {plugin_name}: {e}")
    
    async def _check_alerts(self, plugin_name: str) -> None:
        """Check for alerts for a plugin."""
        try:
            if plugin_name not in self.monitoring_data:
                return
            
            monitoring_data = self.monitoring_data[plugin_name]
            alert_thresholds = self.monitoring_config.get('alert_thresholds', {})
            
            # Check execution time alert
            if self.monitoring_config.get('monitor_execution_time', True):
                max_execution_time = self.monitoring_config.get('max_execution_time', 300)
                execution_time = monitoring_data.get('execution_time', 0)
                
                if execution_time > max_execution_time * alert_thresholds.get('execution_time', 0.8):
                    await self._create_alert(
                        plugin_name, 
                        'execution_time', 
                        f"Execution time ({execution_time:.2f}s) exceeds threshold ({max_execution_time * alert_thresholds.get('execution_time', 0.8):.2f}s)",
                        'warning'
                    )
            
            # Check memory usage alert
            if self.monitoring_config.get('monitor_memory_usage', True):
                max_memory_usage = self.monitoring_config.get('max_memory_usage', 1024 * 1024 * 1024)
                memory_usage = monitoring_data.get('memory_usage', 0)
                
                if memory_usage > max_memory_usage * alert_thresholds.get('memory_usage', 0.8):
                    await self._create_alert(
                        plugin_name, 
                        'memory_usage', 
                        f"Memory usage ({memory_usage / (1024 * 1024):.2f}MB) exceeds threshold ({max_memory_usage * alert_thresholds.get('memory_usage', 0.8) / (1024 * 1024):.2f}MB)",
                        'warning'
                    )
            
            # Check CPU usage alert
            if self.monitoring_config.get('monitor_cpu_usage', True):
                max_cpu_usage = self.monitoring_config.get('max_cpu_usage', 80)
                cpu_usage = monitoring_data.get('cpu_usage', 0)
                
                if cpu_usage > max_cpu_usage * alert_thresholds.get('cpu_usage', 0.8):
                    await self._create_alert(
                        plugin_name, 
                        'cpu_usage', 
                        f"CPU usage ({cpu_usage:.2f}%) exceeds threshold ({max_cpu_usage * alert_thresholds.get('cpu_usage', 0.8):.2f}%)",
                        'warning'
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to check alerts for {plugin_name}: {e}")
    
    async def _create_alert(self, plugin_name: str, alert_type: str, message: str, severity: str) -> None:
        """Create an alert."""
        try:
            alert = {
                'plugin_name': plugin_name,
                'alert_type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': time.time(),
                'acknowledged': False
            }
            
            self.alerts.append(alert)
            self.logger.warning(f"Alert for {plugin_name}: {message}")
            
        except Exception as e:
            self.logger.error(f"Failed to create alert for {plugin_name}: {e}")
    
    def get_monitoring_data(self, plugin_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get monitoring data.
        
        Args:
            plugin_name: Name of plugin to get monitoring data for (None for all)
            
        Returns:
            Monitoring data dictionary
        """
        if plugin_name:
            return self.monitoring_data.get(plugin_name, {})
        return self.monitoring_data.copy()
    
    def get_alerts(self, plugin_name: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get alerts.
        
        Args:
            plugin_name: Name of plugin to get alerts for (None for all)
            severity: Severity level to filter by (None for all)
            
        Returns:
            List of alerts
        """
        alerts = self.alerts.copy()
        
        if plugin_name:
            alerts = [a for a in alerts if a['plugin_name'] == plugin_name]
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return alerts
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_index: Index of alert to acknowledge
            
        Returns:
            True if acknowledgment successful, False otherwise
        """
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index]['acknowledged'] = True
                self.logger.info(f"Acknowledged alert {alert_index}")
                return True
            else:
                self.logger.error(f"Invalid alert index: {alert_index}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {alert_index}: {e}")
            return False
    
    def clear_alerts(self, plugin_name: Optional[str] = None) -> None:
        """
        Clear alerts.
        
        Args:
            plugin_name: Name of plugin to clear alerts for (None for all)
        """
        if plugin_name:
            self.alerts = [a for a in self.alerts if a['plugin_name'] != plugin_name]
        else:
            self.alerts.clear()
        
        self.logger.info(f"Cleared alerts for {plugin_name or 'all plugins'}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get monitoring summary.
        
        Returns:
            Monitoring summary dictionary
        """
        total_plugins = len(self.monitoring_data)
        active_plugins = len(self.monitoring_tasks)
        total_alerts = len(self.alerts)
        unacknowledged_alerts = len([a for a in self.alerts if not a['acknowledged']])
        
        return {
            'total_plugins': total_plugins,
            'active_plugins': active_plugins,
            'total_alerts': total_alerts,
            'unacknowledged_alerts': unacknowledged_alerts,
            'monitoring_active': self.monitoring_active,
            'monitoring_timestamp': time.time()
        }
    
    def update_monitoring_config(self, config: Dict[str, Any]) -> None:
        """
        Update monitoring configuration.
        
        Args:
            config: New monitoring configuration
        """
        self.monitoring_config.update(config)
        self.logger.info("Monitoring configuration updated")
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get monitoring configuration.
        
        Returns:
            Monitoring configuration dictionary
        """
        return self.monitoring_config.copy()
    
    def is_monitoring_enabled(self) -> bool:
        """
        Check if monitoring is enabled.
        
        Returns:
            True if monitoring is enabled, False otherwise
        """
        return self.monitoring_config.get('enabled', True)
    
    def enable_monitoring(self) -> None:
        """Enable monitoring."""
        self.monitoring_config['enabled'] = True
        self.logger.info("Monitoring enabled")
    
    def disable_monitoring(self) -> None:
        """Disable monitoring."""
        self.monitoring_config['enabled'] = False
        self.logger.info("Monitoring disabled")
    
    async def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        try:
            # Stop all monitoring tasks
            for plugin_name in list(self.monitoring_tasks.keys()):
                await self.stop_monitoring(plugin_name)
            
            # Clear monitoring data
            self.monitoring_data.clear()
            self.alerts.clear()
            
            self.logger.info("Plugin monitor cleaned up")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup plugin monitor: {e}")
