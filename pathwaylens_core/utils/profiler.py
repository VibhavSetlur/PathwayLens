"""
Performance profiling utilities for PathwayLens.
"""

import time
import functools
import cProfile
import pstats
import io
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from loguru import logger
import tracemalloc


class PerformanceProfiler:
    """Performance profiling utility for PathwayLens."""
    
    def __init__(self):
        """Initialize the profiler."""
        self.logger = logger.bind(module="profiler")
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.memory_snapshots: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def profile_function(self, function_name: str, enable_memory: bool = True):
        """
        Context manager for profiling a function.
        
        Args:
            function_name: Name of the function being profiled
            enable_memory: Whether to track memory usage
            
        Yields:
            Profiler context
        """
        if enable_memory:
            tracemalloc.start()
        
        start_time = time.perf_counter()
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            yield profiler
        finally:
            profiler.disable()
            end_time = time.perf_counter()
            
            # Get execution time
            execution_time = end_time - start_time
            
            # Get profiling stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_data = {
                'execution_time': execution_time,
                'stats': stats_stream.getvalue(),
                'function_name': function_name
            }
            
            # Get memory usage if enabled
            if enable_memory:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                profile_data['memory_current_mb'] = current / 1024 / 1024
                profile_data['memory_peak_mb'] = peak / 1024 / 1024
            
            self.profiles[function_name] = profile_data
            self.logger.info(
                f"Profiled {function_name}: {execution_time:.3f}s"
                + (f", Peak memory: {profile_data.get('memory_peak_mb', 0):.2f}MB" if enable_memory else "")
            )
    
    def profile_decorator(self, enable_memory: bool = True):
        """
        Decorator for profiling functions.
        
        Args:
            enable_memory: Whether to track memory usage
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                function_name = f"{func.__module__}.{func.__name__}"
                with self.profile_function(function_name, enable_memory):
                    return await func(*args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                function_name = f"{func.__module__}.{func.__name__}"
                with self.profile_function(function_name, enable_memory):
                    return func(*args, **kwargs)
            
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """
        Get summary of all profiles.
        
        Returns:
            Dictionary with profile summary
        """
        if not self.profiles:
            return {'message': 'No profiles available'}
        
        summary = {
            'total_profiles': len(self.profiles),
            'profiles': {}
        }
        
        total_time = 0.0
        total_memory = 0.0
        
        for name, profile in self.profiles.items():
            execution_time = profile['execution_time']
            total_time += execution_time
            
            memory_peak = profile.get('memory_peak_mb', 0)
            total_memory = max(total_memory, memory_peak)
            
            summary['profiles'][name] = {
                'execution_time': execution_time,
                'memory_peak_mb': memory_peak
            }
        
        summary['total_execution_time'] = total_time
        summary['peak_memory_mb'] = total_memory
        
        return summary
    
    def clear_profiles(self):
        """Clear all stored profiles."""
        self.profiles.clear()
        self.memory_snapshots.clear()
        self.logger.info("Cleared all profiles")
    
    def export_profile_report(self, output_path: str):
        """
        Export profile report to file.
        
        Args:
            output_path: Path to output file
        """
        with open(output_path, 'w') as f:
            f.write("PathwayLens Performance Profile Report\n")
            f.write("=" * 50 + "\n\n")
            
            summary = self.get_profile_summary()
            f.write(f"Total Profiles: {summary['total_profiles']}\n")
            f.write(f"Total Execution Time: {summary['total_execution_time']:.3f}s\n")
            f.write(f"Peak Memory Usage: {summary['peak_memory_mb']:.2f}MB\n\n")
            
            for name, profile in self.profiles.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"Function: {name}\n")
                f.write(f"Execution Time: {profile['execution_time']:.3f}s\n")
                if 'memory_peak_mb' in profile:
                    f.write(f"Peak Memory: {profile['memory_peak_mb']:.2f}MB\n")
                f.write(f"\nTop Functions:\n")
                f.write(profile['stats'])
        
        self.logger.info(f"Exported profile report to {output_path}")


# Global profiler instance
_global_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    return _global_profiler


def profile_function(enable_memory: bool = True):
    """
    Decorator for profiling functions using the global profiler.
    
    Args:
        enable_memory: Whether to track memory usage
        
    Returns:
        Decorator function
    """
    return _global_profiler.profile_decorator(enable_memory)



