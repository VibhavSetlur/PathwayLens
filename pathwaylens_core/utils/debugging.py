"""
Debugging utilities and helpers for PathwayLens.

Provides tools for debugging, profiling, and troubleshooting.
"""

import time
import traceback
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import functools
import tracemalloc
import cProfile
import pstats
import io

from loguru import logger


@dataclass
class DebugInfo:
    """Debug information container."""
    timestamp: str
    function_name: str
    execution_time: float
    memory_usage: Optional[float] = None
    error: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class Debugger:
    """Debugging utility class."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize debugger.
        
        Args:
            output_dir: Directory to save debug logs
        """
        self.output_dir = Path(output_dir) if output_dir else Path("debug_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log: List[DebugInfo] = []
        self.logger = logger.bind(module="debugger")
    
    def log_execution(
        self,
        function_name: str,
        execution_time: float,
        memory_usage: Optional[float] = None,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log execution information."""
        debug_info = DebugInfo(
            timestamp=datetime.now().isoformat(),
            function_name=function_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            error=error,
            stack_trace=traceback.format_exc() if error else None,
            context=context
        )
        
        self.debug_log.append(debug_info)
        self.logger.debug(f"Execution logged: {function_name} ({execution_time:.3f}s)")
    
    def save_log(self, filename: Optional[str] = None):
        """Save debug log to file."""
        if not filename:
            filename = f"debug_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_file = self.output_dir / filename
        
        log_data = [asdict(info) for info in self.debug_log]
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.logger.info(f"Debug log saved to {log_file}")
        return log_file
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of debug information."""
        if not self.debug_log:
            return {"message": "No debug information available"}
        
        total_time = sum(info.execution_time for info in self.debug_log)
        errors = [info for info in self.debug_log if info.error]
        
        return {
            "total_executions": len(self.debug_log),
            "total_time": total_time,
            "average_time": total_time / len(self.debug_log) if self.debug_log else 0,
            "errors": len(errors),
            "error_functions": [info.function_name for info in errors],
            "slowest_functions": sorted(
                self.debug_log,
                key=lambda x: x.execution_time,
                reverse=True
            )[:10]
        }


def debug_decorator(output_dir: Optional[Path] = None):
    """
    Decorator to debug function execution.
    
    Args:
        output_dir: Directory to save debug logs
    """
    debugger = Debugger(output_dir)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            tracemalloc.start()
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                debugger.log_execution(
                    function_name=func.__name__,
                    execution_time=elapsed,
                    memory_usage=peak / 1024 / 1024,  # Convert to MB
                    error=error,
                    context={
                        "args": str(args)[:200],  # Limit length
                        "kwargs": str(kwargs)[:200]
                    }
                )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            tracemalloc.start()
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                debugger.log_execution(
                    function_name=func.__name__,
                    execution_time=elapsed,
                    memory_usage=peak / 1024 / 1024,
                    error=error,
                    context={
                        "args": str(args)[:200],
                        "kwargs": str(kwargs)[:200]
                    }
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class Profiler:
    """Performance profiler utility."""
    
    def __init__(self):
        """Initialize profiler."""
        self.profiler = cProfile.Profile()
        self.logger = logger.bind(module="profiler")
    
    def start(self):
        """Start profiling."""
        self.profiler.enable()
        self.logger.debug("Profiling started")
    
    def stop(self):
        """Stop profiling."""
        self.profiler.disable()
        self.logger.debug("Profiling stopped")
    
    def get_stats(self, sort_by: str = "cumulative", limit: int = 20) -> str:
        """
        Get profiling statistics.
        
        Args:
            sort_by: Sort key (cumulative, time, calls, etc.)
            limit: Number of top entries to show
            
        Returns:
            Formatted statistics string
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats(sort_by)
        ps.print_stats(limit)
        return s.getvalue()
    
    def save_stats(self, output_file: Path):
        """Save profiling statistics to file."""
        stats_str = self.get_stats()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(stats_str)
        self.logger.info(f"Profiling stats saved to {output_file}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def print_debug_info(info: Dict[str, Any]):
    """Print formatted debug information."""
    print("\n" + "=" * 60)
    print("DEBUG INFORMATION")
    print("=" * 60)
    
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 60 + "\n")


def get_error_context(exception: Exception) -> Dict[str, Any]:
    """
    Extract context from exception.
    
    Args:
        exception: Exception object
        
    Returns:
        Dictionary with error context
    """
    return {
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat()
    }


# Global debugger instance
_global_debugger: Optional[Debugger] = None


def get_debugger(output_dir: Optional[Path] = None) -> Debugger:
    """Get or create global debugger instance."""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = Debugger(output_dir)
    return _global_debugger

