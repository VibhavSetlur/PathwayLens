"""
Parallel processing utilities for PathwayLens.

Provides utilities for parallel execution of independent analyses.
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Dict
from functools import partial
from loguru import logger
import os


class ParallelProcessor:
    """Parallel processing manager."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_threads: bool = False
    ):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers (None = auto-detect)
            use_threads: Use threads instead of processes (for I/O-bound tasks)
        """
        self.logger = logger.bind(module="parallel_processor")
        
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        
        self.max_workers = max_workers
        self.use_threads = use_threads
        
        self.logger.info(f"Initialized parallel processor with {max_workers} workers")
    
    def process_map(
        self,
        func: Callable,
        items: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Process items in parallel using map.
        
        Args:
            func: Function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results in same order as items
        """
        if not items:
            return []
        
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Create partial function with kwargs
            if kwargs:
                func_partial = partial(func, **kwargs)
            else:
                func_partial = func
            
            # Submit all tasks
            futures = [executor.submit(func_partial, item) for item in items]
            
            # Collect results in order
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in parallel processing: {e}")
                    results.append(None)
            
            return results
    
    async def process_map_async(
        self,
        func: Callable,
        items: List[Any],
        **kwargs
    ) -> List[Any]:
        """
        Process items in parallel using async.
        
        Args:
            func: Async function to apply to each item
            items: List of items to process
            **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results in same order as items
        """
        if not items:
            return []
        
        # Create tasks
        tasks = []
        for item in items:
            if kwargs:
                task = func(item, **kwargs)
            else:
                task = func(item)
            tasks.append(task)
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing item {i}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def process_batches(
        self,
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            func: Function to apply to each batch
            items: List of items to process
            batch_size: Size of each batch (None = auto)
            **kwargs: Additional arguments to pass to func
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        if batch_size is None:
            batch_size = max(1, len(items) // self.max_workers)
        
        # Split into batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, len(items), batch_size)
        ]
        
        # Process batches in parallel
        results = self.process_map(
            lambda batch: [func(item, **kwargs) for item in batch],
            batches
        )
        
        # Flatten results
        flattened = []
        for batch_results in results:
            if batch_results:
                flattened.extend(batch_results)
        
        return flattened


def get_optimal_workers(
    task_count: int,
    task_complexity: str = "medium"
) -> int:
    """
    Calculate optimal number of workers for a task.
    
    Args:
        task_count: Number of tasks
        task_complexity: 'low', 'medium', 'high'
        
    Returns:
        Optimal number of workers
    """
    cpu_count = os.cpu_count() or 1
    
    # Complexity factors
    complexity_factors = {
        'low': 2.0,
        'medium': 1.5,
        'high': 1.0
    }
    
    factor = complexity_factors.get(task_complexity, 1.5)
    optimal = int(cpu_count * factor)
    
    # Don't exceed task count
    optimal = min(optimal, task_count)
    
    # At least 1, at most cpu_count
    return max(1, min(optimal, cpu_count))



