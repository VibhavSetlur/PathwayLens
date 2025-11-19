"""
Batch processing utilities for CLI.

Processes multiple files/datasets in parallel with progress tracking.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from rich.progress import Progress, TaskID

from pathwaylens_core.utils.parallel import ParallelProcessor


class BatchProcessor:
    """Batch processor for CLI operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            max_workers: Maximum number of parallel workers
        """
        self.logger = logger.bind(module="batch_processor")
        self.parallel_processor = ParallelProcessor(max_workers=max_workers)
    
    async def process_files(
        self,
        files: List[Path],
        process_func: Callable,
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files in parallel.
        
        Args:
            files: List of file paths to process
            process_func: Function to process each file
            progress: Rich progress bar (optional)
            task_id: Progress task ID (optional)
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results for each file
        """
        self.logger.info(f"Processing {len(files)} files in batch")
        
        results = []
        
        # Process files in parallel
        async def process_single_file(file_path: Path) -> Dict[str, Any]:
            try:
                result = await process_func(file_path, **kwargs)
                return {
                    'file': str(file_path),
                    'status': 'success',
                    'result': result
                }
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                return {
                    'file': str(file_path),
                    'status': 'error',
                    'error': str(e)
                }
        
        # Create tasks
        tasks = [process_single_file(f) for f in files]
        
        # Execute with progress tracking
        if progress and task_id:
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1
                progress.update(task_id, completed=completed)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'file': str(files[i]),
                        'status': 'error',
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
            results = processed_results
        
        # Log summary
        successful = sum(1 for r in results if r.get('status') == 'success')
        self.logger.info(f"Batch processing complete: {successful}/{len(files)} successful")
        
        return results
    
    def process_sync(
        self,
        files: List[Path],
        process_func: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Synchronous version of process_files.
        
        Args:
            files: List of file paths
            process_func: Function to process each file
            **kwargs: Additional arguments
            
        Returns:
            List of results
        """
        return asyncio.run(self.process_files(files, process_func, **kwargs))



