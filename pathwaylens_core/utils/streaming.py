"""
Streaming utilities for memory-efficient processing of large datasets.

Provides generators and streaming functions for processing data in chunks.
"""

from typing import Iterator, List, Any, Callable, Optional, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import csv


class StreamingProcessor:
    """Streaming data processor for large datasets."""
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize streaming processor.
        
        Args:
            chunk_size: Size of chunks to process
        """
        self.chunk_size = chunk_size
        self.logger = logger.bind(module="streaming_processor")
    
    def stream_csv(
        self,
        file_path: Path,
        chunksize: Optional[int] = None,
        **kwargs
    ) -> Iterator[pd.DataFrame]:
        """
        Stream CSV file in chunks.
        
        Args:
            file_path: Path to CSV file
            chunksize: Size of chunks (None = use default)
            **kwargs: Additional arguments for pd.read_csv
            
        Yields:
            DataFrame chunks
        """
        chunksize = chunksize or self.chunk_size
        
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunksize, **kwargs):
                yield chunk
        except Exception as e:
            self.logger.error(f"Error streaming CSV: {e}")
            raise
    
    def stream_lines(
        self,
        file_path: Path,
        encoding: str = 'utf-8'
    ) -> Iterator[str]:
        """
        Stream file line by line.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Yields:
            Lines from file
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    yield line.strip()
        except Exception as e:
            self.logger.error(f"Error streaming lines: {e}")
            raise
    
    def process_stream(
        self,
        data_stream: Iterator[Any],
        func: Callable,
        **kwargs
    ) -> Iterator[Any]:
        """
        Process a data stream with a function.
        
        Args:
            data_stream: Iterator of data items
            func: Function to apply to each item
            **kwargs: Additional arguments for func
            
        Yields:
            Processed items
        """
        for item in data_stream:
            try:
                if kwargs:
                    result = func(item, **kwargs)
                else:
                    result = func(item)
                yield result
            except Exception as e:
                self.logger.warning(f"Error processing item: {e}")
                yield None
    
    def aggregate_stream(
        self,
        data_stream: Iterator[Any],
        aggregator: Callable,
        initial_value: Any = None
    ) -> Any:
        """
        Aggregate results from a stream.
        
        Args:
            data_stream: Iterator of data items
            aggregator: Aggregation function (accumulator, item) -> accumulator
            initial_value: Initial accumulator value
            
        Returns:
            Aggregated result
        """
        accumulator = initial_value
        
        for item in data_stream:
            if accumulator is None:
                accumulator = item
            else:
                accumulator = aggregator(accumulator, item)
        
        return accumulator
    
    def chunk_iterator(
        self,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> Iterator[List[Any]]:
        """
        Split list into chunks.
        
        Args:
            items: List of items
            chunk_size: Size of chunks (None = use default)
            
        Yields:
            Chunks of items
        """
        chunk_size = chunk_size or self.chunk_size
        
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]
    
    def stream_large_array(
        self,
        array: np.ndarray,
        chunk_size: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """
        Stream large numpy array in chunks.
        
        Args:
            array: Numpy array
            chunk_size: Size of chunks (None = use default)
            
        Yields:
            Array chunks
        """
        chunk_size = chunk_size or self.chunk_size
        
        for i in range(0, len(array), chunk_size):
            yield array[i:i + chunk_size]


def stream_gene_list(
    file_path: Path,
    column: Optional[str] = None
) -> Iterator[str]:
    """
    Stream gene list from file.
    
    Args:
        file_path: Path to file
        column: Column name if CSV (None = first column or line-by-line)
        
    Yields:
        Gene identifiers
    """
    if file_path.suffix.lower() == '.csv':
        # CSV file
        for chunk in pd.read_csv(file_path, chunksize=1000):
            if column:
                for gene in chunk[column]:
                    yield str(gene)
            else:
                for gene in chunk.iloc[:, 0]:
                    yield str(gene)
    else:
        # Text file - line by line
        with open(file_path, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene:
                    yield gene



