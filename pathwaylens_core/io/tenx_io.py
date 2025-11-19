"""
10x Genomics format reader for PathwayLens.

Supports reading 10x Genomics single-cell data formats.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import pandas as pd
import numpy as np

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("anndata library not available. 10x support disabled.")


class TenXIO:
    """Reader for 10x Genomics format."""
    
    def __init__(self):
        """Initialize 10x IO handler."""
        if not ANNDATA_AVAILABLE:
            raise ImportError(
                "anndata library is required. Install with: pip install anndata"
            )
        
        self.logger = logger.bind(module="tenx_io")
    
    def read_10x_mtx(
        self,
        directory: Union[str, Path],
        var_names: str = 'gene_symbols',
        make_unique: bool = True
    ) -> ad.AnnData:
        """
        Read 10x Genomics data from mtx directory.
        
        Args:
            directory: Directory containing matrix.mtx, genes.tsv, barcodes.tsv
            var_names: Variable names to use ('gene_symbols' or 'gene_ids')
            make_unique: Make variable names unique
            
        Returns:
            AnnData object
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        self.logger.info(f"Reading 10x data from {directory}")
        
        try:
            adata = ad.read_10x_mtx(
                directory,
                var_names=var_names,
                make_unique=make_unique
            )
            
            # Transpose to cells x genes
            adata = adata.T
            
            self.logger.info(
                f"Loaded 10x data: {adata.n_obs} cells, {adata.n_vars} genes"
            )
            
            return adata
        
        except Exception as e:
            self.logger.error(f"Failed to read 10x data: {e}")
            raise
    
    def read_10x_h5(
        self,
        file_path: Union[str, Path],
        genome: Optional[str] = None
    ) -> ad.AnnData:
        """
        Read 10x Genomics data from h5 file.
        
        Args:
            file_path: Path to .h5 file
            genome: Genome name (optional)
            
        Returns:
            AnnData object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.logger.info(f"Reading 10x h5 from {file_path}")
        
        try:
            adata = ad.read_10x_h5(file_path, genome=genome)
            
            # Transpose to cells x genes
            adata = adata.T
            
            self.logger.info(
                f"Loaded 10x h5: {adata.n_obs} cells, {adata.n_vars} genes"
            )
            
            return adata
        
        except Exception as e:
            self.logger.error(f"Failed to read 10x h5: {e}")
            raise
    
    def detect_10x_format(self, path: Union[str, Path]) -> Optional[str]:
        """
        Detect 10x Genomics format type.
        
        Args:
            path: Path to check
            
        Returns:
            Format type ('mtx', 'h5', or None)
        """
        path = Path(path)
        
        if path.is_file():
            # Check if it's an h5 file
            if path.suffix in ['.h5', '.hdf5']:
                return 'h5'
        elif path.is_dir():
            # Check for mtx format files
            required_files = ['matrix.mtx', 'genes.tsv', 'barcodes.tsv']
            if all((path / f).exists() for f in required_files):
                return 'mtx'
        
        return None
    
    def read(
        self,
        path: Union[str, Path],
        **kwargs
    ) -> ad.AnnData:
        """
        Auto-detect and read 10x Genomics data.
        
        Args:
            path: Path to 10x data (directory or h5 file)
            **kwargs: Additional arguments
            
        Returns:
            AnnData object
        """
        path = Path(path)
        format_type = self.detect_10x_format(path)
        
        if format_type == 'mtx':
            return self.read_10x_mtx(path, **kwargs)
        elif format_type == 'h5':
            return self.read_10x_h5(path, **kwargs)
        else:
            raise ValueError(
                f"Could not detect 10x format at {path}. "
                "Expected mtx directory or h5 file."
            )



