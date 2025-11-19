"""
AnnData/H5AD reader/writer for PathwayLens.

Supports reading and writing single-cell RNA-seq data in AnnData format.
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
    logger.warning("anndata library not available. AnnData support disabled.")


class AnnDataIO:
    """Reader/writer for AnnData format."""
    
    def __init__(self):
        """Initialize AnnData IO handler."""
        if not ANNDATA_AVAILABLE:
            raise ImportError(
                "anndata library is required. Install with: pip install anndata"
            )
        
        self.logger = logger.bind(module="anndata_io")
    
    def read(self, file_path: Union[str, Path]) -> ad.AnnData:
        """
        Read AnnData file.
        
        Args:
            file_path: Path to .h5ad file
            
        Returns:
            AnnData object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix not in ['.h5ad', '.h5']:
            raise ValueError(f"Expected .h5ad file, got: {file_path.suffix}")
        
        self.logger.info(f"Reading AnnData from {file_path}")
        
        try:
            adata = ad.read_h5ad(file_path)
            self.logger.info(
                f"Loaded AnnData: {adata.n_obs} cells, {adata.n_vars} genes"
            )
            return adata
        except Exception as e:
            self.logger.error(f"Failed to read AnnData: {e}")
            raise
    
    def write(
        self,
        adata: ad.AnnData,
        file_path: Union[str, Path],
        compression: str = "gzip"
    ):
        """
        Write AnnData to file.
        
        Args:
            adata: AnnData object
            file_path: Output file path
            compression: Compression type (gzip, lzf, None)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Writing AnnData to {file_path}")
        
        try:
            adata.write_h5ad(file_path, compression=compression)
            self.logger.info(f"Saved AnnData: {adata.n_obs} cells, {adata.n_vars} genes")
        except Exception as e:
            self.logger.error(f"Failed to write AnnData: {e}")
            raise
    
    def to_dataframe(
        self,
        adata: ad.AnnData,
        layer: Optional[str] = None,
        obs_columns: Optional[List[str]] = None,
        var_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert AnnData to pandas DataFrame.
        
        Args:
            adata: AnnData object
            layer: Layer to extract (default: 'X')
            obs_columns: Observation columns to include
            var_columns: Variable columns to include
            
        Returns:
            DataFrame with genes as columns, cells as rows
        """
        # Extract expression matrix
        if layer is None:
            exp_matrix = adata.X
        else:
            exp_matrix = adata.layers[layer]
        
        # Convert to dense if sparse
        if hasattr(exp_matrix, 'toarray'):
            exp_matrix = exp_matrix.toarray()
        
        # Create DataFrame
        gene_names = adata.var_names.tolist()
        cell_names = adata.obs_names.tolist()
        
        df = pd.DataFrame(
            exp_matrix,
            index=cell_names,
            columns=gene_names
        )
        
        # Add observation metadata
        if obs_columns:
            for col in obs_columns:
                if col in adata.obs.columns:
                    df[col] = adata.obs[col].values
        
        # Add variable metadata as column metadata
        if var_columns:
            for col in var_columns:
                if col in adata.var.columns:
                    # Store as MultiIndex column
                    pass
        
        return df
    
    def generate_pseudobulk(
        self,
        adata: ad.AnnData,
        group_by: str,
        method: str = "mean"
    ) -> pd.DataFrame:
        """
        Generate pseudobulk expression from single-cell data.
        
        Args:
            adata: AnnData object
            group_by: Column in adata.obs to group by (e.g., 'cluster', 'condition')
            method: Aggregation method ('mean', 'sum', 'median')
            
        Returns:
            DataFrame with pseudobulk expression (groups x genes)
        """
        if group_by not in adata.obs.columns:
            raise ValueError(f"Column '{group_by}' not found in adata.obs")
        
        # Extract expression matrix
        exp_matrix = adata.X
        if hasattr(exp_matrix, 'toarray'):
            exp_matrix = exp_matrix.toarray()
        
        # Group by specified column
        groups = adata.obs[group_by].unique()
        pseudobulk = []
        
        for group in groups:
            group_mask = adata.obs[group_by] == group
            group_exp = exp_matrix[group_mask]
            
            if method == "mean":
                agg_exp = np.mean(group_exp, axis=0)
            elif method == "sum":
                agg_exp = np.sum(group_exp, axis=0)
            elif method == "median":
                agg_exp = np.median(group_exp, axis=0)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            pseudobulk.append(agg_exp)
        
        # Create DataFrame
        pseudobulk_df = pd.DataFrame(
            pseudobulk,
            index=groups,
            columns=adata.var_names
        )
        
        self.logger.info(
            f"Generated pseudobulk: {len(groups)} groups, {adata.n_vars} genes"
        )
        
        return pseudobulk_df
    
    def filter_cells(
        self,
        adata: ad.AnnData,
        min_genes: int = 200,
        max_genes: Optional[int] = None,
        min_counts: int = 1000,
        max_counts: Optional[int] = None
    ) -> ad.AnnData:
        """
        Filter cells based on QC metrics.
        
        Args:
            adata: AnnData object
            min_genes: Minimum genes expressed per cell
            max_genes: Maximum genes expressed per cell
            min_counts: Minimum total counts per cell
            max_counts: Maximum total counts per cell
            
        Returns:
            Filtered AnnData object
        """
        # Calculate QC metrics if not present
        if 'n_genes_by_counts' not in adata.obs.columns:
            adata.obs['n_genes_by_counts'] = np.sum(adata.X > 0, axis=1)
        
        if 'total_counts' not in adata.obs.columns:
            adata.obs['total_counts'] = np.sum(adata.X, axis=1)
        
        # Create filter mask
        filter_mask = np.ones(adata.n_obs, dtype=bool)
        
        if min_genes is not None:
            filter_mask &= adata.obs['n_genes_by_counts'] >= min_genes
        
        if max_genes is not None:
            filter_mask &= adata.obs['n_genes_by_counts'] <= max_genes
        
        if min_counts is not None:
            filter_mask &= adata.obs['total_counts'] >= min_counts
        
        if max_counts is not None:
            filter_mask &= adata.obs['total_counts'] <= max_counts
        
        # Apply filter
        filtered_adata = adata[filter_mask].copy()
        
        self.logger.info(
            f"Filtered cells: {adata.n_obs} → {filtered_adata.n_obs} "
            f"({np.sum(filter_mask)}/{len(filter_mask)} passed)"
        )
        
        return filtered_adata
    
    def get_metadata(self, adata: ad.AnnData) -> Dict[str, Any]:
        """
        Extract metadata from AnnData object.
        
        Args:
            adata: AnnData object
            
        Returns:
            Dictionary with metadata
        """
        metadata = {
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "obs_columns": list(adata.obs.columns),
            "var_columns": list(adata.var.columns),
            "layers": list(adata.layers.keys()) if hasattr(adata, 'layers') else [],
            "uns_keys": list(adata.uns.keys()) if hasattr(adata, 'uns') else []
        }
        
        return metadata



