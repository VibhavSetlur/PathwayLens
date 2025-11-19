"""
Differential expression tool result parsers.

Supports parsing results from DESeq2, edgeR, limma, and other DE tools.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from loguru import logger
import pandas as pd
import numpy as np


class DEToolParser:
    """Parser for differential expression tool results."""
    
    def __init__(self):
        """Initialize DE tool parser."""
        self.logger = logger.bind(module="de_tools")
    
    def detect_format(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Auto-detect DE tool format from file.
        
        Args:
            file_path: Path to DE results file
            
        Returns:
            Format type ('deseq2', 'edger', 'limma', 'generic', or None)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        # Read first few lines to detect format
        try:
            df = pd.read_csv(file_path, nrows=5, sep='\t' if file_path.suffix == '.tsv' else ',')
            
            # Check for DESeq2 columns
            if all(col in df.columns for col in ['baseMean', 'log2FoldChange', 'padj']):
                return 'deseq2'
            
            # Check for edgeR columns
            if all(col in df.columns for col in ['logFC', 'PValue', 'FDR']):
                return 'edger'
            
            # Check for limma columns
            if all(col in df.columns for col in ['logFC', 'P.Value', 'adj.P.Val']):
                return 'limma'
            
            # Generic format (must have logFC/log2FoldChange and p-value)
            if any(col in df.columns for col in ['logFC', 'log2FoldChange', 'log2FC']):
                if any(col in df.columns for col in ['PValue', 'pvalue', 'P.Value', 'pval', 'padj', 'FDR', 'adj.P.Val']):
                    return 'generic'
        
        except Exception as e:
            self.logger.warning(f"Failed to detect format: {e}")
        
        return None
    
    def parse_deseq2(
        self,
        file_path: Union[str, Path],
        gene_id_col: str = None,
        logfc_threshold: float = 0.0,
        padj_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Parse DESeq2 results file.
        
        Args:
            file_path: Path to DESeq2 results CSV/TSV
            gene_id_col: Column name for gene IDs (auto-detected if None)
            logfc_threshold: Log2 fold change threshold
            padj_threshold: Adjusted p-value threshold
            
        Returns:
            DataFrame with standardized columns
        """
        file_path = Path(file_path)
        self.logger.info(f"Parsing DESeq2 results from {file_path}")
        
        # Read file
        sep = '\t' if file_path.suffix == '.tsv' else ','
        df = pd.read_csv(file_path, sep=sep)
        
        # Auto-detect gene ID column
        if gene_id_col is None:
            possible_cols = ['gene_id', 'GeneID', 'gene', 'Gene', 'geneName', 'GeneName']
            gene_id_col = next((c for c in possible_cols if c in df.columns), df.columns[0])
        
        # Standardize column names
        standardized = pd.DataFrame()
        standardized['gene_id'] = df[gene_id_col]
        standardized['log2fc'] = df.get('log2FoldChange', df.get('logFC', np.nan))
        standardized['pvalue'] = df.get('pvalue', df.get('PValue', np.nan))
        standardized['padj'] = df.get('padj', df.get('FDR', np.nan))
        standardized['base_mean'] = df.get('baseMean', np.nan)
        standardized['stat'] = df.get('stat', np.nan)
        
        # Filter by thresholds
        if logfc_threshold > 0.0:
            standardized = standardized[
                abs(standardized['log2fc']) >= logfc_threshold
            ]
        
        if padj_threshold < 1.0:
            standardized = standardized[
                standardized['padj'] <= padj_threshold
            ]
        
        # Add significance label
        standardized['significant'] = (
            (abs(standardized['log2fc']) >= logfc_threshold) &
            (standardized['padj'] <= padj_threshold)
        )
        
        self.logger.info(
            f"Parsed {len(standardized)} genes, "
            f"{standardized['significant'].sum()} significant"
        )
        
        return standardized
    
    def parse_edger(
        self,
        file_path: Union[str, Path],
        gene_id_col: str = None,
        logfc_threshold: float = 0.0,
        fdr_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Parse edgeR results file.
        
        Args:
            file_path: Path to edgeR results CSV/TSV
            gene_id_col: Column name for gene IDs
            logfc_threshold: Log2 fold change threshold
            fdr_threshold: FDR threshold
            
        Returns:
            DataFrame with standardized columns
        """
        file_path = Path(file_path)
        self.logger.info(f"Parsing edgeR results from {file_path}")
        
        # Read file
        sep = '\t' if file_path.suffix == '.tsv' else ','
        df = pd.read_csv(file_path, sep=sep)
        
        # Auto-detect gene ID column
        if gene_id_col is None:
            possible_cols = ['gene_id', 'GeneID', 'gene', 'Gene', 'geneName', 'GeneName']
            gene_id_col = next((c for c in possible_cols if c in df.columns), df.columns[0])
        
        # Standardize column names
        standardized = pd.DataFrame()
        standardized['gene_id'] = df[gene_id_col]
        standardized['log2fc'] = df.get('logFC', np.nan)
        standardized['pvalue'] = df.get('PValue', np.nan)
        standardized['padj'] = df.get('FDR', np.nan)
        
        # Filter by thresholds
        if logfc_threshold > 0.0:
            standardized = standardized[
                abs(standardized['log2fc']) >= logfc_threshold
            ]
        
        if fdr_threshold < 1.0:
            standardized = standardized[
                standardized['padj'] <= fdr_threshold
            ]
        
        # Add significance label
        standardized['significant'] = (
            (abs(standardized['log2fc']) >= logfc_threshold) &
            (standardized['padj'] <= fdr_threshold)
        )
        
        self.logger.info(
            f"Parsed {len(standardized)} genes, "
            f"{standardized['significant'].sum()} significant"
        )
        
        return standardized
    
    def parse_limma(
        self,
        file_path: Union[str, Path],
        gene_id_col: str = None,
        logfc_threshold: float = 0.0,
        adjpval_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Parse limma results file.
        
        Args:
            file_path: Path to limma results CSV/TSV
            gene_id_col: Column name for gene IDs
            logfc_threshold: Log2 fold change threshold
            adjpval_threshold: Adjusted p-value threshold
            
        Returns:
            DataFrame with standardized columns
        """
        file_path = Path(file_path)
        self.logger.info(f"Parsing limma results from {file_path}")
        
        # Read file
        sep = '\t' if file_path.suffix == '.tsv' else ','
        df = pd.read_csv(file_path, sep=sep)
        
        # Auto-detect gene ID column
        if gene_id_col is None:
            possible_cols = ['gene_id', 'GeneID', 'gene', 'Gene', 'geneName', 'GeneName']
            gene_id_col = next((c for c in possible_cols if c in df.columns), df.columns[0])
        
        # Standardize column names
        standardized = pd.DataFrame()
        standardized['gene_id'] = df[gene_id_col]
        standardized['log2fc'] = df.get('logFC', np.nan)
        standardized['pvalue'] = df.get('P.Value', df.get('PValue', np.nan))
        standardized['padj'] = df.get('adj.P.Val', df.get('FDR', np.nan))
        
        # Filter by thresholds
        if logfc_threshold > 0.0:
            standardized = standardized[
                abs(standardized['log2fc']) >= logfc_threshold
            ]
        
        if adjpval_threshold < 1.0:
            standardized = standardized[
                standardized['padj'] <= adjpval_threshold
            ]
        
        # Add significance label
        standardized['significant'] = (
            (abs(standardized['log2fc']) >= logfc_threshold) &
            (standardized['padj'] <= adjpval_threshold)
        )
        
        self.logger.info(
            f"Parsed {len(standardized)} genes, "
            f"{standardized['significant'].sum()} significant"
        )
        
        return standardized
    
    def parse(
        self,
        file_path: Union[str, Path],
        format_type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Auto-detect and parse DE tool results.
        
        Args:
            file_path: Path to DE results file
            format_type: Format type (auto-detected if None)
            **kwargs: Additional arguments for parser
            
        Returns:
            DataFrame with standardized columns
        """
        if format_type is None:
            format_type = self.detect_format(file_path)
        
        if format_type == 'deseq2':
            return self.parse_deseq2(file_path, **kwargs)
        elif format_type == 'edger':
            return self.parse_edger(file_path, **kwargs)
        elif format_type == 'limma':
            return self.parse_limma(file_path, **kwargs)
        elif format_type == 'generic':
            # Try to parse as generic format
            return self._parse_generic(file_path, **kwargs)
        else:
            raise ValueError(
                f"Unknown format: {format_type}. "
                "Supported formats: deseq2, edger, limma, generic"
            )
    
    def _parse_generic(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Parse generic DE results format."""
        file_path = Path(file_path)
        sep = '\t' if file_path.suffix == '.tsv' else ','
        df = pd.read_csv(file_path, sep=sep)
        
        # Try to map common column names
        standardized = pd.DataFrame()
        
        # Find logFC column
        logfc_col = next(
            (c for c in df.columns if 'logfc' in c.lower() or 'log2' in c.lower()),
            None
        )
        if logfc_col:
            standardized['log2fc'] = df[logfc_col]
        
        # Find p-value column
        pval_col = next(
            (c for c in df.columns if c.lower() in ['pvalue', 'p.val', 'p_val']),
            None
        )
        if pval_col:
            standardized['pvalue'] = df[pval_col]
        
        # Find adjusted p-value column
        padj_col = next(
            (c for c in df.columns if c.lower() in ['padj', 'fdr', 'adj.p.val']),
            None
        )
        if padj_col:
            standardized['padj'] = df[padj_col]
        
        # Use first column as gene ID if not specified
        gene_id_col = kwargs.get('gene_id_col', df.columns[0])
        standardized['gene_id'] = df[gene_id_col]
        
        return standardized
    
    def get_significant_genes(
        self,
        df: pd.DataFrame,
        logfc_threshold: float = 0.0,
        padj_threshold: float = 0.05,
        direction: str = 'both'  # 'up', 'down', 'both'
    ) -> pd.DataFrame:
        """
        Extract significant genes from DE results.
        
        Args:
            df: DE results DataFrame
            logfc_threshold: Log2 fold change threshold
            padj_threshold: Adjusted p-value threshold
            direction: Direction filter ('up', 'down', 'both')
            
        Returns:
            DataFrame with significant genes only
        """
        significant = df[
            (df['padj'] <= padj_threshold) &
            (abs(df['log2fc']) >= logfc_threshold)
        ].copy()
        
        if direction == 'up':
            significant = significant[significant['log2fc'] > 0]
        elif direction == 'down':
            significant = significant[significant['log2fc'] < 0]
        
        return significant



