"""
High-level Python API for PathwayLens.
Allows programmatic access to core functionality.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd

from .analysis import AnalysisEngine
from .analysis.schemas import AnalysisType, DatabaseType, AnalysisParameters
from .comparison import ComparisonEngine
from .normalization import IDConverter
from .types import OmicType, DataType
from .species import Species

class PathwayLens:
    """Main entry point for programmatic usage."""
    
    def __init__(self):
        self.analysis_engine = AnalysisEngine()
        self.comparison_engine = ComparisonEngine()
        self.id_converter = IDConverter()
        
    async def analyze(
        self,
        gene_list: List[str],
        omic_type: str,
        data_type: str,
        databases: List[str] = ["kegg"],
        species: str = "human",
        background_genes: Optional[List[str]] = None,
        background_size: Optional[int] = None,
        fdr_threshold: float = 0.05,
        lfc_threshold: float = 1.0,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform pathway analysis (ORA).
        
        Args:
            gene_list: List of gene identifiers.
            omic_type: Omic type (transcriptomics, proteomics, etc.).
            data_type: Data type (bulk, singlecell, etc.).
            databases: List of databases to query.
            species: Species name.
            background_genes: Optional list of background genes.
            background_size: Optional size of background.
            fdr_threshold: FDR significance threshold.
            lfc_threshold: Log fold change threshold.
            output_dir: Optional directory to save results.
            
        Returns:
            Dictionary containing analysis results.
        """
        # Resolve enums
        try:
            omic_enum = OmicType(omic_type)
            data_enum = DataType(data_type)
            db_enums = [DatabaseType(db) for db in databases]
        except ValueError as e:
            raise ValueError(f"Invalid parameter: {e}")
            
        species_info = Species.get(species)
        if not species_info:
            raise ValueError(f"Unknown species: {species}")
            
        # Create parameters
        params = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            omic_type=omic_enum,
            data_type=data_enum,
            databases=db_enums,
            species=species_info.common_name,
            significance_threshold=fdr_threshold,
            lfc_threshold=lfc_threshold,
            custom_background=background_genes,
            background_size=background_size
        )
        
        # Run analysis
        result = await self.analysis_engine.analyze(
            input_data=gene_list,
            parameters=params,
            output_dir=output_dir
        )
        
        return result.model_dump()

    async def normalize(
        self,
        identifiers: List[str],
        input_format: str,
        output_format: str,
        species: str = "human",
        services: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Normalize gene identifiers.
        
        Args:
            identifiers: List of identifiers to convert.
            input_format: Input format (symbol, ensembl, etc.).
            output_format: Output format.
            species: Species name.
            services: Optional list of services to use.
            
        Returns:
            List of conversion results.
        """
        from .normalization.schemas import IDType, SpeciesType
        
        try:
            in_type = IDType(input_format)
            out_type = IDType(output_format)
            sp_type = SpeciesType(species)
        except ValueError as e:
            raise ValueError(f"Invalid parameter: {e}")
            
        async with self.id_converter:
            results = await self.id_converter.convert_identifiers(
                identifiers=identifiers,
                input_type=in_type,
                output_type=out_type,
                species=sp_type,
                services=services
            )
            
        return [r.model_dump() for r in results]

    async def analyze_dataframe(
        self,
        df: pd.DataFrame,
        gene_column: Optional[str] = None,
        logfc_column: Optional[str] = None,
        pvalue_column: Optional[str] = None,
        databases: List[str] = ["kegg"],
        species: str = "human",
        analysis_type: str = "ora",
        fdr_threshold: float = 0.05,
        lfc_threshold: float = 1.0,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500
    ) -> Dict[str, Any]:
        """
        Analyze directly from a pandas DataFrame without file I/O.
        
        Auto-detects column mapping if not specified, and logs detection.
        
        Args:
            df: DataFrame containing gene expression/DE data
            gene_column: Column containing gene identifiers (auto-detected if None)
            logfc_column: Column containing log fold changes (auto-detected if None)
            pvalue_column: Column containing p-values (auto-detected if None)
            databases: List of databases to query
            species: Species name
            analysis_type: 'ora' or 'gsea'
            fdr_threshold: FDR significance threshold
            lfc_threshold: Log fold change threshold
            min_pathway_size: Minimum pathway size
            max_pathway_size: Maximum pathway size
            
        Returns:
            Dictionary containing analysis results
        """
        from loguru import logger
        
        # Auto-detect columns if not specified
        detected_cols = self._detect_columns(df, gene_column, logfc_column, pvalue_column)
        gene_column = detected_cols['gene']
        logfc_column = detected_cols['logfc']
        pvalue_column = detected_cols['pvalue']
        
        logger.info(
            f"Column mapping: gene='{gene_column}', "
            f"logfc='{logfc_column}', pvalue='{pvalue_column}'"
        )
        
        # Extract gene list based on analysis type
        if analysis_type.lower() == "ora":
            # For ORA, extract significant genes
            if pvalue_column and pvalue_column in df.columns:
                sig_mask = df[pvalue_column] <= fdr_threshold
            else:
                sig_mask = pd.Series(True, index=df.index)
            
            if logfc_column and logfc_column in df.columns:
                fc_mask = df[logfc_column].abs() >= lfc_threshold
                sig_mask = sig_mask & fc_mask
            
            gene_list = df.loc[sig_mask, gene_column].tolist()
        else:
            # For GSEA, use ranked list
            gene_list = df[gene_column].tolist()
        
        # Build params
        db_enums = [DatabaseType(db) for db in databases]
        species_info = Species.get(species)
        if not species_info:
            raise ValueError(f"Unknown species: {species}")
        
        params = AnalysisParameters(
            analysis_type=AnalysisType.ORA if analysis_type.lower() == "ora" else AnalysisType.GSEA,
            databases=db_enums,
            species=species_info.common_name,
            significance_threshold=fdr_threshold,
            lfc_threshold=lfc_threshold,
            min_pathway_size=min_pathway_size,
            max_pathway_size=max_pathway_size,
            omic_type=OmicType.TRANSCRIPTOMICS,
            data_type=DataType.BULK
        )
        
        result = await self.analysis_engine.analyze(
            input_data=gene_list,
            parameters=params
        )
        
        return result.model_dump()
    
    def _detect_columns(
        self,
        df: pd.DataFrame,
        gene_col: Optional[str],
        logfc_col: Optional[str],
        pvalue_col: Optional[str]
    ) -> Dict[str, Optional[str]]:
        """Auto-detect column names from DataFrame."""
        from loguru import logger
        
        result = {'gene': gene_col, 'logfc': logfc_col, 'pvalue': pvalue_col}
        columns = df.columns.tolist()
        columns_lower = [c.lower() for c in columns]
        
        # Gene column detection
        if gene_col is None:
            gene_patterns = ['gene', 'gene_id', 'geneid', 'gene_symbol', 'symbol', 'name']
            for pattern in gene_patterns:
                for i, col_lower in enumerate(columns_lower):
                    if pattern in col_lower:
                        result['gene'] = columns[i]
                        logger.info(f"Auto-detected gene column: '{columns[i]}'")
                        break
                if result['gene']:
                    break
            if result['gene'] is None:
                result['gene'] = columns[0]
                logger.info(f"Using first column as gene column: '{columns[0]}'")
        
        # LogFC column detection
        if logfc_col is None:
            logfc_patterns = ['log2foldchange', 'logfc', 'log2fc', 'lfc', 'fc']
            for pattern in logfc_patterns:
                for i, col_lower in enumerate(columns_lower):
                    if pattern in col_lower:
                        result['logfc'] = columns[i]
                        logger.info(f"Auto-detected logFC column: '{columns[i]}'")
                        break
                if result['logfc']:
                    break
        
        # P-value column detection
        if pvalue_col is None:
            pval_patterns = ['padj', 'fdr', 'adj.p.val', 'adjusted_pval', 'pvalue', 'p.value', 'pval']
            for pattern in pval_patterns:
                for i, col_lower in enumerate(columns_lower):
                    if pattern in col_lower or col_lower == pattern:
                        result['pvalue'] = columns[i]
                        logger.info(f"Auto-detected p-value column: '{columns[i]}'")
                        break
                if result['pvalue']:
                    break
        
        return result


# Convenience functions for synchronous usage
def analyze(**kwargs):
    """Synchronous wrapper for analyze."""
    client = PathwayLens()
    return asyncio.run(client.analyze(**kwargs))

def normalize(**kwargs):
    """Synchronous wrapper for normalize."""
    client = PathwayLens()
    return asyncio.run(client.normalize(**kwargs))

def analyze_df(
    df: pd.DataFrame,
    gene_column: Optional[str] = None,
    logfc_column: Optional[str] = None,
    pvalue_column: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Synchronous DataFrame analysis - no file I/O required.
    
    Args:
        df: DataFrame with gene expression/DE data
        gene_column: Column with gene IDs (auto-detected if None)
        logfc_column: Column with log fold changes (auto-detected if None)
        pvalue_column: Column with p-values (auto-detected if None)
        **kwargs: Additional arguments passed to analyze_dataframe
        
    Returns:
        Dictionary containing analysis results
    """
    client = PathwayLens()
    return asyncio.run(client.analyze_dataframe(
        df=df,
        gene_column=gene_column,
        logfc_column=logfc_column,
        pvalue_column=pvalue_column,
        **kwargs
    ))

def gsea_df(
    df: pd.DataFrame,
    gene_column: Optional[str] = None,
    ranking_column: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Synchronous GSEA from DataFrame - no file I/O required.
    
    Args:
        df: DataFrame with ranked gene data
        gene_column: Column with gene IDs (auto-detected if None)
        ranking_column: Column with ranking metric (auto-detected if None)
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing GSEA results
    """
    client = PathwayLens()
    return asyncio.run(client.analyze_dataframe(
        df=df,
        gene_column=gene_column,
        logfc_column=ranking_column,
        analysis_type="gsea",
        **kwargs
    ))

