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
            gene_list=gene_list,
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

# Convenience functions for synchronous usage
def analyze(**kwargs):
    """Synchronous wrapper for analyze."""
    client = PathwayLens()
    return asyncio.run(client.analyze(**kwargs))

def normalize(**kwargs):
    """Synchronous wrapper for normalize."""
    client = PathwayLens()
    return asyncio.run(client.normalize(**kwargs))
