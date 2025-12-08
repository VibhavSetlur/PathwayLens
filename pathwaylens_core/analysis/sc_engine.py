"""
Single-cell analysis engine for PathwayLens.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union
from loguru import logger
from dataclasses import dataclass

from ..data.database_manager import DatabaseManager
from .schemas import DatabaseType, PathwayResult


@dataclass
class SingleCellResult:
    """Result of single-cell pathway analysis."""
    cell_ids: List[str]
    pathway_scores: pd.DataFrame  # cells x pathways
    metadata: Dict[str, Any]


class SingleCellEngine:
    """Engine for single-cell pathway analysis."""
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize single-cell engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="sc_engine")
        self.database_manager = database_manager

    async def score_single_cells(
        self,
        expression_matrix: pd.DataFrame,
        database: DatabaseType,
        species: str,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500,
        method: str = "mean_zscore"
    ) -> SingleCellResult:
        """
        Calculate pathway activity scores for each single cell.
        
        Args:
            expression_matrix: DataFrame of gene expression (genes x cells)
            database: Database to use
            species: Species
            min_pathway_size: Minimum pathway size
            max_pathway_size: Maximum pathway size
            method: Scoring method ('mean', 'mean_zscore', 'vision')
            
        Returns:
            SingleCellResult object
        """
        self.logger.info(f"Starting single-cell scoring with {database.value} for {expression_matrix.shape[1]} cells")
        
        # Get pathways
        pathway_data_dict = await self.database_manager.get_pathways(
            databases=[database.value],
            species=species
        )
        pathways = pathway_data_dict.get(database.value, [])
        
        if not pathways:
            self.logger.warning(f"No pathways found for {database.value}")
            return SingleCellResult(
                cell_ids=expression_matrix.columns.tolist(),
                pathway_scores=pd.DataFrame(),
                metadata={"error": "No pathways found"}
            )
        
        # Filter pathways by size and gene overlap
        valid_pathways = []
        matrix_genes = set(expression_matrix.index)
        
        for p in pathways:
            p_genes = set(p.gene_ids if hasattr(p, 'gene_ids') else p.get('genes', []))
            overlap = p_genes.intersection(matrix_genes)
            if min_pathway_size <= len(overlap) <= max_pathway_size:
                valid_pathways.append({
                    'id': p.pathway_id if hasattr(p, 'pathway_id') else p.get('id'),
                    'name': p.name if hasattr(p, 'name') else p.get('name'),
                    'genes': list(overlap)
                })
        
        self.logger.info(f"Scoring {len(valid_pathways)} pathways")
        
        scores = {}
        
        if method == "mean":
            # Simple mean expression of pathway genes
            for p in valid_pathways:
                pathway_genes = p['genes']
                # Calculate mean expression for this pathway across all cells
                # Subset matrix to pathway genes
                sub_matrix = expression_matrix.loc[pathway_genes]
                scores[p['name']] = sub_matrix.mean(axis=0)
                
        elif method == "mean_zscore":
            # Z-score normalization per gene, then mean
            # This highlights relative activity
            z_matrix = (expression_matrix - expression_matrix.mean(axis=1)[:, np.newaxis]) / expression_matrix.std(axis=1)[:, np.newaxis]
            z_matrix = z_matrix.fillna(0) # Handle constant genes
            
            for p in valid_pathways:
                pathway_genes = p['genes']
                sub_matrix = z_matrix.loc[pathway_genes]
                scores[p['name']] = sub_matrix.mean(axis=0)
                
        else:
            raise ValueError(f"Unknown scoring method: {method}")
            
        scores_df = pd.DataFrame(scores, index=expression_matrix.columns)
        
        return SingleCellResult(
            cell_ids=expression_matrix.columns.tolist(),
            pathway_scores=scores_df,
            metadata={
                "database": database.value,
                "species": species,
                "method": method,
                "pathway_count": len(valid_pathways)
            }
        )
