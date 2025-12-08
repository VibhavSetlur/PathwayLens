"""
Single-cell analysis engine for PathwayLens.

Implements single-cell pathway scoring methods including:
- Mean expression scoring
- Z-score normalized mean
- ssGSEA (single-sample Gene Set Enrichment Analysis)
- GSVA (Gene Set Variation Analysis) - basic
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Union, Tuple
from loguru import logger
from dataclasses import dataclass, field
import warnings
import time

from ..data.database_manager import DatabaseManager
from .schemas import DatabaseType, PathwayResult


@dataclass
class SingleCellResult:
    """Result of single-cell pathway analysis."""
    cell_ids: List[str]
    pathway_scores: pd.DataFrame  # cells x pathways
    p_values: Optional[pd.DataFrame] = None  # cells x pathways (for ssGSEA)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceEstimate:
    """Estimated computational resource requirements."""
    estimated_time_seconds: float
    memory_mb: float
    n_cells: int
    n_pathways: int
    n_permutations: int
    warning_level: str  # "none", "moderate", "high", "extreme"
    message: str


class SingleCellEngine:
    """Engine for single-cell pathway analysis."""
    
    METHODS = ["mean", "mean_zscore", "ssgsea", "gsva"]
    
    # Resource estimation constants
    SECONDS_PER_CELL_PATHWAY_PERM = 1e-6  # Empirically tuned
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize single-cell engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="sc_engine")
        self.database_manager = database_manager

    def estimate_resources(
        self,
        n_cells: int,
        n_pathways: int,
        n_permutations: int = 1000,
        method: str = "ssgsea"
    ) -> ResourceEstimate:
        """
        Estimate computational resources required for analysis.
        
        Args:
            n_cells: Number of cells
            n_pathways: Number of pathways
            n_permutations: Number of permutations for p-values
            method: Scoring method
            
        Returns:
            ResourceEstimate with time, memory, and warning level
        """
        if method in ["mean", "mean_zscore"]:
            # Fast methods - minimal resources
            est_time = n_cells * n_pathways * 1e-7
            est_memory = n_cells * n_pathways * 8 / 1e6  # bytes to MB
            warning_level = "none"
            message = "Analysis should complete quickly."
        elif method == "ssgsea":
            # ssGSEA with permutations is expensive
            est_time = n_cells * n_pathways * n_permutations * self.SECONDS_PER_CELL_PATHWAY_PERM
            est_memory = (n_cells * n_pathways * 8 + n_cells * 8 * 1000) / 1e6
            
            if est_time < 60:
                warning_level = "none"
                message = f"Estimated time: {est_time:.1f} seconds."
            elif est_time < 600:
                warning_level = "moderate"
                message = f"Estimated time: {est_time/60:.1f} minutes. Consider reducing permutations."
            elif est_time < 3600:
                warning_level = "high"
                message = (
                    f"Estimated time: {est_time/60:.1f} minutes. "
                    f"Consider using --n-permutations 100 or method 'mean_zscore' for faster results."
                )
            else:
                warning_level = "extreme"
                message = (
                    f"Estimated time: {est_time/3600:.1f} hours! "
                    f"Strongly recommend: (1) reduce permutations to 100-500, "
                    f"(2) subset cells, or (3) use 'mean_zscore' method."
                )
        else:  # gsva
            est_time = n_cells * n_pathways * 5e-6
            est_memory = n_cells * n_pathways * 16 / 1e6
            warning_level = "moderate" if n_cells > 50000 else "none"
            message = f"Estimated time: {est_time:.1f} seconds."
        
        return ResourceEstimate(
            estimated_time_seconds=est_time,
            memory_mb=est_memory,
            n_cells=n_cells,
            n_pathways=n_pathways,
            n_permutations=n_permutations,
            warning_level=warning_level,
            message=message
        )

    async def score_single_cells(
        self,
        expression_matrix: pd.DataFrame,
        database: DatabaseType,
        species: str,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500,
        method: str = "mean_zscore",
        n_permutations: int = 1000,
        estimate_resources: bool = True,
        warn_threshold: str = "moderate"  # "none", "moderate", "high"
    ) -> SingleCellResult:
        """
        Calculate pathway activity scores for each single cell.
        
        Args:
            expression_matrix: DataFrame of gene expression (genes x cells)
            database: Database to use
            species: Species
            min_pathway_size: Minimum pathway size
            max_pathway_size: Maximum pathway size
            method: Scoring method ('mean', 'mean_zscore', 'ssgsea', 'gsva')
            n_permutations: Number of permutations for ssGSEA p-values
            estimate_resources: Whether to estimate and log resource usage
            warn_threshold: Warning level threshold to emit warnings
            
        Returns:
            SingleCellResult object
        """
        self.logger.info(
            f"Starting single-cell scoring with {database.value} "
            f"for {expression_matrix.shape[1]} cells using method '{method}'"
        )
        
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
        
        # Resource estimation
        if estimate_resources and method == "ssgsea":
            estimate = self.estimate_resources(
                n_cells=expression_matrix.shape[1],
                n_pathways=len(valid_pathways),
                n_permutations=n_permutations,
                method=method
            )
            
            self.logger.info(f"Resource estimate: {estimate.message}")
            
            # Emit warning based on threshold
            warning_levels = ["none", "moderate", "high", "extreme"]
            if warning_levels.index(estimate.warning_level) >= warning_levels.index(warn_threshold):
                self.logger.warning(estimate.message)
        
        # Dispatch to appropriate method
        if method == "mean":
            scores, p_values = self._score_mean(expression_matrix, valid_pathways)
        elif method == "mean_zscore":
            scores, p_values = self._score_mean_zscore(expression_matrix, valid_pathways)
        elif method == "ssgsea":
            scores, p_values = self._score_ssgsea(
                expression_matrix, valid_pathways, n_permutations
            )
        elif method == "gsva":
            scores, p_values = self._score_gsva(expression_matrix, valid_pathways)
        else:
            raise ValueError(
                f"Unknown scoring method: {method}. "
                f"Supported: {', '.join(self.METHODS)}"
            )
            
        scores_df = pd.DataFrame(scores, index=expression_matrix.columns)
        p_values_df = pd.DataFrame(p_values, index=expression_matrix.columns) if p_values else None
        
        return SingleCellResult(
            cell_ids=expression_matrix.columns.tolist(),
            pathway_scores=scores_df,
            p_values=p_values_df,
            metadata={
                "database": database.value,
                "species": species,
                "method": method,
                "pathway_count": len(valid_pathways),
                "n_permutations": n_permutations if method == "ssgsea" else None
            }
        )

    def _score_mean(
        self,
        expression_matrix: pd.DataFrame,
        valid_pathways: List[Dict]
    ) -> Tuple[Dict[str, pd.Series], Optional[Dict]]:
        """Simple mean expression of pathway genes."""
        scores = {}
        for p in valid_pathways:
            pathway_genes = p['genes']
            sub_matrix = expression_matrix.loc[pathway_genes]
            scores[p['name']] = sub_matrix.mean(axis=0)
        return scores, None
    
    def _score_mean_zscore(
        self,
        expression_matrix: pd.DataFrame,
        valid_pathways: List[Dict]
    ) -> Tuple[Dict[str, pd.Series], Optional[Dict]]:
        """Z-score normalization per gene, then mean."""
        gene_means = expression_matrix.mean(axis=1)
        gene_stds = expression_matrix.std(axis=1)
        gene_stds[gene_stds == 0] = 1.0
        
        scores = {}
        for p in valid_pathways:
            pathway_genes = p['genes']
            sub_matrix = expression_matrix.loc[pathway_genes]
            
            if hasattr(sub_matrix, "sparse"):
                sub_matrix = sub_matrix.sparse.to_dense()
            
            p_means = gene_means.loc[pathway_genes].values[:, np.newaxis]
            p_stds = gene_stds.loc[pathway_genes].values[:, np.newaxis]
            
            z_sub_matrix = (sub_matrix - p_means) / p_stds
            z_sub_matrix = z_sub_matrix.fillna(0)
            
            scores[p['name']] = z_sub_matrix.mean(axis=0)
        
        return scores, None
    
    def _score_ssgsea(
        self,
        expression_matrix: pd.DataFrame,
        valid_pathways: List[Dict],
        n_permutations: int = 1000
    ) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
        """
        Single-sample GSEA (ssGSEA) with Barbie et al. normalization.
        
        Reference: Barbie et al., 2009. Nature.
        
        For each cell:
        1. Rank genes by expression
        2. Calculate enrichment score using weighted running sum
        3. Apply Barbie normalization (ES / max|ES|)
        4. Permutation test for p-values
        """
        n_genes, n_cells = expression_matrix.shape
        scores = {p['name']: np.zeros(n_cells) for p in valid_pathways}
        p_values = {p['name']: np.ones(n_cells) for p in valid_pathways}
        
        # Process cell by cell
        for cell_idx, cell_id in enumerate(expression_matrix.columns):
            cell_expr = expression_matrix[cell_id].values
            
            # Rank genes (1 = lowest expression)
            ranks = np.argsort(np.argsort(cell_expr)) + 1
            
            for p in valid_pathways:
                pathway_genes = p['genes']
                gene_indices = [
                    expression_matrix.index.get_loc(g) 
                    for g in pathway_genes 
                    if g in expression_matrix.index
                ]
                
                if not gene_indices:
                    continue
                
                # Calculate ssGSEA enrichment score
                es = self._calculate_ssgsea_es(ranks, gene_indices, n_genes)
                scores[p['name']][cell_idx] = es
                
                # Permutation test for p-value
                if n_permutations > 0:
                    null_es = np.zeros(n_permutations)
                    for perm_idx in range(n_permutations):
                        perm_indices = np.random.choice(
                            n_genes, size=len(gene_indices), replace=False
                        )
                        null_es[perm_idx] = self._calculate_ssgsea_es(
                            ranks, perm_indices.tolist(), n_genes
                        )
                    
                    # Two-tailed p-value
                    if es >= 0:
                        p_val = np.mean(null_es >= es)
                    else:
                        p_val = np.mean(null_es <= es)
                    
                    p_values[p['name']][cell_idx] = max(p_val, 1.0 / (n_permutations + 1))
        
        # Convert to Series
        cell_index = expression_matrix.columns
        scores = {k: pd.Series(v, index=cell_index) for k, v in scores.items()}
        p_values = {k: pd.Series(v, index=cell_index) for k, v in p_values.items()}
        
        return scores, p_values
    
    def _calculate_ssgsea_es(
        self,
        ranks: np.ndarray,
        gene_indices: List[int],
        n_genes: int,
        alpha: float = 0.25
    ) -> float:
        """
        Calculate ssGSEA enrichment score.
        
        Args:
            ranks: Gene ranks (1 = lowest)
            gene_indices: Indices of pathway genes
            n_genes: Total number of genes
            alpha: Weight parameter (0.25 for ssGSEA)
            
        Returns:
            Enrichment score
        """
        gene_set = set(gene_indices)
        n_set = len(gene_set)
        n_miss = n_genes - n_set
        
        # Sort by rank
        sorted_indices = np.argsort(ranks)[::-1]  # Highest rank first
        
        # Calculate weighted running sum
        running_sum = 0.0
        max_es = 0.0
        min_es = 0.0
        
        # Weights for genes in set (rank^alpha)
        set_weights = np.zeros(n_genes)
        for idx in gene_indices:
            set_weights[idx] = ranks[idx] ** alpha
        
        norm_factor = np.sum(set_weights[gene_indices])
        if norm_factor == 0:
            return 0.0
        
        miss_penalty = 1.0 / n_miss if n_miss > 0 else 0
        
        for idx in sorted_indices:
            if idx in gene_set:
                running_sum += set_weights[idx] / norm_factor
            else:
                running_sum -= miss_penalty
            
            if running_sum > max_es:
                max_es = running_sum
            if running_sum < min_es:
                min_es = running_sum
        
        # Return the deviation with larger magnitude
        if abs(max_es) > abs(min_es):
            return max_es
        else:
            return min_es
    
    def _score_gsva(
        self,
        expression_matrix: pd.DataFrame,
        valid_pathways: List[Dict]
    ) -> Tuple[Dict[str, pd.Series], Optional[Dict]]:
        """
        Basic GSVA scoring (simplified implementation).
        
        Full GSVA requires kernel density estimation. This is a simplified
        version using z-score ranks.
        """
        scores = {}
        
        for p in valid_pathways:
            pathway_genes = p['genes']
            gene_indices = [
                i for i, g in enumerate(expression_matrix.index) 
                if g in pathway_genes
            ]
            
            if not gene_indices:
                scores[p['name']] = pd.Series(0.0, index=expression_matrix.columns)
                continue
            
            # For each cell, calculate rank-based score
            cell_scores = []
            for cell_id in expression_matrix.columns:
                cell_expr = expression_matrix[cell_id].values
                
                # Z-score of ranks
                ranks = np.argsort(np.argsort(cell_expr))
                pathway_ranks = ranks[gene_indices]
                
                # Mean z-scored rank
                z_rank = (np.mean(pathway_ranks) - len(ranks) / 2) / (len(ranks) / 4)
                cell_scores.append(z_rank)
            
            scores[p['name']] = pd.Series(cell_scores, index=expression_matrix.columns)
        
        return scores, None
