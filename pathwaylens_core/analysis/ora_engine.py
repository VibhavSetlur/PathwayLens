"""
Over-Representation Analysis (ORA) engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from statsmodels.stats.multitest import multipletests
from loguru import logger

from .schemas import (
    AnalysisParameters, DatabaseType, PathwayResult, DatabaseResult,
    CorrectionMethod
)
from ..data import DatabaseManager


class ORAEngine:
    """Over-Representation Analysis engine."""
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the ORA engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="ora_engine")
        self.database_manager = database_manager
    
    async def analyze(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str,
        significance_threshold: float = 0.05,
        correction_method: CorrectionMethod = CorrectionMethod.FDR_BH,
        min_pathway_size: int = 5,
        max_pathway_size: int = 500
    ) -> DatabaseResult:
        """
        Perform Over-Representation Analysis.
        
        Args:
            gene_list: List of input genes
            database: Database to use for analysis
            species: Species for analysis
            significance_threshold: Significance threshold
            correction_method: Multiple testing correction method
            min_pathway_size: Minimum pathway size
            max_pathway_size: Maximum pathway size
            
        Returns:
            DatabaseResult with ORA analysis results
        """
        self.logger.info(f"Starting ORA analysis with {database.value} for {species}")
        
        try:
            # Get pathway data from database
            pathway_data = await self.database_manager.get_pathways(
                database=database,
                species=species,
                min_size=min_pathway_size,
                max_size=max_pathway_size
            )
            
            if not pathway_data:
                self.logger.warning(f"No pathways found for {database.value} in {species}")
                return DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=species,
                    coverage=0.0
                )
            
            # Perform ORA analysis
            pathway_results = []
            total_genes = len(gene_list)
            
            for pathway_id, pathway_info in pathway_data.items():
                pathway_genes = pathway_info['genes']
                pathway_name = pathway_info['name']
                
                # Calculate overlap
                overlapping_genes = list(set(gene_list) & set(pathway_genes))
                overlap_count = len(overlapping_genes)
                pathway_count = len(pathway_genes)
                
                if overlap_count == 0:
                    continue
                
                # Calculate p-value using hypergeometric test
                p_value = self._calculate_hypergeometric_pvalue(
                    overlap_count, total_genes, pathway_count, 
                    self.database_manager.get_background_size(species)
                )
                
                # Calculate enrichment score
                enrichment_score = self._calculate_enrichment_score(
                    overlap_count, total_genes, pathway_count
                )
                
                pathway_result = PathwayResult(
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    database=database,
                    p_value=p_value,
                    adjusted_p_value=p_value,  # Will be corrected later
                    enrichment_score=enrichment_score,
                    overlap_count=overlap_count,
                    pathway_count=pathway_count,
                    input_count=total_genes,
                    overlapping_genes=overlapping_genes,
                    pathway_genes=pathway_genes,
                    pathway_url=pathway_info.get('url'),
                    pathway_description=pathway_info.get('description'),
                    pathway_category=pathway_info.get('category'),
                    analysis_method="ORA"
                )
                
                pathway_results.append(pathway_result)
            
            # Apply multiple testing correction
            if pathway_results:
                p_values = [result.p_value for result in pathway_results]
                corrected_p_values = self._apply_correction(p_values, correction_method)
                
                for i, result in enumerate(pathway_results):
                    result.adjusted_p_value = corrected_p_values[i]
            
            # Filter significant pathways
            significant_pathways = [
                result for result in pathway_results 
                if result.adjusted_p_value <= significance_threshold
            ]
            
            # Calculate coverage
            covered_genes = set()
            for result in pathway_results:
                covered_genes.update(result.overlapping_genes)
            coverage = len(covered_genes) / total_genes if total_genes > 0 else 0.0
            
            # Sort by significance
            pathway_results.sort(key=lambda x: x.adjusted_p_value)
            significant_pathways.sort(key=lambda x: x.adjusted_p_value)
            
            self.logger.info(
                f"ORA analysis completed: {len(significant_pathways)}/{len(pathway_results)} "
                f"significant pathways found"
            )
            
            return DatabaseResult(
                database=database,
                total_pathways=len(pathway_results),
                significant_pathways=len(significant_pathways),
                pathways=pathway_results,
                species=species,
                coverage=coverage,
                database_version=pathway_info.get('version'),
                last_updated=pathway_info.get('last_updated')
            )
            
        except Exception as e:
            self.logger.error(f"ORA analysis failed: {e}")
            raise
    
    def _calculate_hypergeometric_pvalue(
        self, 
        overlap_count: int, 
        total_genes: int, 
        pathway_count: int, 
        background_size: int
    ) -> float:
        """Calculate hypergeometric p-value."""
        # Hypergeometric test: P(X >= overlap_count)
        # X ~ Hypergeometric(N=background_size, K=pathway_count, n=total_genes)
        
        # Use survival function (1 - CDF) for P(X >= overlap_count)
        p_value = stats.hypergeom.sf(
            overlap_count - 1,  # -1 because sf gives P(X > k), we want P(X >= k)
            background_size,
            pathway_count,
            total_genes
        )
        
        return min(max(p_value, 1e-300), 1.0)  # Clamp to avoid numerical issues
    
    def _calculate_enrichment_score(
        self, 
        overlap_count: int, 
        total_genes: int, 
        pathway_count: int
    ) -> float:
        """Calculate enrichment score (fold enrichment)."""
        if total_genes == 0 or pathway_count == 0:
            return 0.0
        
        expected_overlap = (total_genes * pathway_count) / self.database_manager.get_background_size("human")
        if expected_overlap == 0:
            return float('inf') if overlap_count > 0 else 0.0
        
        return overlap_count / expected_overlap
    
    def _apply_correction(
        self, 
        p_values: List[float], 
        correction_method: CorrectionMethod
    ) -> List[float]:
        """Apply multiple testing correction."""
        if not p_values:
            return []
        
        # Convert to numpy array
        p_array = np.array(p_values)
        
        # Apply correction based on method
        if correction_method == CorrectionMethod.BONFERRONI:
            corrected = multipletests(p_array, method='bonferroni')[1]
        elif correction_method == CorrectionMethod.FDR_BH:
            corrected = multipletests(p_array, method='fdr_bh')[1]
        elif correction_method == CorrectionMethod.FDR_BY:
            corrected = multipletests(p_array, method='fdr_by')[1]
        elif correction_method == CorrectionMethod.FDR_TSBH:
            corrected = multipletests(p_array, method='fdr_tsbh')[1]
        elif correction_method == CorrectionMethod.FDR_TSBKY:
            corrected = multipletests(p_array, method='fdr_tsbky')[1]
        elif correction_method == CorrectionMethod.HOLM:
            corrected = multipletests(p_array, method='holm')[1]
        elif correction_method == CorrectionMethod.HOCHBERG:
            corrected = multipletests(p_array, method='hochberg')[1]
        elif correction_method == CorrectionMethod.HOMMEL:
            corrected = multipletests(p_array, method='hommel')[1]
        elif correction_method == CorrectionMethod.SIDAK:
            corrected = multipletests(p_array, method='sidak')[1]
        elif correction_method == CorrectionMethod.SIDAK_SS:
            corrected = multipletests(p_array, method='sidak_ss')[1]
        elif correction_method == CorrectionMethod.SIDAK_SD:
            corrected = multipletests(p_array, method='sidak_sd')[1]
        else:
            # Default to FDR_BH
            corrected = multipletests(p_array, method='fdr_bh')[1]
        
        return corrected.tolist()
    
    def calculate_pathway_statistics(
        self, 
        pathway_results: List[PathwayResult]
    ) -> Dict[str, Any]:
        """Calculate additional pathway statistics."""
        if not pathway_results:
            return {}
        
        p_values = [result.p_value for result in pathway_results]
        adjusted_p_values = [result.adjusted_p_value for result in pathway_results]
        enrichment_scores = [result.enrichment_score for result in pathway_results if result.enrichment_score is not None]
        
        stats_dict = {
            'num_pathways': len(pathway_results),
            'min_p_value': min(p_values) if p_values else 1.0,
            'max_p_value': max(p_values) if p_values else 0.0,
            'mean_p_value': np.mean(p_values) if p_values else 1.0,
            'median_p_value': np.median(p_values) if p_values else 1.0,
            'min_adjusted_p_value': min(adjusted_p_values) if adjusted_p_values else 1.0,
            'max_adjusted_p_value': max(adjusted_p_values) if adjusted_p_values else 0.0,
            'mean_adjusted_p_value': np.mean(adjusted_p_values) if adjusted_p_values else 1.0,
            'median_adjusted_p_value': np.median(adjusted_p_values) if adjusted_p_values else 1.0,
        }
        
        if enrichment_scores:
            stats_dict.update({
                'min_enrichment_score': min(enrichment_scores),
                'max_enrichment_score': max(enrichment_scores),
                'mean_enrichment_score': np.mean(enrichment_scores),
                'median_enrichment_score': np.median(enrichment_scores),
            })
        
        return stats_dict
