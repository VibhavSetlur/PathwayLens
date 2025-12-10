"""
Gene Set Enrichment Analysis (GSEA) engine for PathwayLens.
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


class GSEAEngine:
    """Gene Set Enrichment Analysis engine."""
    
    def __init__(self, database_manager: DatabaseManager):
        """
        Initialize the GSEA engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="gsea_engine")
        self.database_manager = database_manager
    
    async def analyze(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str,
        significance_threshold: float = 0.05,
        correction_method: CorrectionMethod = CorrectionMethod.FDR_BH,
        permutations: int = 1000,
        min_size: int = 15,
        max_size: int = 500
    ) -> DatabaseResult:
        """
        Perform Gene Set Enrichment Analysis.
        
        Args:
            gene_list: List of input genes with expression/ranking information
            database: Database to use for analysis
            species: Species for analysis
            significance_threshold: Significance threshold
            correction_method: Multiple testing correction method
            permutations: Number of permutations for significance testing
            min_size: Minimum gene set size
            max_size: Maximum gene set size
            
        Returns:
            DatabaseResult with GSEA analysis results
        """
        self.logger.info(f"Starting GSEA analysis with {database.value} for {species}")
        
        try:
            # Validate input parameters
            if not self._validate_input_parameters(gene_list, database, species):
                raise ValueError("Invalid input parameters")
            
            if not self._validate_threshold_parameters(significance_threshold, min_size, max_size):
                raise ValueError("Invalid threshold parameters")
                
            if not self._validate_permutation_parameters(permutations):
                raise ValueError("Invalid permutation parameters")

            # Get pathway data from database
            pathway_data_dict = await self.database_manager.get_pathways(
                databases=[database.value],
                species=species
            )
            
            # Extract pathways for the specific database
            pathways = pathway_data_dict.get(database.value, [])
            
            # Filter pathways by size
            pathways = self._filter_pathways_by_size(pathways, min_size, max_size)
            
            if not pathways:
                self.logger.warning(f"No pathways found for {database.value} in {species}")
                return self._create_empty_result(database, species)
            
            # Prepare gene ranking (if not already ranked)
            gene_ranking = self._prepare_gene_ranking(gene_list)
            
            # Perform GSEA analysis
            pathway_results = []
            
            # Perform GSEA analysis
            pathway_results = []
            
            for pathway_info in pathways:
                if isinstance(pathway_info, dict):
                    pathway_id = pathway_info.get('id', 'unknown')
                    pathway_genes = pathway_info.get('genes', [])
                    pathway_name = pathway_info.get('name', 'Unknown Pathway')
                    pathway_url = pathway_info.get('url')
                    pathway_description = pathway_info.get('description')
                    pathway_category = pathway_info.get('category')
                    pathway_version = pathway_info.get('version')
                    pathway_updated = pathway_info.get('last_updated')
                else:
                    # Assume object with attributes
                    pathway_id = getattr(pathway_info, 'pathway_id', 'unknown')
                    pathway_genes = getattr(pathway_info, 'genes', [])
                    pathway_name = getattr(pathway_info, 'name', 'Unknown Pathway')
                    pathway_url = getattr(pathway_info, 'url', None)
                    pathway_description = getattr(pathway_info, 'description', None)
                    pathway_category = getattr(pathway_info, 'category', None)
                    pathway_version = getattr(pathway_info, 'version', None)
                    pathway_updated = getattr(pathway_info, 'last_updated', None)
                
                # Calculate GSEA statistics
                es_result = self._calculate_enrichment_score(
                    gene_ranking, pathway_genes
                )
                
                if es_result is None:
                    continue
                
                enrichment_score, normalized_es, p_value = es_result
                
                # Calculate overlap
                overlapping_genes = list(set(gene_ranking.keys()) & set(pathway_genes))
                overlap_count = len(overlapping_genes)
                pathway_count = len(pathway_genes)
                total_genes = len(gene_ranking)
                
                pathway_result = PathwayResult(
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    database=database,
                    p_value=p_value,
                    adjusted_p_value=p_value,  # Will be corrected later
                    enrichment_score=enrichment_score,
                    normalized_enrichment_score=normalized_es,
                    overlap_count=overlap_count,
                    pathway_count=pathway_count,
                    input_count=total_genes,
                    overlapping_genes=overlapping_genes,
                    pathway_genes=pathway_genes,
                    pathway_url=pathway_info.get('url'),
                    pathway_description=pathway_info.get('description'),
                    pathway_category=pathway_info.get('category'),
                    analysis_method="GSEA"
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
            coverage = len(covered_genes) / len(gene_ranking) if gene_ranking else 0.0
            
            # Sort by significance
            pathway_results.sort(key=lambda x: x.adjusted_p_value)
            significant_pathways.sort(key=lambda x: x.adjusted_p_value)
            
            self.logger.info(
                f"GSEA analysis completed: {len(significant_pathways)}/{len(pathway_results)} "
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
            self.logger.error(f"GSEA analysis failed: {e}")
            raise
    
    def _create_empty_result(
        self,
        database: DatabaseType,
        species: str
    ) -> DatabaseResult:
        """Create an empty database result for when no pathways are found."""
        return DatabaseResult(
            database=database,
            total_pathways=0,
            significant_pathways=0,
            pathways=[],
            species=species,
            coverage=0.0
        )
    
    def _prepare_gene_ranking(self, gene_list: List[str]) -> Dict[str, float]:
        """
        Prepare gene ranking from input gene list.
        
        If gene_list contains tuples of (gene, score), use the scores.
        Otherwise, create a simple ranking based on order.
        """
        gene_ranking = {}
        
        for i, gene_item in enumerate(gene_list):
            if isinstance(gene_item, tuple) and len(gene_item) == 2:
                gene, score = gene_item
                try:
                    gene_ranking[gene] = float(score)
                except (ValueError, TypeError):
                    # If score is not numeric, use position-based ranking
                    gene_ranking[gene] = len(gene_list) - i
            else:
                # Simple ranking based on position (higher position = higher rank)
                gene_ranking[str(gene_item)] = len(gene_list) - i
        
        return gene_ranking
    
    def _calculate_es_only(
        self,
        gene_ranking: Dict[str, float],
        pathway_genes: List[str]
    ) -> Optional[float]:
        """Calculate only the enrichment score without p-value."""
        # Get all genes and their scores
        all_genes = list(gene_ranking.keys())
        all_scores = list(gene_ranking.values())
        
        # Sort genes by score (descending)
        sorted_indices = np.argsort(all_scores)[::-1]
        sorted_genes = [all_genes[i] for i in sorted_indices]
        sorted_scores = [all_scores[i] for i in sorted_indices]
        
        # Create binary vector for pathway genes
        pathway_mask = np.array([gene in pathway_genes for gene in sorted_genes])
        
        if not np.any(pathway_mask):
            return None
        
        # Calculate enrichment score
        n_genes = len(all_genes)
        n_pathway = len(pathway_genes)
        
        # Calculate running sum
        running_sum = np.zeros(n_genes + 1)
        
        # Calculate negative step size
        # If all genes are in pathway, step size is 0 (avoid division by zero)
        neg_step = 1.0 / (n_genes - n_pathway) if n_genes > n_pathway else 0.0
        
        for i in range(n_genes):
            if pathway_mask[i]:
                # Add positive contribution
                running_sum[i + 1] = running_sum[i] + abs(sorted_scores[i]) / n_pathway
            else:
                # Add negative contribution
                running_sum[i + 1] = running_sum[i] - neg_step
        
        # Find maximum deviation from zero
        return np.max(running_sum)

    def _calculate_enrichment_score(
        self, 
        gene_ranking: Dict[str, float], 
        pathway_genes: List[str]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Calculate GSEA enrichment score.
        
        Returns:
            Tuple of (enrichment_score, normalized_es, p_value)
        """
        enrichment_score = self._calculate_es_only(gene_ranking, pathway_genes)
        
        if enrichment_score is None:
            return None
        
        # Normalize by the sum of absolute scores of pathway genes
        n_pathway = len(pathway_genes)
        pathway_scores = [gene_ranking[gene] for gene in pathway_genes if gene in gene_ranking]
        if pathway_scores:
            max_possible_es = sum(abs(score) for score in pathway_scores) / n_pathway
            normalized_es = enrichment_score / max_possible_es if max_possible_es > 0 else 0
        else:
            normalized_es = 0
        
        # Calculate p-value using permutation test
        p_value = self._calculate_gsea_pvalue(
            gene_ranking, pathway_genes, enrichment_score, n_permutations=1000
        )
        
        return enrichment_score, normalized_es, p_value
    
    def _calculate_gsea_pvalue(
        self,
        gene_ranking: Dict[str, float],
        pathway_genes: List[str],
        observed_es: float,
        n_permutations: int = 1000
    ) -> float:
        """Calculate p-value using permutation test."""
        all_genes = list(gene_ranking.keys())
        n_pathway = len(pathway_genes)
        
        # Generate random gene sets of the same size
        null_es_scores = []
        
        for _ in range(n_permutations):
            # Randomly sample genes
            random_genes = np.random.choice(all_genes, size=n_pathway, replace=False)
            
            # Calculate ES for random set
            es_result = self._calculate_es_only(gene_ranking, random_genes.tolist())
            if es_result is not None:
                null_es_scores.append(es_result)
        
        if not null_es_scores:
            return 1.0
        
        # Calculate p-value
        null_es_scores = np.array(null_es_scores)
        p_value = np.mean(np.abs(null_es_scores) >= abs(observed_es))
        
        return min(max(p_value, 1e-300), 1.0)
    
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
        normalized_es = [result.normalized_enrichment_score for result in pathway_results if result.normalized_enrichment_score is not None]
        
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
        
        if normalized_es:
            stats_dict.update({
                'min_normalized_es': min(normalized_es),
                'max_normalized_es': max(normalized_es),
                'mean_normalized_es': np.mean(normalized_es),
                'median_normalized_es': np.median(normalized_es),
            })
        
        return stats_dict
    
    def calculate_cerno(
        self,
        gene_ranking: Dict[str, float],
        pathway_genes: List[str]
    ) -> Optional[Tuple[float, float]]:
        """
        Calculate CERNO (Cumulative Enrichment Ranked by Normalized Order) test.
        
        CERNO is a modified version of GSEA that uses rank-based statistics
        instead of score-based statistics.
        
        Args:
            gene_ranking: Dictionary mapping genes to their scores
            pathway_genes: List of genes in the pathway
            
        Returns:
            Tuple of (cerno_score, p_value) or None if no overlap
        """
        # Get all genes and sort by score (descending)
        all_genes = sorted(gene_ranking.keys(), key=lambda x: gene_ranking[x], reverse=True)
        total_genes = len(all_genes)
        pathway_size = len(pathway_genes)
        
        if pathway_size == 0 or total_genes == 0:
            return None
        
        # Find positions of pathway genes in ranked list
        pathway_positions = []
        for i, gene in enumerate(all_genes):
            if gene in pathway_genes:
                pathway_positions.append(i + 1)  # 1-indexed ranks
        
        if not pathway_positions:
            return None
        
        # Calculate CERNO statistic
        # CERNO = sum((total_genes - rank + 1) for pathway genes)
        cerno_score = sum(total_genes - rank + 1 for rank in pathway_positions)
        
        # Normalize by expected value
        expected_cerno = pathway_size * (total_genes + 1) / 2
        normalized_cerno = cerno_score / expected_cerno if expected_cerno > 0 else 0
        
        # Calculate p-value using permutation test
        p_value = self._calculate_cerno_pvalue(
            gene_ranking, pathway_genes, cerno_score, n_permutations=1000
        )
        
        return normalized_cerno, p_value
    
    def _calculate_cerno_pvalue(
        self,
        gene_ranking: Dict[str, float],
        pathway_genes: List[str],
        observed_cerno: float,
        n_permutations: int = 1000
    ) -> float:
        """Calculate p-value for CERNO test using permutation."""
        all_genes = list(gene_ranking.keys())
        pathway_size = len(pathway_genes)
        total_genes = len(all_genes)
        
        null_cerno_scores = []
        
        for _ in range(n_permutations):
            # Randomly sample genes
            random_genes = np.random.choice(all_genes, size=pathway_size, replace=False)
            
            # Calculate CERNO for random set
            random_positions = []
            sorted_genes = sorted(all_genes, key=lambda x: gene_ranking[x], reverse=True)
            for i, gene in enumerate(sorted_genes):
                if gene in random_genes:
                    random_positions.append(i + 1)
            
            if random_positions:
                random_cerno = sum(total_genes - rank + 1 for rank in random_positions)
                null_cerno_scores.append(random_cerno)
        
        if not null_cerno_scores:
            return 1.0
        
        # Calculate p-value (two-tailed)
        null_cerno_scores = np.array(null_cerno_scores)
        p_value = np.mean(null_cerno_scores >= observed_cerno)
        
        return min(max(p_value, 1e-300), 1.0)

    def _validate_input_parameters(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str
    ) -> bool:
        """Validate input parameters."""
        if not gene_list:
            return False
        if not database:
            return False
        if not species:
            return False
        return True

    def _validate_threshold_parameters(
        self,
        significance_threshold: float,
        min_size: int,
        max_size: int
    ) -> bool:
        """Validate threshold parameters."""
        if not (0 <= significance_threshold <= 1):
            return False
        if min_size > max_size:
            return False
        if min_size < 0 or max_size < 0:
            return False
        return True

    def _validate_permutation_parameters(
        self,
        permutations: int
    ) -> bool:
        """Validate permutation parameters."""
        if permutations <= 0:
            return False
        return True

    def _filter_pathways_by_size(
        self,
        pathways: List[Any],
        min_size: int,
        max_size: int
    ) -> List[Any]:
        """Filter pathways by size."""
        filtered_pathways = []
        for pathway in pathways:
            # Handle PathwayInfo objects
            if hasattr(pathway, 'genes'):
                size = len(pathway.genes)
            # Handle dicts (legacy)
            elif isinstance(pathway, dict):
                size = len(pathway.get('genes', []))
                if 'size' in pathway:
                    size = pathway['size']
            else:
                continue
            
            if min_size <= size <= max_size:
                filtered_pathways.append(pathway)
        return filtered_pathways