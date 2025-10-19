"""
GSVA (Gene Set Variation Analysis) engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger

from .schemas import DatabaseResult, PathwayResult, DatabaseType
from ..data import DatabaseManager


class GSVAEngine:
    """Gene Set Variation Analysis engine for pathway activity scoring."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the GSVA engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="gsva_engine")
        self.database_manager = database_manager or DatabaseManager()
    
    async def analyze(
        self,
        expression_data: pd.DataFrame,
        database: DatabaseType,
        species: str,
        significance_threshold: float = 0.05,
        correction_method: str = "fdr_bh",
        min_size: int = 15,
        max_size: int = 500,
        method: str = "gsva",
        kcdf: str = "Gaussian",
        min_max: bool = True,
        parallel_size: int = 1
    ) -> DatabaseResult:
        """
        Perform GSVA analysis on expression data.
        
        Args:
            expression_data: Gene expression matrix (genes x samples)
            database: Pathway database to use
            species: Species for the analysis
            significance_threshold: P-value threshold for significance
            correction_method: Multiple testing correction method
            min_size: Minimum pathway size
            max_size: Maximum pathway size
            method: GSVA method ('gsva', 'ssgsea', 'zscore', 'plage')
            kcdf: Kernel cumulative distribution function
            min_max: Whether to use min-max normalization
            parallel_size: Number of parallel threads
            
        Returns:
            DatabaseResult with GSVA scores and statistics
        """
        self.logger.info(f"Starting GSVA analysis with {database.value} database")
        
        try:
            # Get pathway definitions from database
            pathway_definitions = await self._get_pathway_definitions(database, species)
            
            if not pathway_definitions:
                self.logger.warning(f"No pathways found for {database.value} in {species.value}")
                return DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=species,
                    coverage=0.0
                )
            
            # Filter pathways by size
            filtered_pathways = self._filter_pathways_by_size(
                pathway_definitions, min_size, max_size
            )
            
            # Calculate GSVA scores
            gsva_scores = await self._calculate_gsva_scores(
                expression_data, filtered_pathways, method, kcdf, min_max, parallel_size
            )
            
            # Perform statistical testing
            pathway_results = await self._perform_statistical_testing(
                gsva_scores, significance_threshold, correction_method
            )
            
            # Calculate coverage
            coverage = self._calculate_coverage(expression_data, filtered_pathways)
            
            # Create database result
            significant_count = sum(1 for p in pathway_results if p.adjusted_p_value <= significance_threshold)
            
            result = DatabaseResult(
                database=database,
                total_pathways=len(pathway_results),
                significant_pathways=significant_count,
                pathways=pathway_results,
                species=species,
                coverage=coverage
            )
            
            self.logger.info(f"GSVA analysis completed: {significant_count}/{len(pathway_results)} significant pathways")
            return result
            
        except Exception as e:
            self.logger.error(f"GSVA analysis failed: {e}")
            return DatabaseResult(
                database=database,
                total_pathways=0,
                significant_pathways=0,
                pathways=[],
                species=species,
                coverage=0.0
            )
    
    async def _get_pathway_definitions(
        self, 
        database: DatabaseType, 
        species: str
    ) -> Dict[str, List[str]]:
        """Get pathway definitions from database."""
        try:
            adapter = self.database_manager.get_adapter(database)
            if not adapter:
                self.logger.error(f"No adapter available for {database.value}")
                return {}
            
            # Get pathways for species
            pathways = await adapter.get_pathways(species)
            
            # Convert to pathway definitions format
            pathway_definitions = {}
            for pathway in pathways:
                pathway_definitions[pathway.pathway_id] = pathway.gene_ids
            
            return pathway_definitions
            
        except Exception as e:
            self.logger.error(f"Failed to get pathway definitions: {e}")
            return {}
    
    def _filter_pathways_by_size(
        self, 
        pathway_definitions: Dict[str, List[str]], 
        min_size: int, 
        max_size: int
    ) -> Dict[str, List[str]]:
        """Filter pathways by size constraints."""
        filtered = {}
        
        for pathway_id, gene_ids in pathway_definitions.items():
            if min_size <= len(gene_ids) <= max_size:
                filtered[pathway_id] = gene_ids
        
        self.logger.info(f"Filtered {len(pathway_definitions)} pathways to {len(filtered)} by size")
        return filtered
    
    async def _calculate_gsva_scores(
        self,
        expression_data: pd.DataFrame,
        pathway_definitions: Dict[str, List[str]],
        method: str,
        kcdf: str,
        min_max: bool,
        parallel_size: int
    ) -> pd.DataFrame:
        """Calculate GSVA scores for pathways."""
        try:
            # Import GSVA implementation (simplified version)
            # In a real implementation, you would use the R GSVA package via rpy2
            
            # For now, create a simplified GSVA-like scoring
            gsva_scores = pd.DataFrame(index=list(pathway_definitions.keys()))
            
            for sample in expression_data.columns:
                sample_scores = []
                
                for pathway_id, gene_ids in pathway_definitions.items():
                    # Get expression values for pathway genes
                    pathway_genes = [g for g in gene_ids if g in expression_data.index]
                    
                    if len(pathway_genes) == 0:
                        score = 0.0
                    else:
                        # Calculate pathway activity score (simplified)
                        pathway_expression = expression_data.loc[pathway_genes, sample]
                        
                        if method == "gsva":
                            # Simplified GSVA scoring
                            score = self._calculate_gsva_score(pathway_expression)
                        elif method == "ssgsea":
                            # Single sample GSEA scoring
                            score = self._calculate_ssgsea_score(pathway_expression)
                        elif method == "zscore":
                            # Z-score method
                            score = self._calculate_zscore(pathway_expression)
                        elif method == "plage":
                            # PLAGE method
                            score = self._calculate_plage_score(pathway_expression)
                        else:
                            score = self._calculate_gsva_score(pathway_expression)
                    
                    sample_scores.append(score)
                
                gsva_scores[sample] = sample_scores
            
            # Apply min-max normalization if requested
            if min_max:
                gsva_scores = (gsva_scores - gsva_scores.min()) / (gsva_scores.max() - gsva_scores.min())
            
            return gsva_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate GSVA scores: {e}")
            return pd.DataFrame()
    
    def _calculate_gsva_score(self, pathway_expression: pd.Series) -> float:
        """Calculate GSVA score for a pathway."""
        if len(pathway_expression) == 0:
            return 0.0
        
        # Sort genes by expression
        sorted_expression = pathway_expression.sort_values(ascending=False)
        n_genes = len(sorted_expression)
        
        # Calculate cumulative sum
        cumsum = sorted_expression.cumsum()
        
        # Calculate ES (Enrichment Score)
        max_es = cumsum.max()
        min_es = cumsum.min()
        
        if abs(max_es) > abs(min_es):
            return max_es / n_genes
        else:
            return min_es / n_genes
    
    def _calculate_ssgsea_score(self, pathway_expression: pd.Series) -> float:
        """Calculate single sample GSEA score."""
        if len(pathway_expression) == 0:
            return 0.0
        
        # Rank genes by expression
        ranks = pathway_expression.rank(ascending=False, method='average')
        
        # Calculate weighted score
        weights = 1.0 / ranks ** 0.25  # GSEA-like weighting
        score = (pathway_expression * weights).sum()
        
        return score / len(pathway_expression)
    
    def _calculate_zscore(self, pathway_expression: pd.Series) -> float:
        """Calculate z-score for pathway."""
        if len(pathway_expression) == 0:
            return 0.0
        
        return pathway_expression.mean()
    
    def _calculate_plage_score(self, pathway_expression: pd.Series) -> float:
        """Calculate PLAGE (Pathway Level Analysis of Gene Expression) score."""
        if len(pathway_expression) == 0:
            return 0.0
        
        # Calculate first principal component
        from sklearn.decomposition import PCA
        
        if len(pathway_expression) == 1:
            return float(pathway_expression.iloc[0])
        
        # For PLAGE, we need multiple samples, so this is simplified
        pca = PCA(n_components=1)
        score = pca.fit_transform(pathway_expression.values.reshape(-1, 1))[0, 0]
        
        return float(score)
    
    async def _perform_statistical_testing(
        self,
        gsva_scores: pd.DataFrame,
        significance_threshold: float,
        correction_method: str
    ) -> List[PathwayResult]:
        """Perform statistical testing on GSVA scores."""
        pathway_results = []
        
        for pathway_id in gsva_scores.index:
            scores = gsva_scores.loc[pathway_id].values
            
            # Perform t-test against zero (simplified)
            from scipy import stats
            
            t_stat, p_value = stats.ttest_1samp(scores, 0)
            
            # Apply multiple testing correction
            # For now, just use the raw p-value
            adjusted_p_value = p_value  # Would apply correction here
            
            # Calculate enrichment score (mean GSVA score)
            enrichment_score = np.mean(scores)
            
            # Create pathway result
            pathway_result = PathwayResult(
                pathway_id=pathway_id,
                pathway_name=pathway_id,  # Would get actual name from database
                p_value=float(p_value),
                adjusted_p_value=float(adjusted_p_value),
                enrichment_score=float(enrichment_score),
                overlap_count=len(scores),
                pathway_count=len(scores),
                overlapping_genes=[]  # Not applicable for GSVA
            )
            
            pathway_results.append(pathway_result)
        
        # Sort by adjusted p-value
        pathway_results.sort(key=lambda x: x.adjusted_p_value)
        
        return pathway_results
    
    def _calculate_coverage(
        self, 
        expression_data: pd.DataFrame, 
        pathway_definitions: Dict[str, List[str]]
    ) -> float:
        """Calculate pathway coverage."""
        all_pathway_genes = set()
        for gene_ids in pathway_definitions.values():
            all_pathway_genes.update(gene_ids)
        
        available_genes = set(expression_data.index)
        covered_genes = all_pathway_genes.intersection(available_genes)
        
        if len(all_pathway_genes) == 0:
            return 0.0
        
        return len(covered_genes) / len(all_pathway_genes)
