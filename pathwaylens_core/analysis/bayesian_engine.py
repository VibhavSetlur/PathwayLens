"""
Bayesian enrichment analysis engine for PathwayLens.

Implements Bayesian methods for pathway enrichment analysis.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from scipy.special import betaln, gammaln
from loguru import logger

from .schemas import DatabaseResult, PathwayResult, DatabaseType, CorrectionMethod
from ..data import DatabaseManager


class BayesianEngine:
    """Bayesian enrichment analysis engine."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the Bayesian engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="bayesian_engine")
        self.database_manager = database_manager or DatabaseManager()
    
    async def analyze(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str,
        significance_threshold: float = 0.05,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        min_size: int = 5,
        max_size: int = 500
    ) -> DatabaseResult:
        """
        Perform Bayesian enrichment analysis.
        
        Uses Bayesian approach to calculate posterior probabilities
        of pathway enrichment.
        
        Args:
            gene_list: List of input genes
            database: Pathway database to use
            species: Species for analysis
            significance_threshold: Posterior probability threshold
            prior_alpha: Prior alpha parameter (Beta prior)
            prior_beta: Prior beta parameter (Beta prior)
            min_size: Minimum pathway size
            max_size: Maximum pathway size
            
        Returns:
            DatabaseResult with Bayesian analysis results
        """
        self.logger.info(f"Starting Bayesian analysis with {database.value} for {species}")
        
        try:
            # Get pathway definitions
            pathway_data = await self._get_pathway_definitions(
                database, species, min_size, max_size
            )
            
            if not pathway_data:
                return self._empty_result(database, species)
            
            # Get background size
            background_size = await self._get_background_size(species)
            
            # Perform Bayesian analysis
            pathway_results = []
            total_genes = len(gene_list)
            
            for pathway_id, pathway_info in pathway_data.items():
                pathway_genes = pathway_info['genes']
                pathway_name = pathway_info.get('name', pathway_id)
                
                # Calculate overlap
                overlapping_genes = list(set(gene_list) & set(pathway_genes))
                overlap_count = len(overlapping_genes)
                pathway_count = len(pathway_genes)
                
                if overlap_count == 0:
                    continue
                
                # Calculate Bayesian posterior probability
                posterior_prob = self._calculate_bayesian_posterior(
                    overlap_count,
                    total_genes,
                    pathway_count,
                    background_size,
                    prior_alpha,
                    prior_beta
                )
                
                # Calculate Bayes factor
                bayes_factor = self._calculate_bayes_factor(
                    overlap_count,
                    total_genes,
                    pathway_count,
                    background_size,
                    prior_alpha,
                    prior_beta
                )
                
                # Convert posterior probability to p-value equivalent
                # (1 - posterior_prob) gives probability of null hypothesis
                p_value_equivalent = 1.0 - posterior_prob
                
                pathway_result = PathwayResult(
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    database=database,
                    p_value=float(p_value_equivalent),
                    adjusted_p_value=float(p_value_equivalent),  # Bayesian doesn't need correction
                    enrichment_score=float(posterior_prob),
                    normalized_enrichment_score=float(bayes_factor),
                    overlap_count=overlap_count,
                    pathway_count=pathway_count,
                    input_count=total_genes,
                    overlapping_genes=overlapping_genes,
                    analysis_method="Bayesian",
                    confidence_score=float(posterior_prob)
                )
                
                pathway_results.append(pathway_result)
            
            # Calculate coverage
            coverage = self._calculate_coverage(gene_list, pathway_data)
            
            # Filter by significance (posterior probability)
            significant_pathways = [
                r for r in pathway_results
                if r.enrichment_score >= (1.0 - significance_threshold)
            ]
            
            return DatabaseResult(
                database=database,
                total_pathways=len(pathway_results),
                significant_pathways=len(significant_pathways),
                pathways=sorted(pathway_results, key=lambda x: 1.0 - x.enrichment_score),
                species=species,
                coverage=coverage
            )
            
        except Exception as e:
            self.logger.error(f"Bayesian analysis failed: {e}")
            return self._empty_result(database, species)
    
    def _calculate_bayesian_posterior(
        self,
        overlap: int,
        total_input: int,
        pathway_size: int,
        background_size: int,
        prior_alpha: float,
        prior_beta: float
    ) -> float:
        """
        Calculate Bayesian posterior probability of enrichment.
        
        Uses Beta-Binomial conjugate prior.
        """
        # Parameters for Beta posterior
        posterior_alpha = prior_alpha + overlap
        posterior_beta = prior_beta + (pathway_size - overlap)
        
        # Expected proportion under null (pathway_size / background_size)
        null_proportion = pathway_size / background_size if background_size > 0 else 0.0
        
        # Calculate posterior probability that true proportion > null_proportion
        # Using Beta distribution CDF
        posterior_prob = 1.0 - stats.beta.cdf(
            null_proportion,
            posterior_alpha,
            posterior_beta
        )
        
        return min(max(posterior_prob, 0.0), 1.0)
    
    def _calculate_bayes_factor(
        self,
        overlap: int,
        total_input: int,
        pathway_size: int,
        background_size: int,
        prior_alpha: float,
        prior_beta: float
    ) -> float:
        """
        Calculate Bayes factor for enrichment.
        
        Bayes factor = P(data | H1) / P(data | H0)
        """
        # Null hypothesis: proportion = pathway_size / background_size
        null_proportion = pathway_size / background_size if background_size > 0 else 0.0
        
        # Alternative hypothesis: proportion follows Beta(prior_alpha, prior_beta)
        
        # Likelihood under null (binomial)
        log_likelihood_null = (
            gammaln(total_input + 1) -
            gammaln(overlap + 1) -
            gammaln(total_input - overlap + 1) +
            overlap * np.log(null_proportion + 1e-10) +
            (total_input - overlap) * np.log(1.0 - null_proportion + 1e-10)
        )
        
        # Likelihood under alternative (Beta-Binomial)
        log_likelihood_alt = (
            betaln(overlap + prior_alpha, total_input - overlap + prior_beta) -
            betaln(prior_alpha, prior_beta) +
            gammaln(total_input + 1) -
            gammaln(overlap + 1) -
            gammaln(total_input - overlap + 1)
        )
        
        # Bayes factor (log space)
        log_bf = log_likelihood_alt - log_likelihood_null
        
        # Convert to linear space
        bayes_factor = np.exp(log_bf)
        
        return float(bayes_factor)
    
    async def _get_pathway_definitions(
        self,
        database: DatabaseType,
        species: str,
        min_size: int,
        max_size: int
    ) -> Dict[str, Dict[str, Any]]:
        """Get pathway definitions from database."""
        try:
            pathways = await self.database_manager.get_pathways(
                databases=[database.value],
                species=species,
                min_size=min_size,
                max_size=max_size
            )
            
            pathway_data = {}
            for pathway in pathways.get(database.value, []):
                if min_size <= len(pathway.gene_ids) <= max_size:
                    pathway_data[pathway.pathway_id] = {
                        'genes': pathway.gene_ids,
                        'name': pathway.name,
                        'description': pathway.description
                    }
            
            return pathway_data
            
        except Exception as e:
            self.logger.error(f"Failed to get pathway definitions: {e}")
            return {}
    
    async def _get_background_size(self, species: str) -> int:
        """Get background gene set size for species."""
        # This would typically query a database
        # Default estimates
        background_sizes = {
            'human': 20000,
            'mouse': 25000,
            'rat': 22000,
            'drosophila': 14000,
            'zebrafish': 26000,
            'c_elegans': 20000,
            's_cerevisiae': 6000
        }
        
        return background_sizes.get(species.lower(), 20000)
    
    def _calculate_coverage(
        self,
        gene_list: List[str],
        pathway_data: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate pathway coverage."""
        all_pathway_genes = set()
        for info in pathway_data.values():
            all_pathway_genes.update(info['genes'])
        
        covered = all_pathway_genes & set(gene_list)
        
        if not all_pathway_genes:
            return 0.0
        
        return len(covered) / len(all_pathway_genes)
    
    def _empty_result(self, database: DatabaseType, species: str) -> DatabaseResult:
        """Return empty result."""
        return DatabaseResult(
            database=database,
            total_pathways=0,
            significant_pathways=0,
            pathways=[],
            species=species,
            coverage=0.0
        )



