"""
Consensus analysis engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from loguru import logger

from .schemas import (
    DatabaseResult, ConsensusResult, ConsensusMethod, PathwayResult, DatabaseType
)


class ConsensusEngine:
    """Consensus analysis engine for combining results across databases."""
    
    def __init__(self):
        """Initialize the consensus engine."""
        self.logger = logger.bind(module="consensus_engine")
    
    async def analyze(
        self,
        database_results: Dict[str, DatabaseResult],
        method: ConsensusMethod = ConsensusMethod.STOUFFER,
        min_databases: int = 2,
        significance_threshold: float = 0.05
    ) -> ConsensusResult:
        """
        Perform consensus analysis across multiple databases.
        
        Args:
            database_results: Results from multiple databases
            method: Consensus method to use
            min_databases: Minimum number of databases required
            significance_threshold: Significance threshold
            
        Returns:
            ConsensusResult with consensus analysis
        """
        self.logger.info(f"Starting consensus analysis using {method.value} method")
        
        try:
            # Filter databases with results
            valid_databases = {
                db_name: result for db_name, result in database_results.items()
                if result.pathways and len(result.pathways) > 0
            }
            
            if len(valid_databases) < min_databases:
                self.logger.warning(
                    f"Not enough databases with results ({len(valid_databases)} < {min_databases})"
                )
                return ConsensusResult(
                    consensus_method=method,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    database_agreement={},
                    consensus_score=0.0,
                    reproducibility=0.0,
                    stability=0.0
                )
            
            # Create pathway mapping across databases
            pathway_mapping = self._create_pathway_mapping(valid_databases)
            
            # Perform consensus analysis
            consensus_pathways = []
            
            for pathway_id, pathway_info in pathway_mapping.items():
                consensus_result = self._calculate_consensus_pathway(
                    pathway_info, method, significance_threshold
                )
                
                if consensus_result:
                    consensus_pathways.append(consensus_result)
            
            # Calculate consensus statistics
            consensus_score = self._calculate_consensus_score(consensus_pathways)
            database_agreement = self._calculate_database_agreement(valid_databases)
            reproducibility = self._calculate_reproducibility(consensus_pathways)
            stability = self._calculate_stability(consensus_pathways)
            
            # Filter significant pathways
            significant_pathways = [
                pathway for pathway in consensus_pathways
                if pathway.adjusted_p_value <= significance_threshold
            ]
            
            # Sort by significance
            consensus_pathways.sort(key=lambda x: x.adjusted_p_value)
            significant_pathways.sort(key=lambda x: x.adjusted_p_value)
            
            self.logger.info(
                f"Consensus analysis completed: {len(significant_pathways)}/{len(consensus_pathways)} "
                f"significant pathways found"
            )
            
            return ConsensusResult(
                consensus_method=method,
                total_pathways=len(consensus_pathways),
                significant_pathways=len(significant_pathways),
                pathways=consensus_pathways,
                database_agreement=database_agreement,
                consensus_score=consensus_score,
                reproducibility=reproducibility,
                stability=stability
            )
            
        except Exception as e:
            self.logger.error(f"Consensus analysis failed: {e}")
            raise
    
    def _create_pathway_mapping(
        self, 
        database_results: Dict[str, DatabaseResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Create mapping of pathways across databases."""
        pathway_mapping = {}
        
        for db_name, db_result in database_results.items():
            for pathway in db_result.pathways:
                # Use pathway ID as key, but also consider pathway name for matching
                pathway_key = pathway.pathway_id
                
                if pathway_key not in pathway_mapping:
                    pathway_mapping[pathway_key] = {
                        'pathway_id': pathway.pathway_id,
                        'pathway_name': pathway.pathway_name,
                        'databases': {},
                        'all_genes': set(),
                        'pathway_genes': set()
                    }
                
                # Store database-specific information
                pathway_mapping[pathway_key]['databases'][db_name] = {
                    'p_value': pathway.p_value,
                    'adjusted_p_value': pathway.adjusted_p_value,
                    'enrichment_score': pathway.enrichment_score,
                    'normalized_enrichment_score': pathway.normalized_enrichment_score,
                    'overlap_count': pathway.overlap_count,
                    'pathway_count': pathway.pathway_count,
                    'overlapping_genes': set(pathway.overlapping_genes),
                    'pathway_genes': set(pathway.pathway_genes)
                }
                
                # Update gene sets
                pathway_mapping[pathway_key]['all_genes'].update(pathway.overlapping_genes)
                pathway_mapping[pathway_key]['pathway_genes'].update(pathway.pathway_genes)
        
        return pathway_mapping
    
    def _calculate_consensus_pathway(
        self,
        pathway_info: Dict[str, Any],
        method: ConsensusMethod,
        significance_threshold: float
    ) -> Optional[PathwayResult]:
        """Calculate consensus result for a single pathway."""
        databases = pathway_info['databases']
        
        if len(databases) < 2:
            return None
        
        # Extract p-values and other statistics
        p_values = [db_info['p_value'] for db_info in databases.values()]
        adjusted_p_values = [db_info['adjusted_p_value'] for db_info in databases.values()]
        enrichment_scores = [db_info['enrichment_score'] for db_info in databases.values() if db_info['enrichment_score'] is not None]
        
        # Calculate consensus p-value
        consensus_p_value = self._combine_p_values(p_values, method)
        
        # Calculate consensus enrichment score
        if enrichment_scores:
            consensus_enrichment_score = np.mean(enrichment_scores)
        else:
            consensus_enrichment_score = None
        
        # Calculate average statistics
        avg_overlap_count = np.mean([db_info['overlap_count'] for db_info in databases.values()])
        avg_pathway_count = np.mean([db_info['pathway_count'] for db_info in databases.values()])
        
        # Get all overlapping genes
        all_overlapping_genes = set()
        for db_info in databases.values():
            all_overlapping_genes.update(db_info['overlapping_genes'])
        
        # Calculate confidence score based on database agreement
        confidence_score = self._calculate_confidence_score(databases)
        
        # Get representative database type
        db_name = list(databases.keys())[0]
        try:
            database_type = DatabaseType(db_name.lower())
        except ValueError:
            # Fallback if not a valid enum value, though this shouldn't happen with valid input
            self.logger.warning(f"Unknown database type: {db_name}, defaulting to KEGG")
            database_type = DatabaseType.KEGG

        return PathwayResult(
            pathway_id=pathway_info['pathway_id'],
            pathway_name=pathway_info['pathway_name'],
            database=database_type,
            p_value=consensus_p_value,
            adjusted_p_value=consensus_p_value,  # Will be corrected later
            enrichment_score=consensus_enrichment_score,
            overlap_count=int(avg_overlap_count),
            pathway_count=int(avg_pathway_count),
            input_count=int(avg_overlap_count),  # Approximate
            overlapping_genes=list(all_overlapping_genes),
            pathway_genes=list(pathway_info['pathway_genes']),
            analysis_method=f"Consensus_{method.value}",
            confidence_score=confidence_score
        )
    
    def _combine_p_values(
        self, 
        p_values: List[float], 
        method: ConsensusMethod
    ) -> float:
        """Combine p-values using specified method."""
        p_values = [p for p in p_values if 0 < p < 1]  # Filter valid p-values
        
        if not p_values:
            return 1.0
        
        if method == ConsensusMethod.STOUFFER:
            return self._stouffer_method(p_values)
        elif method == ConsensusMethod.FISHER:
            return self._fisher_method(p_values)
        elif method == ConsensusMethod.BROWN:
            return self._brown_method(p_values)
        elif method == ConsensusMethod.KOST:
            return self._kost_method(p_values)
        elif method == ConsensusMethod.TIPPETT:
            return self._tippett_method(p_values)
        elif method == ConsensusMethod.MUDHOLKAR_GEORGE:
            return self._mudholkar_george_method(p_values)
        elif method == ConsensusMethod.WILKINSON:
            return self._wilkinson_method(p_values)
        elif method == ConsensusMethod.PEARSON:
            return self._pearson_method(p_values)
        elif method == ConsensusMethod.GEOMETRIC_MEAN:
            return self._geometric_mean_method(p_values)
        else:
            # Default to Stouffer
            return self._stouffer_method(p_values)
    
    def _stouffer_method(self, p_values: List[float]) -> float:
        """Stouffer's method for combining p-values."""
        z_scores = [stats.norm.ppf(1 - p) for p in p_values]
        combined_z = sum(z_scores) / np.sqrt(len(z_scores))
        return 1 - stats.norm.cdf(combined_z)
    
    def _fisher_method(self, p_values: List[float]) -> float:
        """Fisher's method for combining p-values."""
        chi2_stat = -2 * sum(np.log(p) for p in p_values)
        df = 2 * len(p_values)
        return 1 - stats.chi2.cdf(chi2_stat, df)
    
    def _brown_method(self, p_values: List[float]) -> float:
        """Brown's method for combining p-values (assumes correlation)."""
        # Simplified version - in practice, would need correlation matrix
        return self._fisher_method(p_values)
    
    def _kost_method(self, p_values: List[float]) -> float:
        """Kost's method for combining p-values."""
        # Weighted combination based on sample sizes
        weights = [1.0] * len(p_values)  # Simplified - would use actual weights
        weighted_z = sum(w * stats.norm.ppf(1 - p) for w, p in zip(weights, p_values))
        combined_z = weighted_z / np.sqrt(sum(w**2 for w in weights))
        return 1 - stats.norm.cdf(combined_z)
    
    def _tippett_method(self, p_values: List[float]) -> float:
        """Tippett's method for combining p-values (minimum p-value)."""
        min_p = min(p_values)
        return 1 - (1 - min_p) ** len(p_values)
    
    def _mudholkar_george_method(self, p_values: List[float]) -> float:
        """Mudholkar-George method for combining p-values."""
        # Uses logit transformation
        logits = [np.log(p / (1 - p)) for p in p_values]
        combined_logit = sum(logits) / np.sqrt(len(logits))
        return 1 / (1 + np.exp(combined_logit))
    
    def _wilkinson_method(self, p_values: List[float], r: int = None) -> float:
        """
        Wilkinson's method for combining p-values.
        
        Uses the r-th smallest p-value. If r is None, uses median.
        """
        if not p_values:
            return 1.0
        
        sorted_p = sorted(p_values)
        n = len(sorted_p)
        
        if r is None:
            # Use median
            r = (n + 1) // 2
        
        r = max(1, min(r, n))  # Ensure r is in valid range
        
        # Use r-th smallest p-value
        p_r = sorted_p[r - 1]
        
        # Calculate combined p-value using beta distribution
        # P(X <= p_r) where X ~ Beta(r, n - r + 1)
        from scipy.stats import beta
        combined_p = beta.cdf(p_r, r, n - r + 1)
        
        return float(combined_p)
    
    def _pearson_method(self, p_values: List[float]) -> float:
        """
        Pearson's method for combining p-values.
        
        Uses product of p-values with chi-square distribution.
        """
        if not p_values:
            return 1.0
        
        # Product of p-values
        product = np.prod(p_values)
        
        # Use chi-square distribution
        # -2 * log(product) ~ chi2(2n)
        n = len(p_values)
        chi2_stat = -2 * np.log(product)
        df = 2 * n
        
        combined_p = 1 - stats.chi2.cdf(chi2_stat, df)
        
        return float(combined_p)
    
    def _geometric_mean_method(self, p_values: List[float]) -> float:
        """
        Geometric mean method for combining p-values.
        
        Uses geometric mean of p-values, which is more robust to outliers
        than arithmetic mean.
        """
        if not p_values:
            return 1.0
        
        # Geometric mean
        geometric_mean = np.exp(np.mean(np.log(p_values)))
        
        # Adjust for multiple testing
        # Use Bonferroni-like correction
        n = len(p_values)
        adjusted_p = min(1.0, geometric_mean * n)
        
        return float(adjusted_p)
    
    def _calculate_confidence_score(self, databases: Dict[str, Dict[str, Any]]) -> float:
        """Calculate confidence score based on database agreement."""
        if len(databases) < 2:
            return 0.0
        
        # Calculate agreement in significance
        significant_count = sum(
            1 for db_info in databases.values() 
            if db_info['adjusted_p_value'] <= 0.05
        )
        significance_agreement = significant_count / len(databases)
        
        # Calculate agreement in effect direction
        enrichment_scores = [
            db_info['enrichment_score'] for db_info in databases.values()
            if db_info['enrichment_score'] is not None
        ]
        
        if len(enrichment_scores) >= 2:
            positive_count = sum(1 for score in enrichment_scores if score > 1.0)
            direction_agreement = max(positive_count, len(enrichment_scores) - positive_count) / len(enrichment_scores)
        else:
            direction_agreement = 1.0
        
        # Combine agreements
        confidence_score = (significance_agreement + direction_agreement) / 2
        return confidence_score
    
    def _calculate_consensus_score(self, consensus_pathways: List[PathwayResult]) -> float:
        """Calculate overall consensus score."""
        if not consensus_pathways:
            return 0.0
        
        # Use confidence scores if available
        confidence_scores = [
            pathway.confidence_score for pathway in consensus_pathways
            if pathway.confidence_score is not None
        ]
        
        if confidence_scores:
            return np.mean(confidence_scores)
        else:
            # Fallback to significance-based score
            significant_count = sum(
                1 for pathway in consensus_pathways
                if pathway.adjusted_p_value <= 0.05
            )
            return significant_count / len(consensus_pathways)
    
    def _calculate_database_agreement(
        self, 
        database_results: Dict[str, DatabaseResult]
    ) -> Dict[str, float]:
        """Calculate agreement between database pairs."""
        agreement_scores = {}
        db_names = list(database_results.keys())
        
        for i, db1 in enumerate(db_names):
            for db2 in db_names[i+1:]:
                agreement = self._calculate_pairwise_agreement(
                    database_results[db1], database_results[db2]
                )
                agreement_scores[f"{db1}_{db2}"] = agreement
        
        return agreement_scores
    
    def _calculate_pairwise_agreement(
        self, 
        result1: DatabaseResult, 
        result2: DatabaseResult
    ) -> float:
        """Calculate agreement between two database results."""
        # Get pathway IDs
        pathways1 = {p.pathway_id for p in result1.pathways}
        pathways2 = {p.pathway_id for p in result2.pathways}
        
        # Calculate overlap
        overlap = len(pathways1 & pathways2)
        union = len(pathways1 | pathways2)
        
        if union == 0:
            return 0.0
        
        # Jaccard index
        jaccard = overlap / union
        
        # Weight by significance agreement
        significant1 = {p.pathway_id for p in result1.pathways if p.adjusted_p_value <= 0.05}
        significant2 = {p.pathway_id for p in result2.pathways if p.adjusted_p_value <= 0.05}
        
        sig_overlap = len(significant1 & significant2)
        sig_union = len(significant1 | significant2)
        
        if sig_union > 0:
            sig_agreement = sig_overlap / sig_union
        else:
            sig_agreement = 1.0
        
        # Combine Jaccard and significance agreement
        return (jaccard + sig_agreement) / 2
    
    def _calculate_reproducibility(self, consensus_pathways: List[PathwayResult]) -> float:
        """Calculate reproducibility score."""
        if not consensus_pathways:
            return 0.0
        
        # Use confidence scores as a proxy for reproducibility
        confidence_scores = [
            pathway.confidence_score for pathway in consensus_pathways
            if pathway.confidence_score is not None
        ]
        
        if confidence_scores:
            return np.mean(confidence_scores)
        else:
            return 0.5  # Default moderate reproducibility
    
    def _calculate_stability(self, consensus_pathways: List[PathwayResult]) -> float:
        """Calculate stability score."""
        if not consensus_pathways:
            return 0.0
        
        # Calculate coefficient of variation in p-values
        p_values = [pathway.p_value for pathway in consensus_pathways]
        
        if len(p_values) < 2:
            return 1.0
        
        mean_p = np.mean(p_values)
        std_p = np.std(p_values)
        
        if mean_p == 0:
            return 1.0
        
        cv = std_p / mean_p
        stability = 1 / (1 + cv)  # Higher CV = lower stability
        
        return stability
