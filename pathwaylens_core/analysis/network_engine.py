"""
Network-based enrichment analysis engine for PathwayLens.

Implements NEAT (Network Enrichment Analysis Test) and other network-based
methods for pathway analysis.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from statsmodels.stats.multitest import multipletests
from loguru import logger
import networkx as nx

from .schemas import DatabaseResult, PathwayResult, DatabaseType, CorrectionMethod
from ..data import DatabaseManager


class NetworkEngine:
    """Network-based enrichment analysis engine."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the network engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="network_engine")
        self.database_manager = database_manager or DatabaseManager()
    
    async def analyze_neat(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str,
        network: Optional[nx.Graph] = None,
        significance_threshold: float = 0.05,
        correction_method: CorrectionMethod = CorrectionMethod.FDR_BH,
        min_size: int = 5,
        max_size: int = 500,
        directed: bool = False
    ) -> DatabaseResult:
        """
        Perform NEAT (Network Enrichment Analysis Test).
        
        NEAT tests whether a gene set is enriched in a biological network
        by comparing the number of edges between genes in the set to what
        would be expected by chance.
        
        Args:
            gene_list: List of input genes
            database: Pathway database to use
            species: Species for the analysis
            network: Biological network (PPI, co-expression, etc.)
            significance_threshold: P-value threshold
            correction_method: Multiple testing correction
            min_size: Minimum pathway size
            max_size: Maximum pathway size
            directed: Whether network is directed
            
        Returns:
            DatabaseResult with NEAT analysis results
        """
        self.logger.info(f"Starting NEAT analysis with {database.value} for {species}")
        
        try:
            # Get pathway definitions
            pathway_data = await self._get_pathway_definitions(database, species, min_size, max_size)
            
            if not pathway_data:
                return self._empty_result(database, species)
            
            # Build or use provided network
            if network is None:
                network = await self._build_default_network(species)
            
            # Perform NEAT analysis for each pathway
            pathway_results = []
            
            for pathway_id, pathway_info in pathway_data.items():
                pathway_genes = pathway_info['genes']
                pathway_name = pathway_info.get('name', pathway_id)
                
                # Calculate NEAT statistic
                neat_result = self._calculate_neat_statistic(
                    gene_list, pathway_genes, network, directed
                )
                
                if neat_result is None:
                    continue
                
                observed_edges, expected_edges, p_value = neat_result
                
                # Apply multiple testing correction (will be done globally later)
                pathway_result = PathwayResult(
                    pathway_id=pathway_id,
                    pathway_name=pathway_name,
                    database=database,
                    p_value=p_value,
                    adjusted_p_value=p_value,  # Will be corrected later
                    enrichment_score=observed_edges - expected_edges,
                    overlap_count=len(set(gene_list) & set(pathway_genes)),
                    pathway_count=len(pathway_genes),
                    input_count=len(gene_list),
                    overlapping_genes=list(set(gene_list) & set(pathway_genes)),
                    analysis_method="NEAT"
                )
                
                pathway_results.append(pathway_result)
            
            # Apply multiple testing correction
            if pathway_results:
                p_values = [r.p_value for r in pathway_results]
                corrected = self._apply_correction(p_values, correction_method)
                for i, result in enumerate(pathway_results):
                    result.adjusted_p_value = corrected[i]
            
            # Calculate coverage
            coverage = self._calculate_coverage(gene_list, pathway_data)
            
            significant_count = sum(1 for r in pathway_results if r.adjusted_p_value <= significance_threshold)
            
            return DatabaseResult(
                database=database,
                total_pathways=len(pathway_results),
                significant_pathways=significant_count,
                pathways=sorted(pathway_results, key=lambda x: x.adjusted_p_value),
                species=species,
                coverage=coverage
            )
            
        except Exception as e:
            self.logger.error(f"NEAT analysis failed: {e}")
            return self._empty_result(database, species)
    
    def _calculate_neat_statistic(
        self,
        gene_list: List[str],
        pathway_genes: List[str],
        network: nx.Graph,
        directed: bool
    ) -> Optional[Tuple[int, float, float]]:
        """
        Calculate NEAT statistic for a pathway.
        
        Returns:
            Tuple of (observed_edges, expected_edges, p_value)
        """
        # Get genes in both input and pathway
        common_genes = set(gene_list) & set(pathway_genes)
        
        if len(common_genes) < 2:
            return None
        
        # Count edges between common genes in network
        observed_edges = 0
        for gene1 in common_genes:
            for gene2 in common_genes:
                if gene1 != gene2:
                    if network.has_edge(gene1, gene2):
                        observed_edges += 1
        
        # Account for undirected vs directed
        if not directed:
            observed_edges = observed_edges // 2
        
        # Calculate expected number of edges
        n_common = len(common_genes)
        n_total = network.number_of_nodes()
        m_total = network.number_of_edges()
        
        if n_total < 2 or m_total == 0:
            return None
        
        # Expected edges = (n_choose_2 / total_choose_2) * total_edges
        if directed:
            total_possible = n_total * (n_total - 1)
            pathway_possible = n_common * (n_common - 1)
        else:
            total_possible = n_total * (n_total - 1) // 2
            pathway_possible = n_common * (n_common - 1) // 2
        
        if total_possible == 0:
            return None
        
        expected_edges = (pathway_possible / total_possible) * m_total
        
        # Calculate p-value using hypergeometric or normal approximation
        if n_common <= 20:
            # Use exact hypergeometric test
            p_value = self._hypergeometric_test(
                observed_edges, pathway_possible, m_total, total_possible
            )
        else:
            # Use normal approximation
            variance = expected_edges * (1 - pathway_possible / total_possible)
            if variance > 0:
                z_score = (observed_edges - expected_edges) / np.sqrt(variance)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                p_value = 1.0
        
        return int(observed_edges), float(expected_edges), float(p_value)
    
    def _hypergeometric_test(
        self,
        observed: int,
        pathway_possible: int,
        total_edges: int,
        total_possible: int
    ) -> float:
        """Calculate p-value using hypergeometric distribution."""
        # Simplified hypergeometric test
        # P(X >= observed) where X ~ Hypergeometric(total_possible, pathway_possible, total_edges)
        p_value = stats.hypergeom.sf(
            observed - 1,
            total_possible,
            pathway_possible,
            total_edges
        )
        return min(max(p_value, 1e-300), 1.0)
    
    async def _build_default_network(self, species: str) -> nx.Graph:
        """Build default PPI network for species."""
        # This would typically query a PPI database
        # For now, return empty graph - user should provide network
        self.logger.warning("No network provided, using empty network")
        return nx.Graph()
    
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
    
    def _apply_correction(
        self,
        p_values: List[float],
        method: CorrectionMethod
    ) -> List[float]:
        """Apply multiple testing correction."""
        if not p_values:
            return []
        
        p_array = np.array(p_values)
        
        method_map = {
            CorrectionMethod.BONFERRONI: 'bonferroni',
            CorrectionMethod.FDR_BH: 'fdr_bh',
            CorrectionMethod.FDR_BY: 'fdr_by',
            CorrectionMethod.HOLM: 'holm',
            CorrectionMethod.HOCHBERG: 'hochberg',
            CorrectionMethod.HOMMEL: 'hommel',
            CorrectionMethod.SIDAK: 'sidak',
        }
        
        stats_method = method_map.get(method, 'fdr_bh')
        corrected = multipletests(p_array, method=stats_method)[1]
        
        return corrected.tolist()
    
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



