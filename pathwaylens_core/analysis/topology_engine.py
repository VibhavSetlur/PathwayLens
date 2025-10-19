"""
Topology analysis engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from loguru import logger
import networkx as nx

from .schemas import DatabaseResult, PathwayResult, DatabaseType
from ..data import DatabaseManager


class TopologyEngine:
    """Pathway topology analysis engine for network-based pathway analysis."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the topology engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="topology_engine")
        self.database_manager = database_manager or DatabaseManager()
    
    async def analyze(
        self,
        gene_list: List[str],
        database: DatabaseType,
        species: str,
        significance_threshold: float = 0.05,
        correction_method: str = "fdr_bh",
        min_size: int = 5,
        max_size: int = 500,
        topology_method: str = "betweenness",
        network_type: str = "ppi"
    ) -> DatabaseResult:
        """
        Perform topology-based pathway analysis.
        
        Args:
            gene_list: List of input genes
            database: Pathway database to use
            species: Species for the analysis
            significance_threshold: P-value threshold for significance
            correction_method: Multiple testing correction method
            min_size: Minimum pathway size
            max_size: Maximum pathway size
            topology_method: Topology analysis method
            network_type: Type of network to use
            
        Returns:
            DatabaseResult with topology analysis results
        """
        self.logger.info(f"Starting topology analysis with {database.value} database")
        
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
            
            # Build pathway network
            pathway_network = await self._build_pathway_network(filtered_pathways, network_type)
            
            # Calculate topology scores
            topology_scores = await self._calculate_topology_scores(
                gene_list, pathway_network, topology_method
            )
            
            # Perform statistical testing
            pathway_results = await self._perform_statistical_testing(
                topology_scores, significance_threshold, correction_method
            )
            
            # Calculate coverage
            coverage = self._calculate_coverage(gene_list, filtered_pathways)
            
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
            
            self.logger.info(f"Topology analysis completed: {significant_count}/{len(pathway_results)} significant pathways")
            return result
            
        except Exception as e:
            self.logger.error(f"Topology analysis failed: {e}")
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
    
    async def _build_pathway_network(
        self, 
        pathway_definitions: Dict[str, List[str]], 
        network_type: str
    ) -> nx.Graph:
        """Build pathway network based on gene overlaps."""
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add pathway nodes
            for pathway_id in pathway_definitions.keys():
                G.add_node(pathway_id, type='pathway')
            
            # Add edges based on gene overlap
            pathway_ids = list(pathway_definitions.keys())
            for i, pathway1 in enumerate(pathway_ids):
                for pathway2 in pathway_ids[i+1:]:
                    genes1 = set(pathway_definitions[pathway1])
                    genes2 = set(pathway_definitions[pathway2])
                    
                    # Calculate Jaccard similarity
                    intersection = len(genes1.intersection(genes2))
                    union = len(genes1.union(genes2))
                    
                    if union > 0:
                        jaccard = intersection / union
                        if jaccard > 0.1:  # Only connect pathways with >10% overlap
                            G.add_edge(pathway1, pathway2, weight=jaccard, overlap=intersection)
            
            # If network_type is 'ppi', add protein-protein interaction edges
            if network_type == "ppi":
                await self._add_ppi_edges(G, pathway_definitions)
            
            self.logger.info(f"Built pathway network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            self.logger.error(f"Failed to build pathway network: {e}")
            return nx.Graph()
    
    async def _add_ppi_edges(self, G: nx.Graph, pathway_definitions: Dict[str, List[str]]):
        """Add protein-protein interaction edges to the network."""
        try:
            # This would typically query a PPI database
            # For now, we'll simulate PPI interactions based on pathway co-occurrence
            
            pathway_genes = {}
            for pathway_id, genes in pathway_definitions.items():
                pathway_genes[pathway_id] = set(genes)
            
            # Add PPI edges between pathways that share proteins
            pathway_ids = list(pathway_definitions.keys())
            for i, pathway1 in enumerate(pathway_ids):
                for pathway2 in pathway_ids[i+1:]:
                    genes1 = pathway_genes[pathway1]
                    genes2 = pathway_genes[pathway2]
                    
                    # Count shared genes as potential PPIs
                    shared_genes = genes1.intersection(genes2)
                    if len(shared_genes) > 0:
                        # Add PPI edge with weight based on shared genes
                        ppi_weight = len(shared_genes) / min(len(genes1), len(genes2))
                        if ppi_weight > 0.2:  # Only strong PPI connections
                            if G.has_edge(pathway1, pathway2):
                                # Update existing edge
                                G[pathway1][pathway2]['ppi_weight'] = ppi_weight
                            else:
                                # Add new PPI edge
                                G.add_edge(pathway1, pathway2, weight=ppi_weight, type='ppi')
            
        except Exception as e:
            self.logger.error(f"Failed to add PPI edges: {e}")
    
    async def _calculate_topology_scores(
        self,
        gene_list: List[str],
        pathway_network: nx.Graph,
        topology_method: str
    ) -> Dict[str, float]:
        """Calculate topology scores for pathways."""
        try:
            topology_scores = {}
            
            if topology_method == "betweenness":
                scores = nx.betweenness_centrality(pathway_network, weight='weight')
            elif topology_method == "closeness":
                scores = nx.closeness_centrality(pathway_network, distance='weight')
            elif topology_method == "eigenvector":
                scores = nx.eigenvector_centrality(pathway_network, weight='weight')
            elif topology_method == "pagerank":
                scores = nx.pagerank(pathway_network, weight='weight')
            elif topology_method == "degree":
                scores = dict(pathway_network.degree(weight='weight'))
            else:
                # Default to betweenness centrality
                scores = nx.betweenness_centrality(pathway_network, weight='weight')
            
            # Normalize scores
            if scores:
                max_score = max(scores.values())
                min_score = min(scores.values())
                if max_score > min_score:
                    for pathway_id, score in scores.items():
                        normalized_score = (score - min_score) / (max_score - min_score)
                        topology_scores[pathway_id] = normalized_score
                else:
                    topology_scores = scores
            else:
                topology_scores = {}
            
            self.logger.info(f"Calculated topology scores for {len(topology_scores)} pathways")
            return topology_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate topology scores: {e}")
            return {}
    
    async def _perform_statistical_testing(
        self,
        topology_scores: Dict[str, float],
        significance_threshold: float,
        correction_method: str
    ) -> List[PathwayResult]:
        """Perform statistical testing on topology scores."""
        pathway_results = []
        
        if not topology_scores:
            return pathway_results
        
        # Convert scores to array for statistical testing
        scores_array = np.array(list(topology_scores.values()))
        
        # Perform statistical test (simplified - would use proper statistical tests)
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        for pathway_id, score in topology_scores.items():
            # Calculate z-score
            z_score = (score - mean_score) / std_score if std_score > 0 else 0
            
            # Convert to p-value (simplified)
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Apply multiple testing correction (simplified)
            adjusted_p_value = p_value  # Would apply proper correction here
            
            # Create pathway result
            pathway_result = PathwayResult(
                pathway_id=pathway_id,
                pathway_name=pathway_id,  # Would get actual name from database
                database=DatabaseType.KEGG,  # Would use actual database
                p_value=float(p_value),
                adjusted_p_value=float(adjusted_p_value),
                enrichment_score=float(score),
                overlap_count=0,  # Not applicable for topology
                pathway_count=0,  # Not applicable for topology
                input_count=0,  # Not applicable for topology
                overlapping_genes=[],
                analysis_method=f"topology_{topology_method}"
            )
            
            pathway_results.append(pathway_result)
        
        # Sort by adjusted p-value
        pathway_results.sort(key=lambda x: x.adjusted_p_value)
        
        return pathway_results
    
    def _calculate_coverage(
        self, 
        gene_list: List[str], 
        pathway_definitions: Dict[str, List[str]]
    ) -> float:
        """Calculate pathway coverage."""
        all_pathway_genes = set()
        for gene_ids in pathway_definitions.values():
            all_pathway_genes.update(gene_ids)
        
        available_genes = set(gene_list)
        covered_genes = all_pathway_genes.intersection(available_genes)
        
        if len(all_pathway_genes) == 0:
            return 0.0
        
        return len(covered_genes) / len(all_pathway_genes)
