"""
Pathway interaction analysis engine for PathwayLens.

Analyzes interactions between pathways including:
- Pathway-pathway overlaps
- Regulatory relationships
- Functional associations
- Co-expression patterns
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from loguru import logger
import networkx as nx
from scipy import stats
from scipy.spatial.distance import jaccard

from .schemas import DatabaseResult, PathwayResult, DatabaseType
from ..data import DatabaseManager


@dataclass
class PathwayInteraction:
    """Represents an interaction between two pathways."""
    pathway1_id: str
    pathway2_id: str
    interaction_type: str
    strength: float
    overlap_genes: List[str]
    overlap_count: int
    jaccard_index: float
    p_value: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class InteractionAnalysisResult:
    """Results from pathway interaction analysis."""
    pathway_interactions: List[PathwayInteraction]
    interaction_network: nx.Graph
    pathway_clusters: List[List[str]]
    hub_pathways: List[str]
    interaction_statistics: Dict[str, Any]


class PathwayInteractionEngine:
    """Engine for analyzing pathway interactions."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the pathway interaction engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="pathway_interaction_engine")
        self.database_manager = database_manager or DatabaseManager()
    
    async def analyze_interactions(
        self,
        pathway_results: List[PathwayResult],
        database: DatabaseType,
        species: str,
        min_overlap: int = 3,
        significance_threshold: float = 0.05,
        interaction_types: Optional[List[str]] = None
    ) -> InteractionAnalysisResult:
        """
        Analyze interactions between pathways.
        
        Args:
            pathway_results: List of pathway analysis results
            database: Database type
            species: Species for analysis
            min_overlap: Minimum gene overlap to consider interaction
            significance_threshold: P-value threshold for significant interactions
            interaction_types: Types of interactions to analyze
            
        Returns:
            InteractionAnalysisResult with interaction analysis
        """
        self.logger.info(f"Analyzing pathway interactions for {len(pathway_results)} pathways")
        
        try:
            # Get pathway definitions
            pathway_definitions = await self._get_pathway_definitions(
                pathway_results, database, species
            )
            
            if len(pathway_definitions) < 2:
                self.logger.warning("Not enough pathways for interaction analysis")
                return InteractionAnalysisResult(
                    pathway_interactions=[],
                    interaction_network=nx.Graph(),
                    pathway_clusters=[],
                    hub_pathways=[],
                    interaction_statistics={}
                )
            
            # Calculate pairwise interactions
            interactions = await self._calculate_pairwise_interactions(
                pathway_definitions, min_overlap, significance_threshold
            )
            
            # Build interaction network
            interaction_network = self._build_interaction_network(interactions)
            
            # Identify pathway clusters
            clusters = self._identify_pathway_clusters(interaction_network)
            
            # Identify hub pathways
            hubs = self._identify_hub_pathways(interaction_network)
            
            # Calculate statistics
            stats = self._calculate_interaction_statistics(
                interactions, interaction_network, clusters
            )
            
            self.logger.info(
                f"Found {len(interactions)} pathway interactions, "
                f"{len(clusters)} clusters, {len(hubs)} hub pathways"
            )
            
            return InteractionAnalysisResult(
                pathway_interactions=interactions,
                interaction_network=interaction_network,
                pathway_clusters=clusters,
                hub_pathways=hubs,
                interaction_statistics=stats
            )
            
        except Exception as e:
            self.logger.error(f"Pathway interaction analysis failed: {e}")
            raise
    
    async def _get_pathway_definitions(
        self,
        pathway_results: List[PathwayResult],
        database: DatabaseType,
        species: str
    ) -> Dict[str, Set[str]]:
        """Get pathway definitions from results."""
        pathway_definitions = {}
        
        for pathway in pathway_results:
            # Use pathway genes if available, otherwise use overlapping genes
            genes = set(pathway.pathway_genes if pathway.pathway_genes else pathway.overlapping_genes)
            if genes:
                pathway_definitions[pathway.pathway_id] = genes
        
        # If we don't have pathway genes, fetch from database
        if not pathway_definitions:
            adapter = self.database_manager.get_adapter(database)
            if adapter:
                pathways = await adapter.get_pathways(species)
                for pathway in pathways:
                    pathway_definitions[pathway.pathway_id] = set(pathway.gene_ids)
        
        return pathway_definitions
    
    async def _calculate_pairwise_interactions(
        self,
        pathway_definitions: Dict[str, Set[str]],
        min_overlap: int,
        significance_threshold: float
    ) -> List[PathwayInteraction]:
        """Calculate pairwise interactions between pathways."""
        interactions = []
        pathway_ids = list(pathway_definitions.keys())
        
        for i, pathway1_id in enumerate(pathway_ids):
            for pathway2_id in pathway_ids[i+1:]:
                genes1 = pathway_definitions[pathway1_id]
                genes2 = pathway_definitions[pathway2_id]
                
                # Calculate overlap
                overlap_genes = list(genes1.intersection(genes2))
                overlap_count = len(overlap_genes)
                
                if overlap_count < min_overlap:
                    continue
                
                # Calculate Jaccard index
                union_genes = genes1.union(genes2)
                jaccard_idx = overlap_count / len(union_genes) if union_genes else 0.0
                
                # Calculate interaction strength (weighted combination)
                strength = self._calculate_interaction_strength(
                    genes1, genes2, overlap_genes, jaccard_idx
                )
                
                # Calculate statistical significance of overlap
                p_value = self._calculate_overlap_significance(
                    genes1, genes2, overlap_count
                )
                
                # Determine interaction type
                interaction_type = self._determine_interaction_type(
                    genes1, genes2, overlap_genes, jaccard_idx
                )
                
                interaction = PathwayInteraction(
                    pathway1_id=pathway1_id,
                    pathway2_id=pathway2_id,
                    interaction_type=interaction_type,
                    strength=strength,
                    overlap_genes=overlap_genes,
                    overlap_count=overlap_count,
                    jaccard_index=jaccard_idx,
                    p_value=p_value,
                    metadata={
                        'pathway1_size': len(genes1),
                        'pathway2_size': len(genes2),
                        'union_size': len(union_genes)
                    }
                )
                
                # Filter by significance if threshold provided
                if p_value is None or p_value <= significance_threshold:
                    interactions.append(interaction)
        
        # Sort by strength
        interactions.sort(key=lambda x: x.strength, reverse=True)
        
        return interactions
    
    def _calculate_interaction_strength(
        self,
        genes1: Set[str],
        genes2: Set[str],
        overlap_genes: List[str],
        jaccard_idx: float
    ) -> float:
        """Calculate interaction strength between pathways."""
        # Base strength from Jaccard index
        strength = jaccard_idx
        
        # Boost for high overlap relative to pathway sizes
        overlap_ratio1 = len(overlap_genes) / len(genes1) if genes1 else 0.0
        overlap_ratio2 = len(overlap_genes) / len(genes2) if genes2 else 0.0
        avg_overlap_ratio = (overlap_ratio1 + overlap_ratio2) / 2.0
        
        # Combined strength
        strength = 0.6 * jaccard_idx + 0.4 * avg_overlap_ratio
        
        return strength
    
    def _calculate_overlap_significance(
        self,
        genes1: Set[str],
        genes2: Set[str],
        overlap_count: int
    ) -> Optional[float]:
        """Calculate statistical significance of pathway overlap."""
        try:
            # Hypergeometric test for pathway overlap
            # Assuming a background genome size (simplified)
            # In practice, would use actual genome size
            background_size = 20000  # Approximate human genome gene count
            
            pathway1_size = len(genes1)
            pathway2_size = len(genes2)
            
            # Hypergeometric test
            # P(X >= overlap_count) where X ~ Hypergeometric(N, K, n)
            # N = background_size, K = pathway1_size, n = pathway2_size
            p_value = stats.hypergeom.sf(
                overlap_count - 1,
                background_size,
                pathway1_size,
                pathway2_size
            )
            
            return float(p_value)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate overlap significance: {e}")
            return None
    
    def _determine_interaction_type(
        self,
        genes1: Set[str],
        genes2: Set[str],
        overlap_genes: List[str],
        jaccard_idx: float
    ) -> str:
        """Determine the type of interaction between pathways."""
        overlap_ratio1 = len(overlap_genes) / len(genes1) if genes1 else 0.0
        overlap_ratio2 = len(overlap_genes) / len(genes2) if genes2 else 0.0
        
        if jaccard_idx > 0.5:
            return "high_overlap"
        elif jaccard_idx > 0.2:
            return "moderate_overlap"
        elif overlap_ratio1 > 0.3 or overlap_ratio2 > 0.3:
            return "asymmetric_overlap"
        else:
            return "low_overlap"
    
    def _build_interaction_network(
        self,
        interactions: List[PathwayInteraction]
    ) -> nx.Graph:
        """Build a network graph from pathway interactions."""
        G = nx.Graph()
        
        for interaction in interactions:
            # Add nodes
            G.add_node(interaction.pathway1_id)
            G.add_node(interaction.pathway2_id)
            
            # Add edge with weight based on interaction strength
            G.add_edge(
                interaction.pathway1_id,
                interaction.pathway2_id,
                weight=interaction.strength,
                interaction_type=interaction.interaction_type,
                overlap_count=interaction.overlap_count,
                jaccard_index=interaction.jaccard_index,
                p_value=interaction.p_value
            )
        
        return G
    
    def _identify_pathway_clusters(
        self,
        network: nx.Graph,
        min_cluster_size: int = 2
    ) -> List[List[str]]:
        """Identify clusters of interacting pathways."""
        if network.number_of_nodes() == 0:
            return []
        
        # Use community detection (Louvain algorithm if available, else greedy modularity)
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(network, weight='weight')
            clusters = [list(community) for community in communities if len(community) >= min_cluster_size]
        except Exception:
            # Fallback: use connected components
            clusters = [
                list(component) for component in nx.connected_components(network)
                if len(component) >= min_cluster_size
            ]
        
        return clusters
    
    def _identify_hub_pathways(
        self,
        network: nx.Graph,
        top_n: int = 10
    ) -> List[str]:
        """Identify hub pathways (highly connected pathways)."""
        if network.number_of_nodes() == 0:
            return []
        
        # Calculate degree centrality
        degree_centrality = nx.degree_centrality(network)
        
        # Sort by centrality and return top N
        sorted_pathways = sorted(
            degree_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [pathway_id for pathway_id, _ in sorted_pathways[:top_n]]
    
    def _calculate_interaction_statistics(
        self,
        interactions: List[PathwayInteraction],
        network: nx.Graph,
        clusters: List[List[str]]
    ) -> Dict[str, Any]:
        """Calculate statistics about pathway interactions."""
        if not interactions:
            return {}
        
        strengths = [i.strength for i in interactions]
        jaccard_indices = [i.jaccard_index for i in interactions]
        overlap_counts = [i.overlap_count for i in interactions]
        
        stats = {
            'total_interactions': len(interactions),
            'network_nodes': network.number_of_nodes(),
            'network_edges': network.number_of_edges(),
            'num_clusters': len(clusters),
            'avg_cluster_size': np.mean([len(c) for c in clusters]) if clusters else 0.0,
            'avg_interaction_strength': np.mean(strengths),
            'avg_jaccard_index': np.mean(jaccard_indices),
            'avg_overlap_count': np.mean(overlap_counts),
            'max_interaction_strength': np.max(strengths) if strengths else 0.0,
            'min_interaction_strength': np.min(strengths) if strengths else 0.0
        }
        
        return stats



