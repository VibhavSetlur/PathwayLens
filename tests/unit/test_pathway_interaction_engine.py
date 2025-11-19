"""
Unit tests for the Pathway Interaction engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import networkx as nx

from pathwaylens_core.analysis.pathway_interaction_engine import (
    PathwayInteractionEngine,
    PathwayInteraction,
    InteractionAnalysisResult
)
from pathwaylens_core.analysis.schemas import (
    DatabaseType, PathwayResult
)
from pathwaylens_core.data import DatabaseManager


class TestPathwayInteractionEngine:
    """Test cases for the PathwayInteractionEngine class."""

    @pytest.fixture
    def interaction_engine(self):
        """Create a PathwayInteractionEngine instance for testing."""
        db_manager = Mock(spec=DatabaseManager)
        return PathwayInteractionEngine(database_manager=db_manager)

    @pytest.fixture
    def sample_pathway_results(self):
        """Create sample pathway results."""
        return [
            PathwayResult(
                pathway_id="PATH:00010",
                pathway_name="Glycolysis",
                database=DatabaseType.KEGG,
                p_value=0.001,
                adjusted_p_value=0.01,
                enrichment_score=2.0,
                overlapping_genes=["GENE1", "GENE2", "GENE3"],
                overlap_count=3,
                pathway_count=5,
                input_count=100,
                analysis_method="ORA",
                pathway_genes=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
            ),
            PathwayResult(
                pathway_id="PATH:00020",
                pathway_name="TCA Cycle",
                database=DatabaseType.KEGG,
                p_value=0.005,
                adjusted_p_value=0.02,
                enrichment_score=1.5,
                overlapping_genes=["GENE3", "GENE4", "GENE5"],
                overlap_count=3,
                pathway_count=5,
                input_count=100,
                analysis_method="ORA",
                pathway_genes=["GENE3", "GENE4", "GENE5", "GENE6", "GENE7"]
            ),
            PathwayResult(
                pathway_id="PATH:00030",
                pathway_name="Oxidative Phosphorylation",
                database=DatabaseType.KEGG,
                p_value=0.01,
                adjusted_p_value=0.05,
                enrichment_score=1.2,
                overlapping_genes=["GENE6", "GENE7", "GENE8"],
                overlap_count=3,
                pathway_count=5,
                input_count=100,
                analysis_method="ORA",
                pathway_genes=["GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]
            )
        ]

    @pytest.fixture
    def sample_pathway_definitions(self):
        """Create sample pathway definitions."""
        return {
            "PATH:00010": {"GENE1", "GENE2", "GENE3", "GENE4", "GENE5"},
            "PATH:00020": {"GENE3", "GENE4", "GENE5", "GENE6", "GENE7"},
            "PATH:00030": {"GENE6", "GENE7", "GENE8", "GENE9", "GENE10"}
        }

    def test_init(self, interaction_engine):
        """Test PathwayInteractionEngine initialization."""
        assert interaction_engine.database_manager is not None
        assert interaction_engine.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_interactions_basic(
        self, interaction_engine, sample_pathway_results
    ):
        """Test basic pathway interaction analysis."""
        # Mock database manager
        adapter = Mock()
        adapter.get_pathways = AsyncMock(return_value=[])
        interaction_engine.database_manager.get_adapter = Mock(return_value=adapter)
        
        # Run analysis
        result = await interaction_engine.analyze_interactions(
            pathway_results=sample_pathway_results,
            database=DatabaseType.KEGG,
            species="human",
            min_overlap=2
        )
        
        # Verify result structure
        assert isinstance(result, InteractionAnalysisResult)
        assert isinstance(result.pathway_interactions, list)
        assert isinstance(result.interaction_network, nx.Graph)
        assert isinstance(result.pathway_clusters, list)
        assert isinstance(result.hub_pathways, list)
        assert isinstance(result.interaction_statistics, dict)

    @pytest.mark.asyncio
    async def test_analyze_interactions_insufficient_pathways(
        self, interaction_engine
    ):
        """Test interaction analysis with insufficient pathways."""
        # Only one pathway result
        pathway_results = [
            PathwayResult(
                pathway_id="PATH:00010",
                pathway_name="Glycolysis",
                database=DatabaseType.KEGG,
                p_value=0.001,
                adjusted_p_value=0.01,
                enrichment_score=2.0,
                overlapping_genes=["GENE1", "GENE2"],
                overlap_count=2,
                pathway_count=5,
                input_count=100,
                analysis_method="ORA",
                pathway_genes=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
            )
        ]
        
        result = await interaction_engine.analyze_interactions(
            pathway_results=pathway_results,
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify empty result
        assert isinstance(result, InteractionAnalysisResult)
        assert len(result.pathway_interactions) == 0
        assert result.interaction_network.number_of_nodes() == 0

    @pytest.mark.asyncio
    async def test_get_pathway_definitions(
        self, interaction_engine, sample_pathway_results
    ):
        """Test pathway definitions retrieval."""
        definitions = await interaction_engine._get_pathway_definitions(
            sample_pathway_results,
            DatabaseType.KEGG,
            "human"
        )
        
        # Verify definitions
        assert isinstance(definitions, dict)
        assert len(definitions) == len(sample_pathway_results)
        
        for pathway_id in definitions:
            assert isinstance(definitions[pathway_id], set)
            assert len(definitions[pathway_id]) > 0

    @pytest.mark.asyncio
    async def test_calculate_pairwise_interactions(
        self, interaction_engine, sample_pathway_definitions
    ):
        """Test pairwise interaction calculation."""
        interactions = await interaction_engine._calculate_pairwise_interactions(
            sample_pathway_definitions,
            min_overlap=2,
            significance_threshold=0.05
        )
        
        # Verify interactions
        assert isinstance(interactions, list)
        
        for interaction in interactions:
            assert isinstance(interaction, PathwayInteraction)
            assert interaction.pathway1_id is not None
            assert interaction.pathway2_id is not None
            assert interaction.interaction_type is not None
            assert interaction.strength is not None
            assert interaction.overlap_count >= 2  # min_overlap
            assert 0.0 <= interaction.jaccard_index <= 1.0

    def test_build_interaction_network(self, interaction_engine):
        """Test interaction network building."""
        interactions = [
            PathwayInteraction(
                pathway1_id="PATH:00010",
                pathway2_id="PATH:00020",
                interaction_type="high_overlap",
                strength=0.8,
                overlap_genes=["GENE3", "GENE4", "GENE5"],
                overlap_count=3,
                jaccard_index=0.6
            ),
            PathwayInteraction(
                pathway1_id="PATH:00020",
                pathway2_id="PATH:00030",
                interaction_type="moderate_overlap",
                strength=0.5,
                overlap_genes=["GENE6", "GENE7"],
                overlap_count=2,
                jaccard_index=0.4
            )
        ]
        
        network = interaction_engine._build_interaction_network(interactions)
        
        # Verify network structure
        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() == 3  # Three pathways
        assert network.number_of_edges() == 2  # Two interactions

    def test_identify_pathway_clusters(self, interaction_engine):
        """Test pathway cluster identification."""
        # Create a network with clusters
        network = nx.Graph()
        network.add_edge("PATH:00010", "PATH:00020", weight=0.8)
        network.add_edge("PATH:00020", "PATH:00030", weight=0.7)
        network.add_node("PATH:00040")  # Isolated node
        
        clusters = interaction_engine._identify_pathway_clusters(network)
        
        # Verify clusters
        assert isinstance(clusters, list)
        assert len(clusters) >= 1  # At least one cluster

    def test_identify_hub_pathways(self, interaction_engine):
        """Test hub pathway identification."""
        # Create a network with a hub
        network = nx.Graph()
        network.add_edge("PATH:00010", "PATH:00020", weight=0.8)
        network.add_edge("PATH:00010", "PATH:00030", weight=0.7)
        network.add_edge("PATH:00010", "PATH:00040", weight=0.6)
        network.add_edge("PATH:00020", "PATH:00030", weight=0.5)
        
        hubs = interaction_engine._identify_hub_pathways(network)
        
        # Verify hubs
        assert isinstance(hubs, list)
        # PATH:00010 should be a hub (highest degree)
        assert len(hubs) > 0

    def test_calculate_interaction_strength(self, interaction_engine):
        """Test interaction strength calculation."""
        genes1 = {"GENE1", "GENE2", "GENE3", "GENE4", "GENE5"}
        genes2 = {"GENE3", "GENE4", "GENE5", "GENE6", "GENE7"}
        overlap_genes = ["GENE3", "GENE4", "GENE5"]
        jaccard_idx = 0.5
        
        strength = interaction_engine._calculate_interaction_strength(
            genes1, genes2, overlap_genes, jaccard_idx
        )
        
        # Verify strength
        assert isinstance(strength, float)
        assert 0.0 <= strength <= 1.0

    def test_calculate_overlap_significance(self, interaction_engine):
        """Test overlap significance calculation."""
        genes1 = {"GENE1", "GENE2", "GENE3", "GENE4", "GENE5"}
        genes2 = {"GENE3", "GENE4", "GENE5", "GENE6", "GENE7"}
        overlap_genes = ["GENE3", "GENE4", "GENE5"]
        overlap_count = len(overlap_genes)
        
        p_value = interaction_engine._calculate_overlap_significance(
            genes1, genes2, overlap_count
        )
        
        # Verify p-value
        assert isinstance(p_value, (float, type(None)))
        if p_value is not None:
            assert 0.0 <= p_value <= 1.0

    def test_calculate_interaction_statistics(self, interaction_engine):
        """Test interaction statistics calculation."""
        interactions = [
            PathwayInteraction(
                pathway1_id="PATH:00010",
                pathway2_id="PATH:00020",
                interaction_type="high_overlap",
                strength=0.8,
                overlap_genes=["GENE3", "GENE4", "GENE5"],
                overlap_count=3,
                jaccard_index=0.6
            )
        ]
        
        network = nx.Graph()
        network.add_edge("PATH:00010", "PATH:00020", weight=0.8)
        
        clusters = [["PATH:00010", "PATH:00020"]]
        
        stats = interaction_engine._calculate_interaction_statistics(
            interactions, network, clusters
        )
        
        # Verify statistics
        assert isinstance(stats, dict)
        assert "total_interactions" in stats
        assert "avg_interaction_strength" in stats

    def test_pathway_interaction_dataclass(self):
        """Test PathwayInteraction dataclass."""
        interaction = PathwayInteraction(
            pathway1_id="PATH:00010",
            pathway2_id="PATH:00020",
            interaction_type="high_overlap",
            strength=0.8,
            overlap_genes=["GENE1", "GENE2"],
            overlap_count=2,
            jaccard_index=0.5
        )
        
        assert interaction.pathway1_id == "PATH:00010"
        assert interaction.pathway2_id == "PATH:00020"
        assert interaction.interaction_type == "high_overlap"
        assert interaction.strength == 0.8
        assert len(interaction.overlap_genes) == 2
        assert interaction.overlap_count == 2
        assert interaction.jaccard_index == 0.5

    def test_interaction_analysis_result_dataclass(self):
        """Test InteractionAnalysisResult dataclass."""
        network = nx.Graph()
        network.add_node("PATH:00010")
        
        result = InteractionAnalysisResult(
            pathway_interactions=[],
            interaction_network=network,
            pathway_clusters=[],
            hub_pathways=[],
            interaction_statistics={}
        )
        
        assert isinstance(result.pathway_interactions, list)
        assert isinstance(result.interaction_network, nx.Graph)
        assert isinstance(result.pathway_clusters, list)
        assert isinstance(result.hub_pathways, list)
        assert isinstance(result.interaction_statistics, dict)



