"""
Unit tests for NetworkEngine.

Tests network-based enrichment analysis methods including NEAT.
"""

import pytest
import asyncio
import networkx as nx
from pathwaylens_core.analysis.network_engine import NetworkEngine
from pathwaylens_core.analysis.schemas import DatabaseType, CorrectionMethod
from pathwaylens_core.data import DatabaseManager


@pytest.mark.unit
@pytest.mark.analysis
class TestNetworkEngine:
    """Test suite for NetworkEngine."""
    
    @pytest.fixture
    def network_engine(self, database_manager):
        """Create NetworkEngine instance."""
        return NetworkEngine(database_manager)
    
    @pytest.fixture
    def sample_network(self):
        """Create sample PPI network."""
        G = nx.Graph()
        # Add some nodes and edges
        genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6"]
        G.add_nodes_from(genes)
        # Add edges
        G.add_edge("GENE1", "GENE2")
        G.add_edge("GENE2", "GENE3")
        G.add_edge("GENE3", "GENE4")
        G.add_edge("GENE1", "GENE4")
        G.add_edge("GENE5", "GENE6")
        return G
    
    @pytest.mark.asyncio
    async def test_neat_analysis_with_network(
        self,
        network_engine,
        sample_network,
        sample_gene_list
    ):
        """Test NEAT analysis with provided network."""
        result = await network_engine.analyze_neat(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            network=sample_network,
            significance_threshold=0.05,
            correction_method=CorrectionMethod.FDR_BH,
            min_size=2,
            max_size=10
        )
        
        assert result is not None
        assert result.database == DatabaseType.KEGG
        assert result.species == "human"
        assert isinstance(result.total_pathways, int)
        assert isinstance(result.significant_pathways, int)
        assert isinstance(result.pathways, list)
        assert isinstance(result.coverage, float)
        assert 0.0 <= result.coverage <= 1.0
    
    @pytest.mark.asyncio
    async def test_neat_analysis_empty_network(
        self,
        network_engine,
        sample_gene_list
    ):
        """Test NEAT analysis with empty network."""
        empty_network = nx.Graph()
        
        result = await network_engine.analyze_neat(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            network=empty_network,
            significance_threshold=0.05,
            min_size=2,
            max_size=10
        )
        
        # Should handle empty network gracefully
        assert result is not None
        assert result.total_pathways >= 0
    
    @pytest.mark.asyncio
    async def test_neat_analysis_no_network(
        self,
        network_engine,
        sample_gene_list
    ):
        """Test NEAT analysis without providing network."""
        result = await network_engine.analyze_neat(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            network=None,
            significance_threshold=0.05,
            min_size=2,
            max_size=10
        )
        
        # Should handle missing network gracefully
        assert result is not None
    
    def test_calculate_neat_statistic(
        self,
        network_engine,
        sample_network
    ):
        """Test NEAT statistic calculation."""
        gene_list = ["GENE1", "GENE2", "GENE3"]
        pathway_genes = ["GENE1", "GENE2", "GENE4"]
        
        result = network_engine._calculate_neat_statistic(
            gene_list=gene_list,
            pathway_genes=pathway_genes,
            network=sample_network,
            directed=False
        )
        
        if result is not None:
            observed, expected, p_value = result
            assert isinstance(observed, int)
            assert isinstance(expected, (int, float))
            assert isinstance(p_value, float)
            assert 0.0 <= p_value <= 1.0
            assert observed >= 0
            assert expected >= 0
    
    def test_calculate_neat_statistic_insufficient_overlap(
        self,
        network_engine,
        sample_network
    ):
        """Test NEAT statistic with insufficient gene overlap."""
        gene_list = ["GENE1"]
        pathway_genes = ["GENE2"]
        
        result = network_engine._calculate_neat_statistic(
            gene_list=gene_list,
            pathway_genes=pathway_genes,
            network=sample_network,
            directed=False
        )
        
        # Should return None when overlap < 2
        assert result is None
    
    def test_apply_correction(
        self,
        network_engine
    ):
        """Test multiple testing correction."""
        p_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        corrected = network_engine._apply_correction(
            p_values=p_values,
            method=CorrectionMethod.FDR_BH
        )
        
        assert len(corrected) == len(p_values)
        assert all(isinstance(p, float) for p in corrected)
        assert all(0.0 <= p <= 1.0 for p in corrected)
        # Corrected p-values should be >= original
        assert all(corrected[i] >= p_values[i] for i in range(len(p_values)))
    
    def test_apply_correction_empty(self, network_engine):
        """Test correction with empty p-values."""
        corrected = network_engine._apply_correction(
            p_values=[],
            method=CorrectionMethod.FDR_BH
        )
        
        assert corrected == []
    
    def test_apply_correction_all_methods(self, network_engine):
        """Test all correction methods supported by network engine."""
        p_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        
        # Test only methods supported by network engine
        methods = [
            CorrectionMethod.BONFERRONI,
            CorrectionMethod.FDR_BH,
            CorrectionMethod.FDR_BY,
            CorrectionMethod.HOLM,
            CorrectionMethod.HOMMEL,
            CorrectionMethod.SIDAK,
        ]
        
        for method in methods:
            corrected = network_engine._apply_correction(p_values, method)
            assert len(corrected) == len(p_values)
            assert all(0.0 <= p <= 1.0 for p in corrected)
    
    def test_calculate_coverage(
        self,
        network_engine,
        sample_gene_list
    ):
        """Test pathway coverage calculation."""
        pathway_data = {
            "pathway1": {
                "genes": ["GENE1", "GENE2", "GENE3"]
            },
            "pathway2": {
                "genes": ["GENE4", "GENE5"]
            }
        }
        
        coverage = network_engine._calculate_coverage(sample_gene_list, pathway_data)
        
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
    
    def test_calculate_coverage_empty(self, network_engine):
        """Test coverage calculation with empty pathway data."""
        coverage = network_engine._calculate_coverage(
            gene_list=["GENE1"],
            pathway_data={}
        )
        
        assert coverage == 0.0
    
    def test_empty_result(self, network_engine):
        """Test empty result creation."""
        result = network_engine._empty_result(
            database=DatabaseType.KEGG,
            species="human"
        )
        
        assert result is not None
        assert result.database == DatabaseType.KEGG
        assert result.species == "human"
        assert result.total_pathways == 0
        assert result.significant_pathways == 0
        assert result.pathways == []
        assert result.coverage == 0.0
    
    def test_hypergeometric_test(self, network_engine):
        """Test hypergeometric test calculation."""
        p_value = network_engine._hypergeometric_test(
            observed=5,
            pathway_possible=10,
            total_edges=20,
            total_possible=50
        )
        
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0
    
    @pytest.mark.asyncio
    async def test_build_default_network(self, network_engine):
        """Test default network building."""
        network = await network_engine._build_default_network("human")
        
        assert network is not None
        assert isinstance(network, nx.Graph)

