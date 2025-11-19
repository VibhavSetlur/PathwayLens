"""
Unit tests for BayesianEngine.

Tests Bayesian enrichment analysis methods.
"""

import pytest
import asyncio
from pathwaylens_core.analysis.bayesian_engine import BayesianEngine
from pathwaylens_core.analysis.schemas import DatabaseType
from pathwaylens_core.data import DatabaseManager


@pytest.mark.unit
@pytest.mark.analysis
class TestBayesianEngine:
    """Test suite for BayesianEngine."""
    
    @pytest.fixture
    def bayesian_engine(self, database_manager):
        """Create BayesianEngine instance."""
        return BayesianEngine(database_manager)
    
    @pytest.mark.asyncio
    async def test_bayesian_analysis(
        self,
        bayesian_engine,
        sample_gene_list
    ):
        """Test Bayesian enrichment analysis."""
        result = await bayesian_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            significance_threshold=0.05,
            prior_alpha=1.0,
            prior_beta=1.0,
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
    async def test_bayesian_analysis_empty_input(
        self,
        bayesian_engine
    ):
        """Test Bayesian analysis with empty gene list."""
        result = await bayesian_engine.analyze(
            gene_list=[],
            database=DatabaseType.KEGG,
            species="human",
            min_size=2,
            max_size=10
        )
        
        assert result is not None
        assert result.total_pathways == 0
        assert result.significant_pathways == 0
    
    def test_calculate_bayesian_posterior(self, bayesian_engine):
        """Test Bayesian posterior probability calculation."""
        posterior_prob = bayesian_engine._calculate_bayesian_posterior(
            overlap=10,
            total_input=50,
            pathway_size=100,
            background_size=20000,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        
        assert isinstance(posterior_prob, float)
        assert 0.0 <= posterior_prob <= 1.0
    
    def test_calculate_bayesian_posterior_edge_cases(self, bayesian_engine):
        """Test Bayesian posterior with edge cases."""
        # Zero overlap
        prob1 = bayesian_engine._calculate_bayesian_posterior(
            overlap=0,
            total_input=50,
            pathway_size=100,
            background_size=20000,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        assert 0.0 <= prob1 <= 1.0
        
        # Full overlap
        prob2 = bayesian_engine._calculate_bayesian_posterior(
            overlap=50,
            total_input=50,
            pathway_size=100,
            background_size=20000,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        assert 0.0 <= prob2 <= 1.0
    
    def test_calculate_bayes_factor(self, bayesian_engine):
        """Test Bayes factor calculation."""
        bayes_factor = bayesian_engine._calculate_bayes_factor(
            overlap=10,
            total_input=50,
            pathway_size=100,
            background_size=20000,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        
        assert isinstance(bayes_factor, float)
        assert bayes_factor > 0
    
    def test_calculate_bayes_factor_edge_cases(self, bayesian_engine):
        """Test Bayes factor with edge cases."""
        # Zero overlap
        bf1 = bayesian_engine._calculate_bayes_factor(
            overlap=0,
            total_input=50,
            pathway_size=100,
            background_size=20000,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        assert bf1 > 0
        
        # Full overlap
        bf2 = bayesian_engine._calculate_bayes_factor(
            overlap=50,
            total_input=50,
            pathway_size=100,
            background_size=20000,
            prior_alpha=1.0,
            prior_beta=1.0
        )
        assert bf2 > 0
    
    @pytest.mark.asyncio
    async def test_get_background_size(self, bayesian_engine):
        """Test background size retrieval."""
        size = await bayesian_engine._get_background_size("human")
        
        assert isinstance(size, int)
        assert size > 0
        
        # Test with different species
        size_mouse = await bayesian_engine._get_background_size("mouse")
        assert size_mouse > 0
        
        # Test with unknown species (should use default)
        size_unknown = await bayesian_engine._get_background_size("unknown_species")
        assert size_unknown > 0
    
    def test_calculate_coverage(
        self,
        bayesian_engine,
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
        
        coverage = bayesian_engine._calculate_coverage(sample_gene_list, pathway_data)
        
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0
    
    def test_calculate_coverage_empty(self, bayesian_engine):
        """Test coverage calculation with empty pathway data."""
        coverage = bayesian_engine._calculate_coverage(
            gene_list=["GENE1"],
            pathway_data={}
        )
        
        assert coverage == 0.0
    
    def test_empty_result(self, bayesian_engine):
        """Test empty result creation."""
        result = bayesian_engine._empty_result(
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
    
    @pytest.mark.asyncio
    async def test_bayesian_analysis_different_priors(
        self,
        bayesian_engine,
        sample_gene_list
    ):
        """Test Bayesian analysis with different prior parameters."""
        result1 = await bayesian_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            prior_alpha=0.5,
            prior_beta=0.5,
            min_size=2,
            max_size=10
        )
        
        result2 = await bayesian_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            prior_alpha=2.0,
            prior_beta=2.0,
            min_size=2,
            max_size=10
        )
        
        # Both should produce valid results
        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1.total_pathways, int)
        assert isinstance(result2.total_pathways, int)



