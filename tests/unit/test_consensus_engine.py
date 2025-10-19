"""
Unit tests for the Consensus engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.analysis.consensus_engine import ConsensusEngine
from pathwaylens_core.analysis.schemas import (
    DatabaseResult, ConsensusResult, ConsensusMethod, PathwayResult, DatabaseType
)


class TestConsensusEngine:
    """Test cases for the ConsensusEngine class."""

    @pytest.fixture
    def consensus_engine(self):
        """Create a ConsensusEngine instance for testing."""
        return ConsensusEngine()

    @pytest.fixture
    def sample_pathway_results(self):
        """Create sample pathway results."""
        return [
            PathwayResult(
                pathway_id="PATH:00010",
                pathway_name="Glycolysis",
                p_value=0.001,
                adjusted_p_value=0.01,
                gene_overlap=["GENE1", "GENE2"],
                gene_overlap_count=2,
                pathway_size=5
            ),
            PathwayResult(
                pathway_id="PATH:00020",
                pathway_name="TCA Cycle",
                p_value=0.005,
                adjusted_p_value=0.02,
                gene_overlap=["GENE3", "GENE4"],
                gene_overlap_count=2,
                pathway_size=4
            )
        ]

    @pytest.fixture
    def sample_database_results(self, sample_pathway_results):
        """Create sample database results."""
        return {
            "KEGG": DatabaseResult(
                database_name="KEGG",
                database_type=DatabaseType.KEGG,
                species="human",
                pathway_results=sample_pathway_results
            ),
            "REACTOME": DatabaseResult(
                database_name="REACTOME",
                database_type=DatabaseType.REACTOME,
                species="human",
                pathway_results=sample_pathway_results
            )
        }

    def test_init(self, consensus_engine):
        """Test ConsensusEngine initialization."""
        assert consensus_engine.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_basic(self, consensus_engine, sample_database_results):
        """Test basic consensus analysis."""
        # Run analysis
        result = await consensus_engine.analyze(
            database_results=sample_database_results,
            method=ConsensusMethod.STOUFFER,
            min_databases=2,
            significance_threshold=0.05
        )
        
        # Verify result structure
        assert isinstance(result, ConsensusResult)
        assert result.method == ConsensusMethod.STOUFFER
        assert result.min_databases == 2
        assert result.significance_threshold == 0.05
        assert len(result.pathway_results) > 0
        
        # Verify pathway results
        for pathway_result in result.pathway_results:
            assert isinstance(pathway_result, PathwayResult)
            assert pathway_result.pathway_id is not None
            assert pathway_result.pathway_name is not None
            assert 0 <= pathway_result.p_value <= 1

    @pytest.mark.asyncio
    async def test_analyze_with_parameters(self, consensus_engine, sample_database_results):
        """Test consensus analysis with custom parameters."""
        # Run analysis with custom parameters
        result = await consensus_engine.analyze(
            database_results=sample_database_results,
            method=ConsensusMethod.FISHER,
            min_databases=1,
            significance_threshold=0.01
        )
        
        # Verify result structure
        assert isinstance(result, ConsensusResult)
        assert result.method == ConsensusMethod.FISHER
        assert result.min_databases == 1
        assert result.significance_threshold == 0.01

    @pytest.mark.asyncio
    async def test_analyze_empty_database_results(self, consensus_engine):
        """Test consensus analysis with empty database results."""
        # Run analysis with empty database results
        result = await consensus_engine.analyze(
            database_results={},
            method=ConsensusMethod.STOUFFER,
            min_databases=2,
            significance_threshold=0.05
        )
        
        # Verify result structure
        assert isinstance(result, ConsensusResult)
        assert len(result.pathway_results) == 0

    @pytest.mark.asyncio
    async def test_analyze_insufficient_databases(self, consensus_engine, sample_database_results):
        """Test consensus analysis with insufficient databases."""
        # Run analysis with insufficient databases
        result = await consensus_engine.analyze(
            database_results=sample_database_results,
            method=ConsensusMethod.STOUFFER,
            min_databases=5,  # More than available
            significance_threshold=0.05
        )
        
        # Verify result structure
        assert isinstance(result, ConsensusResult)
        assert len(result.pathway_results) == 0

    @pytest.mark.asyncio
    async def test_analyze_different_methods(self, consensus_engine, sample_database_results):
        """Test consensus analysis with different methods."""
        methods = [ConsensusMethod.STOUFFER, ConsensusMethod.FISHER, ConsensusMethod.BROWN]
        
        for method in methods:
            # Run analysis
            result = await consensus_engine.analyze(
                database_results=sample_database_results,
                method=method,
                min_databases=2,
                significance_threshold=0.05
            )
            
            # Verify result structure
            assert isinstance(result, ConsensusResult)
            assert result.method == method

    def test_combine_p_values_stouffer(self, consensus_engine):
        """Test Stouffer's method for combining p-values."""
        p_values = [0.01, 0.05, 0.1]
        combined_p = consensus_engine._combine_p_values_stouffer(p_values)
        
        assert isinstance(combined_p, float)
        assert 0 <= combined_p <= 1

    def test_combine_p_values_fisher(self, consensus_engine):
        """Test Fisher's method for combining p-values."""
        p_values = [0.01, 0.05, 0.1]
        combined_p = consensus_engine._combine_p_values_fisher(p_values)
        
        assert isinstance(combined_p, float)
        assert 0 <= combined_p <= 1

    def test_combine_p_values_brown(self, consensus_engine):
        """Test Brown's method for combining p-values."""
        p_values = [0.01, 0.05, 0.1]
        combined_p = consensus_engine._combine_p_values_brown(p_values)
        
        assert isinstance(combined_p, float)
        assert 0 <= combined_p <= 1

    def test_combine_p_values_edge_cases(self, consensus_engine):
        """Test p-value combination with edge cases."""
        # Empty list
        combined_p = consensus_engine._combine_p_values_stouffer([])
        assert combined_p == 1.0
        
        # Single p-value
        combined_p = consensus_engine._combine_p_values_stouffer([0.05])
        assert combined_p == 0.05
        
        # All p-values are 1
        combined_p = consensus_engine._combine_p_values_stouffer([1.0, 1.0, 1.0])
        assert combined_p == 1.0
        
        # All p-values are 0
        combined_p = consensus_engine._combine_p_values_stouffer([0.0, 0.0, 0.0])
        assert combined_p == 0.0

    def test_combine_p_values_invalid_input(self, consensus_engine):
        """Test p-value combination with invalid input."""
        # Invalid p-values
        with pytest.raises(ValueError):
            consensus_engine._combine_p_values_stouffer([-0.1, 0.5])
        
        with pytest.raises(ValueError):
            consensus_engine._combine_p_values_stouffer([0.5, 1.5])

    def test_calculate_consensus_statistics(self, consensus_engine, sample_database_results):
        """Test consensus statistics calculation."""
        pathway_id = "PATH:00010"
        pathway_results = []
        
        for db_result in sample_database_results.values():
            for pathway_result in db_result.pathway_results:
                if pathway_result.pathway_id == pathway_id:
                    pathway_results.append(pathway_result)
        
        stats = consensus_engine._calculate_consensus_statistics(
            pathway_id=pathway_id,
            pathway_results=pathway_results,
            method=ConsensusMethod.STOUFFER
        )
        
        assert "pathway_id" in stats
        assert "pathway_name" in stats
        assert "p_value" in stats
        assert "gene_overlap" in stats
        assert "gene_overlap_count" in stats
        assert "pathway_size" in stats
        assert stats["pathway_id"] == pathway_id
        assert 0 <= stats["p_value"] <= 1

    def test_calculate_consensus_statistics_no_results(self, consensus_engine):
        """Test consensus statistics calculation with no results."""
        pathway_id = "PATH:00010"
        pathway_results = []
        
        stats = consensus_engine._calculate_consensus_statistics(
            pathway_id=pathway_id,
            pathway_results=pathway_results,
            method=ConsensusMethod.STOUFFER
        )
        
        assert stats["pathway_id"] == pathway_id
        assert stats["p_value"] == 1.0

    def test_filter_pathways_by_significance(self, consensus_engine, sample_pathway_results):
        """Test pathway filtering by significance."""
        filtered_pathways = consensus_engine._filter_pathways_by_significance(
            pathway_results=sample_pathway_results,
            significance_threshold=0.01
        )
        
        assert len(filtered_pathways) <= len(sample_pathway_results)
        for pathway_result in filtered_pathways:
            assert pathway_result.p_value <= 0.01

    def test_filter_pathways_by_significance_no_filter(self, consensus_engine, sample_pathway_results):
        """Test pathway filtering with no significance threshold."""
        filtered_pathways = consensus_engine._filter_pathways_by_significance(
            pathway_results=sample_pathway_results,
            significance_threshold=1.0
        )
        
        assert len(filtered_pathways) == len(sample_pathway_results)

    def test_validate_input_parameters(self, consensus_engine):
        """Test input parameter validation."""
        # Valid parameters
        assert consensus_engine._validate_input_parameters(
            database_results={"KEGG": Mock()},
            method=ConsensusMethod.STOUFFER,
            min_databases=2,
            significance_threshold=0.05
        ) is True
        
        # Invalid database results
        assert consensus_engine._validate_input_parameters(
            database_results={},
            method=ConsensusMethod.STOUFFER,
            min_databases=2,
            significance_threshold=0.05
        ) is False
        
        # Invalid method
        assert consensus_engine._validate_input_parameters(
            database_results={"KEGG": Mock()},
            method=None,
            min_databases=2,
            significance_threshold=0.05
        ) is False
        
        # Invalid min_databases
        assert consensus_engine._validate_input_parameters(
            database_results={"KEGG": Mock()},
            method=ConsensusMethod.STOUFFER,
            min_databases=0,
            significance_threshold=0.05
        ) is False
        
        # Invalid significance threshold
        assert consensus_engine._validate_input_parameters(
            database_results={"KEGG": Mock()},
            method=ConsensusMethod.STOUFFER,
            min_databases=2,
            significance_threshold=1.5
        ) is False

    def test_validate_threshold_parameters(self, consensus_engine):
        """Test threshold parameter validation."""
        # Valid thresholds
        assert consensus_engine._validate_threshold_parameters(
            min_databases=2,
            significance_threshold=0.05
        ) is True
        
        # Invalid min_databases
        assert consensus_engine._validate_threshold_parameters(
            min_databases=0,
            significance_threshold=0.05
        ) is False
        
        # Invalid significance threshold
        assert consensus_engine._validate_threshold_parameters(
            min_databases=2,
            significance_threshold=1.5
        ) is False

    def test_validate_method_parameter(self, consensus_engine):
        """Test method parameter validation."""
        # Valid methods
        assert consensus_engine._validate_method_parameter(ConsensusMethod.STOUFFER) is True
        assert consensus_engine._validate_method_parameter(ConsensusMethod.FISHER) is True
        assert consensus_engine._validate_method_parameter(ConsensusMethod.BROWN) is True
        
        # Invalid method
        assert consensus_engine._validate_method_parameter(None) is False

    def test_validate_database_results(self, consensus_engine):
        """Test database results validation."""
        # Valid database results
        assert consensus_engine._validate_database_results({"KEGG": Mock()}) is True
        
        # Invalid database results
        assert consensus_engine._validate_database_results({}) is False
        assert consensus_engine._validate_database_results(None) is False

    def test_validate_pathway_results(self, consensus_engine, sample_pathway_results):
        """Test pathway results validation."""
        # Valid pathway results
        assert consensus_engine._validate_pathway_results(sample_pathway_results) is True
        
        # Invalid pathway results
        assert consensus_engine._validate_pathway_results([]) is False
        assert consensus_engine._validate_pathway_results(None) is False

    def test_validate_p_values(self, consensus_engine):
        """Test p-values validation."""
        # Valid p-values
        assert consensus_engine._validate_p_values([0.01, 0.05, 0.1]) is True
        
        # Invalid p-values
        assert consensus_engine._validate_p_values([]) is False
        assert consensus_engine._validate_p_values([-0.1, 0.5]) is False
        assert consensus_engine._validate_p_values([0.5, 1.5]) is False
        assert consensus_engine._validate_p_values(None) is False
