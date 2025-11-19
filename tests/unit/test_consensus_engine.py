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
                database=DatabaseType.KEGG,
                p_value=0.001,
                adjusted_p_value=0.01,
                enrichment_score=2.0,
                overlapping_genes=["GENE1", "GENE2"],
                overlap_count=2,
                pathway_count=5,
                input_count=100,
                analysis_method="ORA"
            ),
            PathwayResult(
                pathway_id="PATH:00020",
                pathway_name="TCA Cycle",
                database=DatabaseType.KEGG,
                p_value=0.005,
                adjusted_p_value=0.02,
                enrichment_score=1.5,
                overlapping_genes=["GENE3", "GENE4"],
                overlap_count=2,
                pathway_count=4,
                input_count=100,
                analysis_method="ORA"
            )
        ]

    @pytest.fixture
    def sample_database_results(self, sample_pathway_results):
        """Create sample database results."""
        return {
            "KEGG": DatabaseResult(
                database=DatabaseType.KEGG,
                total_pathways=100,
                significant_pathways=2,
                pathways=sample_pathway_results,
                species="human",
                coverage=0.5
            ),
            "REACTOME": DatabaseResult(
                database=DatabaseType.REACTOME,
                total_pathways=100,
                significant_pathways=2,
                pathways=sample_pathway_results,
                species="human",
                coverage=0.5
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
        assert result.consensus_method == ConsensusMethod.STOUFFER
        assert len(result.pathways) > 0
        
        # Verify pathway results
        for pathway_result in result.pathways:
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
        assert result.consensus_method == ConsensusMethod.FISHER

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
        assert len(result.pathways) == 0

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
        assert len(result.pathways) == 0

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
            assert result.consensus_method == method

    def test_stouffer_method(self, consensus_engine):
        """Test Stouffer's method for combining p-values."""
        p_values = [0.01, 0.05, 0.1]
        combined_p = consensus_engine._stouffer_method(p_values)
        
        assert isinstance(combined_p, float)
        assert 0 <= combined_p <= 1

    def test_fisher_method(self, consensus_engine):
        """Test Fisher's method for combining p-values."""
        p_values = [0.01, 0.05, 0.1]
        combined_p = consensus_engine._fisher_method(p_values)
        
        assert isinstance(combined_p, float)
        assert 0 <= combined_p <= 1

    def test_brown_method(self, consensus_engine):
        """Test Brown's method for combining p-values."""
        p_values = [0.01, 0.05, 0.1]
        combined_p = consensus_engine._brown_method(p_values)
        
        assert isinstance(combined_p, float)
        assert 0 <= combined_p <= 1

    def test_combine_p_values_edge_cases(self, consensus_engine):
        """Test p-value combination with edge cases."""
        # Empty list
        combined_p = consensus_engine._combine_p_values([], ConsensusMethod.STOUFFER)
        assert combined_p == 1.0
        
        # Single p-value
        combined_p = consensus_engine._combine_p_values([0.05], ConsensusMethod.STOUFFER)
        # Stouffer with 1 p-value should return the p-value itself?
        # z = norm.ppf(1-0.05) = 1.645
        # combined_z = 1.645 / sqrt(1) = 1.645
        # p = 1 - norm.cdf(1.645) = 0.05
        assert pytest.approx(combined_p, 0.0001) == 0.05
        
        # All p-values are 1 (filtered out by 0 < p < 1 check in _combine_p_values)
        combined_p = consensus_engine._combine_p_values([1.0, 1.0, 1.0], ConsensusMethod.STOUFFER)
        assert combined_p == 1.0
        
        # All p-values are 0 (filtered out)
        combined_p = consensus_engine._combine_p_values([0.0, 0.0, 0.0], ConsensusMethod.STOUFFER)
        assert combined_p == 1.0
