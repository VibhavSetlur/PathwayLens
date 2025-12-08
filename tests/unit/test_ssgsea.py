"""
Unit tests for ssGSEA single-cell pathway scoring.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, AsyncMock

from pathwaylens_core.analysis.sc_engine import (
    SingleCellEngine, SingleCellResult, ResourceEstimate
)
from pathwaylens_core.analysis.schemas import DatabaseType
from pathwaylens_core.data.adapters.base import PathwayInfo


@pytest.fixture
def mock_database_manager():
    """Create a mock database manager."""
    db_manager = MagicMock()
    db_manager.get_pathways = AsyncMock(return_value={
        "kegg": [
            PathwayInfo(
                pathway_id="path:hsa00010",
                name="Glycolysis",
                genes=["GCK", "HK1", "HK2", "PFKM", "ALDOA", "GAPDH", "ENO1", "PKM"],
                species="human"
            ),
            PathwayInfo(
                pathway_id="path:hsa00020",
                name="TCA Cycle",
                genes=["CS", "ACO1", "IDH1", "OGDH", "SUCLG1", "SDH", "FH", "MDH1"],
                species="human"
            )
        ]
    })
    return db_manager


@pytest.fixture
def expression_matrix():
    """Create a mock expression matrix (genes x cells)."""
    np.random.seed(42)
    genes = ["GCK", "HK1", "HK2", "PFKM", "ALDOA", "GAPDH", "ENO1", "PKM",
             "CS", "ACO1", "IDH1", "OGDH", "RANDOM1", "RANDOM2", "RANDOM3"]
    cells = ["Cell1", "Cell2", "Cell3", "Cell4", "Cell5"]
    
    # Create expression with Glycolysis genes highly expressed in Cell1-2
    data = np.random.rand(len(genes), len(cells))
    data[:8, :2] += 3  # Boost Glycolysis genes in first 2 cells
    
    return pd.DataFrame(data, index=genes, columns=cells)


class TestResourceEstimation:
    """Tests for resource estimation functionality."""
    
    def test_estimate_fast_method(self, mock_database_manager):
        """Test that fast methods have low resource estimates."""
        engine = SingleCellEngine(mock_database_manager)
        
        estimate = engine.estimate_resources(
            n_cells=1000,
            n_pathways=100,
            n_permutations=1000,
            method="mean_zscore"
        )
        
        assert estimate.warning_level == "none"
        assert estimate.estimated_time_seconds < 1
    
    def test_estimate_ssgsea_moderate(self, mock_database_manager):
        """Test ssGSEA with moderate cell count."""
        engine = SingleCellEngine(mock_database_manager)
        
        estimate = engine.estimate_resources(
            n_cells=500,
            n_pathways=100,
            n_permutations=1000,
            method="ssgsea"
        )
        
        assert estimate.warning_level in ["none", "moderate"]
    
    def test_estimate_ssgsea_large(self, mock_database_manager):
        """Test ssGSEA with large dataset warns appropriately."""
        engine = SingleCellEngine(mock_database_manager)
        
        estimate = engine.estimate_resources(
            n_cells=50000,
            n_pathways=500,
            n_permutations=1000,
            method="ssgsea"
        )
        
        assert estimate.warning_level in ["high", "extreme"]
        assert "reduce" in estimate.message.lower() or "recommend" in estimate.message.lower()


class TestSingleCellScoring:
    """Tests for single-cell pathway scoring methods."""
    
    @pytest.mark.asyncio
    async def test_mean_scoring(self, mock_database_manager, expression_matrix):
        """Test mean expression scoring."""
        engine = SingleCellEngine(mock_database_manager)
        
        result = await engine.score_single_cells(
            expression_matrix=expression_matrix,
            database=DatabaseType.KEGG,
            species="human",
            method="mean"
        )
        
        assert isinstance(result, SingleCellResult)
        assert result.pathway_scores.shape[0] == 5  # 5 cells
        assert result.p_values is None  # Mean doesn't generate p-values
    
    @pytest.mark.asyncio
    async def test_mean_zscore_scoring(self, mock_database_manager, expression_matrix):
        """Test z-score normalized mean scoring."""
        engine = SingleCellEngine(mock_database_manager)
        
        result = await engine.score_single_cells(
            expression_matrix=expression_matrix,
            database=DatabaseType.KEGG,
            species="human",
            method="mean_zscore"
        )
        
        assert isinstance(result, SingleCellResult)
        assert result.pathway_scores.shape[0] == 5
    
    @pytest.mark.asyncio
    async def test_ssgsea_scoring(self, mock_database_manager, expression_matrix):
        """Test ssGSEA scoring with permutations."""
        engine = SingleCellEngine(mock_database_manager)
        
        result = await engine.score_single_cells(
            expression_matrix=expression_matrix,
            database=DatabaseType.KEGG,
            species="human",
            method="ssgsea",
            n_permutations=100  # Fewer for testing speed
        )
        
        assert isinstance(result, SingleCellResult)
        assert result.pathway_scores.shape[0] == 5
        assert result.p_values is not None
        assert result.p_values.shape == result.pathway_scores.shape
        
        # P-values should be between 0 and 1
        assert (result.p_values >= 0).all().all()
        assert (result.p_values <= 1).all().all()
    
    @pytest.mark.asyncio
    async def test_ssgsea_enrichment_detection(self, mock_database_manager, expression_matrix):
        """Test that ssGSEA detects enrichment correctly."""
        engine = SingleCellEngine(mock_database_manager)
        
        result = await engine.score_single_cells(
            expression_matrix=expression_matrix,
            database=DatabaseType.KEGG,
            species="human",
            method="ssgsea",
            n_permutations=100
        )
        
        # Cells 1-2 should have higher Glycolysis scores
        glycolysis_scores = result.pathway_scores["Glycolysis"]
        
        # First two cells should have higher scores (they have boosted Glycolysis genes)
        assert glycolysis_scores["Cell1"] > glycolysis_scores["Cell5"]
    
    @pytest.mark.asyncio
    async def test_gsva_scoring(self, mock_database_manager, expression_matrix):
        """Test basic GSVA scoring."""
        engine = SingleCellEngine(mock_database_manager)
        
        result = await engine.score_single_cells(
            expression_matrix=expression_matrix,
            database=DatabaseType.KEGG,
            species="human",
            method="gsva"
        )
        
        assert isinstance(result, SingleCellResult)
        assert result.pathway_scores.shape[0] == 5
    
    @pytest.mark.asyncio
    async def test_invalid_method_raises(self, mock_database_manager, expression_matrix):
        """Test that invalid method raises error."""
        engine = SingleCellEngine(mock_database_manager)
        
        with pytest.raises(ValueError, match="Unknown scoring method"):
            await engine.score_single_cells(
                expression_matrix=expression_matrix,
                database=DatabaseType.KEGG,
                species="human",
                method="invalid_method"
            )
    
    @pytest.mark.asyncio
    async def test_metadata_includes_permutations(self, mock_database_manager, expression_matrix):
        """Test that metadata includes permutation count for ssGSEA."""
        engine = SingleCellEngine(mock_database_manager)
        
        result = await engine.score_single_cells(
            expression_matrix=expression_matrix,
            database=DatabaseType.KEGG,
            species="human",
            method="ssgsea",
            n_permutations=500
        )
        
        assert result.metadata["n_permutations"] == 500
        assert result.metadata["method"] == "ssgsea"
