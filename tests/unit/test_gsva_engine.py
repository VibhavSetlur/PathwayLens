"""
Unit tests for the GSVA engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.analysis.gsva_engine import GSVAEngine
from pathwaylens_core.analysis.schemas import (
    DatabaseType, PathwayResult, DatabaseResult
)
from pathwaylens_core.data import DatabaseManager


class TestGSVAEngine:
    """Test cases for the GSVAEngine class."""

    @pytest.fixture
    def gsva_engine(self):
        """Create a GSVAEngine instance for testing."""
        db_manager = Mock(spec=DatabaseManager)
        return GSVAEngine(database_manager=db_manager)

    @pytest.fixture
    def sample_expression_data(self):
        """Create sample expression data (genes x samples)."""
        return pd.DataFrame({
            "Sample1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "Sample2": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            "Sample3": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        }, index=[f"GENE{i}" for i in range(1, 11)])

    @pytest.fixture
    def sample_pathway_definitions(self):
        """Create sample pathway definitions."""
        return {
            "PATH:00010": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"],
            "PATH:00020": ["GENE3", "GENE4", "GENE5", "GENE6", "GENE7"],
            "PATH:00030": ["GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]
        }

    @pytest.fixture
    def mock_pathway_adapter(self):
        """Create a mock pathway adapter."""
        adapter = Mock()
        pathway_mock = Mock()
        pathway_mock.pathway_id = "PATH:00010"
        pathway_mock.gene_ids = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        adapter.get_pathways = AsyncMock(return_value=[pathway_mock])
        return adapter

    def test_init(self, gsva_engine):
        """Test GSVAEngine initialization."""
        assert gsva_engine.database_manager is not None
        assert gsva_engine.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_basic(self, gsva_engine, sample_expression_data, mock_pathway_adapter):
        """Test basic GSVA analysis."""
        # Mock database manager
        gsva_engine.database_manager.get_adapter = Mock(return_value=mock_pathway_adapter)
        
        # Run analysis
        result = await gsva_engine.analyze(
            expression_data=sample_expression_data,
            database=DatabaseType.KEGG,
            species="human",
            significance_threshold=0.05,
            min_size=5,
            max_size=500
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert result.database == DatabaseType.KEGG
        assert result.species == "human"

    @pytest.mark.asyncio
    async def test_analyze_no_pathways(self, gsva_engine, sample_expression_data):
        """Test GSVA analysis with no pathways available."""
        # Mock adapter returning empty list
        adapter = Mock()
        adapter.get_pathways = AsyncMock(return_value=[])
        gsva_engine.database_manager.get_adapter = Mock(return_value=adapter)
        
        # Run analysis
        result = await gsva_engine.analyze(
            expression_data=sample_expression_data,
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify empty result
        assert isinstance(result, DatabaseResult)
        assert result.total_pathways == 0
        assert len(result.pathways) == 0

    @pytest.mark.asyncio
    async def test_analyze_no_adapter(self, gsva_engine, sample_expression_data):
        """Test GSVA analysis when no adapter is available."""
        # Mock database manager returning None for adapter
        gsva_engine.database_manager.get_adapter = Mock(return_value=None)
        
        # Run analysis
        result = await gsva_engine.analyze(
            expression_data=sample_expression_data,
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify empty result
        assert isinstance(result, DatabaseResult)
        assert result.total_pathways == 0

    def test_filter_pathways_by_size(self, gsva_engine, sample_pathway_definitions):
        """Test pathway filtering by size."""
        filtered = gsva_engine._filter_pathways_by_size(
            sample_pathway_definitions,
            min_size=5,
            max_size=10
        )
        
        # All pathways should pass (size 5)
        assert len(filtered) == 3
        
        # Test with stricter constraints
        filtered_strict = gsva_engine._filter_pathways_by_size(
            sample_pathway_definitions,
            min_size=6,
            max_size=10
        )
        
        # No pathways should pass
        assert len(filtered_strict) == 0

    @pytest.mark.asyncio
    async def test_calculate_gsva_scores(self, gsva_engine, sample_expression_data, sample_pathway_definitions):
        """Test GSVA score calculation."""
        scores = await gsva_engine._calculate_gsva_scores(
            expression_data=sample_expression_data,
            pathway_definitions=sample_pathway_definitions,
            method="gsva",
            kcdf="Gaussian",
            min_max=True,
            parallel_size=1
        )
        
        # Verify scores DataFrame structure
        assert isinstance(scores, pd.DataFrame)
        assert len(scores) == len(sample_pathway_definitions)
        assert len(scores.columns) == len(sample_expression_data.columns)

    def test_calculate_gsva_score_method(self, gsva_engine):
        """Test GSVA score calculation method."""
        # Create sample pathway expression
        pathway_expr = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        score = gsva_engine._calculate_gsva_score(pathway_expr)
        
        # Verify score is a float
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)

    def test_calculate_ssgsea_score(self, gsva_engine):
        """Test single sample GSEA score calculation."""
        pathway_expr = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        score = gsva_engine._calculate_ssgsea_score(pathway_expr)
        
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_calculate_zscore(self, gsva_engine):
        """Test z-score calculation method."""
        pathway_expr = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        score = gsva_engine._calculate_zscore(pathway_expr)
        
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_calculate_plage_score(self, gsva_engine):
        """Test PLAGE score calculation method."""
        pathway_expr = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        
        score = gsva_engine._calculate_plage_score(pathway_expr)
        
        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_calculate_coverage(self, gsva_engine, sample_expression_data, sample_pathway_definitions):
        """Test pathway coverage calculation."""
        coverage = gsva_engine._calculate_coverage(
            sample_expression_data,
            sample_pathway_definitions
        )
        
        # Verify coverage is between 0 and 1
        assert 0.0 <= coverage <= 1.0
        
        # All genes in pathways are in expression data, so coverage should be high
        assert coverage > 0.0

    @pytest.mark.asyncio
    async def test_perform_statistical_testing(self, gsva_engine):
        """Test statistical testing on GSVA scores."""
        # Create sample GSVA scores DataFrame
        gsva_scores = pd.DataFrame({
            "Sample1": [0.5, 0.3, 0.7],
            "Sample2": [0.6, 0.4, 0.8],
            "Sample3": [0.4, 0.5, 0.6]
        }, index=["PATH:00010", "PATH:00020", "PATH:00030"])
        
        results = await gsva_engine._perform_statistical_testing(
            gsva_scores,
            significance_threshold=0.05,
            correction_method="fdr_bh"
        )
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) == len(gsva_scores)
        
        for result in results:
            assert isinstance(result, PathwayResult)
            assert result.pathway_id is not None
            assert result.p_value is not None
            assert result.adjusted_p_value is not None

    def test_calculate_gsva_score_empty(self, gsva_engine):
        """Test GSVA score calculation with empty expression."""
        empty_expr = pd.Series([], dtype=float)
        
        score = gsva_engine._calculate_gsva_score(empty_expr)
        
        assert score == 0.0

    def test_calculate_ssgsea_score_empty(self, gsva_engine):
        """Test SSGSEA score calculation with empty expression."""
        empty_expr = pd.Series([], dtype=float)
        
        score = gsva_engine._calculate_ssgsea_score(empty_expr)
        
        assert score == 0.0

    def test_calculate_zscore_empty(self, gsva_engine):
        """Test z-score calculation with empty expression."""
        empty_expr = pd.Series([], dtype=float)
        
        score = gsva_engine._calculate_zscore(empty_expr)
        
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_analyze_different_methods(self, gsva_engine, sample_expression_data, mock_pathway_adapter):
        """Test GSVA analysis with different methods."""
        gsva_engine.database_manager.get_adapter = Mock(return_value=mock_pathway_adapter)
        
        methods = ["gsva", "ssgsea", "zscore", "plage"]
        
        for method in methods:
            result = await gsva_engine.analyze(
                expression_data=sample_expression_data,
                database=DatabaseType.KEGG,
                species="human",
                method=method
            )
            
            assert isinstance(result, DatabaseResult)
            assert result.database == DatabaseType.KEGG



