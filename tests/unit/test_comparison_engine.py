"""
Unit tests for the Comparison engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from pathwaylens_core.comparison.engine import ComparisonEngine
from pathwaylens_core.comparison.schemas import (
    ComparisonResult, ComparisonParameters, ComparisonType,
    OverlapStatistics, CorrelationResult, ClusteringResult, PathwayConcordance
)
from pathwaylens_core.analysis.schemas import AnalysisResult, DatabaseResult, PathwayResult, DatabaseType


class TestComparisonEngine:
    """Test cases for the ComparisonEngine class."""

    @pytest.fixture
    def comparison_engine(self):
        """Create a ComparisonEngine instance for testing."""
        return ComparisonEngine()

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
    def sample_analysis_results(self, sample_pathway_results):
        """Create sample analysis results."""
        return [
            AnalysisResult(
                analysis_id="analysis_1",
                analysis_name="Analysis 1",
                analysis_type="ORA",
                parameters=Mock(),
                database_results=[
                    DatabaseResult(
                        database_name="KEGG",
                        database_type=DatabaseType.KEGG,
                        species="human",
                        pathway_results=sample_pathway_results
                    )
                ],
                timestamp="2023-01-01T00:00:00"
            ),
            AnalysisResult(
                analysis_id="analysis_2",
                analysis_name="Analysis 2",
                analysis_type="ORA",
                parameters=Mock(),
                database_results=[
                    DatabaseResult(
                        database_name="KEGG",
                        database_type=DatabaseType.KEGG,
                        species="human",
                        pathway_results=sample_pathway_results
                    )
                ],
                timestamp="2023-01-01T00:00:00"
            )
        ]

    @pytest.fixture
    def sample_comparison_parameters(self):
        """Create sample comparison parameters."""
        return ComparisonParameters(
            comparison_name="Test Comparison",
            comparison_type=ComparisonType.OVERLAP,
            significance_threshold=0.05,
            min_pathway_size=5,
            max_pathway_size=500
        )

    def test_init(self, comparison_engine):
        """Test ComparisonEngine initialization."""
        assert comparison_engine.logger is not None

    @pytest.mark.asyncio
    async def test_compare_basic(self, comparison_engine, sample_analysis_results, sample_comparison_parameters):
        """Test basic comparison analysis."""
        # Run comparison
        result = await comparison_engine.compare(
            input_data=sample_analysis_results,
            parameters=sample_comparison_parameters
        )
        
        # Verify result structure
        assert isinstance(result, ComparisonResult)
        assert result.comparison_name == sample_comparison_parameters.comparison_name
        assert result.comparison_type == sample_comparison_parameters.comparison_type
        assert result.parameters == sample_comparison_parameters
        assert len(result.overlap_statistics) > 0

    @pytest.mark.asyncio
    async def test_compare_with_output_dir(self, comparison_engine, sample_analysis_results, sample_comparison_parameters, tmp_path):
        """Test comparison analysis with output directory."""
        output_dir = tmp_path / "comparison_output"
        
        # Run comparison
        result = await comparison_engine.compare(
            input_data=sample_analysis_results,
            parameters=sample_comparison_parameters,
            output_dir=str(output_dir)
        )
        
        # Verify result structure
        assert isinstance(result, ComparisonResult)
        assert output_dir.exists()

    @pytest.mark.asyncio
    async def test_compare_empty_input(self, comparison_engine, sample_comparison_parameters):
        """Test comparison analysis with empty input."""
        # Run comparison with empty input
        result = await comparison_engine.compare(
            input_data=[],
            parameters=sample_comparison_parameters
        )
        
        # Verify result structure
        assert isinstance(result, ComparisonResult)
        assert len(result.overlap_statistics) == 0

    @pytest.mark.asyncio
    async def test_compare_single_input(self, comparison_engine, sample_analysis_results, sample_comparison_parameters):
        """Test comparison analysis with single input."""
        # Run comparison with single input
        result = await comparison_engine.compare(
            input_data=[sample_analysis_results[0]],
            parameters=sample_comparison_parameters
        )
        
        # Verify result structure
        assert isinstance(result, ComparisonResult)
        assert len(result.overlap_statistics) == 0

    @pytest.mark.asyncio
    async def test_compare_different_types(self, comparison_engine, sample_analysis_results):
        """Test comparison analysis with different comparison types."""
        comparison_types = [ComparisonType.OVERLAP, ComparisonType.CORRELATION, ComparisonType.CLUSTERING, ComparisonType.CONCORDANCE]
        
        for comparison_type in comparison_types:
            parameters = ComparisonParameters(
                comparison_name=f"Test {comparison_type.value}",
                comparison_type=comparison_type,
                significance_threshold=0.05,
                min_pathway_size=5,
                max_pathway_size=500
            )
            
            # Run comparison
            result = await comparison_engine.compare(
                input_data=sample_analysis_results,
                parameters=parameters
            )
            
            # Verify result structure
            assert isinstance(result, ComparisonResult)
            assert result.comparison_type == comparison_type

    def test_calculate_overlap_statistics(self, comparison_engine, sample_analysis_results):
        """Test overlap statistics calculation."""
        overlap_stats = comparison_engine._calculate_overlap_statistics(sample_analysis_results)
        
        assert isinstance(overlap_stats, list)
        assert len(overlap_stats) > 0
        
        for stat in overlap_stats:
            assert isinstance(stat, OverlapStatistics)
            assert stat.analysis_1_id is not None
            assert stat.analysis_2_id is not None
            assert 0 <= stat.jaccard_index <= 1
            assert stat.overlap_count >= 0

    def test_calculate_correlation_statistics(self, comparison_engine, sample_analysis_results):
        """Test correlation statistics calculation."""
        correlation_stats = comparison_engine._calculate_correlation_statistics(sample_analysis_results)
        
        assert isinstance(correlation_stats, list)
        assert len(correlation_stats) > 0
        
        for stat in correlation_stats:
            assert isinstance(stat, CorrelationResult)
            assert stat.analysis_1_id is not None
            assert stat.analysis_2_id is not None
            assert -1 <= stat.correlation_coefficient <= 1

    def test_calculate_clustering_statistics(self, comparison_engine, sample_analysis_results):
        """Test clustering statistics calculation."""
        clustering_stats = comparison_engine._calculate_clustering_statistics(sample_analysis_results)
        
        assert isinstance(clustering_stats, list)
        assert len(clustering_stats) > 0
        
        for stat in clustering_stats:
            assert isinstance(stat, ClusteringResult)
            assert stat.analysis_1_id is not None
            assert stat.analysis_2_id is not None
            assert stat.cluster_count > 0

    def test_calculate_concordance_statistics(self, comparison_engine, sample_analysis_results):
        """Test concordance statistics calculation."""
        concordance_stats = comparison_engine._calculate_concordance_statistics(sample_analysis_results)
        
        assert isinstance(concordance_stats, list)
        assert len(concordance_stats) > 0
        
        for stat in concordance_stats:
            assert isinstance(stat, PathwayConcordance)
            assert stat.analysis_1_id is not None
            assert stat.analysis_2_id is not None
            assert 0 <= stat.concordance_score <= 1

    def test_extract_pathway_data(self, comparison_engine, sample_analysis_results):
        """Test pathway data extraction."""
        pathway_data = comparison_engine._extract_pathway_data(sample_analysis_results)
        
        assert isinstance(pathway_data, dict)
        assert len(pathway_data) > 0
        
        for analysis_id, pathways in pathway_data.items():
            assert isinstance(analysis_id, str)
            assert isinstance(pathways, list)
            assert len(pathways) > 0

    def test_extract_pathway_data_empty(self, comparison_engine):
        """Test pathway data extraction with empty input."""
        pathway_data = comparison_engine._extract_pathway_data([])
        
        assert isinstance(pathway_data, dict)
        assert len(pathway_data) == 0

    def test_calculate_jaccard_index(self, comparison_engine):
        """Test Jaccard index calculation."""
        set1 = {"A", "B", "C"}
        set2 = {"B", "C", "D"}
        
        jaccard = comparison_engine._calculate_jaccard_index(set1, set2)
        
        assert isinstance(jaccard, float)
        assert 0 <= jaccard <= 1
        assert jaccard == 0.5  # 2 common elements / 4 total unique elements

    def test_calculate_jaccard_index_edge_cases(self, comparison_engine):
        """Test Jaccard index calculation with edge cases."""
        # Identical sets
        set1 = {"A", "B", "C"}
        set2 = {"A", "B", "C"}
        jaccard = comparison_engine._calculate_jaccard_index(set1, set2)
        assert jaccard == 1.0
        
        # No overlap
        set1 = {"A", "B", "C"}
        set2 = {"D", "E", "F"}
        jaccard = comparison_engine._calculate_jaccard_index(set1, set2)
        assert jaccard == 0.0
        
        # Empty sets
        set1 = set()
        set2 = set()
        jaccard = comparison_engine._calculate_jaccard_index(set1, set2)
        assert jaccard == 0.0

    def test_calculate_correlation_coefficient(self, comparison_engine):
        """Test correlation coefficient calculation."""
        values1 = [1, 2, 3, 4, 5]
        values2 = [2, 4, 6, 8, 10]
        
        correlation = comparison_engine._calculate_correlation_coefficient(values1, values2)
        
        assert isinstance(correlation, float)
        assert -1 <= correlation <= 1
        assert abs(correlation - 1.0) < 0.001  # Should be close to 1.0

    def test_calculate_correlation_coefficient_edge_cases(self, comparison_engine):
        """Test correlation coefficient calculation with edge cases."""
        # Identical values
        values1 = [1, 2, 3, 4, 5]
        values2 = [1, 2, 3, 4, 5]
        correlation = comparison_engine._calculate_correlation_coefficient(values1, values2)
        assert abs(correlation - 1.0) < 0.001
        
        # Negative correlation
        values1 = [1, 2, 3, 4, 5]
        values2 = [5, 4, 3, 2, 1]
        correlation = comparison_engine._calculate_correlation_coefficient(values1, values2)
        assert abs(correlation - (-1.0)) < 0.001
        
        # No correlation
        values1 = [1, 1, 1, 1, 1]
        values2 = [1, 2, 3, 4, 5]
        correlation = comparison_engine._calculate_correlation_coefficient(values1, values2)
        assert correlation == 0.0

    def test_perform_clustering(self, comparison_engine, sample_analysis_results):
        """Test clustering analysis."""
        pathway_data = comparison_engine._extract_pathway_data(sample_analysis_results)
        clusters = comparison_engine._perform_clustering(pathway_data)
        
        assert isinstance(clusters, list)
        assert len(clusters) > 0
        
        for cluster in clusters:
            assert isinstance(cluster, list)
            assert len(cluster) > 0

    def test_perform_clustering_empty_data(self, comparison_engine):
        """Test clustering analysis with empty data."""
        pathway_data = {}
        clusters = comparison_engine._perform_clustering(pathway_data)
        
        assert isinstance(clusters, list)
        assert len(clusters) == 0

    def test_calculate_concordance_score(self, comparison_engine):
        """Test concordance score calculation."""
        pathway1 = {"PATH:00010": 0.001, "PATH:00020": 0.005}
        pathway2 = {"PATH:00010": 0.002, "PATH:00020": 0.006}
        
        concordance = comparison_engine._calculate_concordance_score(pathway1, pathway2)
        
        assert isinstance(concordance, float)
        assert 0 <= concordance <= 1

    def test_calculate_concordance_score_edge_cases(self, comparison_engine):
        """Test concordance score calculation with edge cases."""
        # Identical pathways
        pathway1 = {"PATH:00010": 0.001, "PATH:00020": 0.005}
        pathway2 = {"PATH:00010": 0.001, "PATH:00020": 0.005}
        concordance = comparison_engine._calculate_concordance_score(pathway1, pathway2)
        assert concordance == 1.0
        
        # No overlap
        pathway1 = {"PATH:00010": 0.001}
        pathway2 = {"PATH:00020": 0.005}
        concordance = comparison_engine._calculate_concordance_score(pathway1, pathway2)
        assert concordance == 0.0
        
        # Empty pathways
        pathway1 = {}
        pathway2 = {}
        concordance = comparison_engine._calculate_concordance_score(pathway1, pathway2)
        assert concordance == 0.0

    def test_validate_input_parameters(self, comparison_engine):
        """Test input parameter validation."""
        # Valid parameters
        assert comparison_engine._validate_input_parameters(
            input_data=[Mock()],
            parameters=Mock()
        ) is True
        
        # Invalid input data
        assert comparison_engine._validate_input_parameters(
            input_data=[],
            parameters=Mock()
        ) is False
        
        # Invalid parameters
        assert comparison_engine._validate_input_parameters(
            input_data=[Mock()],
            parameters=None
        ) is False

    def test_validate_comparison_parameters(self, comparison_engine):
        """Test comparison parameters validation."""
        # Valid parameters
        assert comparison_engine._validate_comparison_parameters(Mock()) is True
        
        # Invalid parameters
        assert comparison_engine._validate_comparison_parameters(None) is False

    def test_validate_analysis_results(self, comparison_engine):
        """Test analysis results validation."""
        # Valid analysis results
        assert comparison_engine._validate_analysis_results([Mock()]) is True
        
        # Invalid analysis results
        assert comparison_engine._validate_analysis_results([]) is False
        assert comparison_engine._validate_analysis_results(None) is False

    def test_validate_pathway_data(self, comparison_engine):
        """Test pathway data validation."""
        # Valid pathway data
        assert comparison_engine._validate_pathway_data({"analysis_1": []}) is True
        
        # Invalid pathway data
        assert comparison_engine._validate_pathway_data({}) is False
        assert comparison_engine._validate_pathway_data(None) is False

    def test_validate_statistics(self, comparison_engine):
        """Test statistics validation."""
        # Valid statistics
        assert comparison_engine._validate_statistics([Mock()]) is True
        
        # Invalid statistics
        assert comparison_engine._validate_statistics([]) is False
        assert comparison_engine._validate_statistics(None) is False

    def test_validate_output_directory(self, comparison_engine, tmp_path):
        """Test output directory validation."""
        # Valid output directory
        assert comparison_engine._validate_output_directory(str(tmp_path)) is True
        
        # Invalid output directory
        assert comparison_engine._validate_output_directory("") is False
        assert comparison_engine._validate_output_directory(None) is False

    def test_validate_file_paths(self, comparison_engine, tmp_path):
        """Test file path validation."""
        # Valid file paths
        assert comparison_engine._validate_file_paths([str(tmp_path)]) is True
        
        # Invalid file paths
        assert comparison_engine._validate_file_paths([]) is False
        assert comparison_engine._validate_file_paths(None) is False

    def test_validate_data_types(self, comparison_engine):
        """Test data type validation."""
        # Valid data types
        assert comparison_engine._validate_data_types([Mock()]) is True
        
        # Invalid data types
        assert comparison_engine._validate_data_types([]) is False
        assert comparison_engine._validate_data_types(None) is False

    def test_validate_analysis_ids(self, comparison_engine):
        """Test analysis ID validation."""
        # Valid analysis IDs
        assert comparison_engine._validate_analysis_ids(["analysis_1", "analysis_2"]) is True
        
        # Invalid analysis IDs
        assert comparison_engine._validate_analysis_ids([]) is False
        assert comparison_engine._validate_analysis_ids(None) is False

    def test_validate_pathway_ids(self, comparison_engine):
        """Test pathway ID validation."""
        # Valid pathway IDs
        assert comparison_engine._validate_pathway_ids(["PATH:00010", "PATH:00020"]) is True
        
        # Invalid pathway IDs
        assert comparison_engine._validate_pathway_ids([]) is False
        assert comparison_engine._validate_pathway_ids(None) is False

    def test_validate_p_values(self, comparison_engine):
        """Test p-value validation."""
        # Valid p-values
        assert comparison_engine._validate_p_values([0.01, 0.05, 0.1]) is True
        
        # Invalid p-values
        assert comparison_engine._validate_p_values([]) is False
        assert comparison_engine._validate_p_values([-0.1, 0.5]) is False
        assert comparison_engine._validate_p_values([0.5, 1.5]) is False
        assert comparison_engine._validate_p_values(None) is False
