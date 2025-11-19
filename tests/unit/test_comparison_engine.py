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
from pathwaylens_core.analysis.schemas import (
    AnalysisResult, DatabaseResult, PathwayResult, DatabaseType,
    AnalysisParameters, AnalysisType, CorrectionMethod, ConsensusMethod
)


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
                pathway_id="00010",
                pathway_name="Glycolysis",
                database=DatabaseType.KEGG,
                p_value=0.001,
                adjusted_p_value=0.01,
                enrichment_score=2.5,
                overlap_count=2,
                pathway_count=5,
                input_count=100,
                overlapping_genes=["GENE1", "GENE2"],
                analysis_method="ORA"
            ),
            PathwayResult(
                pathway_id="00020",
                pathway_name="TCA Cycle",
                database=DatabaseType.KEGG,
                p_value=0.005,
                adjusted_p_value=0.02,
                enrichment_score=1.8,
                overlap_count=2,
                pathway_count=4,
                input_count=100,
                overlapping_genes=["GENE3", "GENE4"],
                analysis_method="ORA"
            )
        ]

    @pytest.fixture
    def sample_analysis_parameters(self):
        """Create sample analysis parameters."""
        return AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05,
            correction_method=CorrectionMethod.FDR_BH,
            min_pathway_size=5,
            max_pathway_size=500
        )

    @pytest.fixture
    def sample_analysis_results(self, sample_pathway_results, sample_analysis_parameters):
        """Create sample analysis results."""
        return [
            AnalysisResult(
                job_id="analysis_1",
                analysis_type=AnalysisType.ORA,
                parameters=sample_analysis_parameters,
                input_file="/path/to/file1.txt",
                input_gene_count=100,
                input_species="human",
                database_results={
                    "KEGG": DatabaseResult(
                        database=DatabaseType.KEGG,
                        total_pathways=100,
                        significant_pathways=2,
                        pathways=sample_pathway_results,
                        species="human",
                        coverage=0.5
                    )
                },
                total_pathways=100,
                significant_pathways=2,
                significant_databases=1,
                overall_quality=0.9,
                reproducibility=0.8,
                created_at="2023-01-01T00:00:00",
                completed_at="2023-01-01T00:01:00",
                processing_time=60.0
            ),
            AnalysisResult(
                job_id="analysis_2",
                analysis_type=AnalysisType.ORA,
                parameters=sample_analysis_parameters,
                input_file="/path/to/file2.txt",
                input_gene_count=100,
                input_species="human",
                database_results={
                    "KEGG": DatabaseResult(
                        database=DatabaseType.KEGG,
                        total_pathways=100,
                        significant_pathways=2,
                        pathways=sample_pathway_results,
                        species="human",
                        coverage=0.5
                    )
                },
                total_pathways=100,
                significant_pathways=2,
                significant_databases=1,
                overall_quality=0.9,
                reproducibility=0.8,
                created_at="2023-01-01T00:00:00",
                completed_at="2023-01-01T00:01:00",
                processing_time=60.0
            )
        ]

    @pytest.fixture
    def sample_comparison_parameters(self):
        """Create sample comparison parameters."""
        return ComparisonParameters(
            comparison_name="Test Comparison",
            comparison_type=ComparisonType.GENE_OVERLAP,
            species="human",
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
        # Mock internal methods to avoid complex calculations
        with patch.object(comparison_engine, '_perform_gene_overlap_analysis') as mock_overlap:
            mock_overlap.return_value = {'overlap_statistics': {}}
            
            # Run comparison
            result = await comparison_engine.compare(
                analysis_results=sample_analysis_results,
                parameters=sample_comparison_parameters
            )
            
            # Verify result structure
            assert isinstance(result, ComparisonResult)
            assert result.comparison_type == sample_comparison_parameters.comparison_type
            assert result.parameters == sample_comparison_parameters

    @pytest.mark.asyncio
    async def test_compare_insufficient_input(self, comparison_engine, sample_comparison_parameters, sample_analysis_results):
        """Test comparison analysis with insufficient input."""
        with pytest.raises(ValueError, match="At least 2 analysis results are required"):
            await comparison_engine.compare(
                analysis_results=[sample_analysis_results[0]],
                parameters=sample_comparison_parameters
            )

    @pytest.mark.asyncio
    async def test_compare_different_types(self, comparison_engine, sample_analysis_results):
        """Test comparison analysis with different comparison types."""
        comparison_types = [
            ComparisonType.GENE_OVERLAP,
            ComparisonType.PATHWAY_OVERLAP,
            ComparisonType.PATHWAY_CONCORDANCE,
            ComparisonType.ENRICHMENT_CORRELATION,
            ComparisonType.DATASET_CLUSTERING
        ]
        
        for comparison_type in comparison_types:
            parameters = ComparisonParameters(
                comparison_name=f"Test {comparison_type.value}",
                comparison_type=comparison_type,
                species="human",
                significance_threshold=0.05,
                min_pathway_size=5,
                max_pathway_size=500
            )
            
            # Mock the specific analysis method
            method_name = f"_perform_{comparison_type.value}_analysis"
            if comparison_type == ComparisonType.GENE_OVERLAP:
                method_name = "_perform_gene_overlap_analysis"
            elif comparison_type == ComparisonType.PATHWAY_OVERLAP:
                method_name = "_perform_pathway_overlap_analysis"
            elif comparison_type == ComparisonType.PATHWAY_CONCORDANCE:
                method_name = "_perform_pathway_concordance_analysis"
            elif comparison_type == ComparisonType.ENRICHMENT_CORRELATION:
                method_name = "_perform_enrichment_correlation_analysis"
            elif comparison_type == ComparisonType.DATASET_CLUSTERING:
                method_name = "_perform_dataset_clustering_analysis"
                
            with patch.object(comparison_engine, method_name) as mock_method:
                mock_method.return_value = {}
                
                # Run comparison
                result = await comparison_engine.compare(
                    analysis_results=sample_analysis_results,
                    parameters=parameters
                )
                
                assert isinstance(result, ComparisonResult)
                assert result.comparison_type == comparison_type

    def test_extract_comparison_data(self, comparison_engine, sample_analysis_results, sample_comparison_parameters):
        """Test data extraction for comparison."""
        data = comparison_engine._extract_comparison_data(
            sample_analysis_results,
            sample_comparison_parameters
        )
        
        assert 'datasets' in data
        assert 'all_genes' in data
        assert 'all_pathways' in data
        assert 'pathway_data' in data
        assert len(data['datasets']) == 2

    def test_calculate_gene_overlap(self, comparison_engine):
        """Test gene overlap calculation."""
        genes1 = {"GENE1", "GENE2", "GENE3"}
        genes2 = {"GENE2", "GENE3", "GENE4"}
        
        stats = comparison_engine._calculate_gene_overlap(
            "dataset1", genes1, "dataset2", genes2
        )
        
        assert isinstance(stats, OverlapStatistics)
        assert stats.overlapping_genes == 2
        assert stats.jaccard_index == 0.5  # 2/4
        assert len(stats.overlapping_gene_list) == 2

    def test_calculate_concordance_score(self, comparison_engine, sample_comparison_parameters):
        """Test concordance score calculation."""
        p_values = {"d1": 0.01, "d2": 0.04}  # Both significant
        effect_sizes = {"d1": 2.0, "d2": 1.5}  # Both positive
        
        score = comparison_engine._calculate_concordance_score(
            p_values, effect_sizes, sample_comparison_parameters
        )
        
        assert score == 1.0  # 100% significance concordance + 100% direction concordance

    def test_calculate_enrichment_correlation(self, comparison_engine, sample_comparison_parameters):
        """Test enrichment correlation calculation."""
        # Mock data
        dataset1_info = {
            'database_results': {
                'KEGG': Mock(pathways=[
                    Mock(pathway_id="001", enrichment_score=1.0, database=DatabaseType.KEGG),
                    Mock(pathway_id="002", enrichment_score=2.0, database=DatabaseType.KEGG)
                ], database=DatabaseType.KEGG)
            }
        }
        dataset2_info = {
            'database_results': {
                'KEGG': Mock(pathways=[
                    Mock(pathway_id="001", enrichment_score=1.1, database=DatabaseType.KEGG),
                    Mock(pathway_id="002", enrichment_score=2.1, database=DatabaseType.KEGG)
                ], database=DatabaseType.KEGG)
            }
        }
        
        result = comparison_engine._calculate_enrichment_correlation(
            "d1", dataset1_info, "d2", dataset2_info, sample_comparison_parameters
        )
        
        assert isinstance(result, CorrelationResult)
        assert result.correlation > 0.9  # Should be high correlation
