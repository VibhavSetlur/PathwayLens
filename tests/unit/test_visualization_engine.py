"""
Unit tests for the Visualization engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from pathwaylens_core.visualization.engine import VisualizationEngine
from pathwaylens_core.visualization.schemas import (
    VisualizationResult, VisualizationParameters, PlotType, PlotMetadata,
    DashboardConfig, ExportConfig
)
from pathwaylens_core.analysis.schemas import AnalysisResult, DatabaseResult, PathwayResult, DatabaseType
from pathwaylens_core.comparison.schemas import ComparisonResult, OverlapStatistics


class TestVisualizationEngine:
    """Test cases for the VisualizationEngine class."""

    @pytest.fixture
    def visualization_engine(self):
        """Create a VisualizationEngine instance for testing."""
        return VisualizationEngine()

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
    def sample_analysis_result(self, sample_pathway_results):
        """Create a sample analysis result."""
        return AnalysisResult(
            analysis_id="analysis_1",
            analysis_name="Test Analysis",
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

    @pytest.fixture
    def sample_comparison_result(self):
        """Create a sample comparison result."""
        return ComparisonResult(
            comparison_id="comparison_1",
            comparison_name="Test Comparison",
            comparison_type="OVERLAP",
            parameters=Mock(),
            overlap_statistics=[
                OverlapStatistics(
                    analysis_1_id="analysis_1",
                    analysis_2_id="analysis_2",
                    jaccard_index=0.5,
                    overlap_count=2,
                    total_pathways_1=10,
                    total_pathways_2=8
                )
            ],
            timestamp="2023-01-01T00:00:00"
        )

    @pytest.fixture
    def sample_visualization_parameters(self):
        """Create sample visualization parameters."""
        return VisualizationParameters(
            plot_title="Test Plot",
            plot_type=PlotType.BAR,
            output_format="png",
            width=800,
            height=600
        )

    def test_init(self, visualization_engine):
        """Test VisualizationEngine initialization."""
        assert visualization_engine.logger is not None

    @pytest.mark.asyncio
    async def test_generate_visualizations_basic(self, visualization_engine, sample_analysis_result, sample_visualization_parameters):
        """Test basic visualization generation."""
        # Run visualization generation
        result = await visualization_engine.generate_visualizations(
            input_data=sample_analysis_result,
            parameters=sample_visualization_parameters
        )
        
        # Verify result structure
        assert isinstance(result, VisualizationResult)
        assert result.plot_type == sample_visualization_parameters.plot_type
        assert result.plot_metadata is not None
        assert result.output_path is not None
        assert result.interactive_data is not None

    @pytest.mark.asyncio
    async def test_generate_visualizations_with_output_dir(self, visualization_engine, sample_analysis_result, sample_visualization_parameters, tmp_path):
        """Test visualization generation with output directory."""
        output_dir = tmp_path / "viz_output"
        
        # Run visualization generation
        result = await visualization_engine.generate_visualizations(
            input_data=sample_analysis_result,
            parameters=sample_visualization_parameters,
            output_dir=str(output_dir)
        )
        
        # Verify result structure
        assert isinstance(result, VisualizationResult)
        assert output_dir.exists()

    @pytest.mark.asyncio
    async def test_generate_visualizations_different_plot_types(self, visualization_engine, sample_analysis_result):
        """Test visualization generation with different plot types."""
        plot_types = [PlotType.BAR, PlotType.SCATTER, PlotType.HEATMAP, PlotType.NETWORK, PlotType.VOLCANO]
        
        for plot_type in plot_types:
            parameters = VisualizationParameters(
                plot_title=f"Test {plot_type.value} Plot",
                plot_type=plot_type,
                output_format="png",
                width=800,
                height=600
            )
            
            # Run visualization generation
            result = await visualization_engine.generate_visualizations(
                input_data=sample_analysis_result,
                parameters=parameters
            )
            
            # Verify result structure
            assert isinstance(result, VisualizationResult)
            assert result.plot_type == plot_type

    @pytest.mark.asyncio
    async def test_generate_visualizations_comparison_data(self, visualization_engine, sample_comparison_result, sample_visualization_parameters):
        """Test visualization generation with comparison data."""
        # Run visualization generation
        result = await visualization_engine.generate_visualizations(
            input_data=sample_comparison_result,
            parameters=sample_visualization_parameters
        )
        
        # Verify result structure
        assert isinstance(result, VisualizationResult)
        assert result.plot_metadata is not None

    @pytest.mark.asyncio
    async def test_generate_visualizations_dataframe(self, visualization_engine, sample_visualization_parameters):
        """Test visualization generation with DataFrame."""
        # Create sample DataFrame
        df = pd.DataFrame({
            "pathway": ["PATH:00010", "PATH:00020"],
            "p_value": [0.001, 0.005],
            "gene_count": [2, 2]
        })
        
        # Run visualization generation
        result = await visualization_engine.generate_visualizations(
            input_data=df,
            parameters=sample_visualization_parameters
        )
        
        # Verify result structure
        assert isinstance(result, VisualizationResult)
        assert result.plot_metadata is not None

    @pytest.mark.asyncio
    async def test_generate_visualizations_empty_data(self, visualization_engine, sample_visualization_parameters):
        """Test visualization generation with empty data."""
        # Run visualization generation with empty DataFrame
        result = await visualization_engine.generate_visualizations(
            input_data=pd.DataFrame(),
            parameters=sample_visualization_parameters
        )
        
        # Verify result structure
        assert isinstance(result, VisualizationResult)
        assert result.plot_metadata is not None

    def test_create_bar_plot(self, visualization_engine, sample_analysis_result):
        """Test bar plot creation."""
        plot_data = visualization_engine._create_bar_plot(sample_analysis_result)
        
        assert plot_data is not None
        assert "data" in plot_data
        assert "layout" in plot_data

    def test_create_scatter_plot(self, visualization_engine, sample_analysis_result):
        """Test scatter plot creation."""
        plot_data = visualization_engine._create_scatter_plot(sample_analysis_result)
        
        assert plot_data is not None
        assert "data" in plot_data
        assert "layout" in plot_data

    def test_create_heatmap(self, visualization_engine, sample_analysis_result):
        """Test heatmap creation."""
        plot_data = visualization_engine._create_heatmap(sample_analysis_result)
        
        assert plot_data is not None
        assert "data" in plot_data
        assert "layout" in plot_data

    def test_create_network_plot(self, visualization_engine, sample_analysis_result):
        """Test network plot creation."""
        plot_data = visualization_engine._create_network_plot(sample_analysis_result)
        
        assert plot_data is not None
        assert "data" in plot_data
        assert "layout" in plot_data

    def test_create_volcano_plot(self, visualization_engine, sample_analysis_result):
        """Test volcano plot creation."""
        plot_data = visualization_engine._create_volcano_plot(sample_analysis_result)
        
        assert plot_data is not None
        assert "data" in plot_data
        assert "layout" in plot_data

    def test_create_comparison_plot(self, visualization_engine, sample_comparison_result):
        """Test comparison plot creation."""
        plot_data = visualization_engine._create_comparison_plot(sample_comparison_result)
        
        assert plot_data is not None
        assert "data" in plot_data
        assert "layout" in plot_data

    def test_extract_pathway_data(self, visualization_engine, sample_analysis_result):
        """Test pathway data extraction."""
        pathway_data = visualization_engine._extract_pathway_data(sample_analysis_result)
        
        assert isinstance(pathway_data, pd.DataFrame)
        assert len(pathway_data) > 0
        assert "pathway_id" in pathway_data.columns
        assert "pathway_name" in pathway_data.columns
        assert "p_value" in pathway_data.columns

    def test_extract_pathway_data_empty(self, visualization_engine):
        """Test pathway data extraction with empty data."""
        # Create empty analysis result
        empty_result = AnalysisResult(
            analysis_id="empty",
            analysis_name="Empty Analysis",
            analysis_type="ORA",
            parameters=Mock(),
            database_results=[],
            timestamp="2023-01-01T00:00:00"
        )
        
        pathway_data = visualization_engine._extract_pathway_data(empty_result)
        
        assert isinstance(pathway_data, pd.DataFrame)
        assert len(pathway_data) == 0

    def test_extract_comparison_data(self, visualization_engine, sample_comparison_result):
        """Test comparison data extraction."""
        comparison_data = visualization_engine._extract_comparison_data(sample_comparison_result)
        
        assert isinstance(comparison_data, pd.DataFrame)
        assert len(comparison_data) > 0
        assert "analysis_1_id" in comparison_data.columns
        assert "analysis_2_id" in comparison_data.columns
        assert "jaccard_index" in comparison_data.columns

    def test_extract_comparison_data_empty(self, visualization_engine):
        """Test comparison data extraction with empty data."""
        # Create empty comparison result
        empty_result = ComparisonResult(
            comparison_id="empty",
            comparison_name="Empty Comparison",
            comparison_type="OVERLAP",
            parameters=Mock(),
            overlap_statistics=[],
            timestamp="2023-01-01T00:00:00"
        )
        
        comparison_data = visualization_engine._extract_comparison_data(empty_result)
        
        assert isinstance(comparison_data, pd.DataFrame)
        assert len(comparison_data) == 0

    def test_create_plot_metadata(self, visualization_engine, sample_visualization_parameters):
        """Test plot metadata creation."""
        metadata = visualization_engine._create_plot_metadata(sample_visualization_parameters)
        
        assert isinstance(metadata, PlotMetadata)
        assert metadata.title == sample_visualization_parameters.plot_title
        assert metadata.plot_type == sample_visualization_parameters.plot_type
        assert metadata.width == sample_visualization_parameters.width
        assert metadata.height == sample_visualization_parameters.height

    def test_save_plot(self, visualization_engine, tmp_path):
        """Test plot saving."""
        # Create sample plot data
        plot_data = {
            "data": [{"x": [1, 2, 3], "y": [1, 4, 9], "type": "scatter"}],
            "layout": {"title": "Test Plot"}
        }
        
        output_path = tmp_path / "test_plot.png"
        
        # Save plot
        saved_path = visualization_engine._save_plot(plot_data, str(output_path), "png")
        
        assert saved_path == str(output_path)
        assert output_path.exists()

    def test_save_plot_json(self, visualization_engine, tmp_path):
        """Test plot saving as JSON."""
        # Create sample plot data
        plot_data = {
            "data": [{"x": [1, 2, 3], "y": [1, 4, 9], "type": "scatter"}],
            "layout": {"title": "Test Plot"}
        }
        
        output_path = tmp_path / "test_plot.json"
        
        # Save plot
        saved_path = visualization_engine._save_plot(plot_data, str(output_path), "json")
        
        assert saved_path == str(output_path)
        assert output_path.exists()

    def test_validate_input_parameters(self, visualization_engine):
        """Test input parameter validation."""
        # Valid parameters
        assert visualization_engine._validate_input_parameters(
            input_data=Mock(),
            parameters=Mock()
        ) is True
        
        # Invalid input data
        assert visualization_engine._validate_input_parameters(
            input_data=None,
            parameters=Mock()
        ) is False
        
        # Invalid parameters
        assert visualization_engine._validate_input_parameters(
            input_data=Mock(),
            parameters=None
        ) is False

    def test_validate_visualization_parameters(self, visualization_engine):
        """Test visualization parameters validation."""
        # Valid parameters
        assert visualization_engine._validate_visualization_parameters(Mock()) is True
        
        # Invalid parameters
        assert visualization_engine._validate_visualization_parameters(None) is False

    def test_validate_plot_type(self, visualization_engine):
        """Test plot type validation."""
        # Valid plot types
        assert visualization_engine._validate_plot_type(PlotType.BAR) is True
        assert visualization_engine._validate_plot_type(PlotType.SCATTER) is True
        assert visualization_engine._validate_plot_type(PlotType.HEATMAP) is True
        assert visualization_engine._validate_plot_type(PlotType.NETWORK) is True
        assert visualization_engine._validate_plot_type(PlotType.VOLCANO) is True
        
        # Invalid plot type
        assert visualization_engine._validate_plot_type(None) is False

    def test_validate_output_format(self, visualization_engine):
        """Test output format validation."""
        # Valid output formats
        assert visualization_engine._validate_output_format("png") is True
        assert visualization_engine._validate_output_format("svg") is True
        assert visualization_engine._validate_output_format("pdf") is True
        assert visualization_engine._validate_output_format("html") is True
        assert visualization_engine._validate_output_format("json") is True
        
        # Invalid output format
        assert visualization_engine._validate_output_format("invalid") is False
        assert visualization_engine._validate_output_format("") is False
        assert visualization_engine._validate_output_format(None) is False

    def test_validate_dimensions(self, visualization_engine):
        """Test dimension validation."""
        # Valid dimensions
        assert visualization_engine._validate_dimensions(800, 600) is True
        assert visualization_engine._validate_dimensions(1920, 1080) is True
        
        # Invalid dimensions
        assert visualization_engine._validate_dimensions(0, 600) is False
        assert visualization_engine._validate_dimensions(800, 0) is False
        assert visualization_engine._validate_dimensions(-100, 600) is False
        assert visualization_engine._validate_dimensions(800, -100) is False

    def test_validate_output_directory(self, visualization_engine, tmp_path):
        """Test output directory validation."""
        # Valid output directory
        assert visualization_engine._validate_output_directory(str(tmp_path)) is True
        
        # Invalid output directory
        assert visualization_engine._validate_output_directory("") is False
        assert visualization_engine._validate_output_directory(None) is False

    def test_validate_plot_data(self, visualization_engine):
        """Test plot data validation."""
        # Valid plot data
        plot_data = {
            "data": [{"x": [1, 2, 3], "y": [1, 4, 9], "type": "scatter"}],
            "layout": {"title": "Test Plot"}
        }
        assert visualization_engine._validate_plot_data(plot_data) is True
        
        # Invalid plot data
        assert visualization_engine._validate_plot_data({}) is False
        assert visualization_engine._validate_plot_data(None) is False

    def test_validate_dataframe(self, visualization_engine):
        """Test DataFrame validation."""
        # Valid DataFrame
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        assert visualization_engine._validate_dataframe(df) is True
        
        # Invalid DataFrame
        assert visualization_engine._validate_dataframe(pd.DataFrame()) is False
        assert visualization_engine._validate_dataframe(None) is False

    def test_validate_analysis_result(self, visualization_engine):
        """Test analysis result validation."""
        # Valid analysis result
        assert visualization_engine._validate_analysis_result(Mock()) is True
        
        # Invalid analysis result
        assert visualization_engine._validate_analysis_result(None) is False

    def test_validate_comparison_result(self, visualization_engine):
        """Test comparison result validation."""
        # Valid comparison result
        assert visualization_engine._validate_comparison_result(Mock()) is True
        
        # Invalid comparison result
        assert visualization_engine._validate_comparison_result(None) is False

    def test_validate_pathway_data(self, visualization_engine):
        """Test pathway data validation."""
        # Valid pathway data
        df = pd.DataFrame({
            "pathway_id": ["PATH:00010"],
            "pathway_name": ["Glycolysis"],
            "p_value": [0.001]
        })
        assert visualization_engine._validate_pathway_data(df) is True
        
        # Invalid pathway data
        assert visualization_engine._validate_pathway_data(pd.DataFrame()) is False
        assert visualization_engine._validate_pathway_data(None) is False

    def test_validate_comparison_data(self, visualization_engine):
        """Test comparison data validation."""
        # Valid comparison data
        df = pd.DataFrame({
            "analysis_1_id": ["analysis_1"],
            "analysis_2_id": ["analysis_2"],
            "jaccard_index": [0.5]
        })
        assert visualization_engine._validate_comparison_data(df) is True
        
        # Invalid comparison data
        assert visualization_engine._validate_comparison_data(pd.DataFrame()) is False
        assert visualization_engine._validate_comparison_data(None) is False
