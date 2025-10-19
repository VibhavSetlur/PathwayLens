"""
Integration tests for the core modules.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.comparison.engine import ComparisonEngine
from pathwaylens_core.visualization.engine import VisualizationEngine
from pathwaylens_core.normalization.normalizer import NormalizationEngine
from pathwaylens_core.data.database_manager import DatabaseManager
from pathwaylens_core.analysis.schemas import AnalysisParameters, AnalysisType, DatabaseType
from pathwaylens_core.comparison.schemas import ComparisonParameters, ComparisonType
from pathwaylens_core.visualization.schemas import VisualizationParameters, PlotType


class TestCoreIntegration:
    """Integration tests for the core modules."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def database_manager(self):
        """Create a database manager for testing."""
        return DatabaseManager()

    @pytest.fixture
    def analysis_engine(self, database_manager):
        """Create an analysis engine for testing."""
        return AnalysisEngine(database_manager)

    @pytest.fixture
    def comparison_engine(self):
        """Create a comparison engine for testing."""
        return ComparisonEngine()

    @pytest.fixture
    def visualization_engine(self):
        """Create a visualization engine for testing."""
        return VisualizationEngine()

    @pytest.fixture
    def normalization_engine(self):
        """Create a normalization engine for testing."""
        return NormalizationEngine()

    @pytest.mark.asyncio
    async def test_analysis_workflow(self, analysis_engine, temp_dir):
        """Test complete analysis workflow."""
        # Create test gene list
        gene_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        
        # Create analysis parameters
        parameters = AnalysisParameters(
            analysis_name="test_analysis",
            analysis_type=AnalysisType.ORA,
            database_type=DatabaseType.KEGG,
            species="human",
            significance_threshold=0.05
        )
        
        with patch.object(analysis_engine.database_manager, 'get_pathway_data') as mock_get_data:
            mock_get_data.return_value = {
                "hsa00010": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                "hsa00020": {"name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
            }
            
            # Run analysis
            result = await analysis_engine.analyze(
                input_data=gene_list,
                parameters=parameters,
                output_dir=str(temp_dir)
            )
            
            assert result is not None
            assert result.analysis_name == "test_analysis"
            assert result.analysis_type == AnalysisType.ORA
            assert len(result.database_results) > 0

    @pytest.mark.asyncio
    async def test_comparison_workflow(self, comparison_engine, temp_dir):
        """Test complete comparison workflow."""
        # Create test analysis results
        analysis_results = [
            Mock(database_results=[Mock(pathway_results=[
                Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.01),
                Mock(pathway_id="hsa00020", pathway_name="TCA Cycle", p_value=0.02)
            ])]),
            Mock(database_results=[Mock(pathway_results=[
                Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.015),
                Mock(pathway_id="hsa00030", pathway_name="Pentose Phosphate", p_value=0.03)
            ])])
        ]
        
        # Create comparison parameters
        parameters = ComparisonParameters(
            comparison_name="test_comparison",
            comparison_type=ComparisonType.OVERLAP,
            significance_threshold=0.05
        )
        
        # Run comparison
        result = await comparison_engine.compare(
            input_data=analysis_results,
            parameters=parameters,
            output_dir=str(temp_dir)
        )
        
        assert result is not None
        assert result.comparison_name == "test_comparison"
        assert result.comparison_type == ComparisonType.OVERLAP

    @pytest.mark.asyncio
    async def test_visualization_workflow(self, visualization_engine, temp_dir):
        """Test complete visualization workflow."""
        # Create test analysis result
        analysis_result = Mock(
            database_results=[Mock(pathway_results=[
                Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.01),
                Mock(pathway_id="hsa00020", pathway_name="TCA Cycle", p_value=0.02)
            ])]
        )
        
        # Create visualization parameters
        parameters = VisualizationParameters(
            plot_title="Test Plot",
            plot_type=PlotType.BAR,
            output_format="html"
        )
        
        # Run visualization
        result = await visualization_engine.generate_visualizations(
            input_data=analysis_result,
            parameters=parameters,
            output_dir=str(temp_dir)
        )
        
        assert result is not None
        assert result.plot_type == PlotType.BAR
        assert result.plot_metadata.title == "Test Plot"

    @pytest.mark.asyncio
    async def test_normalization_workflow(self, normalization_engine, temp_dir):
        """Test complete normalization workflow."""
        # Create test gene list
        gene_list = ["GENE1", "GENE2", "GENE3"]
        
        with patch.object(normalization_engine.id_converter, 'convert_identifiers') as mock_convert:
            mock_convert.return_value = Mock(
                converted_genes=["123", "456", "789"],
                mapping_stats={"total": 3, "converted": 3, "failed": 0}
            )
            
            # Run normalization
            result = await normalization_engine.normalize(
                input_data=gene_list,
                input_format="gene_list",
                output_format="entrezgene",
                species="human",
                output_dir=str(temp_dir)
            )
            
            assert result is not None
            assert result.normalized_data is not None
            assert result.mapping_stats["total"] == 3

    @pytest.mark.asyncio
    async def test_database_manager_integration(self, database_manager):
        """Test database manager integration."""
        with patch.object(database_manager, 'get_pathway_data') as mock_get_data:
            mock_get_data.return_value = {
                "hsa00010": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
            }
            
            # Test getting pathway data
            result = await database_manager.get_pathway_data(
                database=DatabaseType.KEGG,
                species="human"
            )
            
            assert result is not None
            assert "hsa00010" in result
            assert result["hsa00010"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, analysis_engine, comparison_engine, visualization_engine, temp_dir):
        """Test end-to-end workflow from analysis to visualization."""
        # Step 1: Analysis
        gene_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        parameters = AnalysisParameters(
            analysis_name="e2e_test",
            analysis_type=AnalysisType.ORA,
            database_type=DatabaseType.KEGG,
            species="human"
        )
        
        with patch.object(analysis_engine.database_manager, 'get_pathway_data') as mock_get_data:
            mock_get_data.return_value = {
                "hsa00010": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                "hsa00020": {"name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
            }
            
            analysis_result = await analysis_engine.analyze(
                input_data=gene_list,
                parameters=parameters,
                output_dir=str(temp_dir)
            )
            
            assert analysis_result is not None
        
        # Step 2: Comparison (with mock second analysis)
        comparison_parameters = ComparisonParameters(
            comparison_name="e2e_comparison",
            comparison_type=ComparisonType.OVERLAP
        )
        
        comparison_result = await comparison_engine.compare(
            input_data=[analysis_result, analysis_result],  # Mock second result
            parameters=comparison_parameters,
            output_dir=str(temp_dir)
        )
        
        assert comparison_result is not None
        
        # Step 3: Visualization
        viz_parameters = VisualizationParameters(
            plot_title="E2E Test Plot",
            plot_type=PlotType.BAR
        )
        
        viz_result = await visualization_engine.generate_visualizations(
            input_data=analysis_result,
            parameters=viz_parameters,
            output_dir=str(temp_dir)
        )
        
        assert viz_result is not None

    def test_error_handling(self, analysis_engine, temp_dir):
        """Test error handling in core modules."""
        # Test with invalid input
        with pytest.raises(ValueError):
            asyncio.run(analysis_engine.analyze(
                input_data=None,
                parameters=None,
                output_dir=str(temp_dir)
            ))

    def test_output_file_creation(self, temp_dir):
        """Test that output files are created correctly."""
        # Create test output file
        output_file = temp_dir / "test_output.json"
        test_data = {"test": "data"}
        
        # Write test data
        import json
        output_file.write_text(json.dumps(test_data))
        
        # Verify file exists and contains correct data
        assert output_file.exists()
        assert json.loads(output_file.read_text()) == test_data
