"""
Unit tests for the analysis engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.analysis.schemas import (
    AnalysisParameters, AnalysisType, DatabaseType, CorrectionMethod,
    AnalysisResult, DatabaseResult, PathwayResult
)
from pathwaylens_core.data import DatabaseManager


class TestAnalysisEngine:
    """Test cases for the AnalysisEngine class."""

    @pytest.fixture
    def analysis_engine(self):
        """Create an AnalysisEngine instance for testing."""
        db_manager = Mock(spec=DatabaseManager)
        return AnalysisEngine(database_manager=db_manager)

    @pytest.fixture
    def sample_parameters(self):
        """Create sample analysis parameters."""
        return AnalysisParameters(
            analysis_name="Test Analysis",
            analysis_type=AnalysisType.ORA,
            database_type=DatabaseType.KEGG,
            species="human",
            significance_threshold=0.05,
            correction_method=CorrectionMethod.FDR_BH,
            min_pathway_size=5,
            max_pathway_size=500
        )

    @pytest.fixture
    def sample_gene_list(self):
        """Create a sample gene list."""
        return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

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
                pathway_size=10
            ),
            PathwayResult(
                pathway_id="PATH:00020",
                pathway_name="TCA Cycle",
                p_value=0.005,
                adjusted_p_value=0.02,
                gene_overlap=["GENE3", "GENE4"],
                gene_overlap_count=2,
                pathway_size=8
            )
        ]

    @pytest.fixture
    def sample_database_result(self, sample_pathway_results):
        """Create a sample database result."""
        return DatabaseResult(
            database_name="KEGG",
            database_type=DatabaseType.KEGG,
            species="human",
            pathway_results=sample_pathway_results
        )

    @pytest.fixture
    def sample_analysis_result(self, sample_parameters, sample_database_result):
        """Create a sample analysis result."""
        return AnalysisResult(
            analysis_id="test_analysis_123",
            analysis_name=sample_parameters.analysis_name,
            analysis_type=sample_parameters.analysis_type,
            parameters=sample_parameters,
            database_results=[sample_database_result],
            timestamp="2024-01-01T00:00:00Z"
        )

    def test_init(self, analysis_engine):
        """Test AnalysisEngine initialization."""
        assert analysis_engine.database_manager is not None
        assert analysis_engine.ora_engine is not None
        assert analysis_engine.gsea_engine is not None
        assert analysis_engine.consensus_engine is not None

    @pytest.mark.asyncio
    async def test_analyze_ora(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test ORA analysis."""
        # Mock the ORA engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.ora_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters
        )
        
        # Verify ORA engine was called
        analysis_engine.ora_engine.analyze.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.analysis_name == sample_parameters.analysis_name
        assert result.analysis_type == sample_parameters.analysis_type

    @pytest.mark.asyncio
    async def test_analyze_gsea(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test GSEA analysis."""
        # Set analysis type to GSEA
        sample_parameters.analysis_type = AnalysisType.GSEA
        
        # Mock the GSEA engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.gsea_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters
        )
        
        # Verify GSEA engine was called
        analysis_engine.gsea_engine.analyze.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.GSEA

    @pytest.mark.asyncio
    async def test_analyze_consensus(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test consensus analysis."""
        # Set analysis type to consensus
        sample_parameters.analysis_type = AnalysisType.CONSENSUS
        
        # Mock the consensus engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.consensus_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters
        )
        
        # Verify consensus engine was called
        analysis_engine.consensus_engine.analyze.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.analysis_type == AnalysisType.CONSENSUS

    @pytest.mark.asyncio
    async def test_analyze_with_dataframe(self, analysis_engine, sample_parameters):
        """Test analysis with DataFrame input."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'gene_id': ['GENE1', 'GENE2', 'GENE3'],
            'expression': [1.5, 2.0, 0.8]
        })
        
        # Mock the ORA engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.ora_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=df,
            parameters=sample_parameters
        )
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)

    @pytest.mark.asyncio
    async def test_analyze_with_file_path(self, analysis_engine, sample_parameters, tmp_path):
        """Test analysis with file path input."""
        # Create sample file
        gene_file = tmp_path / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        # Mock the ORA engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.ora_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=str(gene_file),
            parameters=sample_parameters
        )
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)

    @pytest.mark.asyncio
    async def test_analyze_with_output_dir(self, analysis_engine, sample_parameters, sample_gene_list, tmp_path):
        """Test analysis with output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Mock the ORA engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.ora_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters,
            output_dir=str(output_dir)
        )
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        
        # Verify output files were created
        assert (output_dir / "analysis_result.json").exists()
        assert (output_dir / "analysis_summary.txt").exists()

    @pytest.mark.asyncio
    async def test_analyze_invalid_input(self, analysis_engine, sample_parameters):
        """Test analysis with invalid input."""
        with pytest.raises(ValueError, match="Invalid input data type"):
            await analysis_engine.analyze(
                input_data=123,  # Invalid input type
                parameters=sample_parameters
            )

    @pytest.mark.asyncio
    async def test_analyze_invalid_parameters(self, analysis_engine, sample_gene_list):
        """Test analysis with invalid parameters."""
        with pytest.raises(ValueError, match="Invalid analysis parameters"):
            await analysis_engine.analyze(
                input_data=sample_gene_list,
                parameters=None
            )

    @pytest.mark.asyncio
    async def test_analyze_engine_error(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test analysis when engine raises an error."""
        # Mock the ORA engine to raise an error
        analysis_engine.ora_engine.analyze = AsyncMock(side_effect=Exception("Engine error"))
        
        with pytest.raises(Exception, match="Engine error"):
            await analysis_engine.analyze(
                input_data=sample_gene_list,
                parameters=sample_parameters
            )

    def test_validate_input_data(self, analysis_engine):
        """Test input data validation."""
        # Valid inputs
        assert analysis_engine._validate_input_data(["GENE1", "GENE2"]) is True
        assert analysis_engine._validate_input_data(pd.DataFrame({"gene": ["GENE1"]})) is True
        assert analysis_engine._validate_input_data("path/to/file.txt") is True
        
        # Invalid inputs
        assert analysis_engine._validate_input_data(123) is False
        assert analysis_engine._validate_input_data(None) is False
        assert analysis_engine._validate_input_data([]) is False

    def test_validate_parameters(self, analysis_engine, sample_parameters):
        """Test parameters validation."""
        # Valid parameters
        assert analysis_engine._validate_parameters(sample_parameters) is True
        
        # Invalid parameters
        assert analysis_engine._validate_parameters(None) is False
        assert analysis_engine._validate_parameters("invalid") is False

    def test_prepare_input_data_list(self, analysis_engine, sample_gene_list):
        """Test preparing input data from list."""
        result = analysis_engine._prepare_input_data(sample_gene_list)
        assert result == sample_gene_list

    def test_prepare_input_data_dataframe(self, analysis_engine):
        """Test preparing input data from DataFrame."""
        df = pd.DataFrame({
            'gene_id': ['GENE1', 'GENE2'],
            'expression': [1.5, 2.0]
        })
        
        result = analysis_engine._prepare_input_data(df)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_prepare_input_data_file(self, analysis_engine, tmp_path):
        """Test preparing input data from file."""
        gene_file = tmp_path / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        result = analysis_engine._prepare_input_data(str(gene_file))
        assert isinstance(result, list)
        assert len(result) == 3
        assert "GENE1" in result

    def test_calculate_summary_statistics(self, analysis_engine, sample_analysis_result):
        """Test summary statistics calculation."""
        summary = analysis_engine._calculate_summary_statistics(sample_analysis_result)
        
        assert "total_pathways" in summary
        assert "significant_pathways" in summary
        assert "analysis_duration" in summary
        assert summary["total_pathways"] == 2
        assert summary["significant_pathways"] == 2

    def test_generate_output_files(self, analysis_engine, sample_analysis_result, tmp_path):
        """Test output file generation."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        analysis_engine._generate_output_files(sample_analysis_result, str(output_dir))
        
        # Verify files were created
        assert (output_dir / "analysis_result.json").exists()
        assert (output_dir / "analysis_summary.txt").exists()
        assert (output_dir / "pathway_results.csv").exists()

    def test_generate_output_files_no_dir(self, analysis_engine, sample_analysis_result):
        """Test output file generation without output directory."""
        # Should not raise an error
        analysis_engine._generate_output_files(sample_analysis_result, None)

    @pytest.mark.asyncio
    async def test_analyze_multiple_databases(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test analysis with multiple databases."""
        # Set multiple databases
        sample_parameters.database_type = [DatabaseType.KEGG, DatabaseType.REACTOME]
        
        # Mock the ORA engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.ora_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters
        )
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert len(result.database_results) == 2

    @pytest.mark.asyncio
    async def test_analyze_with_custom_parameters(self, analysis_engine, sample_gene_list):
        """Test analysis with custom parameters."""
        # Create custom parameters
        custom_params = AnalysisParameters(
            analysis_name="Custom Analysis",
            analysis_type=AnalysisType.ORA,
            database_type=DatabaseType.KEGG,
            species="mouse",
            significance_threshold=0.01,
            correction_method=CorrectionMethod.BONFERRONI,
            min_pathway_size=10,
            max_pathway_size=1000,
            custom_parameters={"custom_param": "custom_value"}
        )
        
        # Mock the ORA engine
        mock_result = Mock(spec=DatabaseResult)
        analysis_engine.ora_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=custom_params
        )
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.analysis_name == "Custom Analysis"
        assert result.parameters.species == "mouse"
        assert result.parameters.significance_threshold == 0.01
