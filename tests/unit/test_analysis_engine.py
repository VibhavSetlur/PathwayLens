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
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
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
                database=DatabaseType.KEGG,
                p_value=0.001,
                adjusted_p_value=0.01,
                overlap_count=2,
                pathway_count=10,
                input_count=5,
                overlapping_genes=["GENE1", "GENE2"],
                analysis_method="ORA"
            ),
            PathwayResult(
                pathway_id="PATH:00020",
                pathway_name="TCA Cycle",
                database=DatabaseType.KEGG,
                p_value=0.005,
                adjusted_p_value=0.02,
                overlap_count=2,
                pathway_count=8,
                input_count=5,
                overlapping_genes=["GENE3", "GENE4"],
                analysis_method="ORA"
            )
        ]

    @pytest.fixture
    def sample_database_result(self, sample_pathway_results):
        """Create a sample database result."""
        return DatabaseResult(
            database=DatabaseType.KEGG,
            total_pathways=2,
            significant_pathways=2,
            pathways=sample_pathway_results,
            species="human",
            coverage=0.8
        )

    @pytest.fixture
    def sample_analysis_result(self, sample_parameters, sample_database_result):
        """Create a sample analysis result."""
        return AnalysisResult(
            job_id="test_analysis_123",
            analysis_type=sample_parameters.analysis_type,
            parameters=sample_parameters,
            input_file="test_input.txt",
            input_gene_count=5,
            database_results={"KEGG": sample_database_result}
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
        """Test consensus analysis with multiple databases."""
        # Set multiple databases to trigger consensus analysis
        sample_parameters.databases = [DatabaseType.KEGG, DatabaseType.REACTOME]
        
        # Mock the ORA engine to return proper DatabaseResult objects
        from pathwaylens_core.analysis.schemas import DatabaseResult
        mock_kegg_result = DatabaseResult(
            database=DatabaseType.KEGG,
            total_pathways=10,
            significant_pathways=2,
            pathways=[],
            species="human",
            coverage=0.8
        )
        mock_reactome_result = DatabaseResult(
            database=DatabaseType.REACTOME,
            total_pathways=10,
            significant_pathways=2,
            pathways=[],
            species="human",
            coverage=0.8
        )
        
        async def mock_analyze(*args, **kwargs):
            database = kwargs.get('database')
            if database == DatabaseType.KEGG:
                return mock_kegg_result
            return mock_reactome_result
        
        analysis_engine.ora_engine.analyze = AsyncMock(side_effect=mock_analyze)
        
        # Mock consensus engine
        from pathwaylens_core.analysis.schemas import ConsensusResult, ConsensusMethod
        mock_consensus = ConsensusResult(
            consensus_method=ConsensusMethod.STOUFFER,
            total_pathways=10,
            significant_pathways=2,
            pathways=[],
            database_agreement={},
            consensus_score=0.8,
            reproducibility=0.9,
            stability=0.85
        )
        analysis_engine.consensus_engine.analyze = AsyncMock(return_value=mock_consensus)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters
        )
        
        # Verify consensus engine was called (when multiple databases)
        analysis_engine.consensus_engine.analyze.assert_called_once()
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.consensus_results is not None

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
        
        # Mock the ORA engine to return proper DatabaseResult
        from pathwaylens_core.analysis.schemas import DatabaseResult
        mock_result = DatabaseResult(
            database=DatabaseType.KEGG,
            total_pathways=10,
            significant_pathways=2,
            pathways=[],
            species="human",
            coverage=0.8
        )
        analysis_engine.ora_engine.analyze = AsyncMock(return_value=mock_result)
        
        # Run analysis
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters,
            output_dir=str(output_dir)
        )
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        
        # Verify output files dict is in result
        assert hasattr(result, "output_files") or "output_files" in result.model_dump()

    @pytest.mark.asyncio
    async def test_analyze_invalid_input(self, analysis_engine, sample_parameters):
        """Test analysis with invalid input."""
        # Engine catches errors and returns result with errors
        result = await analysis_engine.analyze(
            input_data=123,  # Invalid input type
            parameters=sample_parameters
        )
        
        # Verify result has errors
        assert isinstance(result, AnalysisResult)
        assert len(result.errors) > 0
        assert "Unsupported input data type" in str(result.errors[0])

    @pytest.mark.asyncio
    async def test_analyze_invalid_parameters(self, analysis_engine, sample_gene_list):
        """Test analysis with invalid parameters."""
        with pytest.raises((ValueError, AttributeError, TypeError)):
            await analysis_engine.analyze(
                input_data=sample_gene_list,
                parameters=None
            )

    @pytest.mark.asyncio
    async def test_analyze_engine_error(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test analysis when engine raises an error."""
        # Mock the ORA engine to raise an error
        analysis_engine.ora_engine.analyze = AsyncMock(side_effect=Exception("Engine error"))
        
        # Engine catches errors in _perform_ora_analysis and creates empty DatabaseResult
        # The analysis completes but with empty results
        result = await analysis_engine.analyze(
            input_data=sample_gene_list,
            parameters=sample_parameters
        )
        
        # Verify result structure - errors are caught and empty results are returned
        assert isinstance(result, AnalysisResult)
        # The database_results will be empty or have empty DatabaseResult
        # Check that the analysis completed (even if with errors)
        assert result.database_results is not None

    def test_validate_parameters(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test parameters validation."""
        # Valid parameters
        analysis_engine._validate_parameters(sample_parameters, sample_gene_list)
        
        # Invalid parameters - should raise ValueError
        with pytest.raises(ValueError, match="Analysis parameters cannot be None"):
            analysis_engine._validate_parameters(None, sample_gene_list)
        
        # Empty gene list
        with pytest.raises(ValueError):
            analysis_engine._validate_parameters(sample_parameters, [])

    @pytest.mark.asyncio
    async def test_prepare_input_data_list(self, analysis_engine, sample_gene_list):
        """Test preparing input data from list."""
        gene_list, input_info = await analysis_engine._prepare_input_data(sample_gene_list)
        assert gene_list == sample_gene_list
        assert isinstance(input_info, dict)

    @pytest.mark.asyncio
    async def test_prepare_input_data_dataframe(self, analysis_engine):
        """Test preparing input data from DataFrame."""
        df = pd.DataFrame({
            'gene_id': ['GENE1', 'GENE2'],
            'expression': [1.5, 2.0]
        })
        
        gene_list, input_info = await analysis_engine._prepare_input_data(df)
        assert isinstance(gene_list, list)
        assert len(gene_list) == 2

    @pytest.mark.asyncio
    async def test_prepare_input_data_file(self, analysis_engine, tmp_path):
        """Test preparing input data from file."""
        gene_file = tmp_path / "genes.csv"
        df = pd.DataFrame({'gene_id': ['GENE1', 'GENE2', 'GENE3']})
        df.to_csv(gene_file, index=False)
        
        gene_list, input_info = await analysis_engine._prepare_input_data(str(gene_file))
        assert isinstance(gene_list, list)
        assert len(gene_list) == 3
        assert "GENE1" in gene_list

    def test_calculate_summary_statistics(self, analysis_engine, sample_database_result):
        """Test summary statistics calculation."""
        database_results = {"KEGG": sample_database_result}
        summary = analysis_engine._calculate_summary_statistics(database_results, None)
        
        assert "total_pathways" in summary
        assert "significant_pathways" in summary
        assert "significant_databases" in summary

    @pytest.mark.asyncio
    async def test_generate_output_files(self, analysis_engine, sample_parameters, sample_database_result, tmp_path):
        """Test output file generation."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        database_results = {"KEGG": sample_database_result}
        job_id = "test_job_123"
        
        output_files = await analysis_engine._generate_output_files(
            job_id=job_id,
            database_results=database_results,
            consensus_results=None,
            parameters=sample_parameters,
            output_dir=str(output_dir)
        )
        
        # Verify output files dict is returned
        assert isinstance(output_files, dict)
        assert "json" in output_files
        assert "csv" in output_files

    @pytest.mark.asyncio
    async def test_analyze_multiple_databases(self, analysis_engine, sample_parameters, sample_gene_list):
        """Test analysis with multiple databases."""
        # Set multiple databases
        sample_parameters.databases = [DatabaseType.KEGG, DatabaseType.REACTOME]
        
        # Mock the ORA engine to return proper DatabaseResult objects
        from pathwaylens_core.analysis.schemas import DatabaseResult
        mock_kegg_result = DatabaseResult(
            database=DatabaseType.KEGG,
            total_pathways=10,
            significant_pathways=2,
            pathways=[],
            species="human",
            coverage=0.8
        )
        mock_reactome_result = DatabaseResult(
            database=DatabaseType.REACTOME,
            total_pathways=10,
            significant_pathways=2,
            pathways=[],
            species="human",
            coverage=0.8
        )
        
        async def mock_analyze(*args, **kwargs):
            database = kwargs.get('database')
            if database == DatabaseType.KEGG:
                return mock_kegg_result
            return mock_reactome_result
        
        analysis_engine.ora_engine.analyze = AsyncMock(side_effect=mock_analyze)
        
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
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="mouse",
            significance_threshold=0.01,
            correction_method=CorrectionMethod.BONFERRONI,
            min_pathway_size=10,
            max_pathway_size=1000
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
        assert result.parameters.species == "mouse"
        assert result.parameters.significance_threshold == 0.01
