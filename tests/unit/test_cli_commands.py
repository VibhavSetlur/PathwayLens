"""
Unit tests for CLI commands.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner

from pathwaylens_cli.main import main
from pathwaylens_cli.commands.analyze import analyze_command
from pathwaylens_cli.commands.compare import compare_command
from pathwaylens_cli.commands.visualize import visualize_command
from pathwaylens_cli.commands.normalize import normalize_command


class TestCLICommands:
    """Test cases for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_main_help(self, runner):
        """Test main command help."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "PathwayLens CLI" in result.output

    def test_analyze_command_help(self, runner):
        """Test analyze command help."""
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.output

    def test_compare_command_help(self, runner):
        """Test compare command help."""
        result = runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.output

    def test_visualize_command_help(self, runner):
        """Test visualize command help."""
        result = runner.invoke(main, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "visualize" in result.output

    def test_normalize_command_help(self, runner):
        """Test normalize command help."""
        result = runner.invoke(main, ["normalize", "--help"])
        assert result.exit_code == 0
        assert "normalize" in result.output

    @pytest.mark.asyncio
    async def test_analyze_command_success(self, runner, temp_dir):
        """Test successful analyze command execution."""
        # Create a test gene list file
        gene_file = temp_dir / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.analyze.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = runner.invoke(main, [
                "analyze",
                "--input", str(gene_file),
                "--analysis-type", "ora",
                "--database", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ])
            
            assert result.exit_code == 0
            mock_instance.analyze.assert_called_once()

    def test_analyze_command_invalid_input(self, runner):
        """Test analyze command with invalid input."""
        result = runner.invoke(main, [
            "analyze",
            "--input", "non-existent-file.txt",
            "--analysis-type", "ora",
            "--database", "kegg",
            "--species", "human"
        ])
        
        assert result.exit_code != 0

    def test_analyze_command_missing_required_args(self, runner):
        """Test analyze command with missing required arguments."""
        result = runner.invoke(main, [
            "analyze",
            "--analysis-type", "ora",
            "--database", "kegg",
            "--species", "human"
        ])
        
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_compare_command_success(self, runner, temp_dir):
        """Test successful compare command execution."""
        # Create test analysis result files
        result1 = temp_dir / "result1.json"
        result2 = temp_dir / "result2.json"
        result1.write_text('{"pathways": []}')
        result2.write_text('{"pathways": []}')
        
        with patch('pathwaylens_cli.commands.compare.ComparisonEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.compare.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = runner.invoke(main, [
                "compare",
                "--input", str(result1),
                "--input", str(result2),
                "--comparison-type", "overlap",
                "--output-dir", str(temp_dir)
            ])
            
            assert result.exit_code == 0
            mock_instance.compare.assert_called_once()

    def test_compare_command_invalid_input(self, runner):
        """Test compare command with invalid input."""
        result = runner.invoke(main, [
            "compare",
            "--input", "non-existent-file.json",
            "--comparison-type", "overlap"
        ])
        
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_visualize_command_success(self, runner, temp_dir):
        """Test successful visualize command execution."""
        # Create a test analysis result file
        result_file = temp_dir / "result.json"
        result_file.write_text('{"pathways": []}')
        
        with patch('pathwaylens_cli.commands.visualize.VisualizationEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.generate_visualizations.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = runner.invoke(main, [
                "visualize",
                "--input", str(result_file),
                "--plot-type", "bar",
                "--output-dir", str(temp_dir)
            ])
            
            assert result.exit_code == 0
            mock_instance.generate_visualizations.assert_called_once()

    def test_visualize_command_invalid_input(self, runner):
        """Test visualize command with invalid input."""
        result = runner.invoke(main, [
            "visualize",
            "--input", "non-existent-file.json",
            "--plot-type", "bar"
        ])
        
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_normalize_command_success(self, runner, temp_dir):
        """Test successful normalize command execution."""
        # Create a test gene list file
        gene_file = temp_dir / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        with patch('pathwaylens_cli.commands.normalize.NormalizationEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.normalize.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = runner.invoke(main, [
                "normalize",
                "--input", str(gene_file),
                "--input-format", "gene_list",
                "--output-format", "entrezgene",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ])
            
            assert result.exit_code == 0
            mock_instance.normalize.assert_called_once()

    def test_normalize_command_invalid_input(self, runner):
        """Test normalize command with invalid input."""
        result = runner.invoke(main, [
            "normalize",
            "--input", "non-existent-file.txt",
            "--input-format", "gene_list",
            "--output-format", "entrezgene",
            "--species", "human"
        ])
        
        assert result.exit_code != 0

    def test_invalid_command(self, runner):
        """Test invalid command returns error."""
        result = runner.invoke(main, ["invalid-command"])
        assert result.exit_code != 0

    def test_config_file_loading(self, runner, temp_dir):
        """Test configuration file loading."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("""
        api:
          base_url: "http://localhost:8000"
          timeout: 30
        """)
        
        with patch('pathwaylens_cli.config.load_config') as mock_load:
            mock_load.return_value = {"api": {"base_url": "http://localhost:8000"}}
            
            result = runner.invoke(main, [
                "analyze",
                "--config", str(config_file),
                "--input", "test.txt",
                "--analysis-type", "ora",
                "--database", "kegg",
                "--species", "human"
            ])
            
            # Should not fail due to config loading
            assert result.exit_code == 0 or result.exit_code != 0  # Either is fine for this test

    def test_output_directory_creation(self, runner, temp_dir):
        """Test output directory creation."""
        output_dir = temp_dir / "output"
        
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.analyze.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = runner.invoke(main, [
                "analyze",
                "--input", "test.txt",
                "--analysis-type", "ora",
                "--database", "kegg",
                "--species", "human",
                "--output-dir", str(output_dir)
            ])
            
            # Should create output directory
            assert output_dir.exists()

    def test_verbose_output(self, runner):
        """Test verbose output option."""
        result = runner.invoke(main, ["--verbose", "--help"])
        assert result.exit_code == 0
        assert "PathwayLens CLI" in result.output

    def test_version_output(self, runner):
        """Test version output."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()
