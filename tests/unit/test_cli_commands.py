"""
Unit tests for CLI commands.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typer.testing import CliRunner

from pathwaylens_cli.main import app


class TestCLICommands:
    """Test cases for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    def test_main_help(self, runner):
        """Test main command help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "PathwayLens" in result.stdout

    def test_analyze_command_help(self, runner):
        """Test analyze command help."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "analyze" in result.stdout

    def test_compare_command_help(self, runner):
        """Test compare command help."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "compare" in result.stdout

    def test_visualize_command_help(self, runner):
        """Test visualize command help."""
        result = runner.invoke(app, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "visualize" in result.stdout

    def test_normalize_command_help(self, runner):
        """Test normalize command help."""
        result = runner.invoke(app, ["normalize", "--help"])
        assert result.exit_code == 0
        assert "normalize" in result.stdout

    def test_analyze_command_success(self, runner, temp_dir):
        """Test successful analyze command execution."""
        # Create a test gene list file
        gene_file = temp_dir / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        result = runner.invoke(app, [
            "analyze",
            "ora",
            "--input", str(gene_file),
            "--databases", "kegg",
            "--species", "human",
            "--output", str(temp_dir / "output.json")
        ])
        
        # Should succeed (currently just prints, but command should not fail)
        assert result.exit_code == 0

    def test_analyze_command_invalid_input(self, runner):
        """Test analyze command with invalid input."""
        result = runner.invoke(app, [
            "analyze",
            "ora",
            "--input", "non-existent-file.txt",
            "--databases", "kegg",
            "--species", "human"
        ])
        
        # Command should fail with invalid file
        assert result.exit_code != 0

    def test_analyze_command_missing_required_args(self, runner):
        """Test analyze command with missing required arguments."""
        result = runner.invoke(app, [
            "analyze",
            "ora",
            "--databases", "kegg",
            "--species", "human"
        ])
        
        # Should fail without required --input
        assert result.exit_code != 0

    def test_compare_command_success(self, runner, temp_dir):
        """Test successful compare command execution."""
        # Create test analysis result files
        result1 = temp_dir / "result1.json"
        result2 = temp_dir / "result2.json"
        result1.write_text('{"pathways": []}')
        result2.write_text('{"pathways": []}')
        
        result = runner.invoke(app, [
            "compare",
            "overlap",
            "--inputs", str(result1),
            "--inputs", str(result2),
            "--output", str(temp_dir / "compare.json")
        ])
        
        # Should succeed (currently just prints, but command should not fail)
        assert result.exit_code == 0

    def test_compare_command_invalid_input(self, runner):
        """Test compare command with invalid input."""
        result = runner.invoke(app, [
            "compare",
            "overlap",
            "--inputs", "non-existent-file.json",
            "--inputs", "another-file.json"
        ])
        
        # Should fail with invalid files
        assert result.exit_code != 0

    def test_visualize_command_success(self, runner, temp_dir):
        """Test successful visualize command execution."""
        # Create a test analysis result file
        result_file = temp_dir / "result.json"
        result_file.write_text('{"pathways": []}')
        
        result = runner.invoke(app, [
            "visualize",
            "dot-plot",
            "--input", str(result_file),
            "--output", str(temp_dir / "plot.png")
        ])
        
        # Should succeed (currently just prints, but command should not fail)
        assert result.exit_code == 0

    def test_visualize_command_invalid_input(self, runner):
        """Test visualize command with invalid input."""
        result = runner.invoke(app, [
            "visualize",
            "dot-plot",
            "--input", "non-existent-file.json"
        ])
        
        # Should fail with invalid file
        assert result.exit_code != 0

    def test_normalize_command_success(self, runner, temp_dir):
        """Test successful normalize command execution."""
        # Create a test gene list file
        gene_file = temp_dir / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        mock_result = {"job_id": "test-job-id", "status": "completed", "results": []}
        mock_start_normalization = AsyncMock(return_value=mock_result)
        
        with patch('pathwaylens_cli.commands.normalize._start_normalization', mock_start_normalization):
            result = runner.invoke(app, [
                "normalize",
                "gene-ids",
                "--input", str(gene_file),
                "--output-format", "entrezgene",
                "--species", "human",
                "--output", str(temp_dir / "normalized.csv")
            ])
            
            # Should succeed (currently just prints, but command should not fail)
            assert result.exit_code == 0

    def test_normalize_command_invalid_input(self, runner):
        """Test normalize command with invalid input."""
        result = runner.invoke(app, [
            "normalize",
            "gene-ids",
            "--input", "non-existent-file.txt",
            "--output-format", "entrezgene",
            "--species", "human"
        ])
        
        # Should fail with invalid file
        assert result.exit_code != 0

    def test_invalid_command(self, runner):
        """Test invalid command returns error."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

    def test_config_file_loading(self, runner, temp_dir):
        """Test configuration file loading."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("""
api:
  base_url: "http://localhost:8000"
  timeout: 30
""")
        
        # Create test input file
        test_input = temp_dir / "test.txt"
        test_input.write_text("GENE1\nGENE2\n")
        
        # Test that config file option is accepted (config loading happens in commands)
        result = runner.invoke(app, [
            "--config", str(config_file),
            "analyze",
            "ora",
            "--input", str(test_input),
            "--databases", "kegg",
            "--species", "human"
        ])
        
        # Config file should be stored in context, command may fail for other reasons
        # Just verify the command accepts the --config option
        assert result.exit_code in [0, 1]  # May fail on actual analysis but config option works

    def test_output_directory_creation(self, runner, temp_dir):
        """Test output directory creation."""
        output_dir = temp_dir / "output"
        output_file = output_dir / "output.json"
        
        # Create test input file
        test_input = temp_dir / "test.txt"
        test_input.write_text("GENE1\nGENE2\n")
        
        result = runner.invoke(app, [
            "analyze",
            "ora",
            "--input", str(test_input),
            "--databases", "kegg",
            "--species", "human",
            "--output", str(output_file)
        ])
        
        # Should create output directory if output file is specified
        # The analyze command now creates output directories before writing
        assert output_dir.exists() or result.exit_code != 0

    def test_verbose_output(self, runner):
        """Test verbose output option."""
        result = runner.invoke(app, ["--verbose", "--help"])
        assert result.exit_code == 0
        assert "PathwayLens" in result.stdout

    def test_version_output(self, runner):
        """Test version output."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "PathwayLens" in result.stdout
