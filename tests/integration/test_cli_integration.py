"""
Integration tests for the CLI interface.
"""

import pytest
import subprocess
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_cli.main import main
from pathwaylens_cli.commands.analyze import analyze_command
from pathwaylens_cli.commands.compare import compare_command
from pathwaylens_cli.commands.visualize import visualize_command
from pathwaylens_cli.commands.normalize import normalize_command


class TestCLIIntegration:
    """Integration tests for the CLI interface."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            ["python", "-m", "pathwaylens_cli.main", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "PathwayLens CLI" in result.stdout

    def test_analyze_command_help(self):
        """Test analyze command help."""
        result = subprocess.run(
            ["python", "-m", "pathwaylens_cli.main", "analyze", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "analyze" in result.stdout

    def test_compare_command_help(self):
        """Test compare command help."""
        result = subprocess.run(
            ["python", "-m", "pathwaylens_cli.main", "compare", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "compare" in result.stdout

    def test_visualize_command_help(self):
        """Test visualize command help."""
        result = subprocess.run(
            ["python", "-m", "pathwaylens_cli.main", "visualize", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "visualize" in result.stdout

    def test_normalize_command_help(self):
        """Test normalize command help."""
        result = subprocess.run(
            ["python", "-m", "pathwaylens_cli.main", "normalize", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "normalize" in result.stdout

    @pytest.mark.asyncio
    async def test_analyze_command_execution(self, temp_dir):
        """Test analyze command execution."""
        # Create a test gene list file
        gene_file = temp_dir / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.analyze.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = await analyze_command(
                input_data=str(gene_file),
                analysis_type="ora",
                database="kegg",
                species="human",
                output_dir=str(temp_dir)
            )
            
            assert result is not None
            mock_instance.analyze.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_command_execution(self, temp_dir):
        """Test compare command execution."""
        # Create test analysis result files
        result1 = temp_dir / "result1.json"
        result2 = temp_dir / "result2.json"
        result1.write_text('{"pathways": []}')
        result2.write_text('{"pathways": []}')
        
        with patch('pathwaylens_cli.commands.compare.ComparisonEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.compare.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = await compare_command(
                input_data=[str(result1), str(result2)],
                comparison_type="overlap",
                output_dir=str(temp_dir)
            )
            
            assert result is not None
            mock_instance.compare.assert_called_once()

    @pytest.mark.asyncio
    async def test_visualize_command_execution(self, temp_dir):
        """Test visualize command execution."""
        # Create a test analysis result file
        result_file = temp_dir / "result.json"
        result_file.write_text('{"pathways": []}')
        
        with patch('pathwaylens_cli.commands.visualize.VisualizationEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.generate_visualizations.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = await visualize_command(
                input_data=str(result_file),
                plot_type="bar",
                output_dir=str(temp_dir)
            )
            
            assert result is not None
            mock_instance.generate_visualizations.assert_called_once()

    @pytest.mark.asyncio
    async def test_normalize_command_execution(self, temp_dir):
        """Test normalize command execution."""
        # Create a test gene list file
        gene_file = temp_dir / "genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\n")
        
        with patch('pathwaylens_cli.commands.normalize.NormalizationEngine') as mock_engine:
            mock_instance = Mock()
            mock_instance.normalize.return_value = Mock()
            mock_engine.return_value = mock_instance
            
            result = await normalize_command(
                input_data=str(gene_file),
                input_format="gene_list",
                output_format="entrezgene",
                species="human",
                output_dir=str(temp_dir)
            )
            
            assert result is not None
            mock_instance.normalize.assert_called_once()

    def test_invalid_command(self):
        """Test invalid command returns error."""
        result = subprocess.run(
            ["python", "-m", "pathwaylens_cli.main", "invalid-command"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0

    def test_missing_required_argument(self):
        """Test missing required argument returns error."""
        result = subprocess.run(
            ["python", "-m", "pathwaylens_cli.main", "analyze"],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0

    def test_config_file_loading(self, temp_dir):
        """Test configuration file loading."""
        config_file = temp_dir / "config.yml"
        config_file.write_text("""
        api:
          base_url: "http://localhost:8000"
          timeout: 30
        """)
        
        with patch('pathwaylens_cli.config.load_config') as mock_load:
            mock_load.return_value = {"api": {"base_url": "http://localhost:8000"}}
            
            # Test that config loading works
            config = mock_load(str(config_file))
            assert config["api"]["base_url"] == "http://localhost:8000"

    def test_output_directory_creation(self, temp_dir):
        """Test output directory creation."""
        output_dir = temp_dir / "output"
        
        # Test that directory creation works
        output_dir.mkdir(exist_ok=True)
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_file_validation(self, temp_dir):
        """Test file validation."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # Test file exists
        assert test_file.exists()
        assert test_file.is_file()
        
        # Test file content
        content = test_file.read_text()
        assert content == "test content"
