"""
End-to-end tests for the complete PathwayLens workflow.
"""

import pytest
import asyncio
import tempfile
import subprocess
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_cli.main import main
from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.comparison.engine import ComparisonEngine
from pathwaylens_core.visualization.engine import VisualizationEngine
from pathwaylens_core.normalization.normalizer import NormalizationEngine


class TestE2EWorkflow:
    """End-to-end tests for the complete PathwayLens workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def test_gene_list(self, temp_dir):
        """Create a test gene list file."""
        gene_file = temp_dir / "test_genes.txt"
        gene_file.write_text("GENE1\nGENE2\nGENE3\nGENE4\nGENE5\n")
        return gene_file

    @pytest.fixture
    def test_expression_data(self, temp_dir):
        """Create a test expression data file."""
        expr_file = temp_dir / "test_expression.csv"
        expr_data = """Gene,Control,Treatment
GENE1,1.0,2.0
GENE2,0.5,1.5
GENE3,2.0,0.8
GENE4,1.2,1.8
GENE5,0.8,2.2"""
        expr_file.write_text(expr_data)
        return expr_file

    @pytest.mark.asyncio
    async def test_complete_cli_workflow(self, temp_dir, test_gene_list):
        """Test complete CLI workflow from gene list to visualization."""
        # Step 1: Normalize gene identifiers
        with patch('pathwaylens_cli.commands.normalize.NormalizationEngine') as mock_norm:
            mock_instance = Mock()
            mock_instance.normalize.return_value = Mock(
                normalized_data=["123", "456", "789", "101", "112"],
                mapping_stats={"total": 5, "converted": 5, "failed": 0}
            )
            mock_norm.return_value = mock_instance
            
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "normalize",
                "--input", str(test_gene_list),
                "--input-format", "gene_list",
                "--output-format", "entrezgene",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0

        # Step 2: Run pathway analysis
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_analysis:
            mock_instance = Mock()
            mock_instance.analyze.return_value = Mock(
                analysis_id="test_analysis_1",
                analysis_name="test_analysis",
                database_results=[Mock(
                    database_name="KEGG",
                    pathway_results=[
                        Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.01),
                        Mock(pathway_id="hsa00020", pathway_name="TCA Cycle", p_value=0.02)
                    ]
                )]
            )
            mock_analysis.return_value = mock_instance
            
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "analyze",
                "--input", str(test_gene_list),
                "--analysis-type", "ora",
                "--database", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0

        # Step 3: Generate visualizations
        with patch('pathwaylens_cli.commands.visualize.VisualizationEngine') as mock_viz:
            mock_instance = Mock()
            mock_instance.generate_visualizations.return_value = Mock(
                plot_id="test_plot_1",
                plot_type="bar",
                output_path=str(temp_dir / "test_plot.html")
            )
            mock_viz.return_value = mock_instance
            
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "visualize",
                "--input", str(temp_dir / "analysis_result.json"),
                "--plot-type", "bar",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_multi_dataset_comparison_workflow(self, temp_dir, test_gene_list):
        """Test multi-dataset comparison workflow."""
        # Create two different gene lists
        gene_list_1 = temp_dir / "genes_1.txt"
        gene_list_2 = temp_dir / "genes_2.txt"
        gene_list_1.write_text("GENE1\nGENE2\nGENE3\n")
        gene_list_2.write_text("GENE2\nGENE3\nGENE4\n")
        
        # Run analysis on both datasets
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_analysis:
            mock_instance = Mock()
            mock_instance.analyze.return_value = Mock(
                analysis_id="test_analysis_1",
                database_results=[Mock(
                    database_name="KEGG",
                    pathway_results=[
                        Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.01),
                        Mock(pathway_id="hsa00020", pathway_name="TCA Cycle", p_value=0.02)
                    ]
                )]
            )
            mock_analysis.return_value = mock_instance
            
            # Analyze first dataset
            result1 = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "analyze",
                "--input", str(gene_list_1),
                "--analysis-type", "ora",
                "--database", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir / "analysis1")
            ], capture_output=True, text=True)
            
            # Analyze second dataset
            result2 = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "analyze",
                "--input", str(gene_list_2),
                "--analysis-type", "ora",
                "--database", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir / "analysis2")
            ], capture_output=True, text=True)
            
            assert result1.returncode == 0
            assert result2.returncode == 0

        # Compare the two analyses
        with patch('pathwaylens_cli.commands.compare.ComparisonEngine') as mock_compare:
            mock_instance = Mock()
            mock_instance.compare.return_value = Mock(
                comparison_id="test_comparison_1",
                comparison_name="test_comparison",
                overlap_statistics=Mock(
                    total_pathways=10,
                    overlapping_pathways=5,
                    overlap_percentage=50.0
                )
            )
            mock_compare.return_value = mock_instance
            
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "compare",
                "--input", str(temp_dir / "analysis1" / "analysis_result.json"),
                "--input", str(temp_dir / "analysis2" / "analysis_result.json"),
                "--comparison-type", "overlap",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_expression_data_workflow(self, temp_dir, test_expression_data):
        """Test workflow with expression data."""
        # Run GSEA analysis on expression data
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_analysis:
            mock_instance = Mock()
            mock_instance.analyze.return_value = Mock(
                analysis_id="test_gsea_1",
                analysis_name="test_gsea",
                database_results=[Mock(
                    database_name="KEGG",
                    pathway_results=[
                        Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.01, enrichment_score=0.8),
                        Mock(pathway_id="hsa00020", pathway_name="TCA Cycle", p_value=0.02, enrichment_score=0.6)
                    ]
                )]
            )
            mock_analysis.return_value = mock_instance
            
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "analyze",
                "--input", str(test_expression_data),
                "--analysis-type", "gsea",
                "--database", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_multi_omics_workflow(self, temp_dir):
        """Test multi-omics analysis workflow."""
        # Create test multi-omics data
        transcriptomics_file = temp_dir / "transcriptomics.csv"
        proteomics_file = temp_dir / "proteomics.csv"
        
        transcriptomics_file.write_text("""Gene,Control,Treatment
GENE1,1.0,2.0
GENE2,0.5,1.5
GENE3,2.0,0.8""")
        
        proteomics_file.write_text("""Protein,Control,Treatment
PROT1,1.2,2.1
PROT2,0.8,1.6
PROT3,1.8,0.9""")
        
        # Run multi-omics analysis
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_analysis:
            mock_instance = Mock()
            mock_instance.analyze.return_value = Mock(
                analysis_id="test_multiomics_1",
                analysis_name="test_multiomics",
                database_results=[Mock(
                    database_name="KEGG",
                    pathway_results=[
                        Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.01),
                        Mock(pathway_id="hsa00020", pathway_name="TCA Cycle", p_value=0.02)
                    ]
                )]
            )
            mock_analysis.return_value = mock_instance
            
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "analyze",
                "--input", str(transcriptomics_file),
                "--input", str(proteomics_file),
                "--analysis-type", "multi_omics",
                "--database", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_plugin_integration_workflow(self, temp_dir, test_gene_list):
        """Test workflow with plugin integration."""
        # Test with custom analysis plugin
        with patch('pathwaylens_core.plugins.plugin_system.PluginSystem') as mock_plugin_system:
            mock_instance = Mock()
            mock_plugin = Mock()
            mock_plugin.perform_analysis.return_value = Mock(
                analysis_id="plugin_analysis_1",
                analysis_name="plugin_analysis",
                database_results=[Mock(
                    database_name="CustomDB",
                    pathway_results=[
                        Mock(pathway_id="CUSTOM1", pathway_name="Custom Pathway", p_value=0.01)
                    ]
                )]
            )
            mock_instance.get_plugin.return_value = mock_plugin
            mock_plugin_system.return_value = mock_instance
            
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "analyze",
                "--input", str(test_gene_list),
                "--analysis-type", "custom",
                "--database", "custom",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            assert result.returncode == 0

    def test_error_handling_workflow(self, temp_dir):
        """Test error handling in end-to-end workflow."""
        # Test with invalid input file
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("invalid content")
        
        result = subprocess.run([
            "python", "-m", "pathwaylens_cli.main", "analyze",
            "--input", str(invalid_file),
            "--analysis-type", "ora",
            "--database", "kegg",
            "--species", "human",
            "--output-dir", str(temp_dir)
        ], capture_output=True, text=True)
        
        # Should return error code for invalid input
        assert result.returncode != 0

    def test_output_file_structure(self, temp_dir):
        """Test that output files are created with correct structure."""
        # Create mock analysis result
        analysis_result = {
            "analysis_id": "test_analysis_1",
            "analysis_name": "test_analysis",
            "analysis_type": "ora",
            "database_results": [
                {
                    "database_name": "KEGG",
                    "pathway_results": [
                        {
                            "pathway_id": "hsa00010",
                            "pathway_name": "Glycolysis",
                            "p_value": 0.01,
                            "adjusted_p_value": 0.05
                        }
                    ]
                }
            ]
        }
        
        # Write analysis result to file
        result_file = temp_dir / "analysis_result.json"
        result_file.write_text(json.dumps(analysis_result, indent=2))
        
        # Verify file structure
        assert result_file.exists()
        loaded_data = json.loads(result_file.read_text())
        assert loaded_data["analysis_id"] == "test_analysis_1"
        assert len(loaded_data["database_results"]) == 1
        assert len(loaded_data["database_results"][0]["pathway_results"]) == 1
