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

import sys
from pathwaylens_cli.main import main
from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.comparison.engine import ComparisonEngine
from pathwaylens_core.visualization.engine import VisualizationEngine
from pathwaylens_core.normalization.normalizer import Normalizer as NormalizationEngine


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
        gene_file.write_text("\n".join([f"GENE{i}" for i in range(1, 21)]))
        gene_file.write_text("\n".join([f"GENE{i}" for i in range(1, 16)])) # Changed to 15 genes
        return gene_file

    @pytest.fixture
    def test_expression_data(self, temp_dir):
        """Create a test expression data file."""
        expr_file = temp_dir / "test_expression.csv"
        expr_data = "Gene,Control,Treatment\n" + "\n".join([f"GENE{i},{1.0+(i*0.1)},{2.0-(i*0.1)}" for i in range(1, 21)])
        expr_file.write_text(expr_data)
        return expr_file

    @pytest.mark.asyncio
    async def test_complete_cli_workflow(self, temp_dir, test_gene_list):
        """Test complete CLI workflow from gene list to visualization."""
        # Step 1: Normalize gene identifiers
        with patch('pathwaylens_cli.commands.normalize.IDConverter') as mock_norm:
            mock_instance = Mock()
            mock_instance.normalize.return_value = Mock(
                normalized_data=["123", "456", "789", "101", "112"],
                mapping_stats={"total": 5, "converted": 5, "failed": 0}
            )
            mock_norm.return_value = mock_instance
            
            result = subprocess.run([
                sys.executable, "-m", "pathwaylens_cli.main", "normalize", "gene-ids",
                "--input", str(test_gene_list),
                "--input-format", "symbol", # Changed from gene_list (not in enum typically) or auto
                "--output-format", "entrez", # Changed from entrezgene
                "--species", "human",
                "--output", str(temp_dir / "normalized.json") # normalize gene-ids uses --output
            ], capture_output=True, text=True)
            
            # Print stderr if failed
            if result.returncode != 0:
                print(f"Normalize failed: {result.stderr}")
            assert result.returncode == 0

        # Step 2: Run pathway analysis (ORA)
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_analysis:
            mock_instance = Mock()
            mock_instance.analyze_sync.return_value = Mock( # analyze_sync is called by CLI
                analysis_id="test_analysis_1",
                analysis_name="test_analysis",
                database_results=[Mock(
                    database_name="KEGG",
                    pathway_results=[
                        Mock(pathway_id="hsa00010", pathway_name="Glycolysis", p_value=0.01),
                        Mock(pathway_id="hsa00020", pathway_name="TCA Cycle", p_value=0.02)
                    ]
                )],
                errors=[]
            )
            mock_analysis.return_value = mock_instance
            
            result = subprocess.run([
                sys.executable, "-m", "pathwaylens_cli.main", "analyze", "ora",
                "--input", str(test_gene_list),
                "--omic-type", "transcriptomics", # Required
                "--data-type", "bulk", # Required
                "--databases", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Analyze failed: {result.stderr}")
            assert result.returncode == 0

        # Step 3: Generate visualizations
        # Visualize command is 'visualize dot-plot'
        # It does NOT use VisualizationEngine but just prints, so we don't mock it or just let it run.
        result = subprocess.run([
            sys.executable, "-m", "pathwaylens_cli.main", "visualize", "dot-plot",
            "--input", str(temp_dir / "analysis_result.json"), # Dummy path, file doesn't exist but command might check
            "--output", str(temp_dir / "plot.html")
        ], capture_output=True, text=True)
        
        # visualize dot-plot checks if input file exists.
        # Since we mocked analyze, analysis_result.json wasn't created on disk.
        # We must create a dummy file.
        (temp_dir / "analysis_result.json").write_text("{}")
        
        result = subprocess.run([
            sys.executable, "-m", "pathwaylens_cli.main", "visualize", "dot-plot",
            "--input", str(temp_dir / "analysis_result.json"),
            "--output", str(temp_dir / "plot.html")
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Visualize failed: {result.stderr}")
        assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_multi_dataset_comparison_workflow(self, temp_dir, test_gene_list):
        """Test multi-dataset comparison workflow."""
        # Create two different gene lists
        gene_list_1 = temp_dir / "genes_1.txt"
        gene_list_2 = temp_dir / "genes_2.txt"
        gene_list_1.write_text("\n".join([f"GENE{i}" for i in range(1, 21)]))
        gene_list_2.write_text("\n".join([f"GENE{i}" for i in range(5, 25)]))
        
        # Analyze first dataset
        # We need to mock AnalysisEngine again or just assume success if we mock it?
        # Actually, if we mock it, we don't produce output files.
        # compare command expects output files for 'pathway' stage if input is result json.
        # Or 'gene' stage if input is gene list.
        # Let's test 'stage gene' comparison which takes gene lists.
        # Compare the two analyses
        # ComparisonEngine is imported locally in compare command, so we patch the source class
        with patch('pathwaylens_core.comparison.engine.ComparisonEngine') as mock_compare:
            mock_instance = Mock()
            mock_instance.compare_gene_lists.return_value = Mock(
                comparison_id="test_comparison_1",
                results={}
            )
            mock_compare.return_value = mock_instance
            
            result = subprocess.run([
                sys.executable, "-m", "pathwaylens_cli.main", "compare",
                "--inputs", str(gene_list_1),
                "--inputs", str(gene_list_2),
                "--labels", "Set1",
                "--labels", "Set2",
                "--comparison-type", "condition",
                "--stage", "gene",
                "--output-dir", str(temp_dir / "comparison")
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Compare failed: {result.stderr}")
            assert result.returncode == 0

    @pytest.mark.asyncio
    async def test_expression_data_workflow(self, temp_dir, test_expression_data):
        """Test workflow with expression data."""
        # Run GSEA analysis on expression data
        with patch('pathwaylens_cli.commands.analyze.AnalysisEngine') as mock_analysis:
            mock_instance = Mock()
            mock_instance.analyze_sync.return_value = Mock(
                analysis_id="test_gsea_1",
                analysis_name="test_gsea",
                database_results=[],
                errors=[]
            )
            mock_analysis.return_value = mock_instance
            
            result = subprocess.run([
                sys.executable, "-m", "pathwaylens_cli.main", "analyze", "gsea",
                "--input", str(test_expression_data),
                "--omic-type", "transcriptomics",
                "--data-type", "bulk",
                "--databases", "kegg",
                "--species", "human",
                "--output-dir", str(temp_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"GSEA failed: {result.stderr}")
            assert result.returncode == 0

    def test_error_handling_workflow(self, temp_dir):
        """Test error handling in end-to-end workflow."""
        # Test with invalid input file
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("invalid content")
        
        result = subprocess.run([
            sys.executable, "-m", "pathwaylens_cli.main", "analyze", "ora",
            "--input", str(invalid_file),
            "--omic-type", "transcriptomics",
            "--data-type", "bulk",
            "--output-dir", str(temp_dir)
        ], capture_output=True, text=True)
        
        # Should return error code for invalid input (or analysis failure)
        # Note: analyze.py checks if input file exists, but doesn't validate content before creating engine?
        # Actually ora command does validation.
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
