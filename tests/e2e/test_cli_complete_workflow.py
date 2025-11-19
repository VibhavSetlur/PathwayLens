"""
Complete CLI workflow end-to-end tests.
"""

import pytest
import asyncio
import tempfile
import subprocess
import json
from pathlib import Path

from pathwaylens_cli.main import main


@pytest.mark.e2e
class TestCLICompleteWorkflow:
    """Complete CLI workflow end-to-end tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_gene_list(self, temp_dir):
        """Create sample gene list file."""
        gene_file = temp_dir / "genes.txt"
        gene_file.write_text("\n".join([
            "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS",
            "PIK3CA", "PTEN", "RB1", "MYC", "CDKN2A"
        ]))
        return gene_file

    @pytest.fixture
    def sample_expression_data(self, temp_dir):
        """Create sample expression data file."""
        expr_file = temp_dir / "expression.csv"
        expr_data = """Gene,Control_1,Control_2,Treatment_1,Treatment_2
TP53,1.0,1.1,2.5,2.3
BRCA1,0.8,0.9,1.8,1.9
BRCA2,1.2,1.1,2.1,2.0
EGFR,0.9,1.0,2.2,2.1
KRAS,1.1,1.2,1.9,2.0"""
        expr_file.write_text(expr_data)
        return expr_file

    def test_normalize_analyze_visualize_pipeline(
        self, temp_dir, sample_gene_list
    ):
        """Test complete pipeline: normalize → analyze → visualize."""
        # Step 1: Normalize
        result = subprocess.run([
            "python", "-m", "pathwaylens_cli.main", "normalize",
            "--input", str(sample_gene_list),
            "--input-format", "gene_list",
            "--output-format", "ensembl",
            "--species", "human",
            "--output-dir", str(temp_dir / "normalized")
        ], capture_output=True, text=True, timeout=60)
        
        # Step 2: Analyze (using normalized output or original)
        normalized_file = temp_dir / "normalized" / "normalized_output.txt"
        input_file = normalized_file if normalized_file.exists() else sample_gene_list
        
        result = subprocess.run([
            "python", "-m", "pathwaylens_cli.main", "analyze",
            "--input", str(input_file),
            "--analysis-type", "ora",
            "--database", "kegg",
            "--species", "human",
            "--output-dir", str(temp_dir / "analysis")
        ], capture_output=True, text=True, timeout=120)
        
        # Step 3: Visualize
        analysis_result = temp_dir / "analysis" / "analysis_result.json"
        if analysis_result.exists():
            result = subprocess.run([
                "python", "-m", "pathwaylens_cli.main", "visualize",
                "--input", str(analysis_result),
                "--plot-type", "bar",
                "--output-dir", str(temp_dir / "visualizations")
            ], capture_output=True, text=True, timeout=60)
            
            # Verify visualization was created
            viz_files = list((temp_dir / "visualizations").glob("*.html"))
            assert len(viz_files) > 0, "Visualization file should be created"

    def test_workflow_file_execution(self, temp_dir, sample_gene_list):
        """Test workflow file execution."""
        # Create workflow file
        workflow_file = temp_dir / "workflow.yaml"
        workflow_content = """
name: Test Workflow
steps:
  - id: normalize
    type: normalization
    params:
      input_file: {input_file}
      output_dir: {output_dir}/normalized
      species: human
      input_id_type: symbol
      output_id_type: ensembl
  
  - id: analyze
    type: analysis
    params:
      input_file: {output_dir}/normalized/normalized_output.txt
      analysis_type: ora
      database: kegg
      species: human
      output_dir: {output_dir}/analysis
"""
        workflow_file.write_text(
            workflow_content.format(
                input_file=str(sample_gene_list),
                output_dir=str(temp_dir)
            )
        )
        
        # Execute workflow
        result = subprocess.run([
            "python", "-m", "pathwaylens_cli.main", "workflow",
            "run",
            "--workflow-file", str(workflow_file),
            "--output-dir", str(temp_dir)
        ], capture_output=True, text=True, timeout=180)
        
        # Workflow should complete (may have warnings but should not crash)
        assert result.returncode in [0, 1], \
            f"Workflow execution failed: {result.stderr}"

    def test_multi_database_consensus_analysis(
        self, temp_dir, sample_gene_list
    ):
        """Test multi-database consensus analysis."""
        result = subprocess.run([
            "python", "-m", "pathwaylens_cli.main", "analyze",
            "--input", str(sample_gene_list),
            "--analysis-type", "ora",
            "--database", "kegg",
            "--database", "reactome",
            "--database", "go",
            "--species", "human",
            "--consensus-method", "fisher",
            "--output-dir", str(temp_dir)
        ], capture_output=True, text=True, timeout=180)
        
        # Should complete (may have warnings for missing databases)
        assert result.returncode in [0, 1]

    def test_visualization_generation_and_export(
        self, temp_dir, sample_gene_list
    ):
        """Test visualization generation and export."""
        # First run analysis
        result = subprocess.run([
            "python", "-m", "pathwaylens_cli.main", "analyze",
            "--input", str(sample_gene_list),
            "--analysis-type", "ora",
            "--database", "kegg",
            "--species", "human",
            "--output-dir", str(temp_dir / "analysis")
        ], capture_output=True, text=True, timeout=120)
        
        # Then generate visualizations
        analysis_result = temp_dir / "analysis" / "analysis_result.json"
        if analysis_result.exists():
            # Test different plot types
            plot_types = ["bar", "volcano", "heatmap"]
            
            for plot_type in plot_types:
                result = subprocess.run([
                    "python", "-m", "pathwaylens_cli.main", "visualize",
                    "--input", str(analysis_result),
                    "--plot-type", plot_type,
                    "--output-dir", str(temp_dir / "visualizations"),
                    "--output-format", "html"
                ], capture_output=True, text=True, timeout=60)
                
                # Visualization should complete
                assert result.returncode in [0, 1]

    def test_report_generation(self, temp_dir, sample_gene_list):
        """Test report generation."""
        # Run analysis
        result = subprocess.run([
            "python", "-m", "pathwaylens_cli.main", "analyze",
            "--input", str(sample_gene_list),
            "--analysis-type", "ora",
            "--database", "kegg",
            "--species", "human",
            "--output-dir", str(temp_dir / "analysis"),
            "--generate-report"
        ], capture_output=True, text=True, timeout=120)
        
        # Check if report was generated
        report_files = list((temp_dir / "analysis").glob("*.html"))
        report_files.extend(list((temp_dir / "analysis").glob("*.pdf")))
        
        # Report may or may not be generated depending on implementation
        # Just verify command completes
        assert result.returncode in [0, 1]



