"""
Performance tests for visualization generation.
"""

import pytest
import time
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from pathwaylens_core.visualization.engine import VisualizationEngine
from pathwaylens_core.analysis.schemas import (
    AnalysisResult, DatabaseResult, PathwayResult, AnalysisType, DatabaseType
)


@pytest.mark.performance
@pytest.mark.slow
class TestVisualizationPerformance:
    """Performance benchmarks for visualization operations."""

    @pytest.fixture
    def visualization_engine(self):
        """Create visualization engine."""
        return VisualizationEngine()

    @pytest.fixture
    def sample_analysis_result(self):
        """Create sample analysis result."""
        pathway_results = [
            PathwayResult(
                pathway_id=f"hsa{i:05d}",
                pathway_name=f"Pathway {i}",
                p_value=0.001 * i,
                adjusted_p_value=0.01 * i,
                gene_overlap=[f"GENE{j}" for j in range(i)],
                gene_overlap_count=i,
                pathway_size=10 + i
            )
            for i in range(1, 101)
        ]
        
        database_result = DatabaseResult(
            database_name="KEGG",
            database_type=DatabaseType.KEGG,
            species="human",
            pathway_results=pathway_results
        )
        
        return AnalysisResult(
            analysis_id="test_analysis",
            analysis_name="Test Analysis",
            analysis_type=AnalysisType.ORA,
            database_results=[database_result]
        )

    @pytest.mark.benchmark
    def test_bar_plot_generation(self, visualization_engine, sample_analysis_result, temp_dir):
        """Benchmark bar plot generation."""
        output_path = temp_dir / "bar_plot.html"
        
        start_time = time.time()
        result = visualization_engine.create_bar_plot(
            analysis_result=sample_analysis_result,
            output_path=str(output_path),
            top_n=50
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert output_path.exists()
        assert elapsed < 5.0, f"Bar plot generation took {elapsed:.2f}s, expected <5s"

    @pytest.mark.benchmark
    def test_volcano_plot_generation(self, visualization_engine, sample_analysis_result, temp_dir):
        """Benchmark volcano plot generation."""
        output_path = temp_dir / "volcano_plot.html"
        
        start_time = time.time()
        result = visualization_engine.create_volcano_plot(
            analysis_result=sample_analysis_result,
            output_path=str(output_path)
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert output_path.exists()
        assert elapsed < 5.0, f"Volcano plot generation took {elapsed:.2f}s, expected <5s"

    @pytest.mark.benchmark
    def test_heatmap_generation(self, visualization_engine, sample_analysis_result, temp_dir):
        """Benchmark heatmap generation."""
        output_path = temp_dir / "heatmap.html"
        
        start_time = time.time()
        result = visualization_engine.create_heatmap(
            analysis_result=sample_analysis_result,
            output_path=str(output_path)
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert output_path.exists()
        assert elapsed < 10.0, f"Heatmap generation took {elapsed:.2f}s, expected <10s"

    @pytest.mark.benchmark
    def test_network_plot_generation(self, visualization_engine, sample_analysis_result, temp_dir):
        """Benchmark network plot generation."""
        output_path = temp_dir / "network_plot.html"
        
        start_time = time.time()
        result = visualization_engine.create_network_plot(
            analysis_result=sample_analysis_result,
            output_path=str(output_path),
            top_n=30
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert output_path.exists()
        # Network plots can be slower due to layout calculations
        assert elapsed < 15.0, f"Network plot generation took {elapsed:.2f}s, expected <15s"

    @pytest.mark.benchmark
    def test_large_dataset_visualization(self, visualization_engine, temp_dir):
        """Benchmark visualization with large dataset."""
        # Create large analysis result
        pathway_results = [
            PathwayResult(
                pathway_id=f"hsa{i:05d}",
                pathway_name=f"Pathway {i}",
                p_value=np.random.uniform(0.001, 0.1),
                adjusted_p_value=np.random.uniform(0.01, 0.2),
                gene_overlap=[f"GENE{j}" for j in range(min(i, 50))],
                gene_overlap_count=min(i, 50),
                pathway_size=10 + i
            )
            for i in range(1, 1001)
        ]
        
        database_result = DatabaseResult(
            database_name="KEGG",
            database_type=DatabaseType.KEGG,
            species="human",
            pathway_results=pathway_results
        )
        
        large_result = AnalysisResult(
            analysis_id="large_analysis",
            analysis_name="Large Analysis",
            analysis_type=AnalysisType.ORA,
            database_results=[database_result]
        )
        
        output_path = temp_dir / "large_bar_plot.html"
        
        start_time = time.time()
        result = visualization_engine.create_bar_plot(
            analysis_result=large_result,
            output_path=str(output_path),
            top_n=100
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert output_path.exists()
        # Large datasets should still complete in reasonable time
        assert elapsed < 30.0, f"Large dataset visualization took {elapsed:.2f}s"



