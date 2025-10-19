"""
Test fixtures and sample data for PathwayLens testing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from pathwaylens_core.analysis.schemas import (
    AnalysisResult, DatabaseResult, PathwayResult, AnalysisParameters,
    AnalysisType, DatabaseType, CorrectionMethod
)
from pathwaylens_core.comparison.schemas import (
    ComparisonResult, ComparisonParameters, ComparisonType,
    OverlapStatistics, CorrelationResult
)
from pathwaylens_core.visualization.schemas import (
    VisualizationResult, VisualizationParameters, PlotType,
    PlotMetadata, DashboardConfig
)


class TestDataFixtures:
    """Test fixtures and sample data for PathwayLens testing."""

    @pytest.fixture
    def sample_gene_list(self) -> List[str]:
        """Sample gene list for testing."""
        return [
            "GENE1", "GENE2", "GENE3", "GENE4", "GENE5",
            "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"
        ]

    @pytest.fixture
    def sample_expression_data(self) -> pd.DataFrame:
        """Sample expression data for testing."""
        np.random.seed(42)
        data = {
            'Gene': [f'GENE{i}' for i in range(1, 11)],
            'Control_1': np.random.normal(1.0, 0.2, 10),
            'Control_2': np.random.normal(1.0, 0.2, 10),
            'Treatment_1': np.random.normal(2.0, 0.3, 10),
            'Treatment_2': np.random.normal(2.0, 0.3, 10)
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_pathway_data(self) -> Dict[str, Dict[str, Any]]:
        """Sample pathway data for testing."""
        return {
            "hsa00010": {
                "name": "Glycolysis / Gluconeogenesis",
                "genes": ["GENE1", "GENE2", "GENE3"],
                "description": "Glycolysis pathway"
            },
            "hsa00020": {
                "name": "Citrate cycle (TCA cycle)",
                "genes": ["GENE4", "GENE5", "GENE6"],
                "description": "TCA cycle pathway"
            },
            "hsa00030": {
                "name": "Pentose phosphate pathway",
                "genes": ["GENE7", "GENE8"],
                "description": "Pentose phosphate pathway"
            },
            "hsa00040": {
                "name": "Pentose and glucuronate interconversions",
                "genes": ["GENE9", "GENE10"],
                "description": "Pentose and glucuronate interconversions"
            }
        }

    @pytest.fixture
    def sample_pathway_results(self) -> List[PathwayResult]:
        """Sample pathway results for testing."""
        return [
            PathwayResult(
                pathway_id="hsa00010",
                pathway_name="Glycolysis / Gluconeogenesis",
                p_value=0.001,
                adjusted_p_value=0.01,
                gene_overlap=["GENE1", "GENE2"],
                gene_overlap_count=2,
                pathway_size=10
            ),
            PathwayResult(
                pathway_id="hsa00020",
                pathway_name="Citrate cycle (TCA cycle)",
                p_value=0.005,
                adjusted_p_value=0.02,
                gene_overlap=["GENE4", "GENE5"],
                gene_overlap_count=2,
                pathway_size=8
            ),
            PathwayResult(
                pathway_id="hsa00030",
                pathway_name="Pentose phosphate pathway",
                p_value=0.01,
                adjusted_p_value=0.03,
                gene_overlap=["GENE7"],
                gene_overlap_count=1,
                pathway_size=6
            )
        ]

    @pytest.fixture
    def sample_database_result(self, sample_pathway_results) -> DatabaseResult:
        """Sample database result for testing."""
        return DatabaseResult(
            database_name="KEGG",
            database_type=DatabaseType.KEGG,
            species="human",
            pathway_results=sample_pathway_results
        )

    @pytest.fixture
    def sample_analysis_result(self, sample_database_result) -> AnalysisResult:
        """Sample analysis result for testing."""
        return AnalysisResult(
            analysis_id="test_analysis_1",
            analysis_name="Test Analysis",
            analysis_type=AnalysisType.ORA,
            parameters=AnalysisParameters(
                analysis_name="Test Analysis",
                analysis_type=AnalysisType.ORA,
                database_type=DatabaseType.KEGG,
                species="human",
                significance_threshold=0.05,
                correction_method=CorrectionMethod.FDR_BH
            ),
            database_results=[sample_database_result],
            timestamp="2024-01-01T00:00:00Z"
        )

    @pytest.fixture
    def sample_comparison_result(self) -> ComparisonResult:
        """Sample comparison result for testing."""
        return ComparisonResult(
            comparison_id="test_comparison_1",
            comparison_name="Test Comparison",
            comparison_type=ComparisonType.OVERLAP,
            parameters=ComparisonParameters(
                comparison_name="Test Comparison",
                comparison_type=ComparisonType.OVERLAP,
                significance_threshold=0.05
            ),
            overlap_statistics=OverlapStatistics(
                total_pathways=20,
                overlapping_pathways=8,
                overlap_percentage=40.0,
                unique_to_dataset1=6,
                unique_to_dataset2=6
            ),
            correlation_result=CorrelationResult(
                correlation_coefficient=0.75,
                p_value=0.001,
                correlation_type="pearson"
            ),
            timestamp="2024-01-01T00:00:00Z"
        )

    @pytest.fixture
    def sample_visualization_result(self) -> VisualizationResult:
        """Sample visualization result for testing."""
        return VisualizationResult(
            plot_id="test_plot_1",
            plot_type=PlotType.BAR,
            plot_metadata=PlotMetadata(
                title="Test Bar Plot",
                description="A test bar plot visualization",
                x_label="Pathways",
                y_label="P-value",
                width=800,
                height=600
            ),
            output_path="/tmp/test_plot.html",
            interactive_data={"data": "test_data"},
            timestamp="2024-01-01T00:00:00Z"
        )

    @pytest.fixture
    def sample_visualization_parameters(self) -> VisualizationParameters:
        """Sample visualization parameters for testing."""
        return VisualizationParameters(
            plot_title="Test Plot",
            plot_type=PlotType.BAR,
            output_format="html",
            width=800,
            height=600,
            color_scheme="viridis",
            show_legend=True
        )

    @pytest.fixture
    def sample_dashboard_config(self) -> DashboardConfig:
        """Sample dashboard configuration for testing."""
        return DashboardConfig(
            title="Test Dashboard",
            description="A test dashboard configuration",
            layout="grid",
            plots=[
                {
                    "plot_id": "plot1",
                    "plot_type": "bar",
                    "position": {"x": 0, "y": 0, "width": 6, "height": 4}
                },
                {
                    "plot_id": "plot2",
                    "plot_type": "scatter",
                    "position": {"x": 6, "y": 0, "width": 6, "height": 4}
                }
            ],
            theme="light"
        )

    @pytest.fixture
    def sample_multi_omics_data(self) -> Dict[str, pd.DataFrame]:
        """Sample multi-omics data for testing."""
        np.random.seed(42)
        
        # Transcriptomics data
        transcriptomics = pd.DataFrame({
            'Gene': [f'GENE{i}' for i in range(1, 11)],
            'Control_1': np.random.normal(1.0, 0.2, 10),
            'Control_2': np.random.normal(1.0, 0.2, 10),
            'Treatment_1': np.random.normal(2.0, 0.3, 10),
            'Treatment_2': np.random.normal(2.0, 0.3, 10)
        })
        
        # Proteomics data
        proteomics = pd.DataFrame({
            'Protein': [f'PROT{i}' for i in range(1, 11)],
            'Control_1': np.random.normal(1.0, 0.2, 10),
            'Control_2': np.random.normal(1.0, 0.2, 10),
            'Treatment_1': np.random.normal(2.0, 0.3, 10),
            'Treatment_2': np.random.normal(2.0, 0.3, 10)
        })
        
        # Metabolomics data
        metabolomics = pd.DataFrame({
            'Metabolite': [f'MET{i}' for i in range(1, 11)],
            'Control_1': np.random.normal(1.0, 0.2, 10),
            'Control_2': np.random.normal(1.0, 0.2, 10),
            'Treatment_1': np.random.normal(2.0, 0.3, 10),
            'Treatment_2': np.random.normal(2.0, 0.3, 10)
        })
        
        return {
            "transcriptomics": transcriptomics,
            "proteomics": proteomics,
            "metabolomics": metabolomics
        }

    @pytest.fixture
    def sample_time_course_data(self) -> pd.DataFrame:
        """Sample time course data for testing."""
        np.random.seed(42)
        time_points = [0, 1, 2, 4, 8, 12, 24]
        genes = [f'GENE{i}' for i in range(1, 11)]
        
        data = {'Gene': genes}
        for time_point in time_points:
            data[f'T{time_point}h'] = np.random.normal(1.0 + time_point * 0.1, 0.2, 10)
        
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_plugin_data(self) -> Dict[str, Any]:
        """Sample plugin data for testing."""
        return {
            "plugin_name": "TestPlugin",
            "plugin_version": "1.0.0",
            "plugin_type": "analysis",
            "plugin_config": {
                "param1": "value1",
                "param2": 42,
                "param3": True
            },
            "plugin_metadata": {
                "author": "Test Author",
                "description": "A test plugin",
                "license": "MIT"
            }
        }

    @pytest.fixture
    def sample_error_data(self) -> Dict[str, Any]:
        """Sample error data for testing error handling."""
        return {
            "error_type": "ValidationError",
            "error_message": "Invalid input data",
            "error_details": {
                "field": "gene_list",
                "value": None,
                "expected_type": "list"
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }

    @pytest.fixture
    def sample_config_data(self) -> Dict[str, Any]:
        """Sample configuration data for testing."""
        return {
            "api": {
                "base_url": "http://localhost:8000",
                "timeout": 30,
                "retry_attempts": 3
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "pathwaylens",
                "user": "pathwaylens_user"
            },
            "cache": {
                "enabled": True,
                "ttl": 3600,
                "max_size": 1000
            },
            "plugins": {
                "enabled": True,
                "plugin_dirs": ["/path/to/plugins"],
                "auto_load": True
            }
        }

    @pytest.fixture
    def sample_file_paths(self, temp_dir) -> Dict[str, Path]:
        """Sample file paths for testing."""
        return {
            "gene_list": temp_dir / "genes.txt",
            "expression_data": temp_dir / "expression.csv",
            "analysis_result": temp_dir / "analysis_result.json",
            "comparison_result": temp_dir / "comparison_result.json",
            "visualization": temp_dir / "plot.html",
            "config": temp_dir / "config.yml"
        }

    @pytest.fixture
    def sample_validation_data(self) -> Dict[str, Any]:
        """Sample validation data for testing."""
        return {
            "valid_gene_list": ["GENE1", "GENE2", "GENE3"],
            "invalid_gene_list": [None, "", "GENE1"],
            "valid_expression_data": pd.DataFrame({
                'Gene': ['GENE1', 'GENE2'],
                'Control': [1.0, 2.0],
                'Treatment': [1.5, 2.5]
            }),
            "invalid_expression_data": pd.DataFrame({
                'Gene': [None, 'GENE2'],
                'Control': [1.0, None],
                'Treatment': [1.5, 2.5]
            })
        }
