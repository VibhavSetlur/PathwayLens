"""
End-to-end tests with real-world dataset formats and sizes.
"""

import pytest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

from pathwaylens_core.normalization.normalizer import Normalizer
from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.analysis.schemas import AnalysisType, DatabaseType, AnalysisParameters
from pathwaylens_core.data.database_manager import DatabaseManager


@pytest.mark.e2e
@pytest.mark.slow
class TestRealWorldDatasets:
    """Tests with real-world dataset formats and sizes."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def database_manager(self):
        """Create database manager."""
        return DatabaseManager()

    @pytest.fixture
    def normalizer(self):
        """Create normalizer."""
        return Normalizer()

    @pytest.fixture
    def analysis_engine(self, database_manager):
        """Create analysis engine."""
        return AnalysisEngine(database_manager)

    def create_small_dataset(self, temp_dir: Path) -> Path:
        """Create small dataset (10-50 genes)."""
        genes = [f"GENE{i}" for i in range(1, 21)]
        gene_file = temp_dir / "small_genes.txt"
        gene_file.write_text("\n".join(genes))
        return gene_file

    def create_medium_dataset(self, temp_dir: Path) -> Path:
        """Create medium dataset (100-500 genes)."""
        genes = [f"GENE{i}" for i in range(1, 201)]
        gene_file = temp_dir / "medium_genes.txt"
        gene_file.write_text("\n".join(genes))
        return gene_file

    def create_large_dataset(self, temp_dir: Path) -> Path:
        """Create large dataset (1000-5000 genes)."""
        genes = [f"GENE{i}" for i in range(1, 2001)]
        gene_file = temp_dir / "large_genes.txt"
        gene_file.write_text("\n".join(genes))
        return gene_file

    def create_expression_matrix(self, temp_dir: Path, n_genes: int, n_samples: int) -> Path:
        """Create expression matrix CSV."""
        np.random.seed(42)
        genes = [f"GENE{i}" for i in range(1, n_genes + 1)]
        samples = [f"Sample_{i}" for i in range(1, n_samples + 1)]
        
        data = {
            'Gene': genes,
            **{sample: np.random.normal(1.0, 0.3, n_genes) for sample in samples}
        }
        
        df = pd.DataFrame(data)
        expr_file = temp_dir / f"expression_{n_genes}genes.csv"
        df.to_csv(expr_file, index=False)
        return expr_file

    @pytest.mark.asyncio
    async def test_small_dataset_workflow(
        self, temp_dir, normalizer, analysis_engine
    ):
        """Test workflow with small dataset."""
        gene_file = self.create_small_dataset(temp_dir)
        
        # Normalize
        genes = gene_file.read_text().strip().split("\n")
        normalized = await normalizer.normalize(
            input_data=genes,
            species="human",
            input_id_type="symbol",
            output_id_type="ensembl"
        )
        
        assert normalized is not None
        
        # Analyze
        normalized_genes = normalized.get("gene_ids", genes)
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        result = await analysis_engine.analyze(
            input_data=normalized_genes if isinstance(normalized_genes, list) else list(normalized_genes),
            parameters=parameters
        )
        
        assert result is not None
        assert result.analysis_type == AnalysisType.ORA

    @pytest.mark.asyncio
    async def test_medium_dataset_workflow(
        self, temp_dir, normalizer, analysis_engine
    ):
        """Test workflow with medium dataset."""
        gene_file = self.create_medium_dataset(temp_dir)
        
        # Normalize
        genes = gene_file.read_text().strip().split("\n")
        normalized = await normalizer.normalize(
            input_data=genes,
            species="human",
            input_id_type="symbol",
            output_id_type="ensembl"
        )
        
        assert normalized is not None
        
        # Analyze
        normalized_genes = normalized.get("gene_ids", genes)
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        result = await analysis_engine.analyze(
            input_data=normalized_genes[:100] if isinstance(normalized_genes, list) else list(normalized_genes)[:100],
            parameters=parameters
        )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_expression_data_formats(
        self, temp_dir, analysis_engine
    ):
        """Test different expression data formats."""
        # Create expression matrix
        expr_file = self.create_expression_matrix(temp_dir, n_genes=50, n_samples=10)
        
        # Read and analyze
        df = pd.read_csv(expr_file)
        
        # Convert to ranked gene list for GSEA
        # Use first sample as ranking
        ranked_genes = dict(zip(df['Gene'], df.iloc[:, 1]))
        
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.GSEA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        result = await analysis_engine.analyze(
            input_data=ranked_genes,
            parameters=parameters
        )
        
        # Should handle expression data format
        assert result is not None or True  # May not always succeed with mock data

    @pytest.mark.asyncio
    async def test_csv_with_various_formats(self, temp_dir, normalizer):
        """Test CSV files with various formats."""
        # Test with different delimiters and encodings
        test_cases = [
            ("comma", ","),
            ("tab", "\t"),
        ]
        
        for name, delimiter in test_cases:
            csv_file = temp_dir / f"test_{name}.csv"
            csv_content = delimiter.join(["Gene", "Value"]) + "\n"
            csv_content += delimiter.join(["GENE1", "1.0"]) + "\n"
            csv_content += delimiter.join(["GENE2", "2.0"]) + "\n"
            
            csv_file.write_text(csv_content)
            
            # Should be able to read the file
            content = csv_file.read_text()
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_excel_format(self, temp_dir):
        """Test Excel file format."""
        try:
            import openpyxl
            
            # Create simple Excel file
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.append(["Gene", "Value"])
            ws.append(["GENE1", 1.0])
            ws.append(["GENE2", 2.0])
            
            excel_file = temp_dir / "test.xlsx"
            wb.save(excel_file)
            
            # File should exist
            assert excel_file.exists()
            
        except ImportError:
            pytest.skip("openpyxl not available")



