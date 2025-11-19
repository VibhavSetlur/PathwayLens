"""
Regression tests for core PathwayLens functionality.

These tests verify that core functionality remains stable across versions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from pathwaylens_core.data import DatabaseManager
from pathwaylens_core.analysis import (
    ORAEngine,
    GSEAEngine,
    NetworkEngine,
    BayesianEngine
)
from pathwaylens_core.normalization import (
    Normalizer,
    IDConverter,
    ConfidenceCalculator
)
from pathwaylens_core.data.utils.file_utils import FileUtils
from pathwaylens_core.analysis.schemas import DatabaseType
from pathwaylens_core.normalization.schemas import SpeciesType


class TestCoreRegression:
    """Regression tests for core analysis engines."""
    
    @pytest.fixture
    def sample_gene_list(self) -> List[str]:
        """Sample gene list for testing."""
        return [
            "BRCA1", "BRCA2", "TP53", "EGFR", "MYC",
            "AKT1", "PIK3CA", "PTEN", "KRAS", "BRAF"
        ]
    
    @pytest.fixture
    def database_manager(self):
        """Database manager fixture."""
        return DatabaseManager()
    
    @pytest.mark.asyncio
    async def test_ora_engine_regression(
        self,
        sample_gene_list: List[str],
        database_manager: DatabaseManager
    ):
        """Test that ORA engine produces consistent results."""
        engine = ORAEngine(database_manager)
        
        result = await engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.GO,
            species=SpeciesType.HUMAN,
            significance_threshold=0.05
        )
        
        # Verify structure
        assert result is not None
        assert hasattr(result, 'database')
        assert hasattr(result, 'pathways')
        assert hasattr(result, 'total_pathways')
        assert hasattr(result, 'significant_pathways')
        
        # Verify pathways are sorted by p-value
        if len(result.pathways) > 1:
            p_values = [p.p_value for p in result.pathways]
            assert p_values == sorted(p_values)
    
    @pytest.mark.asyncio
    async def test_gsea_engine_regression(
        self,
        sample_gene_list: List[str],
        database_manager: DatabaseManager
    ):
        """Test that GSEA engine produces consistent results."""
        # Create ranked gene list
        ranked_genes = [
            ("BRCA1", 2.5), ("BRCA2", 2.3), ("TP53", 2.1),
            ("EGFR", 1.9), ("MYC", 1.7), ("AKT1", 1.5)
        ]
        
        engine = GSEAEngine(database_manager)
        
        result = await engine.analyze(
            gene_list=[gene for gene, _ in ranked_genes],
            database=DatabaseType.GO,
            species=SpeciesType.HUMAN,
            significance_threshold=0.05,
            permutations=100
        )
        
        # Verify structure
        assert result is not None
        assert hasattr(result, 'database')
        assert hasattr(result, 'pathways')
        assert hasattr(result, 'total_pathways')
    
    @pytest.mark.asyncio
    async def test_normalizer_regression(self, sample_gene_list: List[str]):
        """Test that normalizer produces consistent results."""
        normalizer = Normalizer()
        
        result = await normalizer.normalize_list(
            gene_list=sample_gene_list,
            input_type="symbol",
            output_type="ensembl_gene_id",
            species=SpeciesType.HUMAN
        )
        
        # Verify structure
        assert result is not None
        assert result.total_mapped >= 0
        assert result.total_input == len(sample_gene_list)
        
        # Verify conversion stats
        assert result.mapping_rate >= 0.0
        assert result.total_unmapped >= 0


class TestDataIORegression:
    """Regression tests for data I/O functionality."""
    
    @pytest.fixture
    def sample_csv_data(self, tmp_path: Path) -> Path:
        """Create sample CSV file."""
        df = pd.DataFrame({
            'gene_id': ['BRCA1', 'BRCA2', 'TP53', 'EGFR', 'MYC'],
            'expression': [1.5, 2.3, 1.8, 0.9, 2.1],
            'p_value': [0.01, 0.005, 0.02, 0.15, 0.03]
        })
        file_path = tmp_path / "test_data.csv"
        df.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def sample_tsv_data(self, tmp_path: Path) -> Path:
        """Create sample TSV file."""
        df = pd.DataFrame({
            'gene_id': ['BRCA1', 'BRCA2', 'TP53'],
            'log2fc': [1.2, -0.8, 2.1],
            'p_value': [0.001, 0.05, 0.002]
        })
        file_path = tmp_path / "test_data.tsv"
        df.to_csv(file_path, sep='\t', index=False)
        return file_path
    
    @pytest.mark.asyncio
    async def test_csv_reading_regression(
        self,
        sample_csv_data: Path,
        file_utils: FileUtils
    ):
        """Test that CSV reading produces consistent results."""
        data = await file_utils.read_file(str(sample_csv_data), format='csv')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5
        assert 'gene_id' in data.columns
        assert 'expression' in data.columns
        assert 'p_value' in data.columns
    
    @pytest.mark.asyncio
    async def test_tsv_reading_regression(
        self,
        sample_tsv_data: Path,
        file_utils: FileUtils
    ):
        """Test that TSV reading produces consistent results."""
        data = await file_utils.read_file(str(sample_tsv_data), format='tsv')
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert 'gene_id' in data.columns
        assert 'log2fc' in data.columns
    
    @pytest.fixture
    def file_utils(self) -> FileUtils:
        """File utils fixture."""
        return FileUtils()


class TestFormatDetectionRegression:
    """Regression tests for format detection."""
    
    @pytest.mark.asyncio
    async def test_csv_format_detection(self, tmp_path: Path):
        """Test CSV format detection."""
        file_utils = FileUtils()
        
        # Create CSV file
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        csv_file = tmp_path / "test.csv"
        df.to_csv(csv_file, index=False)
        
        detected_format = await file_utils.detect_file_format(str(csv_file))
        assert detected_format == 'csv'
    
    @pytest.mark.asyncio
    async def test_tsv_format_detection(self, tmp_path: Path):
        """Test TSV format detection."""
        file_utils = FileUtils()
        
        # Create TSV file
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        tsv_file = tmp_path / "test.tsv"
        df.to_csv(tsv_file, sep='\t', index=False)
        
        detected_format = await file_utils.detect_file_format(str(tsv_file))
        assert detected_format == 'tsv'


class TestBackwardsCompatibility:
    """Test backwards compatibility of APIs."""
    
    def test_schema_compatibility(self):
        """Test that schemas remain backwards compatible."""
        from pathwaylens_core.analysis.schemas import (
            PathwayResult,
            DatabaseResult,
            AnalysisParameters
        )
        
        # Verify required fields exist
        pathway_fields = PathwayResult.model_fields.keys()
        assert 'pathway_id' in pathway_fields
        assert 'pathway_name' in pathway_fields
        assert 'p_value' in pathway_fields
        
        db_result_fields = DatabaseResult.model_fields.keys()
        assert 'database' in db_result_fields
        assert 'pathways' in db_result_fields
        assert 'total_pathways' in db_result_fields



