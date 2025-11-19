"""
Performance tests for analysis engines.
"""

import pytest
import time
import asyncio
from typing import List

from pathwaylens_core.analysis.engine import AnalysisEngine
from pathwaylens_core.analysis.schemas import (
    AnalysisType, DatabaseType, AnalysisParameters
)
from pathwaylens_core.data.database_manager import DatabaseManager


@pytest.mark.performance
@pytest.mark.slow
class TestAnalysisPerformance:
    """Performance benchmarks for analysis operations."""

    @pytest.fixture
    def database_manager(self):
        """Create database manager."""
        return DatabaseManager()

    @pytest.fixture
    def analysis_engine(self, database_manager):
        """Create analysis engine."""
        return AnalysisEngine(database_manager)

    @pytest.fixture
    def small_gene_list(self) -> List[str]:
        """Small gene list."""
        return [f"GENE{i}" for i in range(1, 21)]

    @pytest.fixture
    def medium_gene_list(self) -> List[str]:
        """Medium gene list."""
        return [f"GENE{i}" for i in range(1, 101)]

    @pytest.fixture
    def large_gene_list(self) -> List[str]:
        """Large gene list."""
        return [f"GENE{i}" for i in range(1, 1001)]

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_ora_analysis_performance(self, analysis_engine, small_gene_list):
        """Benchmark ORA analysis performance."""
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        start_time = time.time()
        result = await analysis_engine.analyze(
            input_data=small_gene_list,
            parameters=parameters
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert elapsed < 10.0, f"ORA analysis took {elapsed:.2f}s, expected <10s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_gsea_analysis_performance(self, analysis_engine, small_gene_list):
        """Benchmark GSEA analysis performance."""
        # Create ranked gene list for GSEA
        ranked_genes = {gene: float(i) for i, gene in enumerate(small_gene_list, 1)}
        
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.GSEA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        start_time = time.time()
        result = await analysis_engine.analyze(
            input_data=ranked_genes,
            parameters=parameters
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert elapsed < 15.0, f"GSEA analysis took {elapsed:.2f}s, expected <15s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_gsva_analysis_performance(self, analysis_engine, medium_gene_list):
        """Benchmark GSVA analysis performance."""
        import pandas as pd
        import numpy as np
        
        # Create expression matrix
        np.random.seed(42)
        expression_data = pd.DataFrame({
            'Gene': medium_gene_list,
            **{f'Sample_{i}': np.random.normal(1.0, 0.2, len(medium_gene_list))
               for i in range(1, 11)}
        })
        
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.GSVA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        start_time = time.time()
        result = await analysis_engine.analyze(
            input_data=expression_data,
            parameters=parameters
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert elapsed < 30.0, f"GSVA analysis took {elapsed:.2f}s, expected <30s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_multi_database_analysis_performance(
        self, analysis_engine, small_gene_list
    ):
        """Benchmark analysis with multiple databases."""
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG, DatabaseType.REACTOME, DatabaseType.GO],
            species="human",
            significance_threshold=0.05
        )
        
        start_time = time.time()
        result = await analysis_engine.analyze(
            input_data=small_gene_list,
            parameters=parameters
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert len(result.database_results) >= 1
        # Multi-database should take longer but still be reasonable
        assert elapsed < 30.0, f"Multi-database analysis took {elapsed:.2f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_concurrent_analyses(self, analysis_engine, small_gene_list):
        """Test performance of concurrent analyses."""
        parameters = AnalysisParameters(
            analysis_type=AnalysisType.ORA,
            databases=[DatabaseType.KEGG],
            species="human",
            significance_threshold=0.05
        )
        
        async def run_analysis():
            return await analysis_engine.analyze(
                input_data=small_gene_list,
                parameters=parameters
            )
        
        async def run_concurrent():
            tasks = [run_analysis() for _ in range(3)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time
            return results, elapsed
        
        results, elapsed = await run_concurrent()
        
        assert len(results) == 3
        assert all(r is not None for r in results)
        assert elapsed < 30.0, f"Concurrent analyses took {elapsed:.2f}s"



