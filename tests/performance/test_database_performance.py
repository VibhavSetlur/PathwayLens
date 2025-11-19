"""
Performance tests for database operations.
"""

import pytest
import time
import asyncio
from typing import List

from pathwaylens_core.data.database_manager import DatabaseManager


@pytest.mark.performance
@pytest.mark.slow
class TestDatabasePerformance:
    """Performance benchmarks for database operations."""

    @pytest.fixture
    def database_manager(self):
        """Create database manager."""
        return DatabaseManager()

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_pathway_query_performance(self, database_manager):
        """Benchmark pathway query performance."""
        start_time = time.time()
        pathways = await database_manager.get_pathways(
            database="kegg",
            species="human",
            limit=100
        )
        elapsed = time.time() - start_time
        
        assert pathways is not None
        assert elapsed < 5.0, f"Pathway query took {elapsed:.2f}s, expected <5s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_gene_pathway_query_performance(self, database_manager):
        """Benchmark gene-pathway query performance."""
        gene_ids = [f"ENSG{i:012d}" for i in range(1, 101)]
        
        start_time = time.time()
        results = await database_manager.get_gene_pathways(
            gene_ids=gene_ids,
            database="kegg",
            species="human"
        )
        elapsed = time.time() - start_time
        
        assert results is not None
        assert elapsed < 10.0, f"Gene-pathway query took {elapsed:.2f}s, expected <10s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_batch_gene_pathway_query(self, database_manager):
        """Benchmark batch gene-pathway query performance."""
        gene_ids = [f"ENSG{i:012d}" for i in range(1, 1001)]
        
        start_time = time.time()
        results = await database_manager.get_gene_pathways_batch(
            gene_ids=gene_ids,
            database="kegg",
            species="human",
            batch_size=100
        )
        elapsed = time.time() - start_time
        
        assert results is not None
        # Batch processing should be more efficient
        assert elapsed < 30.0, f"Batch query took {elapsed:.2f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_parallel_queries(self, database_manager):
        """Test performance of parallel database queries."""
        async def query_pathways():
            return await database_manager.get_pathways(
                database="kegg",
                species="human",
                limit=50
            )
        
        async def run_parallel():
            tasks = [query_pathways() for _ in range(5)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time
            return results, elapsed
        
        results, elapsed = await run_parallel()
        
        assert len(results) == 5
        assert all(r is not None for r in results)
        assert elapsed < 15.0, f"Parallel queries took {elapsed:.2f}s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_cached_query_performance(self, database_manager):
        """Benchmark cached query performance."""
        # First query (cache miss)
        start_time = time.time()
        pathways1 = await database_manager.get_pathways(
            database="kegg",
            species="human",
            limit=100
        )
        elapsed_first = time.time() - start_time
        
        # Second query (cache hit)
        start_time = time.time()
        pathways2 = await database_manager.get_pathways(
            database="kegg",
            species="human",
            limit=100
        )
        elapsed_second = time.time() - start_time
        
        assert pathways1 is not None
        assert pathways2 is not None
        # Cached query should be faster
        assert elapsed_second < elapsed_first or elapsed_second < 1.0, \
            f"Cached query ({elapsed_second:.2f}s) should be faster than first ({elapsed_first:.2f}s)"



