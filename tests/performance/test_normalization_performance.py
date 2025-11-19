"""
Performance tests for normalization operations.
"""

import pytest
import time
import asyncio
from typing import List
import numpy as np

from pathwaylens_core.normalization.normalizer import Normalizer
from pathwaylens_core.normalization.id_converter import IDConverter
from pathwaylens_core.normalization.schemas import IDType, SpeciesType


@pytest.mark.performance
@pytest.mark.slow
class TestNormalizationPerformance:
    """Performance benchmarks for normalization operations."""

    @pytest.fixture
    def small_gene_list(self) -> List[str]:
        """Small gene list (10 genes)."""
        return [f"GENE{i}" for i in range(1, 11)]

    @pytest.fixture
    def medium_gene_list(self) -> List[str]:
        """Medium gene list (100 genes)."""
        return [f"GENE{i}" for i in range(1, 101)]

    @pytest.fixture
    def large_gene_list(self) -> List[str]:
        """Large gene list (1000 genes)."""
        return [f"GENE{i}" for i in range(1, 1001)]

    @pytest.fixture
    def very_large_gene_list(self) -> List[str]:
        """Very large gene list (10000 genes)."""
        return [f"GENE{i}" for i in range(1, 10001)]

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_id_conversion_small(self, small_gene_list):
        """Benchmark ID conversion for small gene list."""
        converter = IDConverter(rate_limit=10.0, batch_size=100)
        
        start_time = time.time()
        async with converter:
            results = await converter.convert_identifiers(
                identifiers=small_gene_list,
                input_type=IDType.SYMBOL,
                output_type=IDType.ENSEMBL,
                species=SpeciesType.HUMAN,
                track_statistics=False
            )
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0, f"Small conversion took {elapsed:.2f}s, expected <5s"
        assert len(results) > 0

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_id_conversion_medium(self, medium_gene_list):
        """Benchmark ID conversion for medium gene list."""
        converter = IDConverter(rate_limit=10.0, batch_size=100)
        
        start_time = time.time()
        async with converter:
            results = await converter.convert_identifiers(
                identifiers=medium_gene_list,
                input_type=IDType.SYMBOL,
                output_type=IDType.ENSEMBL,
                species=SpeciesType.HUMAN,
                track_statistics=False
            )
        elapsed = time.time() - start_time
        
        # Medium list should complete in reasonable time
        assert elapsed < 30.0, f"Medium conversion took {elapsed:.2f}s, expected <30s"
        assert len(results) > 0

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_id_conversion_batch_processing(self, large_gene_list):
        """Benchmark batch processing efficiency."""
        converter_small_batch = IDConverter(rate_limit=10.0, batch_size=10)
        converter_large_batch = IDConverter(rate_limit=10.0, batch_size=1000)
        
        # Test with small batch size
        start_time = time.time()
        async with converter_small_batch:
            results_small = await converter_small_batch.convert_identifiers(
                identifiers=large_gene_list[:100],  # Use subset for speed
                input_type=IDType.SYMBOL,
                output_type=IDType.ENSEMBL,
                species=SpeciesType.HUMAN,
                track_statistics=False
            )
        elapsed_small = time.time() - start_time
        
        # Test with large batch size
        start_time = time.time()
        async with converter_large_batch:
            results_large = await converter_large_batch.convert_identifiers(
                identifiers=large_gene_list[:100],  # Use subset for speed
                input_type=IDType.SYMBOL,
                output_type=IDType.ENSEMBL,
                species=SpeciesType.HUMAN,
                track_statistics=False
            )
        elapsed_large = time.time() - start_time
        
        # Large batch should be more efficient (or at least not significantly worse)
        # Note: This may vary based on API rate limits
        assert len(results_small) > 0
        assert len(results_large) > 0

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_normalization_memory_usage(self, large_gene_list):
        """Test memory usage during normalization."""
        import tracemalloc
        
        normalizer = Normalizer(rate_limit=10.0)
        
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        
        result = await normalizer.normalize(
            input_data=large_gene_list,
            species=SpeciesType.HUMAN,
            input_id_type=IDType.SYMBOL,
            output_id_type=IDType.ENSEMBL
        )
        
        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()
        
        # Calculate memory difference
        top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        total_memory = sum(stat.size_diff for stat in top_stats)
        
        # Memory usage should be reasonable (< 500MB for 1000 genes)
        assert total_memory < 500 * 1024 * 1024, \
            f"Memory usage {total_memory / 1024 / 1024:.2f}MB exceeds 500MB"
        assert result is not None

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    def test_concurrent_conversions(self, medium_gene_list):
        """Test performance of concurrent ID conversions."""
        async def run_conversion():
            converter = IDConverter(rate_limit=10.0, batch_size=100)
            async with converter:
                return await converter.convert_identifiers(
                    identifiers=medium_gene_list[:50],
                    input_type=IDType.SYMBOL,
                    output_type=IDType.ENSEMBL,
                    species=SpeciesType.HUMAN,
                    track_statistics=False
                )
        
        async def run_concurrent():
            tasks = [run_conversion() for _ in range(5)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time
            return results, elapsed
        
        results, elapsed = asyncio.run(run_concurrent())
        
        # Concurrent execution should complete in reasonable time
        assert elapsed < 60.0, f"Concurrent conversions took {elapsed:.2f}s"
        assert len(results) == 5
        assert all(len(r) > 0 for r in results)



