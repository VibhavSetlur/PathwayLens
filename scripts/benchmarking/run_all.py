
import time
import psutil
import os
import sys
import pandas as pd
import numpy as np
import asyncio
from typing import List
from loguru import logger

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from pathwaylens import PathwayLens
from pathwaylens_core.analysis.schemas import DatabaseType
from pathwaylens_core.normalization.schemas import SpeciesType

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

async def benchmark_ora(num_genes: int, database: str = "kegg"):
    print(f"--- Benchmarking ORA with {num_genes} genes ({database}) ---")
    
    # Generate synthetic gene list (Entrez IDs)
    # Using a range that likely overlaps with real genes
    gene_list = [str(i) for i in range(1, num_genes + 1)]
    
    pl = PathwayLens()
    
    start_mem = get_memory_usage()
    start_time = time.time()
    
    try:
        # We use use_gprofiler=False to test internal engine speed if possible,
        # but PathwayLens defaults might use API.
        # For benchmarking, we want to test the default path or explicit internal if we loaded DBs.
        # Since we haven't pre-loaded DBs in this script, it might hit the API or fail if offline.
        # Let's assume we want to test the 'analyze' function end-to-end.
        
        results = await pl.analyze(
            gene_list=gene_list,
            omic_type="transcriptomics",
            data_type="bulk",
            databases=[database],
            species="human"
        )
        
        end_time = time.time()
        end_mem = get_memory_usage()
        
        print(f"Runtime: {end_time - start_time:.4f} seconds")
        print(f"Memory Delta: {end_mem - start_mem:.2f} MB")
        
        # Handle dict or object
        if isinstance(results, dict):
            total = results.get('total_pathways', 0)
            sig = results.get('significant_pathways', 0)
            # Check nested database results if top level is analysis result
            if 'database_results' in results:
                # database_results is a Dict[str, DatabaseResult]
                for db_res in results['database_results'].values():
                    if isinstance(db_res, dict):
                        total = db_res.get('total_pathways', 0)
                        sig = db_res.get('significant_pathways', 0)
                        break # Just take the first one for now
        else:
            total = results.total_pathways
            sig = results.significant_pathways
            
        print(f"Pathways Found: {total}")
        print(f"Significant: {sig}")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

async def run_benchmarks():
    print("Starting PathwayLens Benchmarks...")
    print(f"System: {os.uname().sysname} {os.uname().release}")
    print(f"Python: {sys.version.split()[0]}")
    
    # ORA Benchmarks
    await benchmark_ora(100, "kegg")
    await benchmark_ora(1000, "kegg")
    # await benchmark_ora(10000, "kegg") # Uncomment for longer run

if __name__ == "__main__":
    asyncio.run(run_benchmarks())
