"""
Gold Standard validation test for PathwayLens.
Ensures statistical accuracy on known datasets.
"""

import pytest
import pandas as pd
import asyncio
from pathlib import Path
from pathwaylens_core.analysis import AnalysisEngine
from pathwaylens_core.analysis.schemas import AnalysisParameters, AnalysisType, DatabaseType
from pathwaylens_core.types import OmicType, DataType
from pathwaylens_core.species import Species

@pytest.mark.asyncio
async def test_gold_standard_ora():
    """
    Test ORA on a synthetic dataset with known enrichment.
    We create a gene list that is perfectly enriched for a specific KEGG pathway.
    """
    # 1. Setup
    # Let's assume we are looking for "Glycolysis / Gluconeogenesis" (hsa00010)
    # We will pick genes from this pathway.
    # Note: This test requires the KEGG database to be accessible or cached.
    # For CI/CD, we would mock the database adapter, but for "Gold Standard" we want end-to-end.
    
    # Mocking the database manager to return a known pathway for deterministic testing
    from pathwaylens_core.data import DatabaseManager
    from pathwaylens_core.data.adapters.base import PathwayInfo
    from unittest.mock import MagicMock, AsyncMock
    
    # Create a mock adapter
    mock_adapter = MagicMock()
    mock_adapter.get_pathways = AsyncMock(return_value=[
        PathwayInfo(
            pathway_id="path:hsa00010",
            name="Glycolysis / Gluconeogenesis",
            genes=["HK1", "HK2", "HK3", "GCK", "GPI", "PFKM", "PFKP", "PFKL", "ALDOA", "ALDOB"],
            species="human"
        ),
        PathwayInfo(
            pathway_id="path:hsa00020",
            name="Citrate cycle (TCA cycle)",
            genes=["CS", "ACO1", "ACO2", "IDH1", "IDH2", "IDH3A", "IDH3B", "IDH3G", "OGDH", "DLST"],
            species="human"
        )
    ])
    
    # Mock database manager
    db_manager = DatabaseManager()
    db_manager.adapters["kegg"] = mock_adapter
    db_manager.availability_status["kegg"] = True
    
    # 2. Create input data
    # Input list contains 8 genes from Glycolysis and 2 random genes
    input_genes = ["HK1", "HK2", "HK3", "GCK", "GPI", "PFKM", "PFKP", "PFKL", "RANDOM1", "RANDOM2"]
    
    # 3. Run Analysis
    engine = AnalysisEngine(database_manager=db_manager, use_gprofiler=False)
    
    params = AnalysisParameters(
        analysis_type=AnalysisType.ORA,
        databases=[DatabaseType.KEGG],
        species="human",
        min_pathway_size=5,
        max_pathway_size=500,
        significance_threshold=0.05,
        background_size=20000, # Assume 20k background genes
        omic_type=OmicType.TRANSCRIPTOMICS,
        data_type=DataType.BULK
    )
    
    result = await engine.analyze(
        input_data=input_genes,
        parameters=params
    )
    
    # 4. Verification
    assert result.significant_pathways > 0
    
    kegg_result = result.database_results["kegg"]
    top_pathway = kegg_result.pathways[0]
    
    assert top_pathway.pathway_id == "path:hsa00010"
    assert top_pathway.pathway_name == "Glycolysis / Gluconeogenesis"
    assert top_pathway.overlap_count == 8
    assert top_pathway.p_value < 1e-5 # Should be very significant
    
    print(f"Gold Standard Test Passed: {top_pathway.pathway_name} p={top_pathway.p_value}")

if __name__ == "__main__":
    asyncio.run(test_gold_standard_ora())
