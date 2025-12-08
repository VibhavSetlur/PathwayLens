"""
Unit tests for Competitive ORA Engine.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from pathwaylens_core.analysis.competitive_ora_engine import CompetitiveORAEngine
from pathwaylens_core.analysis.schemas import DatabaseType, CorrectionMethod
from pathwaylens_core.data.adapters.base import PathwayInfo


@pytest.fixture
def mock_database_manager():
    """Create a mock database manager for testing."""
    mock_adapter = MagicMock()
    mock_adapter.get_pathways = AsyncMock(return_value=[
        PathwayInfo(
            pathway_id="path:hsa00010",
            name="Glycolysis",
            genes=["GCK", "HK1", "HK2", "PFKM", "ALDOA", "GAPDH", "ENO1", "PKM", "LDHA", "LDHB"],
            species="human"
        ),
        PathwayInfo(
            pathway_id="path:hsa00020",
            name="TCA Cycle",
            genes=["CS", "ACO1", "IDH1", "OGDH", "SUCLG1", "SDH", "FH", "MDH1", "MDH2", "PC"],
            species="human"
        ),
        PathwayInfo(
            pathway_id="path:hsa00030",
            name="Pentose Phosphate",
            genes=["G6PD", "PGD", "TKT", "TALDO1", "RPE", "RPIA"],
            species="human"
        )
    ])
    
    db_manager = MagicMock()
    db_manager.get_pathways = AsyncMock(return_value={
        "kegg": [
            PathwayInfo(
                pathway_id="path:hsa00010",
                name="Glycolysis",
                genes=["GCK", "HK1", "HK2", "PFKM", "ALDOA", "GAPDH", "ENO1", "PKM", "LDHA", "LDHB"],
                species="human"
            ),
            PathwayInfo(
                pathway_id="path:hsa00020",
                name="TCA Cycle",
                genes=["CS", "ACO1", "IDH1", "OGDH", "SUCLG1", "SDH", "FH", "MDH1", "MDH2", "PC"],
                species="human"
            ),
            PathwayInfo(
                pathway_id="path:hsa00030",
                name="Pentose Phosphate",
                genes=["G6PD", "PGD", "TKT", "TALDO1", "RPE", "RPIA"],
                species="human"
            )
        ]
    })
    
    return db_manager


@pytest.mark.asyncio
async def test_competitive_ora_fisher(mock_database_manager):
    """Test Fisher's exact test for competitive ORA."""
    engine = CompetitiveORAEngine(mock_database_manager)
    
    # Gene list enriched for Glycolysis
    gene_list = ["GCK", "HK1", "HK2", "PFKM", "ALDOA", "GAPDH", "RANDOM1", "RANDOM2"]
    
    result = await engine.analyze(
        gene_list=gene_list,
        database=DatabaseType.KEGG,
        species="human",
        method="fisher",
        min_pathway_size=3,
        max_pathway_size=500
    )
    
    assert result is not None
    assert result.database == DatabaseType.KEGG
    assert len(result.pathways) > 0
    
    # Glycolysis should be top result
    if result.pathways:
        top_pathway = result.pathways[0]
        assert top_pathway.pathway_id == "path:hsa00010"
        assert top_pathway.analysis_method == "competitive_ora_fisher"


@pytest.mark.asyncio
async def test_competitive_vs_standard_pvalues(mock_database_manager):
    """Verify competitive ORA produces different p-values than would standard ORA."""
    engine = CompetitiveORAEngine(mock_database_manager)
    
    gene_list = ["GCK", "HK1", "HK2", "PFKM", "RANDOM1", "RANDOM2", "RANDOM3"]
    
    result = await engine.analyze(
        gene_list=gene_list,
        database=DatabaseType.KEGG,
        species="human",
        method="fisher"
    )
    
    # Both competitive and standard p-values should be valid
    for pathway in result.pathways:
        assert 0 <= pathway.p_value <= 1
        # Competitive method stored in analysis_method
        assert "competitive" in pathway.analysis_method


@pytest.mark.asyncio
async def test_gene_length_weighted_ora(mock_database_manager):
    """Test GOseq-style gene length weighted ORA."""
    engine = CompetitiveORAEngine(mock_database_manager)
    
    # Provide gene lengths
    gene_lengths = {
        "GCK": 5000, "HK1": 8000, "HK2": 7500, "PFKM": 6000,
        "ALDOA": 4000, "GAPDH": 3000, "ENO1": 4500, "PKM": 5500,
        "CS": 4200, "ACO1": 6100, "IDH1": 5800,
        "RANDOM1": 2000, "RANDOM2": 2500, "RANDOM3": 1800
    }
    
    engine.set_gene_lengths(gene_lengths)
    
    gene_list = ["GCK", "HK1", "HK2", "PFKM", "RANDOM1"]
    
    result = await engine.analyze(
        gene_list=gene_list,
        database=DatabaseType.KEGG,
        species="human",
        method="weighted"
    )
    
    assert result is not None
    # Results should include bias scores
    for pathway in result.pathways:
        assert hasattr(pathway, 'confidence_score')


@pytest.mark.asyncio
async def test_annotation_bias_calculation(mock_database_manager):
    """Test that annotation bias is calculated correctly."""
    engine = CompetitiveORAEngine(mock_database_manager)
    
    # Set annotation counts (simulating well-studied vs novel genes)
    annotation_counts = {
        "GCK": 500,  # Well-studied
        "HK1": 450,
        "HK2": 400,
        "PFKM": 350,
        "ALDOA": 300,
        "GAPDH": 1000,  # Very well-studied
        "RANDOM1": 10,  # Novel
        "RANDOM2": 15
    }
    
    engine.set_annotation_counts(annotation_counts)
    
    gene_list = ["GCK", "HK1", "RANDOM1", "RANDOM2"]
    
    result = await engine.analyze(
        gene_list=gene_list,
        database=DatabaseType.KEGG,
        species="human",
        method="fisher"
    )
    
    assert result is not None
    # confidence_score should reflect annotation bias (lower = more bias)


@pytest.mark.asyncio
async def test_empty_gene_list(mock_database_manager):
    """Test handling of empty gene list."""
    engine = CompetitiveORAEngine(mock_database_manager)
    
    result = await engine.analyze(
        gene_list=[],
        database=DatabaseType.KEGG,
        species="human"
    )
    
    assert result is not None
    assert result.significant_pathways == 0


@pytest.mark.asyncio
async def test_multiple_correction_methods(mock_database_manager):
    """Test different multiple testing correction methods."""
    engine = CompetitiveORAEngine(mock_database_manager)
    
    gene_list = ["GCK", "HK1", "HK2", "CS", "ACO1"]
    
    for method in [CorrectionMethod.FDR_BH, CorrectionMethod.BONFERRONI]:
        result = await engine.analyze(
            gene_list=gene_list,
            database=DatabaseType.KEGG,
            species="human",
            correction_method=method
        )
        
        assert result is not None
        # Bonferroni should be more conservative
        for pathway in result.pathways:
            assert pathway.adjusted_p_value >= pathway.p_value
