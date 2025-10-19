"""
Unit tests for the GSEA engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.analysis.gsea_engine import GSEAEngine
from pathwaylens_core.analysis.schemas import (
    DatabaseType, PathwayResult, DatabaseResult, CorrectionMethod
)
from pathwaylens_core.data import DatabaseManager


class TestGSEAEngine:
    """Test cases for the GSEAEngine class."""

    @pytest.fixture
    def gsea_engine(self):
        """Create a GSEAEngine instance for testing."""
        db_manager = Mock(spec=DatabaseManager)
        return GSEAEngine(database_manager=db_manager)

    @pytest.fixture
    def sample_gene_list(self):
        """Create a sample gene list."""
        return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

    @pytest.fixture
    def sample_expression_data(self):
        """Create sample expression data."""
        return pd.DataFrame({
            "GENE1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "GENE2": [2.0, 4.0, 6.0, 8.0, 10.0],
            "GENE3": [0.5, 1.0, 1.5, 2.0, 2.5],
            "GENE4": [3.0, 6.0, 9.0, 12.0, 15.0],
            "GENE5": [1.5, 3.0, 4.5, 6.0, 7.5]
        })

    @pytest.fixture
    def sample_pathway_data(self):
        """Create sample pathway data."""
        return {
            "PATH:00010": {
                "name": "Glycolysis",
                "genes": ["GENE1", "GENE2", "GENE6", "GENE7", "GENE8"],
                "size": 5
            },
            "PATH:00020": {
                "name": "TCA Cycle",
                "genes": ["GENE3", "GENE4", "GENE9", "GENE10"],
                "size": 4
            },
            "PATH:00030": {
                "name": "Oxidative Phosphorylation",
                "genes": ["GENE11", "GENE12", "GENE13", "GENE14", "GENE15", "GENE16"],
                "size": 6
            }
        }

    @pytest.fixture
    def sample_pathway_results(self):
        """Create sample pathway results."""
        return [
            PathwayResult(
                pathway_id="PATH:00010",
                pathway_name="Glycolysis",
                p_value=0.001,
                adjusted_p_value=0.01,
                gene_overlap=["GENE1", "GENE2"],
                gene_overlap_count=2,
                pathway_size=5
            ),
            PathwayResult(
                pathway_id="PATH:00020",
                pathway_name="TCA Cycle",
                p_value=0.005,
                adjusted_p_value=0.02,
                gene_overlap=["GENE3", "GENE4"],
                gene_overlap_count=2,
                pathway_size=4
            )
        ]

    def test_init(self, gsea_engine):
        """Test GSEAEngine initialization."""
        assert gsea_engine.database_manager is not None
        assert gsea_engine.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_basic(self, gsea_engine, sample_gene_list, sample_pathway_data):
        """Test basic GSEA analysis."""
        # Mock database manager
        gsea_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
        gsea_engine.database_manager.get_background_genes = AsyncMock(return_value=set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]))
        
        # Run analysis
        result = await gsea_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert result.database_name == "KEGG"
        assert result.database_type == DatabaseType.KEGG
        assert result.species == "human"
        assert len(result.pathway_results) > 0
        
        # Verify pathway results
        for pathway_result in result.pathway_results:
            assert isinstance(pathway_result, PathwayResult)
            assert pathway_result.pathway_id is not None
            assert pathway_result.pathway_name is not None
            assert 0 <= pathway_result.p_value <= 1
            assert pathway_result.gene_overlap_count > 0

    @pytest.mark.asyncio
    async def test_analyze_with_parameters(self, gsea_engine, sample_gene_list, sample_pathway_data):
        """Test GSEA analysis with custom parameters."""
        # Mock database manager
        gsea_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
        gsea_engine.database_manager.get_background_genes = AsyncMock(return_value=set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]))
        
        # Run analysis with custom parameters
        result = await gsea_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            significance_threshold=0.01,
            correction_method=CorrectionMethod.BONFERRONI,
            permutations=500,
            min_size=3,
            max_size=10
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert len(result.pathway_results) > 0

    @pytest.mark.asyncio
    async def test_analyze_empty_gene_list(self, gsea_engine, sample_pathway_data):
        """Test GSEA analysis with empty gene list."""
        # Mock database manager
        gsea_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
        gsea_engine.database_manager.get_background_genes = AsyncMock(return_value=set())
        
        # Run analysis with empty gene list
        result = await gsea_engine.analyze(
            gene_list=[],
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert len(result.pathway_results) == 0

    @pytest.mark.asyncio
    async def test_analyze_no_pathways(self, gsea_engine, sample_gene_list):
        """Test GSEA analysis with no pathways."""
        # Mock database manager to return empty pathway data
        gsea_engine.database_manager.get_pathway_data = AsyncMock(return_value={})
        gsea_engine.database_manager.get_background_genes = AsyncMock(return_value=set(sample_gene_list))
        
        # Run analysis
        result = await gsea_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert len(result.pathway_results) == 0

    @pytest.mark.asyncio
    async def test_analyze_database_error(self, gsea_engine, sample_gene_list):
        """Test GSEA analysis when database raises an error."""
        # Mock database manager to raise an error
        gsea_engine.database_manager.get_pathway_data = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(Exception, match="Database error"):
            await gsea_engine.analyze(
                gene_list=sample_gene_list,
                database=DatabaseType.KEGG,
                species="human"
            )

    def test_calculate_enrichment_score(self, gsea_engine, sample_gene_list, sample_pathway_data):
        """Test enrichment score calculation."""
        pathway_id = "PATH:00010"
        pathway_info = sample_pathway_data[pathway_id]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"])
        
        es = gsea_engine._calculate_enrichment_score(
            pathway_id=pathway_id,
            pathway_info=pathway_info,
            gene_list=sample_gene_list,
            background_genes=background_genes
        )
        
        assert isinstance(es, float)
        assert -1 <= es <= 1

    def test_calculate_enrichment_score_no_overlap(self, gsea_engine, sample_pathway_data):
        """Test enrichment score calculation with no gene overlap."""
        pathway_id = "PATH:00030"  # No overlap with sample genes
        pathway_info = sample_pathway_data[pathway_id]
        gene_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE11", "GENE12", "GENE13", "GENE14", "GENE15", "GENE16"])
        
        es = gsea_engine._calculate_enrichment_score(
            pathway_id=pathway_id,
            pathway_info=pathway_info,
            gene_list=gene_list,
            background_genes=background_genes
        )
        
        assert es == 0.0

    def test_calculate_normalized_enrichment_score(self, gsea_engine):
        """Test normalized enrichment score calculation."""
        es = 0.5
        nes = gsea_engine._calculate_normalized_enrichment_score(es)
        
        assert isinstance(nes, float)
        assert -1 <= nes <= 1

    def test_calculate_false_discovery_rate(self, gsea_engine):
        """Test false discovery rate calculation."""
        es = 0.5
        fdr = gsea_engine._calculate_false_discovery_rate(es)
        
        assert isinstance(fdr, float)
        assert 0 <= fdr <= 1

    def test_calculate_leading_edge_subset(self, gsea_engine, sample_gene_list, sample_pathway_data):
        """Test leading edge subset calculation."""
        pathway_id = "PATH:00010"
        pathway_info = sample_pathway_data[pathway_id]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"])
        
        les = gsea_engine._calculate_leading_edge_subset(
            pathway_id=pathway_id,
            pathway_info=pathway_info,
            gene_list=sample_gene_list,
            background_genes=background_genes
        )
        
        assert isinstance(les, list)
        assert all(isinstance(gene, str) for gene in les)

    def test_calculate_leading_edge_subset_no_overlap(self, gsea_engine, sample_pathway_data):
        """Test leading edge subset calculation with no gene overlap."""
        pathway_id = "PATH:00030"  # No overlap with sample genes
        pathway_info = sample_pathway_data[pathway_id]
        gene_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE11", "GENE12", "GENE13", "GENE14", "GENE15", "GENE16"])
        
        les = gsea_engine._calculate_leading_edge_subset(
            pathway_id=pathway_id,
            pathway_info=pathway_info,
            gene_list=gene_list,
            background_genes=background_genes
        )
        
        assert les == []

    def test_apply_multiple_testing_correction(self, gsea_engine):
        """Test multiple testing correction."""
        p_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        # Test FDR correction
        corrected_p_values = gsea_engine._apply_multiple_testing_correction(
            p_values=p_values,
            method=CorrectionMethod.FDR_BH
        )
        
        assert len(corrected_p_values) == len(p_values)
        assert all(0 <= p <= 1 for p in corrected_p_values)
        
        # Test Bonferroni correction
        corrected_p_values = gsea_engine._apply_multiple_testing_correction(
            p_values=p_values,
            method=CorrectionMethod.BONFERRONI
        )
        
        assert len(corrected_p_values) == len(p_values)
        assert all(0 <= p <= 1 for p in corrected_p_values)

    def test_filter_pathways_by_size(self, gsea_engine, sample_pathway_data):
        """Test pathway filtering by size."""
        filtered_pathways = gsea_engine._filter_pathways_by_size(
            pathway_data=sample_pathway_data,
            min_size=3,
            max_size=10
        )
        
        assert len(filtered_pathways) <= len(sample_pathway_data)
        for pathway_id, pathway_info in filtered_pathways.items():
            assert 3 <= pathway_info["size"] <= 10

    def test_filter_pathways_by_size_no_filter(self, gsea_engine, sample_pathway_data):
        """Test pathway filtering with no size restrictions."""
        filtered_pathways = gsea_engine._filter_pathways_by_size(
            pathway_data=sample_pathway_data,
            min_size=0,
            max_size=float('inf')
        )
        
        assert len(filtered_pathways) == len(sample_pathway_data)

    def test_calculate_pathway_statistics(self, gsea_engine, sample_gene_list, sample_pathway_data):
        """Test pathway statistics calculation."""
        pathway_id = "PATH:00010"
        pathway_info = sample_pathway_data[pathway_id]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"])
        
        stats = gsea_engine._calculate_pathway_statistics(
            pathway_id=pathway_id,
            pathway_info=pathway_info,
            gene_list=sample_gene_list,
            background_genes=background_genes
        )
        
        assert "pathway_id" in stats
        assert "pathway_name" in stats
        assert "p_value" in stats
        assert "gene_overlap" in stats
        assert "gene_overlap_count" in stats
        assert "pathway_size" in stats
        assert stats["pathway_id"] == pathway_id
        assert stats["pathway_name"] == pathway_info["name"]
        assert 0 <= stats["p_value"] <= 1
        assert stats["gene_overlap_count"] > 0

    def test_calculate_pathway_statistics_no_overlap(self, gsea_engine, sample_pathway_data):
        """Test pathway statistics calculation with no gene overlap."""
        pathway_id = "PATH:00030"  # No overlap with sample genes
        pathway_info = sample_pathway_data[pathway_id]
        gene_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE11", "GENE12", "GENE13", "GENE14", "GENE15", "GENE16"])
        
        stats = gsea_engine._calculate_pathway_statistics(
            pathway_id=pathway_id,
            pathway_info=pathway_info,
            gene_list=gene_list,
            background_genes=background_genes
        )
        
        assert stats["gene_overlap_count"] == 0
        assert stats["p_value"] == 1.0

    @pytest.mark.asyncio
    async def test_analyze_different_databases(self, gsea_engine, sample_gene_list):
        """Test GSEA analysis with different databases."""
        databases = [DatabaseType.KEGG, DatabaseType.REACTOME, DatabaseType.GO, DatabaseType.WIKIPATHWAYS]
        
        for database in databases:
            # Mock database manager
            gsea_engine.database_manager.get_pathway_data = AsyncMock(return_value={})
            gsea_engine.database_manager.get_background_genes = AsyncMock(return_value=set(sample_gene_list))
            
            # Run analysis
            result = await gsea_engine.analyze(
                gene_list=sample_gene_list,
                database=database,
                species="human"
            )
            
            # Verify result structure
            assert isinstance(result, DatabaseResult)
            assert result.database_type == database

    @pytest.mark.asyncio
    async def test_analyze_different_species(self, gsea_engine, sample_gene_list, sample_pathway_data):
        """Test GSEA analysis with different species."""
        species_list = ["human", "mouse", "rat", "yeast", "drosophila"]
        
        for species in species_list:
            # Mock database manager
            gsea_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
            gsea_engine.database_manager.get_background_genes = AsyncMock(return_value=set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]))
            
            # Run analysis
            result = await gsea_engine.analyze(
                gene_list=sample_gene_list,
                database=DatabaseType.KEGG,
                species=species
            )
            
            # Verify result structure
            assert isinstance(result, DatabaseResult)
            assert result.species == species

    def test_validate_input_parameters(self, gsea_engine):
        """Test input parameter validation."""
        # Valid parameters
        assert gsea_engine._validate_input_parameters(
            gene_list=["GENE1", "GENE2"],
            database=DatabaseType.KEGG,
            species="human"
        ) is True
        
        # Invalid gene list
        assert gsea_engine._validate_input_parameters(
            gene_list=[],
            database=DatabaseType.KEGG,
            species="human"
        ) is False
        
        # Invalid database
        assert gsea_engine._validate_input_parameters(
            gene_list=["GENE1", "GENE2"],
            database=None,
            species="human"
        ) is False
        
        # Invalid species
        assert gsea_engine._validate_input_parameters(
            gene_list=["GENE1", "GENE2"],
            database=DatabaseType.KEGG,
            species=""
        ) is False

    def test_validate_threshold_parameters(self, gsea_engine):
        """Test threshold parameter validation."""
        # Valid thresholds
        assert gsea_engine._validate_threshold_parameters(
            significance_threshold=0.05,
            min_size=5,
            max_size=500
        ) is True
        
        # Invalid significance threshold
        assert gsea_engine._validate_threshold_parameters(
            significance_threshold=1.5,
            min_size=5,
            max_size=500
        ) is False
        
        # Invalid pathway size range
        assert gsea_engine._validate_threshold_parameters(
            significance_threshold=0.05,
            min_size=10,
            max_size=5
        ) is False

    def test_validate_permutation_parameters(self, gsea_engine):
        """Test permutation parameter validation."""
        # Valid permutations
        assert gsea_engine._validate_permutation_parameters(permutations=1000) is True
        
        # Invalid permutations
        assert gsea_engine._validate_permutation_parameters(permutations=0) is False
        assert gsea_engine._validate_permutation_parameters(permutations=-1) is False
