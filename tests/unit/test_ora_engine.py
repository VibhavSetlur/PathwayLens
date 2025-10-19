"""
Unit tests for the ORA engine.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.analysis.ora_engine import ORAEngine
from pathwaylens_core.analysis.schemas import (
    DatabaseType, PathwayResult, DatabaseResult, CorrectionMethod
)
from pathwaylens_core.data import DatabaseManager


class TestORAEngine:
    """Test cases for the ORAEngine class."""

    @pytest.fixture
    def ora_engine(self):
        """Create an ORAEngine instance for testing."""
        db_manager = Mock(spec=DatabaseManager)
        return ORAEngine(database_manager=db_manager)

    @pytest.fixture
    def sample_gene_list(self):
        """Create a sample gene list."""
        return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

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

    def test_init(self, ora_engine):
        """Test ORAEngine initialization."""
        assert ora_engine.database_manager is not None
        assert ora_engine.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_basic(self, ora_engine, sample_gene_list, sample_pathway_data):
        """Test basic ORA analysis."""
        # Mock database manager
        ora_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
        ora_engine.database_manager.get_background_genes = AsyncMock(return_value=set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]))
        
        # Run analysis
        result = await ora_engine.analyze(
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
    async def test_analyze_with_parameters(self, ora_engine, sample_gene_list, sample_pathway_data):
        """Test ORA analysis with custom parameters."""
        # Mock database manager
        ora_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
        ora_engine.database_manager.get_background_genes = AsyncMock(return_value=set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]))
        
        # Run analysis with custom parameters
        result = await ora_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human",
            significance_threshold=0.01,
            correction_method=CorrectionMethod.BONFERRONI,
            min_pathway_size=3,
            max_pathway_size=10
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert len(result.pathway_results) > 0

    @pytest.mark.asyncio
    async def test_analyze_empty_gene_list(self, ora_engine, sample_pathway_data):
        """Test ORA analysis with empty gene list."""
        # Mock database manager
        ora_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
        ora_engine.database_manager.get_background_genes = AsyncMock(return_value=set())
        
        # Run analysis with empty gene list
        result = await ora_engine.analyze(
            gene_list=[],
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert len(result.pathway_results) == 0

    @pytest.mark.asyncio
    async def test_analyze_no_pathways(self, ora_engine, sample_gene_list):
        """Test ORA analysis with no pathways."""
        # Mock database manager to return empty pathway data
        ora_engine.database_manager.get_pathway_data = AsyncMock(return_value={})
        ora_engine.database_manager.get_background_genes = AsyncMock(return_value=set(sample_gene_list))
        
        # Run analysis
        result = await ora_engine.analyze(
            gene_list=sample_gene_list,
            database=DatabaseType.KEGG,
            species="human"
        )
        
        # Verify result structure
        assert isinstance(result, DatabaseResult)
        assert len(result.pathway_results) == 0

    @pytest.mark.asyncio
    async def test_analyze_database_error(self, ora_engine, sample_gene_list):
        """Test ORA analysis when database raises an error."""
        # Mock database manager to raise an error
        ora_engine.database_manager.get_pathway_data = AsyncMock(side_effect=Exception("Database error"))
        
        with pytest.raises(Exception, match="Database error"):
            await ora_engine.analyze(
                gene_list=sample_gene_list,
                database=DatabaseType.KEGG,
                species="human"
            )

    def test_calculate_hypergeometric_p_value(self, ora_engine):
        """Test hypergeometric p-value calculation."""
        # Test case: 2 successes in 5 draws from 10 total with 3 successes
        p_value = ora_engine._calculate_hypergeometric_p_value(
            n_successes=2,
            n_draws=5,
            n_total=10,
            n_successes_total=3
        )
        
        assert 0 <= p_value <= 1
        assert isinstance(p_value, float)

    def test_calculate_hypergeometric_p_value_edge_cases(self, ora_engine):
        """Test hypergeometric p-value calculation edge cases."""
        # Test case: no successes
        p_value = ora_engine._calculate_hypergeometric_p_value(
            n_successes=0,
            n_draws=5,
            n_total=10,
            n_successes_total=3
        )
        assert 0 <= p_value <= 1
        
        # Test case: all successes
        p_value = ora_engine._calculate_hypergeometric_p_value(
            n_successes=3,
            n_draws=3,
            n_total=10,
            n_successes_total=3
        )
        assert 0 <= p_value <= 1

    def test_calculate_odds_ratio(self, ora_engine):
        """Test odds ratio calculation."""
        # Test case: 2 successes in 5 draws from 10 total with 3 successes
        odds_ratio = ora_engine._calculate_odds_ratio(
            n_successes=2,
            n_draws=5,
            n_total=10,
            n_successes_total=3
        )
        
        assert odds_ratio > 0
        assert isinstance(odds_ratio, float)

    def test_calculate_confidence_interval(self, ora_engine):
        """Test confidence interval calculation."""
        # Test case: 2 successes in 5 draws from 10 total with 3 successes
        ci = ora_engine._calculate_confidence_interval(
            n_successes=2,
            n_draws=5,
            n_total=10,
            n_successes_total=3,
            confidence_level=0.95
        )
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert all(isinstance(x, float) for x in ci)

    def test_apply_multiple_testing_correction(self, ora_engine):
        """Test multiple testing correction."""
        p_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        
        # Test FDR correction
        corrected_p_values = ora_engine._apply_multiple_testing_correction(
            p_values=p_values,
            method=CorrectionMethod.FDR_BH
        )
        
        assert len(corrected_p_values) == len(p_values)
        assert all(0 <= p <= 1 for p in corrected_p_values)
        
        # Test Bonferroni correction
        corrected_p_values = ora_engine._apply_multiple_testing_correction(
            p_values=p_values,
            method=CorrectionMethod.BONFERRONI
        )
        
        assert len(corrected_p_values) == len(p_values)
        assert all(0 <= p <= 1 for p in corrected_p_values)

    def test_filter_pathways_by_size(self, ora_engine, sample_pathway_data):
        """Test pathway filtering by size."""
        filtered_pathways = ora_engine._filter_pathways_by_size(
            pathway_data=sample_pathway_data,
            min_size=3,
            max_size=10
        )
        
        assert len(filtered_pathways) <= len(sample_pathway_data)
        for pathway_id, pathway_info in filtered_pathways.items():
            assert 3 <= pathway_info["size"] <= 10

    def test_filter_pathways_by_size_no_filter(self, ora_engine, sample_pathway_data):
        """Test pathway filtering with no size restrictions."""
        filtered_pathways = ora_engine._filter_pathways_by_size(
            pathway_data=sample_pathway_data,
            min_size=0,
            max_size=float('inf')
        )
        
        assert len(filtered_pathways) == len(sample_pathway_data)

    def test_calculate_pathway_statistics(self, ora_engine, sample_gene_list, sample_pathway_data):
        """Test pathway statistics calculation."""
        pathway_id = "PATH:00010"
        pathway_info = sample_pathway_data[pathway_id]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"])
        
        stats = ora_engine._calculate_pathway_statistics(
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

    def test_calculate_pathway_statistics_no_overlap(self, ora_engine, sample_pathway_data):
        """Test pathway statistics calculation with no gene overlap."""
        pathway_id = "PATH:00030"  # No overlap with sample genes
        pathway_info = sample_pathway_data[pathway_id]
        gene_list = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
        background_genes = set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE11", "GENE12", "GENE13", "GENE14", "GENE15", "GENE16"])
        
        stats = ora_engine._calculate_pathway_statistics(
            pathway_id=pathway_id,
            pathway_info=pathway_info,
            gene_list=gene_list,
            background_genes=background_genes
        )
        
        assert stats["gene_overlap_count"] == 0
        assert stats["p_value"] == 1.0

    @pytest.mark.asyncio
    async def test_analyze_different_databases(self, ora_engine, sample_gene_list):
        """Test ORA analysis with different databases."""
        databases = [DatabaseType.KEGG, DatabaseType.REACTOME, DatabaseType.GO, DatabaseType.WIKIPATHWAYS]
        
        for database in databases:
            # Mock database manager
            ora_engine.database_manager.get_pathway_data = AsyncMock(return_value={})
            ora_engine.database_manager.get_background_genes = AsyncMock(return_value=set(sample_gene_list))
            
            # Run analysis
            result = await ora_engine.analyze(
                gene_list=sample_gene_list,
                database=database,
                species="human"
            )
            
            # Verify result structure
            assert isinstance(result, DatabaseResult)
            assert result.database_type == database

    @pytest.mark.asyncio
    async def test_analyze_different_species(self, ora_engine, sample_gene_list, sample_pathway_data):
        """Test ORA analysis with different species."""
        species_list = ["human", "mouse", "rat", "yeast", "drosophila"]
        
        for species in species_list:
            # Mock database manager
            ora_engine.database_manager.get_pathway_data = AsyncMock(return_value=sample_pathway_data)
            ora_engine.database_manager.get_background_genes = AsyncMock(return_value=set(["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8", "GENE9", "GENE10"]))
            
            # Run analysis
            result = await ora_engine.analyze(
                gene_list=sample_gene_list,
                database=DatabaseType.KEGG,
                species=species
            )
            
            # Verify result structure
            assert isinstance(result, DatabaseResult)
            assert result.species == species

    def test_validate_input_parameters(self, ora_engine):
        """Test input parameter validation."""
        # Valid parameters
        assert ora_engine._validate_input_parameters(
            gene_list=["GENE1", "GENE2"],
            database=DatabaseType.KEGG,
            species="human"
        ) is True
        
        # Invalid gene list
        assert ora_engine._validate_input_parameters(
            gene_list=[],
            database=DatabaseType.KEGG,
            species="human"
        ) is False
        
        # Invalid database
        assert ora_engine._validate_input_parameters(
            gene_list=["GENE1", "GENE2"],
            database=None,
            species="human"
        ) is False
        
        # Invalid species
        assert ora_engine._validate_input_parameters(
            gene_list=["GENE1", "GENE2"],
            database=DatabaseType.KEGG,
            species=""
        ) is False

    def test_validate_threshold_parameters(self, ora_engine):
        """Test threshold parameter validation."""
        # Valid thresholds
        assert ora_engine._validate_threshold_parameters(
            significance_threshold=0.05,
            min_pathway_size=5,
            max_pathway_size=500
        ) is True
        
        # Invalid significance threshold
        assert ora_engine._validate_threshold_parameters(
            significance_threshold=1.5,
            min_pathway_size=5,
            max_pathway_size=500
        ) is False
        
        # Invalid pathway size range
        assert ora_engine._validate_threshold_parameters(
            significance_threshold=0.05,
            min_pathway_size=10,
            max_pathway_size=5
        ) is False
