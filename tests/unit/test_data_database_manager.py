"""
Unit tests for the DatabaseManager class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.data.database_manager import DatabaseManager


class TestDatabaseManager:
    """Test cases for the DatabaseManager class."""

    @pytest.fixture
    def database_manager(self):
        """Create a DatabaseManager instance for testing."""
        return DatabaseManager()

    def test_init(self, database_manager):
        """Test DatabaseManager initialization."""
        assert database_manager.logger is not None
        assert database_manager.adapters is not None
        assert len(database_manager.adapters) > 0

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, database_manager):
        """Test getting pathway data."""
        with patch.object(database_manager.adapters["KEGG"], 'get_pathway_data') as mock_get:
            mock_get.return_value = {
                "hsa00010": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                "hsa00020": {"name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
            }

            result = await database_manager.get_pathway_data("KEGG", "human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "hsa00010" in result
            assert result["hsa00010"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_invalid_database(self, database_manager):
        """Test getting pathway data with invalid database."""
        result = await database_manager.get_pathway_data("INVALID_DB", "human")

        assert isinstance(result, dict)
        assert "error" in result
        assert "Unsupported database" in result["error"]

    @pytest.mark.asyncio
    async def test_get_pathway_data_database_error(self, database_manager):
        """Test getting pathway data with database error."""
        with patch.object(database_manager.adapters["KEGG"], 'get_pathway_data') as mock_get:
            mock_get.side_effect = Exception("Database error")

            result = await database_manager.get_pathway_data("KEGG", "human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "Database error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_background_genes(self, database_manager):
        """Test getting background genes."""
        with patch.object(database_manager.adapters["KEGG"], 'get_background_genes') as mock_get:
            mock_get.return_value = {"GENE1", "GENE2", "GENE3", "GENE4", "GENE5"}

            result = await database_manager.get_background_genes("KEGG", "human")

            assert isinstance(result, set)
            assert len(result) > 0
            assert "GENE1" in result

    @pytest.mark.asyncio
    async def test_get_background_genes_invalid_database(self, database_manager):
        """Test getting background genes with invalid database."""
        result = await database_manager.get_background_genes("INVALID_DB", "human")

        assert isinstance(result, set)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_background_genes_database_error(self, database_manager):
        """Test getting background genes with database error."""
        with patch.object(database_manager.adapters["KEGG"], 'get_background_genes') as mock_get:
            mock_get.side_effect = Exception("Database error")

            result = await database_manager.get_background_genes("KEGG", "human")

            assert isinstance(result, set)
            assert len(result) == 0

    def test_get_supported_databases(self, database_manager):
        """Test getting supported databases."""
        databases = database_manager.get_supported_databases()

        assert isinstance(databases, list)
        assert len(databases) > 0
        assert "KEGG" in databases
        assert "Reactome" in databases
        assert "GO" in databases

    def test_get_supported_species(self, database_manager):
        """Test getting supported species."""
        species = database_manager.get_supported_species()

        assert isinstance(species, list)
        assert len(species) > 0
        assert "human" in species
        assert "mouse" in species

    def test_validate_database(self, database_manager):
        """Test database validation."""
        # Valid databases
        assert database_manager._validate_database("KEGG") is True
        assert database_manager._validate_database("Reactome") is True
        assert database_manager._validate_database("GO") is True

        # Invalid databases
        assert database_manager._validate_database("INVALID_DB") is False
        assert database_manager._validate_database("") is False
        assert database_manager._validate_database(None) is False

    def test_validate_species(self, database_manager):
        """Test species validation."""
        # Valid species
        assert database_manager._validate_species("human") is True
        assert database_manager._validate_species("mouse") is True
        assert database_manager._validate_species("rat") is True

        # Invalid species
        assert database_manager._validate_species("") is False
        assert database_manager._validate_species(None) is False

    def test_get_adapter(self, database_manager):
        """Test getting database adapter."""
        # Valid adapter
        adapter = database_manager._get_adapter("KEGG")
        assert adapter is not None
        assert adapter.database_name == "KEGG"

        # Invalid adapter
        adapter = database_manager._get_adapter("INVALID_DB")
        assert adapter is None

    def test_validate_parameters(self, database_manager):
        """Test parameter validation."""
        # Valid parameters
        assert database_manager._validate_parameters("KEGG", "human") is True
        assert database_manager._validate_parameters("Reactome", "mouse") is True

        # Invalid parameters
        assert database_manager._validate_parameters("INVALID_DB", "human") is False
        assert database_manager._validate_parameters("KEGG", "invalid_species") is False
        assert database_manager._validate_parameters("", "human") is False
        assert database_manager._validate_parameters("KEGG", "") is False
        assert database_manager._validate_parameters(None, "human") is False
        assert database_manager._validate_parameters("KEGG", None) is False

    def test_validate_database_species_combination(self, database_manager):
        """Test database-species combination validation."""
        # Valid combinations
        assert database_manager._validate_database_species_combination("KEGG", "human") is True
        assert database_manager._validate_database_species_combination("Reactome", "mouse") is True

        # Invalid combinations
        assert database_manager._validate_database_species_combination("INVALID_DB", "human") is False
        assert database_manager._validate_database_species_combination("KEGG", "invalid_species") is False

    def test_validate_pathway_data(self, database_manager):
        """Test pathway data validation."""
        # Valid pathway data
        valid_data = {
            "PATH:00010": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
            "PATH:00020": {"name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
        }
        assert database_manager._validate_pathway_data(valid_data) is True

        # Invalid pathway data
        assert database_manager._validate_pathway_data({}) is False
        assert database_manager._validate_pathway_data(None) is False

    def test_validate_background_genes(self, database_manager):
        """Test background genes validation."""
        # Valid background genes
        valid_genes = {"GENE1", "GENE2", "GENE3"}
        assert database_manager._validate_background_genes(valid_genes) is True

        # Invalid background genes
        assert database_manager._validate_background_genes(set()) is False
        assert database_manager._validate_background_genes(None) is False

    def test_validate_adapter(self, database_manager):
        """Test adapter validation."""
        # Valid adapter
        adapter = database_manager.adapters["KEGG"]
        assert database_manager._validate_adapter(adapter) is True

        # Invalid adapter
        assert database_manager._validate_adapter(None) is False

    def test_validate_adapter_method(self, database_manager):
        """Test adapter method validation."""
        # Valid adapter with method
        adapter = database_manager.adapters["KEGG"]
        assert database_manager._validate_adapter_method(adapter, "get_pathway_data") is True

        # Invalid adapter
        assert database_manager._validate_adapter_method(None, "get_pathway_data") is False

        # Valid adapter without method
        assert database_manager._validate_adapter_method(adapter, "nonexistent_method") is False

    def test_validate_error_response(self, database_manager):
        """Test error response validation."""
        # Valid error response
        error_response = {"error": "Test error message"}
        assert database_manager._validate_error_response(error_response) is True

        # Invalid error response
        assert database_manager._validate_error_response({}) is False
        assert database_manager._validate_error_response(None) is False

    def test_validate_success_response(self, database_manager):
        """Test success response validation."""
        # Valid success response
        success_response = {"data": "test data"}
        assert database_manager._validate_success_response(success_response) is True

        # Invalid success response
        assert database_manager._validate_success_response({}) is False
        assert database_manager._validate_success_response(None) is False

    def test_validate_response(self, database_manager):
        """Test response validation."""
        # Valid response
        response = {"data": "test data"}
        assert database_manager._validate_response(response) is True

        # Invalid response
        assert database_manager._validate_response(None) is False

    def test_validate_async_method(self, database_manager):
        """Test async method validation."""
        # Valid async method
        assert database_manager._validate_async_method(database_manager.get_pathway_data) is True

        # Invalid async method
        assert database_manager._validate_async_method(None) is False

    def test_validate_async_result(self, database_manager):
        """Test async result validation."""
        # Valid async result
        result = {"data": "test data"}
        assert database_manager._validate_async_result(result) is True

        # Invalid async result
        assert database_manager._validate_async_result(None) is False