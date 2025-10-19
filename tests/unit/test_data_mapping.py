"""
Unit tests for the Data mapping modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.data.mapping.gene_mapper import GeneMapper
from pathwaylens_core.data.mapping.ortholog_mapper import OrthologMapper
from pathwaylens_core.data.mapping.pathway_mapper import PathwayMapper


class TestGeneMapper:
    """Test cases for the GeneMapper class."""

    @pytest.fixture
    def gene_mapper(self):
        """Create a GeneMapper instance for testing."""
        return GeneMapper()

    @pytest.fixture
    def sample_gene_ids(self):
        """Create sample gene IDs."""
        return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

    def test_init(self, gene_mapper):
        """Test GeneMapper initialization."""
        assert gene_mapper.logger is not None
        assert gene_mapper.mg is not None

    @pytest.mark.asyncio
    async def test_map_genes_basic(self, gene_mapper, sample_gene_ids):
        """Test basic gene mapping."""
        with patch.object(gene_mapper.mg, 'querymany') as mock_query:
            mock_query.return_value = [
                {"query": "GENE1", "entrezgene": "12345"},
                {"query": "GENE2", "entrezgene": "67890"},
                {"query": "GENE3", "entrezgene": "11111"}
            ]

            result = await gene_mapper.map_genes(
                gene_ids=sample_gene_ids,
                current_species="human",
                target_species="mouse",
                input_type="symbol",
                output_type="entrezgene"
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "input_id" in result.columns
            assert "output_id" in result.columns

    @pytest.mark.asyncio
    async def test_map_genes_same_species(self, gene_mapper, sample_gene_ids):
        """Test gene mapping within the same species."""
        with patch.object(gene_mapper.mg, 'querymany') as mock_query:
            mock_query.return_value = [
                {"query": "GENE1", "entrezgene": "12345"},
                {"query": "GENE2", "entrezgene": "67890"}
            ]

            result = await gene_mapper.map_genes(
                gene_ids=sample_gene_ids,
                current_species="human",
                target_species="human",
                input_type="symbol",
                output_type="entrezgene"
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_map_genes_empty_input(self, gene_mapper):
        """Test gene mapping with empty input."""
        result = await gene_mapper.map_genes(
            gene_ids=[],
            current_species="human",
            target_species="mouse",
            input_type="symbol",
            output_type="entrezgene"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_map_genes_api_error(self, gene_mapper, sample_gene_ids):
        """Test gene mapping with API error."""
        with patch.object(gene_mapper.mg, 'querymany') as mock_query:
            mock_query.side_effect = Exception("API Error")

            result = await gene_mapper.map_genes(
                gene_ids=sample_gene_ids,
                current_species="human",
                target_species="mouse",
                input_type="symbol",
                output_type="entrezgene"
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_validate_gene_ids(self, gene_mapper):
        """Test gene ID validation."""
        # Valid gene IDs
        assert gene_mapper._validate_gene_ids(["GENE1", "GENE2"]) is True

        # Invalid gene IDs
        assert gene_mapper._validate_gene_ids([]) is False
        assert gene_mapper._validate_gene_ids(None) is False

    def test_validate_species(self, gene_mapper):
        """Test species validation."""
        # Valid species
        assert gene_mapper._validate_species("human") is True
        assert gene_mapper._validate_species("mouse") is True

        # Invalid species
        assert gene_mapper._validate_species("") is False
        assert gene_mapper._validate_species(None) is False

    def test_validate_id_types(self, gene_mapper):
        """Test ID type validation."""
        # Valid ID types
        assert gene_mapper._validate_id_types("symbol", "entrezgene") is True
        assert gene_mapper._validate_id_types("ensembl_gene_id", "symbol") is True

        # Invalid ID types
        assert gene_mapper._validate_id_types("", "entrezgene") is False
        assert gene_mapper._validate_id_types("symbol", "") is False
        assert gene_mapper._validate_id_types(None, "entrezgene") is False
        assert gene_mapper._validate_id_types("symbol", None) is False

    def test_validate_drop_unmapped(self, gene_mapper):
        """Test drop unmapped validation."""
        # Valid drop unmapped
        assert gene_mapper._validate_drop_unmapped(True) is True
        assert gene_mapper._validate_drop_unmapped(False) is True

        # Invalid drop unmapped
        assert gene_mapper._validate_drop_unmapped(None) is False


class TestOrthologMapper:
    """Test cases for the OrthologMapper class."""

    @pytest.fixture
    def ortholog_mapper(self):
        """Create an OrthologMapper instance for testing."""
        return OrthologMapper()

    @pytest.fixture
    def sample_gene_ids(self):
        """Create sample gene IDs."""
        return ["ENSG00000123456", "ENSG00000789012", "ENSG00000111111"]

    def test_init(self, ortholog_mapper):
        """Test OrthologMapper initialization."""
        assert ortholog_mapper.logger is not None
        assert ortholog_mapper.ensembl_rest_url is not None

    @pytest.mark.asyncio
    async def test_map_orthologs_basic(self, ortholog_mapper, sample_gene_ids):
        """Test basic ortholog mapping."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [
                    {"id": "ENSG00000123456", "homologs": [{"id": "ENSMUSG00000012345"}]},
                    {"id": "ENSG00000789012", "homologs": [{"id": "ENSMUSG00000078901"}]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await ortholog_mapper.map_orthologs(
                gene_ids=sample_gene_ids,
                source_species="human",
                target_species="mouse",
                input_type="ensembl_gene_id",
                output_type="ensembl_gene_id"
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert "source_id" in result.columns
            assert "target_id" in result.columns

    @pytest.mark.asyncio
    async def test_map_orthologs_empty_input(self, ortholog_mapper):
        """Test ortholog mapping with empty input."""
        result = await ortholog_mapper.map_orthologs(
            gene_ids=[],
            source_species="human",
            target_species="mouse",
            input_type="ensembl_gene_id",
            output_type="ensembl_gene_id"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_map_orthologs_api_error(self, ortholog_mapper, sample_gene_ids):
        """Test ortholog mapping with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await ortholog_mapper.map_orthologs(
                gene_ids=sample_gene_ids,
                source_species="human",
                target_species="mouse",
                input_type="ensembl_gene_id",
                output_type="ensembl_gene_id"
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_validate_gene_ids(self, ortholog_mapper):
        """Test gene ID validation."""
        # Valid gene IDs
        assert ortholog_mapper._validate_gene_ids(["ENSG00000123456", "ENSG00000789012"]) is True

        # Invalid gene IDs
        assert ortholog_mapper._validate_gene_ids([]) is False
        assert ortholog_mapper._validate_gene_ids(None) is False

    def test_validate_species(self, ortholog_mapper):
        """Test species validation."""
        # Valid species
        assert ortholog_mapper._validate_species("human", "mouse") is True
        assert ortholog_mapper._validate_species("mouse", "rat") is True

        # Invalid species
        assert ortholog_mapper._validate_species("", "mouse") is False
        assert ortholog_mapper._validate_species("human", "") is False
        assert ortholog_mapper._validate_species(None, "mouse") is False
        assert ortholog_mapper._validate_species("human", None) is False

    def test_validate_id_types(self, ortholog_mapper):
        """Test ID type validation."""
        # Valid ID types
        assert ortholog_mapper._validate_id_types("ensembl_gene_id", "ensembl_gene_id") is True
        assert ortholog_mapper._validate_id_types("symbol", "symbol") is True

        # Invalid ID types
        assert ortholog_mapper._validate_id_types("", "ensembl_gene_id") is False
        assert ortholog_mapper._validate_id_types("ensembl_gene_id", "") is False
        assert ortholog_mapper._validate_id_types(None, "ensembl_gene_id") is False
        assert ortholog_mapper._validate_id_types("ensembl_gene_id", None) is False

    def test_validate_species_pair(self, ortholog_mapper):
        """Test species pair validation."""
        # Valid species pairs
        assert ortholog_mapper._validate_species_pair("human", "mouse") is True
        assert ortholog_mapper._validate_species_pair("mouse", "rat") is True

        # Invalid species pairs
        assert ortholog_mapper._validate_species_pair("human", "human") is False
        assert ortholog_mapper._validate_species_pair("", "mouse") is False
        assert ortholog_mapper._validate_species_pair("human", "") is False


class TestPathwayMapper:
    """Test cases for the PathwayMapper class."""

    @pytest.fixture
    def pathway_mapper(self):
        """Create a PathwayMapper instance for testing."""
        return PathwayMapper()

    @pytest.fixture
    def sample_pathway_ids(self):
        """Create sample pathway IDs."""
        return ["PATH:00010", "PATH:00020", "PATH:00030"]

    def test_init(self, pathway_mapper):
        """Test PathwayMapper initialization."""
        assert pathway_mapper.logger is not None

    @pytest.mark.asyncio
    async def test_map_pathways_basic(self, pathway_mapper, sample_pathway_ids):
        """Test basic pathway mapping."""
        result = await pathway_mapper.map_pathways(
            pathway_ids=sample_pathway_ids,
            source_database="KEGG",
            target_database="Reactome",
            species="human"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "source_id" in result.columns
        assert "target_id" in result.columns

    @pytest.mark.asyncio
    async def test_map_pathways_empty_input(self, pathway_mapper):
        """Test pathway mapping with empty input."""
        result = await pathway_mapper.map_pathways(
            pathway_ids=[],
            source_database="KEGG",
            target_database="Reactome",
            species="human"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_map_pathways_same_database(self, pathway_mapper, sample_pathway_ids):
        """Test pathway mapping within the same database."""
        result = await pathway_mapper.map_pathways(
            pathway_ids=sample_pathway_ids,
            source_database="KEGG",
            target_database="KEGG",
            species="human"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_validate_pathway_ids(self, pathway_mapper):
        """Test pathway ID validation."""
        # Valid pathway IDs
        assert pathway_mapper._validate_pathway_ids(["PATH:00010", "PATH:00020"]) is True

        # Invalid pathway IDs
        assert pathway_mapper._validate_pathway_ids([]) is False
        assert pathway_mapper._validate_pathway_ids(None) is False

    def test_validate_databases(self, pathway_mapper):
        """Test database validation."""
        # Valid databases
        assert pathway_mapper._validate_databases("KEGG", "Reactome") is True
        assert pathway_mapper._validate_databases("Reactome", "GO") is True

        # Invalid databases
        assert pathway_mapper._validate_databases("", "Reactome") is False
        assert pathway_mapper._validate_databases("KEGG", "") is False
        assert pathway_mapper._validate_databases(None, "Reactome") is False
        assert pathway_mapper._validate_databases("KEGG", None) is False

    def test_validate_species(self, pathway_mapper):
        """Test species validation."""
        # Valid species
        assert pathway_mapper._validate_species("human") is True
        assert pathway_mapper._validate_species("mouse") is True

        # Invalid species
        assert pathway_mapper._validate_species("") is False
        assert pathway_mapper._validate_species(None) is False

    def test_validate_database_pair(self, pathway_mapper):
        """Test database pair validation."""
        # Valid database pairs
        assert pathway_mapper._validate_database_pair("KEGG", "Reactome") is True
        assert pathway_mapper._validate_database_pair("Reactome", "GO") is True

        # Invalid database pairs
        assert pathway_mapper._validate_database_pair("KEGG", "KEGG") is False
        assert pathway_mapper._validate_database_pair("", "Reactome") is False
        assert pathway_mapper._validate_database_pair("KEGG", "") is False
