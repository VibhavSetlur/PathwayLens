"""
Unit tests for the Normalization modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.normalization.normalizer import Normalizer
from pathwaylens_core.normalization.id_converter import IDConverter
from pathwaylens_core.normalization.species_mapper import SpeciesMapper
from pathwaylens_core.normalization.format_detector import FormatDetector
from pathwaylens_core.normalization.validation import ValidationEngine
from pathwaylens_core.normalization.schemas import (
    NormalizationResult, IDMappingResult, SpeciesMappingResult,
    FormatDetectionResult, ValidationResult
)


class TestNormalizer:
    """Test cases for the Normalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Create a Normalizer instance for testing."""
        return Normalizer()

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

    def test_init(self, normalizer):
        """Test Normalizer initialization."""
        assert normalizer.logger is not None
        assert normalizer.id_converter is not None
        assert normalizer.species_mapper is not None
        assert normalizer.format_detector is not None
        assert normalizer.validation_engine is not None

    @pytest.mark.asyncio
    async def test_normalize_basic(self, normalizer, sample_gene_list):
        """Test basic normalization."""
        result = await normalizer.normalize(
            input_data=sample_gene_list,
            input_format="gene_list",
            target_format="entrezgene",
            species="human"
        )

        assert isinstance(result, NormalizationResult)
        assert result.input_format == "gene_list"
        assert result.target_format == "entrezgene"
        assert result.species == "human"
        assert len(result.normalized_data) > 0

    @pytest.mark.asyncio
    async def test_normalize_with_expression_data(self, normalizer, sample_expression_data):
        """Test normalization with expression data."""
        result = await normalizer.normalize(
            input_data=sample_expression_data,
            input_format="expression_matrix",
            target_format="entrezgene",
            species="human"
        )

        assert isinstance(result, NormalizationResult)
        assert result.input_format == "expression_matrix"
        assert result.target_format == "entrezgene"
        assert result.species == "human"
        assert len(result.normalized_data) > 0

    @pytest.mark.asyncio
    async def test_normalize_empty_input(self, normalizer):
        """Test normalization with empty input."""
        result = await normalizer.normalize(
            input_data=[],
            input_format="gene_list",
            target_format="entrezgene",
            species="human"
        )

        assert isinstance(result, NormalizationResult)
        assert len(result.normalized_data) == 0

    @pytest.mark.asyncio
    async def test_normalize_different_formats(self, normalizer, sample_gene_list):
        """Test normalization with different input and target formats."""
        formats = [
            ("gene_list", "entrezgene"),
            ("gene_list", "ensembl_gene_id"),
            ("gene_list", "symbol"),
            ("gene_list", "uniprot")
        ]

        for input_format, target_format in formats:
            result = await normalizer.normalize(
                input_data=sample_gene_list,
                input_format=input_format,
                target_format=target_format,
                species="human"
            )

            assert isinstance(result, NormalizationResult)
            assert result.input_format == input_format
            assert result.target_format == target_format

    @pytest.mark.asyncio
    async def test_normalize_different_species(self, normalizer, sample_gene_list):
        """Test normalization with different species."""
        species_list = ["human", "mouse", "rat", "yeast", "drosophila"]

        for species in species_list:
            result = await normalizer.normalize(
                input_data=sample_gene_list,
                input_format="gene_list",
                target_format="entrezgene",
                species=species
            )

            assert isinstance(result, NormalizationResult)
            assert result.species == species

    def test_validate_input_parameters(self, normalizer):
        """Test input parameter validation."""
        # Valid parameters
        assert normalizer._validate_input_parameters(
            input_data=["GENE1", "GENE2"],
            input_format="gene_list",
            target_format="entrezgene",
            species="human"
        ) is True

        # Invalid input data
        assert normalizer._validate_input_parameters(
            input_data=None,
            input_format="gene_list",
            target_format="entrezgene",
            species="human"
        ) is False

        # Invalid input format
        assert normalizer._validate_input_parameters(
            input_data=["GENE1", "GENE2"],
            input_format="",
            target_format="entrezgene",
            species="human"
        ) is False

        # Invalid target format
        assert normalizer._validate_input_parameters(
            input_data=["GENE1", "GENE2"],
            input_format="gene_list",
            target_format="",
            species="human"
        ) is False

        # Invalid species
        assert normalizer._validate_input_parameters(
            input_data=["GENE1", "GENE2"],
            input_format="gene_list",
            target_format="entrezgene",
            species=""
        ) is False

    def test_validate_input_data(self, normalizer):
        """Test input data validation."""
        # Valid gene list
        assert normalizer._validate_input_data(["GENE1", "GENE2"]) is True

        # Valid DataFrame
        df = pd.DataFrame({"GENE1": [1, 2, 3]})
        assert normalizer._validate_input_data(df) is True

        # Invalid input data
        assert normalizer._validate_input_data(None) is False
        assert normalizer._validate_input_data([]) is False
        assert normalizer._validate_input_data(pd.DataFrame()) is False

    def test_validate_format(self, normalizer):
        """Test format validation."""
        # Valid formats
        assert normalizer._validate_format("gene_list") is True
        assert normalizer._validate_format("expression_matrix") is True
        assert normalizer._validate_format("entrezgene") is True
        assert normalizer._validate_format("ensembl_gene_id") is True
        assert normalizer._validate_format("symbol") is True
        assert normalizer._validate_format("uniprot") is True

        # Invalid format
        assert normalizer._validate_format("") is False
        assert normalizer._validate_format(None) is False

    def test_validate_species(self, normalizer):
        """Test species validation."""
        # Valid species
        assert normalizer._validate_species("human") is True
        assert normalizer._validate_species("mouse") is True
        assert normalizer._validate_species("rat") is True
        assert normalizer._validate_species("yeast") is True
        assert normalizer._validate_species("drosophila") is True

        # Invalid species
        assert normalizer._validate_species("") is False
        assert normalizer._validate_species(None) is False


class TestIDConverter:
    """Test cases for the IDConverter class."""

    @pytest.fixture
    def id_converter(self):
        """Create an IDConverter instance for testing."""
        return IDConverter()

    @pytest.fixture
    def sample_gene_ids(self):
        """Create sample gene IDs."""
        return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

    def test_init(self, id_converter):
        """Test IDConverter initialization."""
        assert id_converter.logger is not None
        assert id_converter.mygene_client is not None

    @pytest.mark.asyncio
    async def test_convert_ids_basic(self, id_converter, sample_gene_ids):
        """Test basic ID conversion."""
        with patch.object(id_converter.mygene_client, 'querymany') as mock_query:
            mock_query.return_value = [
                {"query": "GENE1", "entrezgene": "12345"},
                {"query": "GENE2", "entrezgene": "67890"},
                {"query": "GENE3", "entrezgene": "11111"}
            ]

            result = await id_converter.convert_ids(
                gene_ids=sample_gene_ids,
                input_type="symbol",
                output_type="entrezgene",
                species="human"
            )

            assert isinstance(result, IDMappingResult)
            assert result.input_type == "symbol"
            assert result.output_type == "entrezgene"
            assert result.species == "human"
            assert len(result.mapped_ids) > 0

    @pytest.mark.asyncio
    async def test_convert_ids_empty_input(self, id_converter):
        """Test ID conversion with empty input."""
        result = await id_converter.convert_ids(
            gene_ids=[],
            input_type="symbol",
            output_type="entrezgene",
            species="human"
        )

        assert isinstance(result, IDMappingResult)
        assert len(result.mapped_ids) == 0

    @pytest.mark.asyncio
    async def test_convert_ids_api_error(self, id_converter, sample_gene_ids):
        """Test ID conversion with API error."""
        with patch.object(id_converter.mygene_client, 'querymany') as mock_query:
            mock_query.side_effect = Exception("API Error")

            result = await id_converter.convert_ids(
                gene_ids=sample_gene_ids,
                input_type="symbol",
                output_type="entrezgene",
                species="human"
            )

            assert isinstance(result, IDMappingResult)
            assert "error" in result.metadata
            assert "API Error" in result.metadata["error"]

    def test_validate_gene_ids(self, id_converter):
        """Test gene ID validation."""
        # Valid gene IDs
        assert id_converter._validate_gene_ids(["GENE1", "GENE2"]) is True

        # Invalid gene IDs
        assert id_converter._validate_gene_ids([]) is False
        assert id_converter._validate_gene_ids(None) is False

    def test_validate_id_types(self, id_converter):
        """Test ID type validation."""
        # Valid ID types
        assert id_converter._validate_id_types("symbol", "entrezgene") is True
        assert id_converter._validate_id_types("ensembl_gene_id", "symbol") is True
        assert id_converter._validate_id_types("uniprot", "entrezgene") is True

        # Invalid ID types
        assert id_converter._validate_id_types("", "entrezgene") is False
        assert id_converter._validate_id_types("symbol", "") is False
        assert id_converter._validate_id_types(None, "entrezgene") is False
        assert id_converter._validate_id_types("symbol", None) is False

    def test_validate_species(self, id_converter):
        """Test species validation."""
        # Valid species
        assert id_converter._validate_species("human") is True
        assert id_converter._validate_species("mouse") is True

        # Invalid species
        assert id_converter._validate_species("") is False
        assert id_converter._validate_species(None) is False


class TestSpeciesMapper:
    """Test cases for the SpeciesMapper class."""

    @pytest.fixture
    def species_mapper(self):
        """Create a SpeciesMapper instance for testing."""
        return SpeciesMapper()

    @pytest.fixture
    def sample_gene_ids(self):
        """Create sample gene IDs."""
        return ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]

    def test_init(self, species_mapper):
        """Test SpeciesMapper initialization."""
        assert species_mapper.logger is not None
        assert species_mapper.ensembl_rest_url is not None

    @pytest.mark.asyncio
    async def test_map_species_basic(self, species_mapper, sample_gene_ids):
        """Test basic species mapping."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "data": [
                    {"id": "GENE1", "homologs": [{"id": "MOUSE_GENE1"}]},
                    {"id": "GENE2", "homologs": [{"id": "MOUSE_GENE2"}]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await species_mapper.map_species(
                gene_ids=sample_gene_ids,
                source_species="human",
                target_species="mouse"
            )

            assert isinstance(result, SpeciesMappingResult)
            assert result.source_species == "human"
            assert result.target_species == "mouse"
            assert len(result.mapped_ids) > 0

    @pytest.mark.asyncio
    async def test_map_species_empty_input(self, species_mapper):
        """Test species mapping with empty input."""
        result = await species_mapper.map_species(
            gene_ids=[],
            source_species="human",
            target_species="mouse"
        )

        assert isinstance(result, SpeciesMappingResult)
        assert len(result.mapped_ids) == 0

    @pytest.mark.asyncio
    async def test_map_species_api_error(self, species_mapper, sample_gene_ids):
        """Test species mapping with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await species_mapper.map_species(
                gene_ids=sample_gene_ids,
                source_species="human",
                target_species="mouse"
            )

            assert isinstance(result, SpeciesMappingResult)
            assert "error" in result.metadata
            assert "API Error" in result.metadata["error"]

    def test_validate_gene_ids(self, species_mapper):
        """Test gene ID validation."""
        # Valid gene IDs
        assert species_mapper._validate_gene_ids(["GENE1", "GENE2"]) is True

        # Invalid gene IDs
        assert species_mapper._validate_gene_ids([]) is False
        assert species_mapper._validate_gene_ids(None) is False

    def test_validate_species(self, species_mapper):
        """Test species validation."""
        # Valid species
        assert species_mapper._validate_species("human", "mouse") is True
        assert species_mapper._validate_species("mouse", "rat") is True

        # Invalid species
        assert species_mapper._validate_species("", "mouse") is False
        assert species_mapper._validate_species("human", "") is False
        assert species_mapper._validate_species(None, "mouse") is False
        assert species_mapper._validate_species("human", None) is False

    def test_validate_species_pair(self, species_mapper):
        """Test species pair validation."""
        # Valid species pairs
        assert species_mapper._validate_species_pair("human", "mouse") is True
        assert species_mapper._validate_species_pair("mouse", "rat") is True

        # Invalid species pairs
        assert species_mapper._validate_species_pair("human", "human") is False
        assert species_mapper._validate_species_pair("", "mouse") is False
        assert species_mapper._validate_species_pair("human", "") is False


class TestFormatDetector:
    """Test cases for the FormatDetector class."""

    @pytest.fixture
    def format_detector(self):
        """Create a FormatDetector instance for testing."""
        return FormatDetector()

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

    def test_init(self, format_detector):
        """Test FormatDetector initialization."""
        assert format_detector.logger is not None
        assert format_detector.supported_formats is not None

    def test_detect_format_basic(self, format_detector, sample_gene_list):
        """Test basic format detection."""
        result = format_detector.detect_format(sample_gene_list)

        assert isinstance(result, FormatDetectionResult)
        assert result.detected_format is not None
        assert result.confidence > 0
        assert result.confidence <= 1

    def test_detect_format_expression_data(self, format_detector, sample_expression_data):
        """Test format detection with expression data."""
        result = format_detector.detect_format(sample_expression_data)

        assert isinstance(result, FormatDetectionResult)
        assert result.detected_format is not None
        assert result.confidence > 0
        assert result.confidence <= 1

    def test_detect_format_empty_input(self, format_detector):
        """Test format detection with empty input."""
        result = format_detector.detect_format([])

        assert isinstance(result, FormatDetectionResult)
        assert result.detected_format is None
        assert result.confidence == 0

    def test_detect_format_invalid_input(self, format_detector):
        """Test format detection with invalid input."""
        result = format_detector.detect_format(None)

        assert isinstance(result, FormatDetectionResult)
        assert result.detected_format is None
        assert result.confidence == 0

    def test_validate_input_data(self, format_detector):
        """Test input data validation."""
        # Valid input data
        assert format_detector._validate_input_data(["GENE1", "GENE2"]) is True
        assert format_detector._validate_input_data(pd.DataFrame({"GENE1": [1, 2, 3]})) is True

        # Invalid input data
        assert format_detector._validate_input_data(None) is False
        assert format_detector._validate_input_data([]) is False
        assert format_detector._validate_input_data(pd.DataFrame()) is False

    def test_validate_detected_format(self, format_detector):
        """Test detected format validation."""
        # Valid detected format
        assert format_detector._validate_detected_format("gene_list") is True
        assert format_detector._validate_detected_format("expression_matrix") is True

        # Invalid detected format
        assert format_detector._validate_detected_format("") is False
        assert format_detector._validate_detected_format(None) is False

    def test_validate_confidence(self, format_detector):
        """Test confidence validation."""
        # Valid confidence
        assert format_detector._validate_confidence(0.5) is True
        assert format_detector._validate_confidence(0.0) is True
        assert format_detector._validate_confidence(1.0) is True

        # Invalid confidence
        assert format_detector._validate_confidence(-0.1) is False
        assert format_detector._validate_confidence(1.1) is False
        assert format_detector._validate_confidence(None) is False


class TestValidationEngine:
    """Test cases for the ValidationEngine class."""

    @pytest.fixture
    def validation_engine(self):
        """Create a ValidationEngine instance for testing."""
        return ValidationEngine()

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

    def test_init(self, validation_engine):
        """Test ValidationEngine initialization."""
        assert validation_engine.logger is not None
        assert validation_engine.validation_rules is not None

    def test_validate_basic(self, validation_engine, sample_gene_list):
        """Test basic validation."""
        result = validation_engine.validate(
            input_data=sample_gene_list,
            data_type="gene_list"
        )

        assert isinstance(result, ValidationResult)
        assert result.data_type == "gene_list"
        assert result.is_valid is not None
        assert len(result.validation_errors) >= 0

    def test_validate_expression_data(self, validation_engine, sample_expression_data):
        """Test validation with expression data."""
        result = validation_engine.validate(
            input_data=sample_expression_data,
            data_type="expression_matrix"
        )

        assert isinstance(result, ValidationResult)
        assert result.data_type == "expression_matrix"
        assert result.is_valid is not None
        assert len(result.validation_errors) >= 0

    def test_validate_empty_input(self, validation_engine):
        """Test validation with empty input."""
        result = validation_engine.validate(
            input_data=[],
            data_type="gene_list"
        )

        assert isinstance(result, ValidationResult)
        assert result.data_type == "gene_list"
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_validate_invalid_input(self, validation_engine):
        """Test validation with invalid input."""
        result = validation_engine.validate(
            input_data=None,
            data_type="gene_list"
        )

        assert isinstance(result, ValidationResult)
        assert result.data_type == "gene_list"
        assert result.is_valid is False
        assert len(result.validation_errors) > 0

    def test_validate_different_data_types(self, validation_engine, sample_gene_list):
        """Test validation with different data types."""
        data_types = ["gene_list", "expression_matrix", "pathway_data", "metadata"]

        for data_type in data_types:
            result = validation_engine.validate(
                input_data=sample_gene_list,
                data_type=data_type
            )

            assert isinstance(result, ValidationResult)
            assert result.data_type == data_type

    def test_validate_input_data(self, validation_engine):
        """Test input data validation."""
        # Valid input data
        assert validation_engine._validate_input_data(["GENE1", "GENE2"]) is True
        assert validation_engine._validate_input_data(pd.DataFrame({"GENE1": [1, 2, 3]})) is True

        # Invalid input data
        assert validation_engine._validate_input_data(None) is False
        assert validation_engine._validate_input_data([]) is False
        assert validation_engine._validate_input_data(pd.DataFrame()) is False

    def test_validate_data_type(self, validation_engine):
        """Test data type validation."""
        # Valid data types
        assert validation_engine._validate_data_type("gene_list") is True
        assert validation_engine._validate_data_type("expression_matrix") is True
        assert validation_engine._validate_data_type("pathway_data") is True
        assert validation_engine._validate_data_type("metadata") is True

        # Invalid data type
        assert validation_engine._validate_data_type("") is False
        assert validation_engine._validate_data_type(None) is False

    def test_validate_validation_result(self, validation_engine):
        """Test validation result validation."""
        # Valid validation result
        result = ValidationResult(
            data_type="gene_list",
            is_valid=True,
            validation_errors=[]
        )
        assert validation_engine._validate_validation_result(result) is True

        # Invalid validation result
        assert validation_engine._validate_validation_result(None) is False

    def test_validate_validation_errors(self, validation_engine):
        """Test validation errors validation."""
        # Valid validation errors
        assert validation_engine._validate_validation_errors([]) is True
        assert validation_engine._validate_validation_errors(["Error 1", "Error 2"]) is True

        # Invalid validation errors
        assert validation_engine._validate_validation_errors(None) is False
