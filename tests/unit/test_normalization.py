"""
Unit tests for the Normalization modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from pathwaylens_core.normalization.normalizer import Normalizer
from pathwaylens_core.normalization.id_converter import IDConverter
from pathwaylens_core.normalization.species_mapper import SpeciesMapper
from pathwaylens_core.normalization.format_detector import FormatDetector
from pathwaylens_core.normalization.validation import InputValidator
from pathwaylens_core.normalization.schemas import (
    NormalizationResult, ConversionMapping, CrossSpeciesMapping,
    FormatDetectionResult, ValidationResult, SpeciesType, IDType
)


class TestNormalizer:
    """Test cases for the Normalizer class."""

    @pytest.fixture
    def normalizer(self):
        """Create a Normalizer instance for testing."""
        return Normalizer()

    def test_init(self, normalizer):
        """Test Normalizer initialization."""
        assert normalizer.logger is not None
        assert normalizer.id_converter is not None
        assert normalizer.species_mapper is not None
        assert normalizer.format_detector is not None
        assert normalizer.validator is not None

    @pytest.mark.asyncio
    async def test_normalize_file_basic(self, normalizer, tmp_path):
        """Test basic file normalization."""
        # Create a dummy input file
        input_file = tmp_path / "test_input.csv"
        input_file.write_text("gene_id,value\nGENE1,1.0\nGENE2,2.0")
        
        # Mock internal components to avoid external calls
        with patch.object(normalizer.format_detector, 'detect_format') as mock_detect:
            mock_detect.return_value = FormatDetectionResult(
                format_type='csv', confidence=1.0, delimiter=',', encoding='utf-8', has_header=True
            )
            
            with patch.object(normalizer, '_read_file', return_value=pd.DataFrame({'gene_id': ['GENE1', 'GENE2'], 'value': [1.0, 2.0]})):
                with patch.object(normalizer, '_detect_species', return_value=SpeciesType.HUMAN):
                    with patch.object(normalizer, '_detect_id_type', return_value=IDType.SYMBOL):
                        with patch.object(normalizer, '_extract_gene_ids', return_value=['GENE1', 'GENE2']):
                            with patch.object(normalizer, '_convert_identifiers', return_value=[]):
                                with patch.object(normalizer, '_create_normalized_table') as mock_create:
                                    mock_create.return_value = Mock()
                                    with patch.object(normalizer, '_save_output'):
                                        
                                        result = await normalizer.normalize_file(
                                            input_file=str(input_file),
                                            target_id_type=IDType.ENTREZ,
                                            target_species=SpeciesType.HUMAN
                                        )
                                        
                                        assert isinstance(result, NormalizationResult)
                                        assert result.input_file == str(input_file)


class TestIDConverter:
    """Test cases for the IDConverter class."""

    @pytest.fixture
    def id_converter(self):
        """Create an IDConverter instance for testing."""
        return IDConverter()

    def test_init(self, id_converter):
        """Test IDConverter initialization."""
        assert id_converter.logger is not None
        assert id_converter.mygene_url is not None

    @pytest.mark.asyncio
    async def test_convert_ids_basic(self, id_converter):
        """Test basic ID conversion."""
        # Mock the internal conversion method to avoid network calls
        from pathwaylens_core.normalization.id_converter import ConversionResult
        with patch.object(id_converter, '_convert_via_mygene') as mock_convert:
            mock_convert.return_value = [
                ConversionResult(
                    input_id="GENE1",
                    output_id="12345",
                    confidence=0.9,
                    source="mygene"
                )
            ]
            
            # Mock other methods to return empty lists
            with patch.object(id_converter, '_convert_via_ensembl', return_value=[]):
                with patch.object(id_converter, '_convert_via_ncbi', return_value=[]):
                    
                    result = await id_converter.convert_identifiers(
                        identifiers=["GENE1"],
                        input_type=IDType.SYMBOL,
                        output_type=IDType.ENTREZ,
                        species=SpeciesType.HUMAN
                    )
                    
                    assert len(result) > 0
                    assert result[0].input_id == "GENE1"
                    assert result[0].output_id == "12345"

    @pytest.mark.asyncio
    async def test_convert_ids_empty_input(self, id_converter):
        """Test ID conversion with empty input."""
        result = await id_converter.convert_identifiers(
            identifiers=[],
            input_type=IDType.SYMBOL,
            output_type=IDType.ENTREZ,
            species=SpeciesType.HUMAN
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_convert_ids_api_error(self, id_converter):
        """Test ID conversion with API error."""
        # Mock methods to simulate failure
        with patch.object(id_converter, '_convert_via_mygene', side_effect=Exception("API Error")):
            with patch.object(id_converter, '_convert_via_ensembl', return_value=[]):
                with patch.object(id_converter, '_convert_via_ncbi', return_value=[]):
                    
                    result = await id_converter.convert_identifiers(
                        identifiers=["GENE1"],
                        input_type=IDType.SYMBOL,
                        output_type=IDType.ENTREZ,
                        species=SpeciesType.HUMAN
                    )
                    
                    # Should return empty list or partial results, not raise exception
                    assert isinstance(result, list)


class TestSpeciesMapper:
    """Test cases for the SpeciesMapper class."""

    @pytest.fixture
    def species_mapper(self):
        """Create a SpeciesMapper instance for testing."""
        return SpeciesMapper()

    def test_init(self, species_mapper):
        """Test SpeciesMapper initialization."""
        assert species_mapper.logger is not None
        assert species_mapper.ensembl_url is not None

    @pytest.mark.asyncio
    async def test_map_species_basic(self, species_mapper):
        """Test basic species mapping."""
        # Mock internal mapping method
        with patch.object(species_mapper, '_map_via_ensembl') as mock_map:
            from pathwaylens_core.normalization.species_mapper import OrthologResult
            mock_map.return_value = [
                OrthologResult(
                    source_id="GENE1",
                    target_id="MOUSE_GENE1",
                    source_species=SpeciesType.HUMAN,
                    target_species=SpeciesType.MOUSE,
                    ortholog_type="ortholog_one2one",
                    confidence=1.0,
                    method="ensembl"
                )
            ]
            
            with patch.object(species_mapper, '_map_via_homologene', return_value=[]):
                with patch.object(species_mapper, '_map_via_orthodb', return_value=[]):
                    
                    result = await species_mapper.map_orthologs(
                        identifiers=["GENE1"],
                        source_species=SpeciesType.HUMAN,
                        target_species=SpeciesType.MOUSE
                    )
                    
                    assert len(result) > 0
                    assert result[0].source_id == "GENE1"
                    assert result[0].target_id == "MOUSE_GENE1"

    @pytest.mark.asyncio
    async def test_map_species_empty_input(self, species_mapper):
        """Test species mapping with empty input."""
        result = await species_mapper.map_orthologs(
            identifiers=[],
            source_species=SpeciesType.HUMAN,
            target_species=SpeciesType.MOUSE
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_map_species_api_error(self, species_mapper):
        """Test species mapping with API error."""
        with patch.object(species_mapper, '_map_via_ensembl', side_effect=Exception("API Error")):
            with patch.object(species_mapper, '_map_via_homologene', return_value=[]):
                with patch.object(species_mapper, '_map_via_orthodb', return_value=[]):
                    
                    result = await species_mapper.map_orthologs(
                        identifiers=["GENE1"],
                        source_species=SpeciesType.HUMAN,
                        target_species=SpeciesType.MOUSE
                    )
                    
                    assert isinstance(result, list)


class TestFormatDetector:
    """Test cases for the FormatDetector class."""

    @pytest.fixture
    def format_detector(self):
        """Create a FormatDetector instance for testing."""
        return FormatDetector()

    def test_init(self, format_detector):
        """Test FormatDetector initialization."""
        assert format_detector.logger is not None
        assert format_detector.SUPPORTED_FORMATS is not None

    def test_detect_format_csv(self, format_detector, tmp_path):
        """Test CSV format detection."""
        f = tmp_path / "test.csv"
        f.write_text("col1,col2\n1,2")
        result = format_detector.detect_format(str(f))
        assert result.format_type == 'csv'
        assert result.confidence > 0.5

    def test_detect_format_json(self, format_detector, tmp_path):
        """Test JSON format detection."""
        f = tmp_path / "test.json"
        f.write_text('[{"col1": 1, "col2": 2}]')
        result = format_detector.detect_format(str(f))
        assert result.format_type == 'json'
        assert result.confidence > 0.5

    def test_detect_format_invalid_file(self, format_detector):
        """Test detection with non-existent file."""
        with pytest.raises(FileNotFoundError):
            format_detector.detect_format("non_existent_file.txt")


class TestInputValidator:
    """Test cases for InputValidator."""

    @pytest.fixture
    def input_validator(self):
        """Create an InputValidator instance for testing."""
        return InputValidator()

    def test_init(self, input_validator):
        """Test initialization."""
        assert input_validator is not None

    def test_validate_dataframe_basic(self, input_validator):
        """Test basic DataFrame validation."""
        df = pd.DataFrame({
            'gene': ['g1', 'g2'],
            'value': [1.0, 2.0]
        })
        result = input_validator.validate_dataframe(df)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_dataframe_empty(self, input_validator):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        result = input_validator.validate_dataframe(df)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_analysis_parameters(self, input_validator):
        """Test analysis parameters validation."""
        params = {
            'species': SpeciesType.HUMAN,
            'databases': ['kegg']
        }
        result = input_validator.validate_analysis_parameters(params)
        assert result.is_valid

    def test_validate_analysis_parameters_missing(self, input_validator):
        """Test validation with missing parameters."""
        params = {}
        result = input_validator.validate_analysis_parameters(params)
        assert not result.is_valid
        assert len(result.errors) > 0
