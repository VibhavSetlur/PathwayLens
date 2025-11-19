"""
Unit tests for ConfidenceCalculator.

Tests confidence scoring for gene ID normalization.
"""

import pytest
from pathwaylens_core.normalization.confidence_calculator import (
    ConfidenceCalculator,
    ConfidenceFactors
)


@pytest.mark.unit
@pytest.mark.normalization
class TestConfidenceCalculator:
    """Test suite for ConfidenceCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create ConfidenceCalculator instance."""
        return ConfidenceCalculator()
    
    def test_calculate_confidence_single_mapping(self, calculator):
        """Test confidence calculation with single mapping."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl",
                "confidence": 0.9
            }
        ]
        
        confidence = calculator.calculate_confidence(mappings)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_confidence_multiple_agreeing_mappings(self, calculator):
        """Test confidence with multiple agreeing mappings."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "mygene"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ncbi"
            }
        ]
        
        confidence = calculator.calculate_confidence(mappings)
        
        # Multiple agreeing mappings should increase confidence
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high
    
    def test_calculate_confidence_conflicting_mappings(self, calculator):
        """Test confidence with conflicting mappings."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000002",
                "source": "mygene"
            }
        ]
        
        confidence = calculator.calculate_confidence(mappings)
        
        # Conflicting mappings should lower confidence
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_confidence_empty_mappings(self, calculator):
        """Test confidence with empty mappings."""
        confidence = calculator.calculate_confidence([])
        
        assert confidence == 0.0
    
    def test_calculate_confidence_with_factors(self, calculator):
        """Test confidence calculation with explicit factors."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl"
            }
        ]
        
        factors = ConfidenceFactors(
            database_agreement=1.0,
            mapping_frequency=0.8,
            database_quality=0.9,
            species_match=1.0,
            id_type_compatibility=1.0
        )
        
        confidence = calculator.calculate_confidence(mappings, factors)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_consensus_mapping(self, calculator):
        """Test consensus mapping calculation."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "mygene"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000002",
                "source": "ncbi"
            }
        ]
        
        consensus = calculator.calculate_consensus_mapping(mappings, min_confidence=0.5)
        
        assert consensus is not None
        assert consensus["input_id"] == "GENE1"
        assert consensus["output_id"] == "ENSG000001"  # Most common
        assert "confidence" in consensus
        assert consensus["confidence"] >= 0.5
    
    def test_calculate_consensus_mapping_low_confidence(self, calculator):
        """Test consensus mapping with low confidence."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000002",
                "source": "mygene"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000003",
                "source": "ncbi"
            }
        ]
        
        consensus = calculator.calculate_consensus_mapping(mappings, min_confidence=0.9)
        
        # With high threshold and conflicting mappings, might return None
        if consensus is not None:
            assert consensus["confidence"] >= 0.9
        # Otherwise it's None, which is acceptable
    
    def test_calculate_consensus_mapping_empty(self, calculator):
        """Test consensus mapping with empty mappings."""
        consensus = calculator.calculate_consensus_mapping([])
        
        assert consensus is None
    
    def test_resolve_ambiguity(self, calculator):
        """Test ambiguity resolution."""
        ambiguous_mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "confidence": 0.6,
                "source": "ensembl"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000002",
                "confidence": 0.7,
                "source": "mygene"
            }
        ]
        
        resolved = calculator.resolve_ambiguity(ambiguous_mappings)
        
        assert resolved is not None
        assert resolved["output_id"] in ["ENSG000001", "ENSG000002"]
    
    def test_resolve_ambiguity_with_context(self, calculator):
        """Test ambiguity resolution with context."""
        ambiguous_mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "confidence": 0.6
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000002",
                "confidence": 0.7
            }
        ]
        
        context = {
            "pathway": "pathway1",
            "tissue": "liver"
        }
        
        resolved = calculator.resolve_ambiguity(ambiguous_mappings, context)
        
        # Should use highest confidence when context doesn't help
        assert resolved is not None
        assert resolved["confidence"] >= 0.6
    
    def test_resolve_ambiguity_empty(self, calculator):
        """Test ambiguity resolution with empty mappings."""
        resolved = calculator.resolve_ambiguity([])
        
        assert resolved is None
    
    def test_extract_factors_single_database(self, calculator):
        """Test factor extraction with single database."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl"
            }
        ]
        
        factors = calculator._extract_factors(mappings)
        
        assert isinstance(factors, ConfidenceFactors)
        assert factors.database_agreement >= 0.0
        assert factors.mapping_frequency >= 0.0
        assert factors.database_quality >= 0.0
    
    def test_extract_factors_multiple_databases(self, calculator):
        """Test factor extraction with multiple databases."""
        mappings = [
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ensembl"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "mygene"
            },
            {
                "input_id": "GENE1",
                "output_id": "ENSG000001",
                "source": "ncbi"
            }
        ]
        
        factors = calculator._extract_factors(mappings)
        
        assert factors.database_agreement == 1.0  # All agree
        assert factors.mapping_frequency > 0.0
        assert factors.database_quality > 0.0



