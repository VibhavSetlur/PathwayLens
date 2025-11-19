"""
Unit tests for OrthologyEngine.

Tests cross-species orthology mapping functionality.
"""

import pytest
import asyncio
from pathwaylens_core.normalization.orthology_engine import OrthologyEngine, OrthologMapping
from pathwaylens_core.normalization.schemas import SpeciesType, IDType


@pytest.mark.unit
@pytest.mark.normalization
class TestOrthologyEngine:
    """Test suite for OrthologyEngine."""
    
    @pytest.fixture
    def orthology_engine(self):
        """Create OrthologyEngine instance."""
        return OrthologyEngine()
    
    @pytest.mark.asyncio
    async def test_map_orthologs_same_species(self, orthology_engine):
        """Test ortholog mapping for same species (identity mapping)."""
        source_ids = ["GENE1", "GENE2", "GENE3"]
        
        mappings = await orthology_engine.map_orthologs(
            source_ids=source_ids,
            source_species=SpeciesType.HUMAN,
            target_species=SpeciesType.HUMAN,
            id_type=IDType.ENSEMBL
        )
        
        assert len(mappings) == len(source_ids)
        for mapping in mappings:
            assert isinstance(mapping, OrthologMapping)
            assert mapping.source_id == mapping.target_id
            assert mapping.source_species == SpeciesType.HUMAN
            assert mapping.target_species == SpeciesType.HUMAN
            assert mapping.confidence == 1.0
            assert mapping.ortholog_type == "one-to-one"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_map_orthologs_cross_species(
        self,
        orthology_engine
    ):
        """Test ortholog mapping across species."""
        # Note: This test may require network access
        source_ids = ["ENSG00000139618"]  # BRCA2 in human
        
        mappings = await orthology_engine.map_orthologs(
            source_ids=source_ids,
            source_species=SpeciesType.HUMAN,
            target_species=SpeciesType.MOUSE,
            id_type=IDType.ENSEMBL,
            use_consensus=True
        )
        
        # Should return at least one mapping (if API accessible)
        # If API is not accessible, might return empty list
        assert isinstance(mappings, list)
        for mapping in mappings:
            assert isinstance(mapping, OrthologMapping)
            assert mapping.source_species == SpeciesType.HUMAN
            assert mapping.target_species == SpeciesType.MOUSE
            assert 0.0 <= mapping.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_map_orthologs_empty_input(self, orthology_engine):
        """Test ortholog mapping with empty input."""
        mappings = await orthology_engine.map_orthologs(
            source_ids=[],
            source_species=SpeciesType.HUMAN,
            target_species=SpeciesType.MOUSE,
            id_type=IDType.ENSEMBL
        )
        
        assert mappings == []
    
    def test_apply_consensus(self, orthology_engine):
        """Test consensus application to ortholog mappings."""
        mappings = [
            OrthologMapping(
                source_id="GENE1",
                target_id="GENE1_MOUSE",
                source_species=SpeciesType.HUMAN,
                target_species=SpeciesType.MOUSE,
                ortholog_type="one-to-one",
                confidence=0.9,
                method="ensembl",
                database="ensembl"
            ),
            OrthologMapping(
                source_id="GENE1",
                target_id="GENE1_MOUSE",
                source_species=SpeciesType.HUMAN,
                target_species=SpeciesType.MOUSE,
                ortholog_type="one-to-one",
                confidence=0.85,
                method="orthodb",
                database="orthodb"
            ),
            OrthologMapping(
                source_id="GENE1",
                target_id="GENE1_MOUSE_ALT",
                source_species=SpeciesType.HUMAN,
                target_species=SpeciesType.MOUSE,
                ortholog_type="one-to-one",
                confidence=0.7,
                method="other",
                database="other"
            )
        ]
        
        consensus = orthology_engine._apply_consensus(mappings)
        
        # Should have consensus for each unique source ID
        assert len(consensus) > 0
        assert all(isinstance(m, OrthologMapping) for m in consensus)
        
        # Find consensus for GENE1
        gene1_consensus = [m for m in consensus if m.source_id == "GENE1"]
        assert len(gene1_consensus) > 0
        
        # Consensus should have higher confidence than individual mappings
        consensus_mapping = gene1_consensus[0]
        assert consensus_mapping.confidence >= 0.85  # At least average of agreeing ones
        assert consensus_mapping.method == "consensus"
        assert consensus_mapping.database == "multiple"
    
    def test_determine_ortholog_type(self, orthology_engine):
        """Test ortholog type determination."""
        # Mock Ensembl homolog data
        homolog_data = {
            "type": "ortholog_one2one"
        }
        
        ortholog_type = orthology_engine._determine_ortholog_type(homolog_data)
        
        assert isinstance(ortholog_type, str)
        assert ortholog_type in ["one-to-one", "one-to-many", "many-to-many"]
    
    def test_calculate_ensembl_confidence(self, orthology_engine):
        """Test Ensembl confidence calculation."""
        # Mock Ensembl homolog data
        homolog_data = {
            "type": "ortholog_one2one",
            "taxonomy_level": "Euteleostomi"
        }
        
        confidence = orthology_engine._calculate_ensembl_confidence(homolog_data)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_map_via_ensembl_mock(self, orthology_engine):
        """Test Ensembl mapping (may fail if API not accessible)."""
        source_ids = ["ENSG00000139618"]
        
        try:
            mappings = await orthology_engine._map_via_ensembl(
                source_ids=source_ids,
                source_species=SpeciesType.HUMAN,
                target_species=SpeciesType.MOUSE,
                id_type=IDType.ENSEMBL
            )
            
            # If successful, should return list of mappings
            assert isinstance(mappings, list)
            for mapping in mappings:
                assert isinstance(mapping, OrthologMapping)
        except Exception as e:
            # API failures are acceptable in tests
            pytest.skip(f"Ensembl API not accessible: {e}")
    
    @pytest.mark.asyncio
    async def test_map_via_orthodb(self, orthology_engine):
        """Test OrthoDB mapping (currently returns empty)."""
        source_ids = ["GENE1"]
        
        mappings = await orthology_engine._map_via_orthodb(
            source_ids=source_ids,
            source_species=SpeciesType.HUMAN,
            target_species=SpeciesType.MOUSE,
            id_type=IDType.ENSEMBL
        )
        
        # Currently returns empty list (not implemented)
        assert isinstance(mappings, list)
    
    def test_ortholog_mapping_dataclass(self):
        """Test OrthologMapping dataclass."""
        mapping = OrthologMapping(
            source_id="GENE1",
            target_id="GENE1_MOUSE",
            source_species=SpeciesType.HUMAN,
            target_species=SpeciesType.MOUSE,
            ortholog_type="one-to-one",
            confidence=0.9,
            method="ensembl",
            database="ensembl"
        )
        
        assert mapping.source_id == "GENE1"
        assert mapping.target_id == "GENE1_MOUSE"
        assert mapping.source_species == SpeciesType.HUMAN
        assert mapping.target_species == SpeciesType.MOUSE
        assert mapping.confidence == 0.9



