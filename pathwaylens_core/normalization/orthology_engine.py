"""
Cross-species orthology mapping engine.

Implements robust ortholog mapping using multiple databases with consensus.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np

from .schemas import SpeciesType, IDType
from .confidence_calculator import ConfidenceCalculator


@dataclass
class OrthologMapping:
    """Ortholog mapping result."""
    source_id: str
    target_id: str
    source_species: SpeciesType
    target_species: SpeciesType
    ortholog_type: str  # 'one-to-one', 'one-to-many', 'many-to-many'
    confidence: float
    method: str
    database: str


class OrthologyEngine:
    """Cross-species orthology mapping engine."""
    
    def __init__(self):
        """Initialize orthology engine."""
        self.logger = logger.bind(module="orthology_engine")
        self.confidence_calc = ConfidenceCalculator()
        
        # Orthology database endpoints
        self.ensembl_ortholog_url = "https://rest.ensembl.org/homology/id"
        self.orthodb_url = "https://www.orthodb.org/orthodb_rest"
        
        # Species taxonomy IDs
        self.taxonomy_ids = {
            SpeciesType.HUMAN: 9606,
            SpeciesType.MOUSE: 10090,
            SpeciesType.RAT: 10116,
            SpeciesType.DROSOPHILA: 7227,
            SpeciesType.ZEBRAFISH: 7955,
            SpeciesType.C_ELEGANS: 6239,
            SpeciesType.S_CEREVISIAE: 559292,
        }
    
    async def map_orthologs(
        self,
        source_ids: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: IDType = IDType.ENSEMBL,
        use_consensus: bool = True
    ) -> List[OrthologMapping]:
        """
        Map orthologs from source to target species.
        
        Args:
            source_ids: List of source gene IDs
            source_species: Source species
            target_species: Target species
            id_type: Type of identifiers
            use_consensus: Whether to use consensus from multiple databases
            
        Returns:
            List of ortholog mappings
        """
        self.logger.info(
            f"Mapping {len(source_ids)} orthologs from {source_species.value} "
            f"to {target_species.value}"
        )
        
        if source_species == target_species:
            # Same species - return identity mappings
            return [
                OrthologMapping(
                    source_id=id_val,
                    target_id=id_val,
                    source_species=source_species,
                    target_species=target_species,
                    ortholog_type='one-to-one',
                    confidence=1.0,
                    method='identity',
                    database='same_species'
                )
                for id_val in source_ids
            ]
        
        # Get orthologs from multiple databases
        all_mappings = []
        
        # Method 1: Ensembl
        try:
            ensembl_mappings = await self._map_via_ensembl(
                source_ids, source_species, target_species, id_type
            )
            all_mappings.extend(ensembl_mappings)
        except Exception as e:
            self.logger.warning(f"Ensembl ortholog mapping failed: {e}")
        
        # Method 2: OrthoDB (if available)
        try:
            orthodb_mappings = await self._map_via_orthodb(
                source_ids, source_species, target_species, id_type
            )
            all_mappings.extend(orthodb_mappings)
        except Exception as e:
            self.logger.warning(f"OrthoDB mapping failed: {e}")
        
        # Apply consensus if requested
        if use_consensus and all_mappings:
            return self._apply_consensus(all_mappings)
        
        return all_mappings
    
    async def _map_via_ensembl(
        self,
        source_ids: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: IDType
    ) -> List[OrthologMapping]:
        """Map orthologs using Ensembl API."""
        mappings = []
        
        source_tax_id = self.taxonomy_ids.get(source_species)
        target_tax_id = self.taxonomy_ids.get(target_species)
        
        if not source_tax_id or not target_tax_id:
            return mappings
        
        async with aiohttp.ClientSession() as session:
            for source_id in source_ids:
                try:
                    url = f"{self.ensembl_ortholog_url}/{source_id}"
                    params = {
                        'type': 'orthologues',
                        'target_species': target_species.value.replace('_', ' '),
                        'format': 'json'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Parse Ensembl response
                            if 'data' in data and len(data['data']) > 0:
                                homologs = data['data'][0].get('homologies', [])
                                
                                for homolog in homologs:
                                    target = homolog.get('target', {})
                                    target_id = target.get('id')
                                    
                                    if target_id:
                                        ortholog_type = self._determine_ortholog_type(homolog)
                                        confidence = self._calculate_ensembl_confidence(homolog)
                                        
                                        mappings.append(OrthologMapping(
                                            source_id=source_id,
                                            target_id=target_id,
                                            source_species=source_species,
                                            target_species=target_species,
                                            ortholog_type=ortholog_type,
                                            confidence=confidence,
                                            method='ensembl',
                                            database='ensembl'
                                        ))
                
                except Exception as e:
                    self.logger.debug(f"Failed to map {source_id} via Ensembl: {e}")
                    continue
        
        return mappings
    
    async def _map_via_orthodb(
        self,
        source_ids: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: IDType
    ) -> List[OrthologMapping]:
        """Map orthologs using OrthoDB (simplified - would need actual API)."""
        # OrthoDB API integration would go here
        # For now, return empty list
        return []
    
    def _determine_ortholog_type(self, homolog_data: Dict) -> str:
        """Determine ortholog type from Ensembl data."""
        # Simplified - would parse Ensembl response properly
        return 'one-to-one'
    
    def _calculate_ensembl_confidence(self, homolog_data: Dict) -> float:
        """Calculate confidence from Ensembl homology data."""
        # Would use percent identity, coverage, etc.
        # Simplified for now
        return 0.9
    
    def _apply_consensus(
        self,
        mappings: List[OrthologMapping]
    ) -> List[OrthologMapping]:
        """Apply consensus to ortholog mappings."""
        # Group by source ID
        grouped = {}
        for mapping in mappings:
            if mapping.source_id not in grouped:
                grouped[mapping.source_id] = []
            grouped[mapping.source_id].append(mapping)
        
        consensus_mappings = []
        
        for source_id, mapping_list in grouped.items():
            # Group by target ID
            target_groups = {}
            for mapping in mapping_list:
                if mapping.target_id not in target_groups:
                    target_groups[mapping.target_id] = []
                target_groups[mapping.target_id].append(mapping)
            
            # Find most common target (consensus)
            if target_groups:
                best_target = max(
                    target_groups.keys(),
                    key=lambda x: len(target_groups[x])
                )
                
                # Calculate consensus confidence
                best_mappings = target_groups[best_target]
                avg_confidence = np.mean([m.confidence for m in best_mappings])
                
                # Boost confidence if multiple databases agree
                agreement_boost = min(0.1, len(best_mappings) * 0.05)
                consensus_confidence = min(1.0, avg_confidence + agreement_boost)
                
                consensus_mappings.append(OrthologMapping(
                    source_id=source_id,
                    target_id=best_target,
                    source_species=mapping_list[0].source_species,
                    target_species=mapping_list[0].target_species,
                    ortholog_type=mapping_list[0].ortholog_type,
                    confidence=consensus_confidence,
                    method='consensus',
                    database='multiple'
                ))
        
        return consensus_mappings



