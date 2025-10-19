"""
Reactome database adapter for PathwayLens.
"""

import json
from typing import Dict, List, Optional, Any
from loguru import logger

from .base import BaseAdapter, PathwayInfo, GeneInfo
from ...normalization.schemas import SpeciesType, IDType


class ReactomeAdapter(BaseAdapter):
    """Adapter for Reactome database."""
    
    def __init__(self, base_url: str = "https://reactome.org/AnalysisService", rate_limit: float = 10.0):
        """
        Initialize Reactome adapter.
        
        Args:
            base_url: Base URL for Reactome API
            rate_limit: Rate limit in requests per second
        """
        super().__init__(base_url, rate_limit)
        self.logger = logger.bind(module="reactome_adapter")
        
        # Reactome species codes
        self.species_codes = {
            SpeciesType.HUMAN: "Homo sapiens",
            SpeciesType.MOUSE: "Mus musculus",
            SpeciesType.RAT: "Rattus norvegicus",
            SpeciesType.DROSOPHILA: "Drosophila melanogaster",
            SpeciesType.ZEBRAFISH: "Danio rerio",
            SpeciesType.C_ELEGANS: "Caenorhabditis elegans",
            SpeciesType.S_CEREVISIAE: "Saccharomyces cerevisiae"
        }
    
    async def get_pathways(self, species: SpeciesType) -> List[PathwayInfo]:
        """Get all pathways for a species."""
        species_name = self.species_codes.get(species)
        if not species_name:
            self.logger.warning(f"Species {species} not supported by Reactome")
            return []
        
        try:
            # Get pathway list
            response = await self._make_request(f"species/{species_name}/pathways")
            
            pathways = []
            for pathway_data in response:
                pathways.append(PathwayInfo(
                    pathway_id=pathway_data.get('stId', ''),
                    name=pathway_data.get('displayName', ''),
                    description=pathway_data.get('summation', [{}])[0].get('text', '') if pathway_data.get('summation') else None,
                    species=species_name,
                    url=f"https://reactome.org/content/detail/{pathway_data.get('stId', '')}"
                ))
            
            return pathways
            
        except Exception as e:
            self.logger.error(f"Error getting pathways: {e}")
            return []
    
    async def get_pathway_genes(self, pathway_id: str, species: SpeciesType) -> List[str]:
        """Get genes in a pathway."""
        try:
            response = await self._make_request(f"pathway/{pathway_id}/participatingMolecules")
            
            genes = []
            for molecule in response:
                if molecule.get('type') == 'Gene':
                    genes.append(molecule.get('identifier', ''))
            
            return genes
            
        except Exception as e:
            self.logger.error(f"Error getting pathway genes: {e}")
            return []
    
    async def get_gene_pathways(self, gene_id: str, species: SpeciesType) -> List[str]:
        """Get pathways containing a gene."""
        try:
            response = await self._make_request(f"molecule/{gene_id}/pathways")
            
            pathways = []
            for pathway in response:
                pathways.append(pathway.get('stId', ''))
            
            return pathways
            
        except Exception as e:
            self.logger.error(f"Error getting gene pathways: {e}")
            return []
    
    async def search_pathways(self, query: str, species: SpeciesType) -> List[PathwayInfo]:
        """Search for pathways by name or description."""
        try:
            # Get all pathways and filter by query
            all_pathways = await self.get_pathways(species)
            
            matching_pathways = []
            query_lower = query.lower()
            
            for pathway in all_pathways:
                if (query_lower in pathway.name.lower() or 
                    (pathway.description and query_lower in pathway.description.lower())):
                    matching_pathways.append(pathway)
            
            return matching_pathways
            
        except Exception as e:
            self.logger.error(f"Error searching pathways: {e}")
            return []
    
    async def get_gene_info(self, gene_id: str, species: SpeciesType) -> Optional[GeneInfo]:
        """Get information about a gene."""
        try:
            response = await self._make_request(f"molecule/{gene_id}")
            
            return GeneInfo(
                gene_id=gene_id,
                symbol=response.get('displayName', ''),
                name=response.get('displayName', ''),
                description=response.get('summation', [{}])[0].get('text', '') if response.get('summation') else None,
                species=species.value,
                url=f"https://reactome.org/content/detail/{gene_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Error getting gene info: {e}")
            return None
    
    async def convert_gene_id(self, gene_id: str, from_type: IDType, to_type: IDType, species: SpeciesType) -> Optional[str]:
        """Convert gene identifier from one type to another."""
        # Reactome primarily uses its own gene IDs
        # This is a simplified implementation
        if from_type == to_type:
            return gene_id
        
        # For now, return the same ID
        # In a full implementation, this would use Reactome's conversion services
        return gene_id
    
    async def get_supported_species(self) -> List[SpeciesType]:
        """Get list of supported species."""
        return list(self.species_codes.keys())
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        return {
            "name": "Reactome",
            "description": "Reactome Pathway Database",
            "version": "Latest",
            "url": "https://reactome.org/",
            "api_url": self.base_url,
            "supported_species": [s.value for s in self.species_codes.keys()],
            "rate_limit": self.rate_limit
        }
