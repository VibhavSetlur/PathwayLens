"""
Pathway Commons database adapter for PathwayLens.
"""

import json
from typing import Dict, List, Optional, Any
from loguru import logger

from .base import BaseAdapter, PathwayInfo, GeneInfo
from ...normalization.schemas import SpeciesType, IDType


class PathwayCommonsAdapter(BaseAdapter):
    """Adapter for Pathway Commons database."""
    
    def __init__(self, base_url: str = "https://www.pathwaycommons.org", rate_limit: float = 5.0):
        """
        Initialize Pathway Commons adapter.
        
        Args:
            base_url: Base URL for Pathway Commons API
            rate_limit: Rate limit in requests per second
        """
        super().__init__(base_url, rate_limit)
        self.logger = logger.bind(module="pathway_commons_adapter")
        
        # Pathway Commons species codes
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
            self.logger.warning(f"Species {species} not supported by Pathway Commons")
            return []
        
        try:
            # Get pathway list
            response = await self._make_request(f"pc2/pathways?organism={species_name}")
            
            pathways = []
            for pathway_data in response.get('searchHit', []):
                pathway_info = pathway_data.get('pathway', {})
                pathways.append(PathwayInfo(
                    pathway_id=pathway_info.get('uri', ''),
                    name=pathway_info.get('displayName', ''),
                    description=pathway_info.get('comment', ''),
                    species=species_name,
                    url=pathway_info.get('uri', '')
                ))
            
            return pathways
            
        except Exception as e:
            self.logger.error(f"Error getting pathways: {e}")
            return []
    
    async def get_pathway_genes(self, pathway_id: str, species: SpeciesType) -> List[str]:
        """Get genes in a pathway."""
        try:
            response = await self._make_request(f"pc2/pathway/{pathway_id}")
            
            genes = []
            for participant in response.get('pathway', {}).get('participant', []):
                if participant.get('type') == 'Gene':
                    genes.append(participant.get('uri', ''))
            
            return genes
            
        except Exception as e:
            self.logger.error(f"Error getting pathway genes: {e}")
            return []
    
    async def get_gene_pathways(self, gene_id: str, species: SpeciesType) -> List[str]:
        """Get pathways containing a gene."""
        try:
            response = await self._make_request(f"pc2/gene/{gene_id}")
            
            pathways = []
            for pathway in response.get('pathway', []):
                pathways.append(pathway.get('uri', ''))
            
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
            response = await self._make_request(f"pc2/gene/{gene_id}")
            
            gene_info = response.get('gene', {})
            return GeneInfo(
                gene_id=gene_id,
                symbol=gene_info.get('displayName', ''),
                name=gene_info.get('displayName', ''),
                description=gene_info.get('comment', ''),
                species=species.value,
                url=gene_info.get('uri', '')
            )
            
        except Exception as e:
            self.logger.error(f"Error getting gene info: {e}")
            return None
    
    async def convert_gene_id(self, gene_id: str, from_type: IDType, to_type: IDType, species: SpeciesType) -> Optional[str]:
        """Convert gene identifier from one type to another."""
        # Pathway Commons primarily uses its own gene IDs
        # This is a simplified implementation
        if from_type == to_type:
            return gene_id
        
        # For now, return the same ID
        # In a full implementation, this would use Pathway Commons' conversion services
        return gene_id
    
    async def get_supported_species(self) -> List[SpeciesType]:
        """Get list of supported species."""
        return list(self.species_codes.keys())
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        return {
            "name": "Pathway Commons",
            "description": "Pathway Commons Database",
            "version": "Latest",
            "url": "https://www.pathwaycommons.org/",
            "api_url": self.base_url,
            "supported_species": [s.value for s in self.species_codes.keys()],
            "rate_limit": self.rate_limit
        }
