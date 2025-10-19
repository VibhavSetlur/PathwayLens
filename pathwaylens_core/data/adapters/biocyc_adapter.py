"""
BioCyc database adapter for PathwayLens.
"""

import json
from typing import Dict, List, Optional, Any
from loguru import logger

from .base import BaseAdapter, PathwayInfo, GeneInfo
from ...normalization.schemas import SpeciesType, IDType


class BioCycAdapter(BaseAdapter):
    """Adapter for BioCyc database."""
    
    def __init__(self, base_url: str = "https://biocyc.org", rate_limit: float = 5.0):
        """
        Initialize BioCyc adapter.
        
        Args:
            base_url: Base URL for BioCyc API
            rate_limit: Rate limit in requests per second
        """
        super().__init__(base_url, rate_limit)
        self.logger = logger.bind(module="biocyc_adapter")
        
        # BioCyc species codes
        self.species_codes = {
            SpeciesType.HUMAN: "HUMAN",
            SpeciesType.MOUSE: "MOUSE",
            SpeciesType.RAT: "RAT",
            SpeciesType.DROSOPHILA: "DROSOPHILA",
            SpeciesType.ZEBRAFISH: "ZEBRAFISH",
            SpeciesType.C_ELEGANS: "C_ELEGANS",
            SpeciesType.S_CEREVISIAE: "S_CEREVISIAE"
        }
    
    async def get_pathways(self, species: SpeciesType) -> List[PathwayInfo]:
        """Get all pathways for a species."""
        species_code = self.species_codes.get(species)
        if not species_code:
            self.logger.warning(f"Species {species} not supported by BioCyc")
            return []
        
        try:
            # Get pathway list
            response = await self._make_request(f"pathways/{species_code}")
            
            pathways = []
            for pathway_data in response:
                pathways.append(PathwayInfo(
                    pathway_id=pathway_data.get('id', ''),
                    name=pathway_data.get('name', ''),
                    description=pathway_data.get('description', ''),
                    category=pathway_data.get('category', ''),
                    species=species_code,
                    url=f"https://biocyc.org/pathway/{pathway_data.get('id', '')}"
                ))
            
            return pathways
            
        except Exception as e:
            self.logger.error(f"Error getting pathways: {e}")
            return []
    
    async def get_pathway_genes(self, pathway_id: str, species: SpeciesType) -> List[str]:
        """Get genes in a pathway."""
        try:
            response = await self._make_request(f"pathway/{pathway_id}/genes")
            
            genes = []
            for gene_data in response:
                genes.append(gene_data.get('id', ''))
            
            return genes
            
        except Exception as e:
            self.logger.error(f"Error getting pathway genes: {e}")
            return []
    
    async def get_gene_pathways(self, gene_id: str, species: SpeciesType) -> List[str]:
        """Get pathways containing a gene."""
        try:
            response = await self._make_request(f"gene/{gene_id}/pathways")
            
            pathways = []
            for pathway_data in response:
                pathways.append(pathway_data.get('id', ''))
            
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
            response = await self._make_request(f"gene/{gene_id}")
            
            return GeneInfo(
                gene_id=gene_id,
                symbol=response.get('symbol', ''),
                name=response.get('name', ''),
                description=response.get('description', ''),
                species=species.value,
                url=f"https://biocyc.org/gene/{gene_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Error getting gene info: {e}")
            return None
    
    async def convert_gene_id(self, gene_id: str, from_type: IDType, to_type: IDType, species: SpeciesType) -> Optional[str]:
        """Convert gene identifier from one type to another."""
        # BioCyc primarily uses its own gene IDs
        # This is a simplified implementation
        if from_type == to_type:
            return gene_id
        
        # For now, return the same ID
        # In a full implementation, this would use BioCyc's conversion services
        return gene_id
    
    async def get_supported_species(self) -> List[SpeciesType]:
        """Get list of supported species."""
        return list(self.species_codes.keys())
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        return {
            "name": "BioCyc",
            "description": "BioCyc Database Collection",
            "version": "Latest",
            "url": "https://biocyc.org/",
            "api_url": self.base_url,
            "supported_species": [s.value for s in self.species_codes.keys()],
            "rate_limit": self.rate_limit
        }
