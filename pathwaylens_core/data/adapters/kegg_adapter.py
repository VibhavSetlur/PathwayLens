"""
KEGG database adapter for PathwayLens.
"""

import re
from typing import Dict, List, Optional, Any
from loguru import logger

from .base import BaseAdapter, PathwayInfo, GeneInfo
from ...normalization.schemas import SpeciesType, IDType


class KEGGAdapter(BaseAdapter):
    """Adapter for KEGG database."""
    
    def __init__(self, base_url: str = "https://rest.kegg.jp", rate_limit: float = 10.0):
        """
        Initialize KEGG adapter.
        
        Args:
            base_url: Base URL for KEGG API
            rate_limit: Rate limit in requests per second
        """
        super().__init__(base_url, rate_limit)
        self.logger = logger.bind(module="kegg_adapter")
        
        # KEGG species codes
        self.species_codes = {
            SpeciesType.HUMAN: "hsa",
            SpeciesType.MOUSE: "mmu", 
            SpeciesType.RAT: "rno",
            SpeciesType.DROSOPHILA: "dme",
            SpeciesType.ZEBRAFISH: "dre",
            SpeciesType.C_ELEGANS: "cel",
            SpeciesType.S_CEREVISIAE: "sce"
        }
    
    async def get_pathways(self, species: SpeciesType) -> List[PathwayInfo]:
        """Get all pathways for a species."""
        species_code = self.species_codes.get(species)
        if not species_code:
            self.logger.warning(f"Species {species} not supported by KEGG")
            return []
        
        try:
            # Get pathway list
            response = await self._make_request(f"list/pathway/{species_code}")
            
            pathways = []
            for line in response.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[0]
                        name = parts[1]
                        
                        pathways.append(PathwayInfo(
                            pathway_id=pathway_id,
                            name=name,
                            species=species_code,
                            url=f"https://www.genome.jp/kegg-bin/show_pathway?{pathway_id}"
                        ))
            
            return pathways
            
        except Exception as e:
            self.logger.error(f"Error getting pathways: {e}")
            return []
    
    async def get_pathway_genes(self, pathway_id: str, species: SpeciesType) -> List[str]:
        """Get genes in a pathway."""
        try:
            response = await self._make_request(f"link/{species.value}/pathway/{pathway_id}")
            
            genes = []
            for line in response.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        gene_id = parts[1]
                        genes.append(gene_id)
            
            return genes
            
        except Exception as e:
            self.logger.error(f"Error getting pathway genes: {e}")
            return []
    
    async def get_gene_pathways(self, gene_id: str, species: SpeciesType) -> List[str]:
        """Get pathways containing a gene."""
        try:
            response = await self._make_request(f"link/pathway/{species.value}:{gene_id}")
            
            pathways = []
            for line in response.split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id = parts[1]
                        pathways.append(pathway_id)
            
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
            species_code = self.species_codes.get(species)
            if not species_code:
                return None
            
            response = await self._make_request(f"get/{species_code}:{gene_id}")
            
            # Parse gene information
            lines = response.split('\n')
            symbol = None
            name = None
            description = None
            
            for line in lines:
                if line.startswith('NAME'):
                    name = line.split('NAME')[1].strip()
                elif line.startswith('DEFINITION'):
                    description = line.split('DEFINITION')[1].strip()
                elif line.startswith('SYMBOL'):
                    symbol = line.split('SYMBOL')[1].strip()
            
            return GeneInfo(
                gene_id=gene_id,
                symbol=symbol,
                name=name,
                description=description,
                species=species_code,
                url=f"https://www.genome.jp/dbget-bin/www_bget?{species_code}:{gene_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Error getting gene info: {e}")
            return None
    
    async def convert_gene_id(self, gene_id: str, from_type: IDType, to_type: IDType, species: SpeciesType) -> Optional[str]:
        """Convert gene identifier from one type to another."""
        # KEGG primarily uses its own gene IDs
        # This is a simplified implementation
        if from_type == to_type:
            return gene_id
        
        # For now, return the same ID
        # In a full implementation, this would use KEGG's conversion services
        return gene_id
    
    async def get_supported_species(self) -> List[SpeciesType]:
        """Get list of supported species."""
        return list(self.species_codes.keys())
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        return {
            "name": "KEGG",
            "description": "Kyoto Encyclopedia of Genes and Genomes",
            "version": "Latest",
            "url": "https://www.genome.jp/kegg/",
            "api_url": self.base_url,
            "supported_species": [s.value for s in self.species_codes.keys()],
            "rate_limit": self.rate_limit
        }
