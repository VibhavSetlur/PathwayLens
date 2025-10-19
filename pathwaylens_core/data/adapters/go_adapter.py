"""
Gene Ontology (GO) database adapter for PathwayLens.
"""

import json
from typing import Dict, List, Optional, Any
from loguru import logger

from .base import BaseAdapter, PathwayInfo, GeneInfo
from ...normalization.schemas import SpeciesType, IDType


class GOAdapter(BaseAdapter):
    """Adapter for Gene Ontology database."""
    
    def __init__(self, base_url: str = "http://current.geneontology.org", rate_limit: float = 5.0):
        """
        Initialize GO adapter.
        
        Args:
            base_url: Base URL for GO API
            rate_limit: Rate limit in requests per second
        """
        super().__init__(base_url, rate_limit)
        self.logger = logger.bind(module="go_adapter")
        
        # GO namespaces
        self.namespaces = {
            "biological_process": "BP",
            "molecular_function": "MF", 
            "cellular_component": "CC"
        }
    
    async def get_pathways(self, species: SpeciesType) -> List[PathwayInfo]:
        """Get all pathways for a species."""
        try:
            # Get GO terms (pathways)
            response = await self._make_request(f"ontology/go.json")
            
            pathways = []
            for term_id, term_data in response.get('graphs', [{}])[0].get('nodes', {}).items():
                if term_data.get('type') == 'CLASS':
                    pathways.append(PathwayInfo(
                        pathway_id=term_id,
                        name=term_data.get('lbl', ''),
                        description=term_data.get('meta', {}).get('definition', {}).get('val', ''),
                        category=term_data.get('meta', {}).get('basicPropertyValues', [{}])[0].get('val', '') if term_data.get('meta', {}).get('basicPropertyValues') else None,
                        species=species.value,
                        url=f"http://amigo.geneontology.org/amigo/term/{term_id}"
                    ))
            
            return pathways
            
        except Exception as e:
            self.logger.error(f"Error getting pathways: {e}")
            return []
    
    async def get_pathway_genes(self, pathway_id: str, species: SpeciesType) -> List[str]:
        """Get genes in a pathway."""
        try:
            response = await self._make_request(f"annotations/{species.value}.json")
            
            genes = []
            for annotation in response.get('annotations', []):
                if annotation.get('object', {}).get('id') == pathway_id:
                    genes.append(annotation.get('subject', {}).get('id', ''))
            
            return genes
            
        except Exception as e:
            self.logger.error(f"Error getting pathway genes: {e}")
            return []
    
    async def get_gene_pathways(self, gene_id: str, species: SpeciesType) -> List[str]:
        """Get pathways containing a gene."""
        try:
            response = await self._make_request(f"annotations/{species.value}.json")
            
            pathways = []
            for annotation in response.get('annotations', []):
                if annotation.get('subject', {}).get('id') == gene_id:
                    pathways.append(annotation.get('object', {}).get('id', ''))
            
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
            response = await self._make_request(f"annotations/{species.value}.json")
            
            # Find gene information
            for annotation in response.get('annotations', []):
                if annotation.get('subject', {}).get('id') == gene_id:
                    return GeneInfo(
                        gene_id=gene_id,
                        symbol=annotation.get('subject', {}).get('label', ''),
                        name=annotation.get('subject', {}).get('label', ''),
                        description=annotation.get('subject', {}).get('description', ''),
                        species=species.value,
                        url=f"http://amigo.geneontology.org/amigo/gene_product/{gene_id}"
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting gene info: {e}")
            return None
    
    async def convert_gene_id(self, gene_id: str, from_type: IDType, to_type: IDType, species: SpeciesType) -> Optional[str]:
        """Convert gene identifier from one type to another."""
        # GO primarily uses its own gene IDs
        # This is a simplified implementation
        if from_type == to_type:
            return gene_id
        
        # For now, return the same ID
        # In a full implementation, this would use GO's conversion services
        return gene_id
    
    async def get_supported_species(self) -> List[SpeciesType]:
        """Get list of supported species."""
        return [SpeciesType.HUMAN, SpeciesType.MOUSE, SpeciesType.RAT, SpeciesType.DROSOPHILA, SpeciesType.ZEBRAFISH, SpeciesType.C_ELEGANS, SpeciesType.S_CEREVISIAE]
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        return {
            "name": "Gene Ontology",
            "description": "Gene Ontology Database",
            "version": "Latest",
            "url": "http://geneontology.org/",
            "api_url": self.base_url,
            "supported_species": [s.value for s in self.get_supported_species()],
            "rate_limit": self.rate_limit
        }
