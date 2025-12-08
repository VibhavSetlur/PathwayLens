"""
Custom database adapter for loading local gene set files (GMT/GMX).
"""

import os
from typing import List, Dict, Optional, Any
from loguru import logger

from .base import BaseAdapter, PathwayInfo, GeneInfo
from ...normalization.schemas import SpeciesType, IDType


class CustomAdapter(BaseAdapter):
    """Adapter for custom local databases (GMT/GMX files)."""
    
    def __init__(self, file_path: str, name: str = "custom"):
        """
        Initialize custom adapter.
        
        Args:
            file_path: Path to the GMT/GMX file
            name: Name of the custom database
        """
        super().__init__(base_url="local", rate_limit=float('inf'))
        self.file_path = file_path
        self.name = name
        self.pathways: Dict[str, PathwayInfo] = {}
        self.gene_to_pathways: Dict[str, List[str]] = {}
        self._loaded = False
    
    async def _load_data(self):
        """Load data from file."""
        if self._loaded:
            return

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Custom database file not found: {self.file_path}")
        
        try:
            with open(self.file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 3:
                        continue
                    
                    pathway_name = parts[0]
                    description = parts[1]
                    genes = parts[2:]
                    
                    # Remove empty strings
                    genes = [g for g in genes if g]
                    
                    pathway_id = pathway_name  # Use name as ID for custom sets
                    
                    info = PathwayInfo(
                        pathway_id=pathway_id,
                        name=pathway_name,
                        description=description,
                        genes=genes,
                        category="custom",
                        species="unknown" # Custom sets might not specify species
                    )
                    
                    self.pathways[pathway_id] = info
                    
                    for gene in genes:
                        if gene not in self.gene_to_pathways:
                            self.gene_to_pathways[gene] = []
                        self.gene_to_pathways[gene].append(pathway_id)
            
            self._loaded = True
            self.logger.info(f"Loaded {len(self.pathways)} pathways from {self.file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load custom database: {e}")
            raise

    async def get_pathways(self, species: SpeciesType) -> List[PathwayInfo]:
        """Get all pathways."""
        await self._load_data()
        return list(self.pathways.values())

    async def get_pathway_genes(self, pathway_id: str, species: SpeciesType) -> List[str]:
        """Get genes in a pathway."""
        await self._load_data()
        pathway = self.pathways.get(pathway_id)
        return pathway.genes if pathway else []

    async def get_gene_pathways(self, gene_id: str, species: SpeciesType) -> List[str]:
        """Get pathways containing a gene."""
        await self._load_data()
        return self.gene_to_pathways.get(gene_id, [])

    async def search_pathways(self, query: str, species: SpeciesType) -> List[PathwayInfo]:
        """Search pathways."""
        await self._load_data()
        query = query.lower()
        return [
            p for p in self.pathways.values() 
            if query in p.name.lower() or (p.description and query in p.description.lower())
        ]

    async def get_gene_info(self, gene_id: str, species: SpeciesType) -> Optional[GeneInfo]:
        """Get gene info (minimal support)."""
        return GeneInfo(gene_id=gene_id)

    async def convert_gene_id(self, gene_id: str, from_type: IDType, to_type: IDType, species: SpeciesType) -> Optional[str]:
        """ID conversion not supported for custom flat files."""
        return None

    async def get_supported_species(self) -> List[SpeciesType]:
        """Return generic list."""
        return [SpeciesType.HUMAN, SpeciesType.MOUSE] # Placeholder

    async def get_database_info(self) -> Dict[str, Any]:
        """Get database info."""
        await self._load_data()
        return {
            "name": self.name,
            "version": "custom",
            "pathway_count": len(self.pathways),
            "gene_count": len(self.gene_to_pathways)
        }
