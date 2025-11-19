"""
STRING database API integration for PathwayLens.

Provides protein-protein interaction and pathway analysis using STRING.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential


class StringAPIClient:
    """Client for STRING database API."""
    
    BASE_URL = "https://string-db.org/api"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize STRING API client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.logger = logger.bind(module="string_api")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def enrich(
        self,
        gene_list: List[str],
        species: str = "9606",  # Human NCBI tax ID
        required_score: int = 400,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform pathway enrichment using STRING.
        
        Args:
            gene_list: List of protein identifiers
            species: NCBI taxonomy ID (9606 for human, 10090 for mouse)
            required_score: Minimum interaction score (0-1000)
            **kwargs: Additional parameters
            
        Returns:
            Enrichment results dictionary
        """
        self.logger.info(f"Querying STRING for {len(gene_list)} proteins")
        
        # Map species names to taxonomy IDs
        species_map = {
            "human": "9606",
            "mouse": "10090",
            "rat": "10116",
            "drosophila": "7227",
            "zebrafish": "7955"
        }
        
        if species.lower() in species_map:
            species_id = species_map[species.lower()]
        else:
            species_id = species
        
        url = f"{self.BASE_URL}/json/enrichment"
        
        params = {
            "identifiers": "%0d".join(gene_list),
            "species": species_id,
            "required_score": required_score,
            **kwargs
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_results(data)
                else:
                    error_text = await response.text()
                    self.logger.error(f"STRING API error: {error_text}")
                    raise Exception(f"STRING API returned status {response.status}: {error_text}")
    
    def _parse_results(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse STRING API response."""
        enriched_pathways = []
        
        for item in data:
            enriched_pathways.append({
                "pathway_id": item.get("category", "") + ":" + item.get("term", ""),
                "name": item.get("description", ""),
                "p_value": item.get("p_value", 1.0),
                "fdr": item.get("fdr", 1.0),
                "category": item.get("category", ""),
                "number_of_genes": item.get("number_of_genes", 0),
                "number_of_genes_in_background": item.get("number_of_genes_in_background", 0),
                "input_genes": item.get("inputGenes", [])
            })
        
        return {
            "tool": "string",
            "total_results": len(enriched_pathways),
            "result": enriched_pathways
        }
    
    async def get_interactions(
        self,
        gene_list: List[str],
        species: str = "9606",
        required_score: int = 400
    ) -> Dict[str, Any]:
        """
        Get protein-protein interactions from STRING.
        
        Args:
            gene_list: List of protein identifiers
            species: NCBI taxonomy ID
            required_score: Minimum interaction score
            
        Returns:
            Interaction network dictionary
        """
        self.logger.info(f"Getting STRING interactions for {len(gene_list)} proteins")
        
        species_map = {
            "human": "9606",
            "mouse": "10090",
            "rat": "10116"
        }
        
        if species.lower() in species_map:
            species_id = species_map[species.lower()]
        else:
            species_id = species
        
        url = f"{self.BASE_URL}/json/network"
        
        params = {
            "identifiers": "%0d".join(gene_list),
            "species": species_id,
            "required_score": required_score
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "tool": "string",
                        "interactions": data
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"STRING network API error: {error_text}")
                    raise Exception(f"STRING network API returned status {response.status}")



