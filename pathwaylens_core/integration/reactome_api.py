"""
Reactome pathway API integration for PathwayLens.

Provides pathway analysis using the Reactome database.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential


class ReactomeAPIClient:
    """Client for Reactome API."""
    
    BASE_URL = "https://reactome.org/AnalysisService"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize Reactome API client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.logger = logger.bind(module="reactome_api")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def analyze_pathways(
        self,
        gene_list: List[str],
        species: str = "Homo sapiens",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform pathway analysis using Reactome.
        
        Args:
            gene_list: List of gene identifiers
            species: Species name (e.g., "Homo sapiens", "Mus musculus")
            **kwargs: Additional parameters
            
        Returns:
            Pathway analysis results dictionary
        """
        self.logger.info(f"Querying Reactome for {len(gene_list)} genes")
        
        url = f"{self.BASE_URL}/identifiers/query"
        
        payload = {
            "ids": gene_list,
            "species": species,
            **kwargs
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_results(data)
                else:
                    error_text = await response.text()
                    self.logger.error(f"Reactome API error: {error_text}")
                    raise Exception(f"Reactome API returned status {response.status}: {error_text}")
    
    def _parse_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Reactome API response."""
        pathways = []
        
        # Extract pathway results
        pathway_results = data.get("pathways", [])
        
        for pathway in pathway_results:
            pathways.append({
                "pathway_id": pathway.get("stId", pathway.get("id", "")),
                "name": pathway.get("name", ""),
                "p_value": pathway.get("entities", {}).get("pValue", 1.0),
                "fdr": pathway.get("entities", {}).get("fdr", 1.0),
                "reaction_count": pathway.get("entities", {}).get("found", 0),
                "total_reactions": pathway.get("entities", {}).get("total", 0),
                "url": pathway.get("url", ""),
                "database_name": pathway.get("databaseName", "Reactome")
            })
        
        return {
            "tool": "reactome",
            "total_pathways": len(pathways),
            "pathways": pathways,
            "identifiers_found": data.get("identifiersFound", 0),
            "identifiers_total": data.get("identifiersTotal", 0),
            "summary": data.get("summary", {})
        }
    
    async def get_pathway_details(
        self,
        pathway_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific pathway.
        
        Args:
            pathway_id: Reactome pathway stable identifier
            **kwargs: Additional parameters
            
        Returns:
            Pathway details dictionary
        """
        url = f"{self.BASE_URL}/pathway/{pathway_id}"
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(url, params=kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    self.logger.error(f"Reactome pathway details error: {error_text}")
                    raise Exception(f"Reactome pathway details returned status {response.status}")



