"""
Enrichr API integration for PathwayLens.

Provides gene set enrichment analysis using the Enrichr service.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential


class EnrichrClient:
    """Client for Enrichr API."""
    
    BASE_URL = "https://maayanlab.cloud/Enrichr"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize Enrichr client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.logger = logger.bind(module="enrichr")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def enrich(
        self,
        gene_list: List[str],
        gene_set_libraries: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform enrichment analysis using Enrichr.
        
        Args:
            gene_list: List of gene identifiers
            gene_set_libraries: List of gene set libraries to query
            **kwargs: Additional parameters
            
        Returns:
            Enrichment results dictionary
        """
        self.logger.info(f"Querying Enrichr for {len(gene_list)} genes")
        
        if gene_set_libraries is None:
            gene_set_libraries = [
                "GO_Biological_Process_2021",
                "GO_Molecular_Function_2021",
                "GO_Cellular_Component_2021",
                "KEGG_2021_Human",
                "Reactome_2022"
            ]
        
        # Step 1: Add gene list
        add_url = f"{self.BASE_URL}/addList"
        payload = {
            "list": ",".join(gene_list),
            "description": kwargs.get("description", "")
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(add_url, data=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Enrichr addList error: {error_text}")
                    raise Exception(f"Enrichr addList returned status {response.status}")
                
                add_data = await response.json()
                user_list_id = add_data.get("userListId")
                short_id = add_data.get("shortId")
                
                if not user_list_id:
                    raise Exception("Failed to add gene list to Enrichr")
            
            # Step 2: Query libraries
            results = {}
            for library in gene_set_libraries:
                try:
                    query_url = f"{self.BASE_URL}/enrich"
                    params = {
                        "userListId": user_list_id,
                        "backgroundType": library
                    }
                    
                    async with session.get(query_url, params=params) as query_response:
                        if query_response.status == 200:
                            library_results = await query_response.json()
                            results[library] = library_results.get(library, [])
                        else:
                            self.logger.warning(f"Failed to query library {library}")
                
                except Exception as e:
                    self.logger.warning(f"Error querying library {library}: {e}")
                    continue
            
            return {
                "tool": "enrichr",
                "user_list_id": user_list_id,
                "short_id": short_id,
                "results": results,
                "libraries_queried": gene_set_libraries
            }
    
    async def get_libraries(self) -> List[str]:
        """Get list of available gene set libraries."""
        url = f"{self.BASE_URL}/datasetStatistics"
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return list(data.keys())
                else:
                    self.logger.warning("Failed to get Enrichr libraries")
                    return []



