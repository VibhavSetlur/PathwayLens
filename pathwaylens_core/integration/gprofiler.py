"""
g:Profiler API integration for PathwayLens.

Provides enrichment analysis using the g:Profiler service.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential


class GProfilerClient:
    """Client for g:Profiler API."""
    
    BASE_URL = "https://biit.cs.ut.ee/gprofiler/api"
    
    def __init__(self, timeout: int = 30):
        """
        Initialize g:Profiler client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.logger = logger.bind(module="gprofiler")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def enrich(
        self,
        gene_list: List[str],
        species: str = "hsapiens",
        sources: Optional[List[str]] = None,
        significance_threshold: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform enrichment analysis using g:Profiler.
        
        Args:
            gene_list: List of gene identifiers
            species: Species code (e.g., 'hsapiens', 'mmusculus')
            sources: List of data sources to query (GO, KEGG, REACTOME, etc.)
            significance_threshold: P-value threshold
            **kwargs: Additional parameters
            
        Returns:
            Enrichment results dictionary
        """
        self.logger.info(f"Querying g:Profiler for {len(gene_list)} genes")
        
        if sources is None:
            sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REACTOME"]
        
        url = f"{self.BASE_URL}/gost/profile"
        
        payload = {
            "organism": species,
            "query": gene_list,
            "sources": sources,
            "threshold_algorithm": "analytical",
            "domain_scope": "annotated",
            "user_threshold": significance_threshold,
            **kwargs
        }
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_results(data)
                else:
                    error_text = await response.text()
                    self.logger.error(f"g:Profiler API error: {error_text}")
                    raise Exception(f"g:Profiler API returned status {response.status}: {error_text}")
    
    def _parse_results(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse g:Profiler API response."""
        result = data.get("result", [])
        
        enriched_pathways = []
        for item in result:
            enriched_pathways.append({
                "pathway_id": item.get("native", item.get("id", "")),
                "name": item.get("name", ""),
                "p_value": item.get("p_value", 1.0),
                "adjusted_p_value": item.get("significant", False) and item.get("p_value", 1.0),
                "term_size": item.get("term_size", 0),
                "query_size": item.get("query_size", 0),
                "intersection_size": item.get("intersection_size", 0),
                "precision": item.get("precision", 0.0),
                "recall": item.get("recall", 0.0),
                "source": item.get("source", ""),
                "description": item.get("description", "")
            })
        
        return {
            "tool": "gprofiler",
            "total_results": len(enriched_pathways),
            "result": enriched_pathways,
            "meta": data.get("meta", {})
        }



