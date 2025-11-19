"""
Unified API manager for multiple bioinformatics tools.

Provides a unified interface for interacting with multiple API services
with rate limiting, caching, and error handling.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
import numpy as np
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from .gprofiler import GProfilerClient
from .enrichr import EnrichrClient
from .string_api import StringAPIClient
from .reactome_api import ReactomeAPIClient


@dataclass
class APIResult:
    """Result from an API call."""
    tool: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class APIManager:
    """
    Unified manager for multiple bioinformatics API services.
    
    Provides rate limiting, caching, and result aggregation across
    multiple enrichment analysis tools.
    """
    
    def __init__(
        self,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        rate_limit_per_second: float = 10.0
    ):
        """
        Initialize the API manager.
        
        Args:
            enable_caching: Whether to enable result caching
            cache_ttl: Cache time-to-live in seconds
            rate_limit_per_second: Maximum requests per second
        """
        self.logger = logger.bind(module="api_manager")
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.rate_limit = rate_limit_per_second
        
        # Initialize API clients
        self.gprofiler = GProfilerClient()
        self.enrichr = EnrichrClient()
        self.string = StringAPIClient()
        self.reactome = ReactomeAPIClient()
        
        # Cache storage
        self.cache: Dict[str, APIResult] = {}
        
        # Rate limiting
        self._last_request_time: Dict[str, datetime] = {}
        self._request_lock = asyncio.Lock()
    
    async def enrich_pathways(
        self,
        gene_list: List[str],
        species: str = "human",
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, APIResult]:
        """
        Perform enrichment analysis using multiple tools.
        
        Args:
            gene_list: List of gene identifiers
            species: Species for analysis (human, mouse, etc.)
            tools: List of tools to use (gprofiler, enrichr, etc.). If None, uses all.
            **kwargs: Additional parameters for specific tools
            
        Returns:
            Dictionary mapping tool names to APIResult objects
        """
        if tools is None:
            tools = ["gprofiler", "enrichr", "reactome"]
        
        self.logger.info(f"Running enrichment with {len(tools)} tools for {len(gene_list)} genes")
        
        # Create cache key
        cache_key = self._make_cache_key(gene_list, species, tools, kwargs)
        
        # Check cache
        if self.enable_caching and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            age = (datetime.now() - cached_result.timestamp).total_seconds()
            if age < self.cache_ttl:
                self.logger.debug(f"Using cached result for {cache_key[:50]}...")
                return {tool: cached_result for tool in tools}
        
        # Run enrichment with multiple tools
        results = {}
        tasks = []
        
        if "gprofiler" in tools:
            tasks.append(self._run_gprofiler(gene_list, species, **kwargs))
        
        if "enrichr" in tools:
            tasks.append(self._run_enrichr(gene_list, **kwargs))
        
        if "reactome" in tools:
            tasks.append(self._run_reactome(gene_list, species, **kwargs))
        
        if "string" in tools:
            tasks.append(self._run_string(gene_list, species, **kwargs))
        
        # Execute in parallel with rate limiting
        api_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        tool_names = [t for t in tools if t in ["gprofiler", "enrichr", "reactome", "string"]]
        for tool_name, result in zip(tool_names, api_results):
            if isinstance(result, Exception):
                results[tool_name] = APIResult(
                    tool=tool_name,
                    success=False,
                    error=str(result)
                )
            else:
                results[tool_name] = result
        
        # Cache results
        if self.enable_caching:
            self.cache[cache_key] = results.get(tools[0], APIResult(tool=tools[0], success=False))
        
        return results
    
    async def aggregate_results(
        self,
        results: Dict[str, APIResult],
        min_agreement: int = 2
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple tools.
        
        Args:
            results: Dictionary of APIResult objects
            min_agreement: Minimum number of tools that must agree on a pathway
            
        Returns:
            Aggregated results with consensus pathways
        """
        successful_results = {
            tool: r.data for tool, r in results.items()
            if r.success and r.data
        }
        
        if not successful_results:
            return {"pathways": [], "consensus_pathways": []}
        
        # Extract pathways from each tool
        all_pathways: Dict[str, Dict[str, Any]] = {}
        
        for tool, data in successful_results.items():
            pathways = self._extract_pathways(tool, data)
            for pathway_id, pathway_info in pathways.items():
                if pathway_id not in all_pathways:
                    all_pathways[pathway_id] = {
                        "pathway_id": pathway_id,
                        "name": pathway_info.get("name", pathway_id),
                        "tools": [],
                        "p_values": [],
                        "enrichment_scores": []
                    }
                
                all_pathways[pathway_id]["tools"].append(tool)
                if "p_value" in pathway_info:
                    all_pathways[pathway_id]["p_values"].append(pathway_info["p_value"])
                if "enrichment_score" in pathway_info:
                    all_pathways[pathway_id]["enrichment_scores"].append(
                        pathway_info["enrichment_score"]
                    )
        
        # Find consensus pathways (appear in multiple tools)
        consensus_pathways = [
            pathway for pathway in all_pathways.values()
            if len(pathway["tools"]) >= min_agreement
        ]
        
        # Calculate consensus scores
        for pathway in consensus_pathways:
            if pathway["p_values"]:
                pathway["consensus_p_value"] = min(pathway["p_values"])
            if pathway["enrichment_scores"]:
                pathway["consensus_enrichment"] = sum(pathway["enrichment_scores"]) / len(
                    pathway["enrichment_scores"]
                )
        
        return {
            "pathways": list(all_pathways.values()),
            "consensus_pathways": sorted(
                consensus_pathways,
                key=lambda x: x.get("consensus_p_value", 1.0)
            ),
            "tools_used": list(successful_results.keys())
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _run_gprofiler(
        self,
        gene_list: List[str],
        species: str,
        **kwargs
    ) -> APIResult:
        """Run g:Profiler enrichment."""
        await self._rate_limit("gprofiler")
        try:
            result = await self.gprofiler.enrich(
                gene_list=gene_list,
                species=species,
                **kwargs
            )
            return APIResult(tool="gprofiler", success=True, data=result)
        except Exception as e:
            self.logger.error(f"g:Profiler enrichment failed: {e}")
            return APIResult(tool="gprofiler", success=False, error=str(e))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _run_enrichr(
        self,
        gene_list: List[str],
        **kwargs
    ) -> APIResult:
        """Run Enrichr enrichment."""
        await self._rate_limit("enrichr")
        try:
            result = await self.enrichr.enrich(
                gene_list=gene_list,
                **kwargs
            )
            return APIResult(tool="enrichr", success=True, data=result)
        except Exception as e:
            self.logger.error(f"Enrichr enrichment failed: {e}")
            return APIResult(tool="enrichr", success=False, error=str(e))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _run_reactome(
        self,
        gene_list: List[str],
        species: str,
        **kwargs
    ) -> APIResult:
        """Run Reactome enrichment."""
        await self._rate_limit("reactome")
        try:
            result = await self.reactome.analyze_pathways(
                gene_list=gene_list,
                species=species,
                **kwargs
            )
            return APIResult(tool="reactome", success=True, data=result)
        except Exception as e:
            self.logger.error(f"Reactome enrichment failed: {e}")
            return APIResult(tool="reactome", success=False, error=str(e))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _run_string(
        self,
        gene_list: List[str],
        species: str,
        **kwargs
    ) -> APIResult:
        """Run STRING enrichment."""
        await self._rate_limit("string")
        try:
            result = await self.string.enrich(
                gene_list=gene_list,
                species=species,
                **kwargs
            )
            return APIResult(tool="string", success=True, data=result)
        except Exception as e:
            self.logger.error(f"STRING enrichment failed: {e}")
            return APIResult(tool="string", success=False, error=str(e))
    
    async def _rate_limit(self, tool: str):
        """Implement rate limiting."""
        async with self._request_lock:
            now = datetime.now()
            if tool in self._last_request_time:
                time_since = (now - self._last_request_time[tool]).total_seconds()
                min_interval = 1.0 / self.rate_limit
                if time_since < min_interval:
                    await asyncio.sleep(min_interval - time_since)
            self._last_request_time[tool] = datetime.now()
    
    def _make_cache_key(
        self,
        gene_list: List[str],
        species: str,
        tools: List[str],
        kwargs: Dict[str, Any]
    ) -> str:
        """Create cache key from parameters."""
        gene_str = ",".join(sorted(gene_list))
        tools_str = ",".join(sorted(tools))
        kwargs_str = str(sorted(kwargs.items()))
        return f"{species}:{gene_str}:{tools_str}:{kwargs_str}"
    
    def _extract_pathways(self, tool: str, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract pathways from tool-specific response format."""
        pathways = {}
        
        if tool == "gprofiler" and "result" in data:
            for item in data["result"]:
                pathway_id = item.get("native", item.get("id"))
                pathways[pathway_id] = {
                    "name": item.get("name", ""),
                    "p_value": item.get("p_value", 1.0),
                    "enrichment_score": -np.log10(item.get("p_value", 1.0))
                }
        
        elif tool == "enrichr" and isinstance(data, dict):
            for library, terms in data.items():
                for term in terms:
                    pathway_id = term[1] if len(term) > 1 else term[0]
                    pathways[pathway_id] = {
                        "name": term[0] if len(term) > 0 else pathway_id,
                        "p_value": term[2] if len(term) > 2 else 1.0,
                        "enrichment_score": term[3] if len(term) > 3 else 0.0
                    }
        
        elif tool == "reactome" and "pathways" in data:
            for pathway in data["pathways"]:
                pathway_id = pathway.get("id", "")
                pathways[pathway_id] = {
                    "name": pathway.get("name", ""),
                    "p_value": pathway.get("p_value", 1.0),
                    "enrichment_score": pathway.get("enrichment_score", 0.0)
                }
        
        return pathways

