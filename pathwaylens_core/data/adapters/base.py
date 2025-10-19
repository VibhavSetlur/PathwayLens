"""
Base adapter class for database connections.
"""

import asyncio
import aiohttp
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from loguru import logger
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from ...normalization.schemas import SpeciesType, IDType


@dataclass
class PathwayInfo:
    """Information about a pathway."""
    pathway_id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    species: Optional[str] = None
    genes: List[str] = None
    url: Optional[str] = None
    
    def __post_init__(self):
        if self.genes is None:
            self.genes = []


@dataclass
class GeneInfo:
    """Information about a gene."""
    gene_id: str
    symbol: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    species: Optional[str] = None
    aliases: List[str] = None
    url: Optional[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class BaseAdapter(ABC):
    """Base class for database adapters."""
    
    def __init__(self, base_url: str, rate_limit: float = 1.0, timeout: int = 30):
        """
        Initialize the adapter.
        
        Args:
            base_url: Base URL for the database API
            rate_limit: Rate limit in requests per second
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.logger = logger.bind(module=self.__class__.__name__)
        self.session = None
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # seconds
        self.rate_limit_start = time.time()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.rate_limit_start >= self.rate_limit_window:
            self.request_count = 0
            self.rate_limit_start = current_time
        
        # Check if we need to wait
        if self.request_count >= self.rate_limit * self.rate_limit_window:
            time.sleep(1)
            self.request_count = 0
            self.rate_limit_start = current_time
        
        # Wait between requests
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a request to the database API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        self._rate_limit()
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    self.logger.warning("Rate limited, waiting...")
                    await asyncio.sleep(5)
                    raise Exception("Rate limited")
                else:
                    raise Exception(f"API error: {response.status}")
        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            raise
    
    @abstractmethod
    async def get_pathways(self, species: SpeciesType) -> List[PathwayInfo]:
        """
        Get all pathways for a species.
        
        Args:
            species: Species to get pathways for
            
        Returns:
            List of pathway information
        """
        pass
    
    @abstractmethod
    async def get_pathway_genes(self, pathway_id: str, species: SpeciesType) -> List[str]:
        """
        Get genes in a pathway.
        
        Args:
            pathway_id: Pathway identifier
            species: Species
            
        Returns:
            List of gene identifiers
        """
        pass
    
    @abstractmethod
    async def get_gene_pathways(self, gene_id: str, species: SpeciesType) -> List[str]:
        """
        Get pathways containing a gene.
        
        Args:
            gene_id: Gene identifier
            species: Species
            
        Returns:
            List of pathway identifiers
        """
        pass
    
    @abstractmethod
    async def search_pathways(self, query: str, species: SpeciesType) -> List[PathwayInfo]:
        """
        Search for pathways by name or description.
        
        Args:
            query: Search query
            species: Species
            
        Returns:
            List of matching pathway information
        """
        pass
    
    @abstractmethod
    async def get_gene_info(self, gene_id: str, species: SpeciesType) -> Optional[GeneInfo]:
        """
        Get information about a gene.
        
        Args:
            gene_id: Gene identifier
            species: Species
            
        Returns:
            Gene information or None if not found
        """
        pass
    
    @abstractmethod
    async def convert_gene_id(self, gene_id: str, from_type: IDType, to_type: IDType, species: SpeciesType) -> Optional[str]:
        """
        Convert gene identifier from one type to another.
        
        Args:
            gene_id: Gene identifier to convert
            from_type: Source identifier type
            to_type: Target identifier type
            species: Species
            
        Returns:
            Converted gene identifier or None if conversion failed
        """
        pass
    
    @abstractmethod
    async def get_supported_species(self) -> List[SpeciesType]:
        """
        Get list of supported species.
        
        Returns:
            List of supported species
        """
        pass
    
    @abstractmethod
    async def get_database_info(self) -> Dict[str, Any]:
        """
        Get information about the database.
        
        Returns:
            Database information
        """
        pass
    
    def get_sync(self, method_name: str, *args, **kwargs):
        """
        Synchronous wrapper for async methods.
        
        Args:
            method_name: Name of the method to call
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result
        """
        method = getattr(self, method_name)
        return asyncio.run(method(*args, **kwargs))
    
    def __str__(self) -> str:
        """String representation of the adapter."""
        return f"{self.__class__.__name__}({self.base_url})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the adapter."""
        return f"{self.__class__.__name__}(base_url='{self.base_url}', rate_limit={self.rate_limit}, timeout={self.timeout})"
