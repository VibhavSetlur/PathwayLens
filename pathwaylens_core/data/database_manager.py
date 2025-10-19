"""
Database manager for PathwayLens.

This module provides a unified interface to all pathway and gene databases,
managing connections, caching, and data retrieval across multiple sources.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from loguru import logger
from pathlib import Path

from .adapters import (
    KEGGAdapter, ReactomeAdapter, GOAdapter, BioCycAdapter,
    PathwayCommonsAdapter, MSigDBAdapter, PantherAdapter, WikiPathwaysAdapter
)
from .adapters.base import BaseAdapter, PathwayInfo, GeneInfo
from ..normalization.schemas import SpeciesType, IDType
from .cache import CacheManager


class DatabaseManager:
    """Manages connections to multiple pathway and gene databases."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            cache_dir: Directory for caching database responses
        """
        self.logger = logger.bind(module="database_manager")
        self.cache_manager = CacheManager(cache_dir) if cache_dir else None
        
        # Initialize database adapters
        self.adapters: Dict[str, BaseAdapter] = {
            "kegg": KEGGAdapter(),
            "reactome": ReactomeAdapter(),
            "go": GOAdapter(),
            "biocyc": BioCycAdapter(),
            "pathway_commons": PathwayCommonsAdapter(),
            "msigdb": MSigDBAdapter(),
            "panther": PantherAdapter(),
            "wikipathways": WikiPathwaysAdapter()
        }
        
        # Database availability status
        self.availability_status: Dict[str, bool] = {}
    
    async def initialize(self):
        """Initialize all database connections."""
        self.logger.info("Initializing database connections...")
        
        for name, adapter in self.adapters.items():
            try:
                # Test connection
                await adapter.__aenter__()
                info = await adapter.get_database_info()
                self.availability_status[name] = True
                self.logger.info(f"✅ {name.upper()} database connected: {info['name']}")
            except Exception as e:
                self.availability_status[name] = False
                self.logger.warning(f"❌ {name.upper()} database unavailable: {e}")
    
    async def cleanup(self):
        """Clean up database connections."""
        for adapter in self.adapters.values():
            try:
                await adapter.__aexit__(None, None, None)
            except Exception as e:
                self.logger.error(f"Error cleaning up adapter: {e}")
    
    async def get_available_databases(self) -> List[str]:
        """Get list of available databases."""
        return [name for name, available in self.availability_status.items() if available]
    
    async def get_database_info(self, database_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific database."""
        if database_name not in self.adapters:
            return None
        
        try:
            adapter = self.adapters[database_name]
            return await adapter.get_database_info()
        except Exception as e:
            self.logger.error(f"Error getting database info for {database_name}: {e}")
            return None
    
    async def get_all_database_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all databases."""
        info = {}
        for name in self.adapters.keys():
            db_info = await self.get_database_info(name)
            if db_info:
                info[name] = db_info
        return info
    
    async def get_pathways(
        self, 
        databases: List[str], 
        species: SpeciesType,
        use_cache: bool = True
    ) -> Dict[str, List[PathwayInfo]]:
        """
        Get pathways from multiple databases.
        
        Args:
            databases: List of database names to query
            species: Species to get pathways for
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping database names to lists of pathway information
        """
        results = {}
        
        for db_name in databases:
            if db_name not in self.adapters:
                self.logger.warning(f"Unknown database: {db_name}")
                continue
            
            if not self.availability_status.get(db_name, False):
                self.logger.warning(f"Database {db_name} is not available")
                continue
            
            # Check cache first
            cache_key = f"pathways_{db_name}_{species.value}"
            if use_cache and self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    results[db_name] = cached_result
                    continue
            
            try:
                adapter = self.adapters[db_name]
                pathways = await adapter.get_pathways(species)
                results[db_name] = pathways
                
                # Cache the result
                if use_cache and self.cache_manager:
                    await self.cache_manager.set(cache_key, pathways)
                
                self.logger.info(f"Retrieved {len(pathways)} pathways from {db_name}")
                
            except Exception as e:
                self.logger.error(f"Error getting pathways from {db_name}: {e}")
                results[db_name] = []
        
        return results
    
    async def get_pathway_genes(
        self, 
        pathway_id: str, 
        databases: List[str], 
        species: SpeciesType,
        use_cache: bool = True
    ) -> Dict[str, List[str]]:
        """
        Get genes in a pathway from multiple databases.
        
        Args:
            pathway_id: Pathway identifier
            databases: List of database names to query
            species: Species
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping database names to lists of gene identifiers
        """
        results = {}
        
        for db_name in databases:
            if db_name not in self.adapters:
                continue
            
            if not self.availability_status.get(db_name, False):
                continue
            
            # Check cache first
            cache_key = f"pathway_genes_{db_name}_{pathway_id}_{species.value}"
            if use_cache and self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    results[db_name] = cached_result
                    continue
            
            try:
                adapter = self.adapters[db_name]
                genes = await adapter.get_pathway_genes(pathway_id, species)
                results[db_name] = genes
                
                # Cache the result
                if use_cache and self.cache_manager:
                    await self.cache_manager.set(cache_key, genes)
                
            except Exception as e:
                self.logger.error(f"Error getting pathway genes from {db_name}: {e}")
                results[db_name] = []
        
        return results
    
    async def get_gene_pathways(
        self, 
        gene_id: str, 
        databases: List[str], 
        species: SpeciesType,
        use_cache: bool = True
    ) -> Dict[str, List[str]]:
        """
        Get pathways containing a gene from multiple databases.
        
        Args:
            gene_id: Gene identifier
            databases: List of database names to query
            species: Species
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping database names to lists of pathway identifiers
        """
        results = {}
        
        for db_name in databases:
            if db_name not in self.adapters:
                continue
            
            if not self.availability_status.get(db_name, False):
                continue
            
            # Check cache first
            cache_key = f"gene_pathways_{db_name}_{gene_id}_{species.value}"
            if use_cache and self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    results[db_name] = cached_result
                    continue
            
            try:
                adapter = self.adapters[db_name]
                pathways = await adapter.get_gene_pathways(gene_id, species)
                results[db_name] = pathways
                
                # Cache the result
                if use_cache and self.cache_manager:
                    await self.cache_manager.set(cache_key, pathways)
                
            except Exception as e:
                self.logger.error(f"Error getting gene pathways from {db_name}: {e}")
                results[db_name] = []
        
        return results
    
    async def search_pathways(
        self, 
        query: str, 
        databases: List[str], 
        species: SpeciesType,
        use_cache: bool = True
    ) -> Dict[str, List[PathwayInfo]]:
        """
        Search for pathways across multiple databases.
        
        Args:
            query: Search query
            databases: List of database names to query
            species: Species
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping database names to lists of matching pathway information
        """
        results = {}
        
        for db_name in databases:
            if db_name not in self.adapters:
                continue
            
            if not self.availability_status.get(db_name, False):
                continue
            
            # Check cache first
            cache_key = f"search_pathways_{db_name}_{query}_{species.value}"
            if use_cache and self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    results[db_name] = cached_result
                    continue
            
            try:
                adapter = self.adapters[db_name]
                pathways = await adapter.search_pathways(query, species)
                results[db_name] = pathways
                
                # Cache the result
                if use_cache and self.cache_manager:
                    await self.cache_manager.set(cache_key, pathways)
                
            except Exception as e:
                self.logger.error(f"Error searching pathways in {db_name}: {e}")
                results[db_name] = []
        
        return results
    
    async def get_gene_info(
        self, 
        gene_id: str, 
        databases: List[str], 
        species: SpeciesType,
        use_cache: bool = True
    ) -> Dict[str, Optional[GeneInfo]]:
        """
        Get gene information from multiple databases.
        
        Args:
            gene_id: Gene identifier
            databases: List of database names to query
            species: Species
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping database names to gene information
        """
        results = {}
        
        for db_name in databases:
            if db_name not in self.adapters:
                continue
            
            if not self.availability_status.get(db_name, False):
                continue
            
            # Check cache first
            cache_key = f"gene_info_{db_name}_{gene_id}_{species.value}"
            if use_cache and self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    results[db_name] = cached_result
                    continue
            
            try:
                adapter = self.adapters[db_name]
                gene_info = await adapter.get_gene_info(gene_id, species)
                results[db_name] = gene_info
                
                # Cache the result
                if use_cache and self.cache_manager:
                    await self.cache_manager.set(cache_key, gene_info)
                
            except Exception as e:
                self.logger.error(f"Error getting gene info from {db_name}: {e}")
                results[db_name] = None
        
        return results
    
    async def convert_gene_id(
        self, 
        gene_id: str, 
        from_type: IDType, 
        to_type: IDType, 
        databases: List[str], 
        species: SpeciesType,
        use_cache: bool = True
    ) -> Dict[str, Optional[str]]:
        """
        Convert gene identifier across multiple databases.
        
        Args:
            gene_id: Gene identifier to convert
            from_type: Source identifier type
            to_type: Target identifier type
            databases: List of database names to query
            species: Species
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary mapping database names to converted gene identifiers
        """
        results = {}
        
        for db_name in databases:
            if db_name not in self.adapters:
                continue
            
            if not self.availability_status.get(db_name, False):
                continue
            
            # Check cache first
            cache_key = f"convert_gene_id_{db_name}_{gene_id}_{from_type.value}_{to_type.value}_{species.value}"
            if use_cache and self.cache_manager:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    results[db_name] = cached_result
                    continue
            
            try:
                adapter = self.adapters[db_name]
                converted_id = await adapter.convert_gene_id(gene_id, from_type, to_type, species)
                results[db_name] = converted_id
                
                # Cache the result
                if use_cache and self.cache_manager:
                    await self.cache_manager.set(cache_key, converted_id)
                
            except Exception as e:
                self.logger.error(f"Error converting gene ID in {db_name}: {e}")
                results[db_name] = None
        
        return results
    
    async def get_supported_species(self, databases: List[str]) -> Dict[str, List[SpeciesType]]:
        """
        Get supported species for multiple databases.
        
        Args:
            databases: List of database names to query
            
        Returns:
            Dictionary mapping database names to lists of supported species
        """
        results = {}
        
        for db_name in databases:
            if db_name not in self.adapters:
                continue
            
            if not self.availability_status.get(db_name, False):
                continue
            
            try:
                adapter = self.adapters[db_name]
                species = await adapter.get_supported_species()
                results[db_name] = species
                
            except Exception as e:
                self.logger.error(f"Error getting supported species from {db_name}: {e}")
                results[db_name] = []
        
        return results
    
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
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    def __str__(self) -> str:
        """String representation of the database manager."""
        available = len([name for name, available in self.availability_status.items() if available])
        total = len(self.adapters)
        return f"DatabaseManager({available}/{total} databases available)"
    
    def __repr__(self) -> str:
        """Detailed string representation of the database manager."""
        available = [name for name, available in self.availability_status.items() if available]
        return f"DatabaseManager(available_databases={available}, cache_enabled={self.cache_manager is not None})"
