"""
Cross-species ortholog mapping functionality.
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import pandas as pd
import time
from tenacity import retry, stop_after_attempt, wait_exponential

from .schemas import SpeciesType, CrossSpeciesMapping


@dataclass
class OrthologResult:
    """Result of ortholog mapping."""
    source_id: str
    target_id: str
    source_species: SpeciesType
    target_species: SpeciesType
    ortholog_type: str
    confidence: float
    method: str
    is_ambiguous: bool = False
    alternative_mappings: List[str] = None
    
    def __post_init__(self):
        if self.alternative_mappings is None:
            self.alternative_mappings = []


class SpeciesMapper:
    """Maps genes across different species using ortholog databases."""
    
    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize species mapper.
        
        Args:
            rate_limit: Rate limit in requests per second
        """
        self.rate_limit = rate_limit
        self.logger = logger.bind(module="species_mapper")
        self.session = None
        
        # API endpoints
        self.ensembl_url = "https://rest.ensembl.org"
        self.orthodb_url = "https://www.orthodb.org"
        self.homologene_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Rate limiting
        self.last_request_time = 0
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            time.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def map_orthologs(
        self,
        identifiers: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: str = "ensembl"
    ) -> List[OrthologResult]:
        """
        Map orthologs between species.
        
        Args:
            identifiers: List of gene identifiers
            source_species: Source species
            target_species: Target species
            id_type: Type of identifiers (ensembl, symbol, entrez)
            
        Returns:
            List of ortholog mapping results
        """
        self.logger.info(f"Mapping {len(identifiers)} genes from {source_species} to {target_species}")
        
        # If same species, return identity mapping
        if source_species == target_species:
            return [
                OrthologResult(
                    source_id=id_val,
                    target_id=id_val,
                    source_species=source_species,
                    target_species=target_species,
                    ortholog_type="identity",
                    confidence=1.0,
                    method="identity"
                )
                for id_val in identifiers
            ]
        
        # Try different mapping methods
        results = []
        
        # Method 1: Ensembl Compara
        try:
            ensembl_results = await self._map_via_ensembl(
                identifiers, source_species, target_species, id_type
            )
            results.extend(ensembl_results)
        except Exception as e:
            self.logger.warning(f"Ensembl ortholog mapping failed: {e}")
        
        # Method 2: NCBI HomoloGene
        try:
            homologene_results = await self._map_via_homologene(
                identifiers, source_species, target_species, id_type
            )
            results.extend(homologene_results)
        except Exception as e:
            self.logger.warning(f"HomoloGene mapping failed: {e}")
        
        # Method 3: OrthoDB
        try:
            orthodb_results = await self._map_via_orthodb(
                identifiers, source_species, target_species, id_type
            )
            results.extend(orthodb_results)
        except Exception as e:
            self.logger.warning(f"OrthoDB mapping failed: {e}")
        
        # Merge and deduplicate results
        merged_results = self._merge_ortholog_results(results)
        
        self.logger.info(f"Ortholog mapping completed: {len(merged_results)} results")
        return merged_results
    
    async def _map_via_ensembl(
        self,
        identifiers: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: str
    ) -> List[OrthologResult]:
        """Map orthologs using Ensembl Compara."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Map species to Ensembl format
        species_map = {
            SpeciesType.HUMAN: "homo_sapiens",
            SpeciesType.MOUSE: "mus_musculus",
            SpeciesType.RAT: "rattus_norvegicus",
            SpeciesType.DROSOPHILA: "drosophila_melanogaster",
            SpeciesType.ZEBRAFISH: "danio_rerio",
            SpeciesType.C_ELEGANS: "caenorhabditis_elegans",
            SpeciesType.S_CEREVISIAE: "saccharomyces_cerevisiae"
        }
        
        source_ensembl = species_map.get(source_species)
        target_ensembl = species_map.get(target_species)
        
        if not source_ensembl or not target_ensembl:
            return []
        
        # Prepare query
        query = {
            "species": source_ensembl,
            "target_species": target_ensembl,
            "type": "orthologues"
        }
        
        results = []
        
        # Process identifiers in batches
        batch_size = 50
        for i in range(0, len(identifiers), batch_size):
            batch = identifiers[i:i + batch_size]
            
            self._rate_limit()
            
            async with self.session.get(
                f"{self.ensembl_url}/homology/id/{','.join(batch)}",
                params=query,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ensembl API error: {response.status}")
                
                data = await response.json()
            
            # Parse results
            for item in data.get('data', []):
                source_id = item.get('id')
                if not source_id:
                    continue
                
                for homolog in item.get('homologies', []):
                    target_id = homolog.get('target', {}).get('id')
                    if not target_id:
                        continue
                    
                    # Determine ortholog type
                    ortholog_type = homolog.get('type', 'ortholog')
                    confidence = 0.9 if ortholog_type == 'ortholog_one2one' else 0.7
                    
                    results.append(OrthologResult(
                        source_id=source_id,
                        target_id=target_id,
                        source_species=source_species,
                        target_species=target_species,
                        ortholog_type=ortholog_type,
                        confidence=confidence,
                        method="ensembl_compara"
                    ))
        
        return results
    
    async def _map_via_homologene(
        self,
        identifiers: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: str
    ) -> List[OrthologResult]:
        """Map orthologs using NCBI HomoloGene."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Map species to NCBI taxonomy ID
        species_map = {
            SpeciesType.HUMAN: "9606",
            SpeciesType.MOUSE: "10090",
            SpeciesType.RAT: "10116",
            SpeciesType.DROSOPHILA: "7227",
            SpeciesType.ZEBRAFISH: "7955",
            SpeciesType.C_ELEGANS: "6239",
            SpeciesType.S_CEREVISIAE: "559292"
        }
        
        source_tax_id = species_map.get(source_species)
        target_tax_id = species_map.get(target_species)
        
        if not source_tax_id or not target_tax_id:
            return []
        
        # Prepare query
        query = {
            "db": "homologene",
            "id": ",".join(identifiers),
            "retmode": "json",
            "retmax": len(identifiers)
        }
        
        self._rate_limit()
        
        async with self.session.get(
            f"{self.homologene_url}/esummary.fcgi",
            params=query,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise Exception(f"NCBI API error: {response.status}")
            
            data = await response.json()
        
        # Parse results
        results = []
        if 'result' in data and 'uids' in data['result']:
            for uid in data['result']['uids']:
                if uid in data['result']:
                    homolog_data = data['result'][uid]
                    source_id = uid
                    
                    # Find target species in homolog group
                    target_id = None
                    for gene in homolog_data.get('genes', []):
                        if gene.get('tax_id') == target_tax_id:
                            target_id = gene.get('gene_id')
                            break
                    
                    if target_id:
                        results.append(OrthologResult(
                            source_id=source_id,
                            target_id=str(target_id),
                            source_species=source_species,
                            target_species=target_species,
                            ortholog_type="ortholog",
                            confidence=0.8,
                            method="ncbi_homologene"
                        ))
        
        return results
    
    async def _map_via_orthodb(
        self,
        identifiers: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: str
    ) -> List[OrthologResult]:
        """Map orthologs using OrthoDB."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Map species to OrthoDB format
        species_map = {
            SpeciesType.HUMAN: "9606",
            SpeciesType.MOUSE: "10090",
            SpeciesType.RAT: "10116",
            SpeciesType.DROSOPHILA: "7227",
            SpeciesType.ZEBRAFISH: "7955",
            SpeciesType.C_ELEGANS: "6239",
            SpeciesType.S_CEREVISIAE: "559292"
        }
        
        source_tax_id = species_map.get(source_species)
        target_tax_id = species_map.get(target_species)
        
        if not source_tax_id or not target_tax_id:
            return []
        
        # Prepare query
        query = {
            "query": ",".join(identifiers),
            "species": source_tax_id,
            "target_species": target_tax_id,
            "level": "1"
        }
        
        self._rate_limit()
        
        async with self.session.get(
            f"{self.orthodb_url}/search",
            params=query,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise Exception(f"OrthoDB API error: {response.status}")
            
            data = await response.json()
        
        # Parse results
        results = []
        for item in data.get('data', []):
            source_id = item.get('id')
            if not source_id:
                continue
            
            # Find target species ortholog
            for ortholog in item.get('orthologs', []):
                if ortholog.get('species') == target_tax_id:
                    target_id = ortholog.get('id')
                    if target_id:
                        results.append(OrthologResult(
                            source_id=source_id,
                            target_id=str(target_id),
                            source_species=source_species,
                            target_species=target_species,
                            ortholog_type="ortholog",
                            confidence=0.85,
                            method="orthodb"
                        ))
        
        return results
    
    def _merge_ortholog_results(self, results: List[OrthologResult]) -> List[OrthologResult]:
        """Merge and deduplicate ortholog results."""
        # Group by source ID
        grouped = {}
        for result in results:
            if result.source_id not in grouped:
                grouped[result.source_id] = []
            grouped[result.source_id].append(result)
        
        # Merge results
        merged_results = []
        for source_id, result_list in grouped.items():
            if len(result_list) == 1:
                merged_results.append(result_list[0])
            else:
                # Multiple results for same source ID
                # Use highest confidence result
                best_result = max(result_list, key=lambda x: x.confidence)
                best_result.is_ambiguous = True
                best_result.alternative_mappings = [
                    r.target_id for r in result_list 
                    if r.target_id != best_result.target_id
                ]
                merged_results.append(best_result)
        
        return merged_results
    
    def map_orthologs_sync(
        self,
        identifiers: List[str],
        source_species: SpeciesType,
        target_species: SpeciesType,
        id_type: str = "ensembl"
    ) -> List[OrthologResult]:
        """
        Synchronous version of map_orthologs.
        
        Args:
            identifiers: List of gene identifiers
            source_species: Source species
            target_species: Target species
            id_type: Type of identifiers (ensembl, symbol, entrez)
            
        Returns:
            List of ortholog mapping results
        """
        return asyncio.run(self.map_orthologs(
            identifiers, source_species, target_species, id_type
        ))
