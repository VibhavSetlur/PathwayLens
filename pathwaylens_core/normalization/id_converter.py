"""
Gene identifier conversion using multiple databases and APIs.
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

from .schemas import IDType, SpeciesType, ConversionMapping, AmbiguityPolicy


@dataclass
class ConversionResult:
    """Result of identifier conversion."""
    input_id: str
    output_id: Optional[str]
    confidence: float
    source: str
    is_ambiguous: bool = False
    alternative_mappings: List[str] = None
    
    def __post_init__(self):
        if self.alternative_mappings is None:
            self.alternative_mappings = []


class IDConverter:
    """Converts gene identifiers across different formats and databases."""
    
    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize ID converter.
        
        Args:
            rate_limit: Rate limit in requests per second
        """
        self.rate_limit = rate_limit
        self.logger = logger.bind(module="id_converter")
        self.session = None
        
        # API endpoints
        self.mygene_url = "https://mygene.info/v3"
        self.ensembl_url = "https://rest.ensembl.org"
        self.ncbi_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
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
    async def convert_identifiers(
        self,
        identifiers: List[str],
        input_type: IDType,
        output_type: IDType,
        species: SpeciesType,
        ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.EXPAND
    ) -> List[ConversionResult]:
        """
        Convert identifiers from input type to output type.
        
        Args:
            identifiers: List of input identifiers
            input_type: Type of input identifiers
            output_type: Type of output identifiers
            species: Species of the identifiers
            ambiguity_policy: How to handle ambiguous mappings
            
        Returns:
            List of conversion results
        """
        self.logger.info(f"Converting {len(identifiers)} identifiers from {input_type} to {output_type}")
        
        # If input and output types are the same, return identity mapping
        if input_type == output_type:
            return [
                ConversionResult(
                    input_id=id_val,
                    output_id=id_val,
                    confidence=1.0,
                    source="identity"
                )
                for id_val in identifiers
            ]
        
        # Try different conversion methods
        results = []
        
        # Method 1: MyGene.info
        try:
            mygene_results = await self._convert_via_mygene(
                identifiers, input_type, output_type, species
            )
            results.extend(mygene_results)
        except Exception as e:
            self.logger.warning(f"MyGene conversion failed: {e}")
        
        # Method 2: Ensembl
        try:
            ensembl_results = await self._convert_via_ensembl(
                identifiers, input_type, output_type, species
            )
            results.extend(ensembl_results)
        except Exception as e:
            self.logger.warning(f"Ensembl conversion failed: {e}")
        
        # Method 3: NCBI
        try:
            ncbi_results = await self._convert_via_ncbi(
                identifiers, input_type, output_type, species
            )
            results.extend(ncbi_results)
        except Exception as e:
            self.logger.warning(f"NCBI conversion failed: {e}")
        
        # Merge and deduplicate results
        merged_results = self._merge_conversion_results(results, ambiguity_policy)
        
        self.logger.info(f"Conversion completed: {len(merged_results)} results")
        return merged_results
    
    async def _convert_via_mygene(
        self,
        identifiers: List[str],
        input_type: IDType,
        output_type: IDType,
        species: SpeciesType
    ) -> List[ConversionResult]:
        """Convert identifiers using MyGene.info API."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Map species to MyGene format
        species_map = {
            SpeciesType.HUMAN: "human",
            SpeciesType.MOUSE: "mouse",
            SpeciesType.RAT: "rat",
            SpeciesType.DROSOPHILA: "fly",
            SpeciesType.ZEBRAFISH: "zebrafish",
            SpeciesType.C_ELEGANS: "worm",
            SpeciesType.S_CEREVISIAE: "yeast"
        }
        
        mygene_species = species_map.get(species, "human")
        
        # Map ID types to MyGene format
        id_type_map = {
            IDType.SYMBOL: "symbol",
            IDType.ENSEMBL: "ensembl.gene",
            IDType.ENTREZ: "entrezgene",
            IDType.UNIPROT: "uniprot",
            IDType.REFSEQ: "refseq"
        }
        
        input_field = id_type_map.get(input_type, "symbol")
        output_field = id_type_map.get(output_type, "symbol")
        
        # Prepare query
        query = {
            "q": identifiers,
            "scopes": input_field,
            "fields": f"{output_field},name,symbol",
            "species": mygene_species,
            "size": len(identifiers)
        }
        
        self._rate_limit()
        
        async with self.session.post(
            f"{self.mygene_url}/query",
            json=query,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise Exception(f"MyGene API error: {response.status}")
            
            data = await response.json()
        
        # Parse results
        results = []
        for item in data:
            if 'query' in item and 'notfound' not in item:
                input_id = item['query']
                output_id = item.get(output_field)
                
                if output_id:
                    # Handle multiple mappings
                    if isinstance(output_id, list):
                        for alt_id in output_id:
                            results.append(ConversionResult(
                                input_id=input_id,
                                output_id=str(alt_id),
                                confidence=0.9,
                                source="mygene",
                                is_ambiguous=len(output_id) > 1,
                                alternative_mappings=[str(x) for x in output_id if x != alt_id]
                            ))
                    else:
                        results.append(ConversionResult(
                            input_id=input_id,
                            output_id=str(output_id),
                            confidence=0.9,
                            source="mygene"
                        ))
        
        return results
    
    async def _convert_via_ensembl(
        self,
        identifiers: List[str],
        input_type: IDType,
        output_type: IDType,
        species: SpeciesType
    ) -> List[ConversionResult]:
        """Convert identifiers using Ensembl REST API."""
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
        
        ensembl_species = species_map.get(species, "homo_sapiens")
        
        # Map ID types to Ensembl format
        id_type_map = {
            IDType.SYMBOL: "external_name",
            IDType.ENSEMBL: "ensembl_gene_id",
            IDType.ENTREZ: "entrezgene"
        }
        
        input_field = id_type_map.get(input_type)
        output_field = id_type_map.get(output_type)
        
        if not input_field or not output_field:
            return []  # Ensembl doesn't support this conversion
        
        # Prepare query
        query = {
            "ids": identifiers,
            "species": ensembl_species
        }
        
        self._rate_limit()
        
        async with self.session.post(
            f"{self.ensembl_url}/lookup/id",
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise Exception(f"Ensembl API error: {response.status}")
            
            data = await response.json()
        
        # Parse results
        results = []
        for item in data:
            if isinstance(item, dict) and 'id' in item:
                input_id = item['id']
                output_id = item.get(output_field)
                
                if output_id:
                    results.append(ConversionResult(
                        input_id=input_id,
                        output_id=str(output_id),
                        confidence=0.95,
                        source="ensembl"
                    ))
        
        return results
    
    async def _convert_via_ncbi(
        self,
        identifiers: List[str],
        input_type: IDType,
        output_type: IDType,
        species: SpeciesType
    ) -> List[ConversionResult]:
        """Convert identifiers using NCBI E-utilities."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # NCBI is primarily for Entrez IDs
        if input_type != IDType.ENTREZ and output_type != IDType.ENTREZ:
            return []
        
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
        
        tax_id = species_map.get(species, "9606")
        
        # Prepare query
        query = {
            "db": "gene",
            "id": ",".join(identifiers),
            "retmode": "json",
            "retmax": len(identifiers)
        }
        
        self._rate_limit()
        
        async with self.session.get(
            f"{self.ncbi_url}/esummary.fcgi",
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
                    gene_data = data['result'][uid]
                    input_id = uid
                    
                    if output_type == IDType.SYMBOL:
                        output_id = gene_data.get('name')
                    elif output_type == IDType.ENSEMBL:
                        output_id = gene_data.get('ensembl_gene_id')
                    else:
                        output_id = None
                    
                    if output_id:
                        results.append(ConversionResult(
                            input_id=input_id,
                            output_id=str(output_id),
                            confidence=0.9,
                            source="ncbi"
                        ))
        
        return results
    
    def _merge_conversion_results(
        self,
        results: List[ConversionResult],
        ambiguity_policy: AmbiguityPolicy
    ) -> List[ConversionResult]:
        """Merge and deduplicate conversion results."""
        # Group by input ID
        grouped = {}
        for result in results:
            if result.input_id not in grouped:
                grouped[result.input_id] = []
            grouped[result.input_id].append(result)
        
        # Apply ambiguity policy
        merged_results = []
        for input_id, result_list in grouped.items():
            if len(result_list) == 1:
                merged_results.append(result_list[0])
            else:
                # Multiple results for same input ID
                if ambiguity_policy == AmbiguityPolicy.EXPAND:
                    # Include all mappings
                    merged_results.extend(result_list)
                elif ambiguity_policy == AmbiguityPolicy.COLLAPSE:
                    # Use highest confidence result
                    best_result = max(result_list, key=lambda x: x.confidence)
                    best_result.is_ambiguous = True
                    best_result.alternative_mappings = [
                        r.output_id for r in result_list 
                        if r.output_id != best_result.output_id
                    ]
                    merged_results.append(best_result)
                elif ambiguity_policy == AmbiguityPolicy.SKIP:
                    # Skip ambiguous entries
                    continue
                elif ambiguity_policy == AmbiguityPolicy.ERROR:
                    # Raise error
                    raise ValueError(f"Ambiguous mapping for {input_id}: {[r.output_id for r in result_list]}")
        
        return merged_results
    
    def convert_sync(
        self,
        identifiers: List[str],
        input_type: IDType,
        output_type: IDType,
        species: SpeciesType,
        ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.EXPAND
    ) -> List[ConversionResult]:
        """
        Synchronous version of convert_identifiers.
        
        Args:
            identifiers: List of input identifiers
            input_type: Type of input identifiers
            output_type: Type of output identifiers
            species: Species of the identifiers
            ambiguity_policy: How to handle ambiguous mappings
            
        Returns:
            List of conversion results
        """
        return asyncio.run(self.convert_identifiers(
            identifiers, input_type, output_type, species, ambiguity_policy
        ))
