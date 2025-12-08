"""
Gene identifier conversion using multiple databases and APIs.
"""

import asyncio
import aiohttp
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
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


@dataclass
class ConversionStatistics:
    """Statistics for a batch conversion operation."""
    total_input: int
    successful: int
    failed: int
    ambiguous: int
    average_confidence: float
    source_breakdown: Dict[str, int]
    error_summary: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'total_input': self.total_input,
            'successful': self.successful,
            'failed': self.failed,
            'ambiguous': self.ambiguous,
            'success_rate': self.successful / self.total_input if self.total_input > 0 else 0.0,
            'average_confidence': self.average_confidence,
            'source_breakdown': self.source_breakdown,
            'error_summary': self.error_summary
        }


class IDConverter:
    """Converts gene identifiers across different formats and databases."""
    
    # Batch size for efficient processing
    DEFAULT_BATCH_SIZE = 1000
    MAX_BATCH_SIZE = 10000
    
    def __init__(self, rate_limit: float = 1.0, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initialize ID converter.
        
        Args:
            rate_limit: Rate limit in requests per second
            batch_size: Number of identifiers to process per batch
        """
        self.rate_limit = rate_limit
        self.batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        self.logger = logger.bind(module="id_converter")
        self.session = None
        
        # API endpoints
        self.mygene_url = "https://mygene.info/v3"
        self.ensembl_url = "https://rest.ensembl.org"
        self.ncbi_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Rate limiting
        self.last_request_time = 0
        
        # Statistics tracking
        self._conversion_stats: Optional[ConversionStatistics] = None
        
        # Simple JSON cache
        self.cache_file = Path.home() / ".pathwaylens" / "id_cache.json"
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                import json
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return {}
        
    def _save_cache(self):
        """Save cache to disk."""
        try:
            import json
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
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
        ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.EXPAND,
        track_statistics: bool = True,
        services: Optional[List[str]] = None
    ) -> List[ConversionResult]:
        """
        Convert identifiers from input type to output type.
        
        Supports batch processing for large identifier lists to improve efficiency.
        
        Args:
            identifiers: List of input identifiers
            input_type: Type of input identifiers
            output_type: Type of output identifiers
            species: Species of the identifiers
            ambiguity_policy: How to handle ambiguous mappings
            track_statistics: Whether to track conversion statistics
            
        Returns:
            List of conversion results
            
        Example:
            >>> converter = IDConverter()
            >>> async with converter:
            ...     results = await converter.convert_identifiers(
            ...         ["BRCA1", "TP53", "EGFR"],
            ...         IDType.SYMBOL,
            ...         IDType.ENSEMBL,
            ...         SpeciesType.HUMAN
            ...     )
            >>> print(f"Converted {len(results)} identifiers")
        """
        if not identifiers:
            self.logger.warning("Empty identifier list provided")
            return []
        
        self.logger.info(f"Converting {len(identifiers)} identifiers from {input_type} to {output_type}")
        
        # If input and output types are the same, return identity mapping
        if input_type == output_type:
            identity_results = [
                ConversionResult(
                    input_id=id_val,
                    output_id=id_val,
                    confidence=1.0,
                    source="identity"
                )
                for id_val in identifiers
            ]
            if track_statistics:
                self._update_statistics(identity_results, {})
            return identity_results
        
        # Process in batches for large lists
        if len(identifiers) > self.batch_size:
            return await self._convert_in_batches(
                identifiers, input_type, output_type, species, ambiguity_policy, track_statistics, services
            )
        
        # Check cache
        cached_results = []
        uncached_identifiers = []
        
        for id_val in identifiers:
            cache_key = f"{id_val}:{input_type.value}:{output_type.value}:{species.value}"
            if cache_key in self.cache:
                # Reconstruct result from cache
                cached_data = self.cache[cache_key]
                cached_results.append(ConversionResult(**cached_data))
            else:
                uncached_identifiers.append(id_val)
        
        if not uncached_identifiers:
            self.logger.info("All identifiers found in cache")
            return cached_results
            
        # Process uncached identifiers
        results = []
        errors: Dict[str, int] = {}
        
        # Update identifiers list to only process uncached
        identifiers = uncached_identifiers
        
        # Default services if not specified
        if not services:
            services = ["mygene", "ensembl", "ncbi"]
            
        # Method 1: MyGene.info
        if "mygene" in services:
            try:
                mygene_results = await self._convert_via_mygene(
                    identifiers, input_type, output_type, species
                )
                results.extend(mygene_results)
            except Exception as e:
                error_msg = str(e)
                errors[error_msg] = errors.get(error_msg, 0) + 1
                self.logger.warning(f"MyGene conversion failed: {e}. This may affect conversion accuracy.")
        
        # Method 2: Ensembl
        if "ensembl" in services:
            try:
                ensembl_results = await self._convert_via_ensembl(
                    identifiers, input_type, output_type, species
                )
                results.extend(ensembl_results)
            except Exception as e:
                error_msg = str(e)
                errors[error_msg] = errors.get(error_msg, 0) + 1
                self.logger.warning(f"Ensembl conversion failed: {e}. This may affect conversion accuracy.")
        
        # Method 3: NCBI
        if "ncbi" in services:
            try:
                ncbi_results = await self._convert_via_ncbi(
                    identifiers, input_type, output_type, species
                )
                results.extend(ncbi_results)
            except Exception as e:
                error_msg = str(e)
                errors[error_msg] = errors.get(error_msg, 0) + 1
                self.logger.warning(f"NCBI conversion failed: {e}. This may affect conversion accuracy.")
        
        # Merge and deduplicate results
        merged_results = self._merge_conversion_results(results, ambiguity_policy)
        
        # Update cache
        for result in merged_results:
            cache_key = f"{result.input_id}:{input_type.value}:{output_type.value}:{species.value}"
            # Store as dict
            self.cache[cache_key] = {
                "input_id": result.input_id,
                "output_id": result.output_id,
                "confidence": result.confidence,
                "source": result.source,
                "is_ambiguous": result.is_ambiguous,
                "alternative_mappings": result.alternative_mappings
            }
        self._save_cache()
        
        # Combine with cached results
        merged_results.extend(cached_results)
        
        # Track statistics
        if track_statistics:
            self._update_statistics(merged_results, errors)
        
        self.logger.info(f"Conversion completed: {len(merged_results)} results")
        return merged_results
    
    async def _convert_in_batches(
        self,
        identifiers: List[str],
        input_type: IDType,
        output_type: IDType,
        species: SpeciesType,
        ambiguity_policy: AmbiguityPolicy,
        track_statistics: bool,
        services: Optional[List[str]] = None
    ) -> List[ConversionResult]:
        """Convert identifiers in batches for efficient processing."""
        self.logger.info(f"Processing {len(identifiers)} identifiers in batches of {self.batch_size}")
        
        all_results = []
        total_batches = (len(identifiers) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(identifiers), self.batch_size):
            batch = identifiers[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            self.logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} identifiers)")
            
            try:
                batch_results = await self.convert_identifiers(
                    batch, input_type, output_type, species, ambiguity_policy, track_statistics=False, services=services
                )
                all_results.extend(batch_results)
            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {e}. Continuing with remaining batches.")
                # Add failed results
                for id_val in batch:
                    all_results.append(ConversionResult(
                        input_id=id_val,
                        output_id=None,
                        confidence=0.0,
                        source="error",
                        is_ambiguous=False
                    ))
        
        # Update statistics for entire batch
        if track_statistics:
            self._update_statistics(all_results, {})
        
        return all_results
    
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
        ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.EXPAND,
        track_statistics: bool = True
    ) -> List[ConversionResult]:
        """
        Synchronous version of convert_identifiers.
        
        Args:
            identifiers: List of input identifiers
            input_type: Type of input identifiers
            output_type: Type of output identifiers
            species: Species of the identifiers
            ambiguity_policy: How to handle ambiguous mappings
            track_statistics: Whether to track conversion statistics
            
        Returns:
            List of conversion results
        """
        return asyncio.run(self.convert_identifiers(
            identifiers, input_type, output_type, species, ambiguity_policy, track_statistics
        ))
    
    def _update_statistics(
        self, 
        results: List[ConversionResult], 
        errors: Dict[str, int]
    ) -> None:
        """Update conversion statistics."""
        if not results:
            return
        
        successful = sum(1 for r in results if r.output_id is not None)
        failed = sum(1 for r in results if r.output_id is None)
        ambiguous = sum(1 for r in results if r.is_ambiguous)
        
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        source_breakdown: Dict[str, int] = {}
        for r in results:
            source_breakdown[r.source] = source_breakdown.get(r.source, 0) + 1
        
        self._conversion_stats = ConversionStatistics(
            total_input=len(results),
            successful=successful,
            failed=failed,
            ambiguous=ambiguous,
            average_confidence=avg_confidence,
            source_breakdown=source_breakdown,
            error_summary=errors
        )
    
    def get_statistics(self) -> Optional[ConversionStatistics]:
        """
        Get conversion statistics from the last conversion operation.
        
        Returns:
            ConversionStatistics object or None if no conversion has been performed
        """
        return self._conversion_stats
    
    def get_conversion_report(self) -> Dict[str, Any]:
        """
        Get a detailed conversion report.
        
        Returns:
            Dictionary containing conversion statistics and recommendations
        """
        if not self._conversion_stats:
            return {
                'status': 'no_conversions',
                'message': 'No conversion operations have been performed yet'
            }
        
        stats_dict = self._conversion_stats.to_dict()
        success_rate = stats_dict['success_rate']
        
        # Add recommendations
        recommendations = []
        if success_rate < 0.5:
            recommendations.append("Low success rate detected. Consider checking input identifier format.")
        if self._conversion_stats.ambiguous > 0:
            recommendations.append(f"{self._conversion_stats.ambiguous} ambiguous mappings found. Review ambiguity policy.")
        if self._conversion_stats.average_confidence < 0.7:
            recommendations.append("Low average confidence. Consider using multiple conversion sources.")
        
        return {
            'statistics': stats_dict,
            'recommendations': recommendations,
            'summary': {
                'total_processed': self._conversion_stats.total_input,
                'success_rate': f"{success_rate * 100:.1f}%",
                'average_confidence': f"{self._conversion_stats.average_confidence:.3f}"
            }
        }
