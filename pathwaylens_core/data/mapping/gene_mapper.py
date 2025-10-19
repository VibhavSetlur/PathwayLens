"""
Gene ID mapping utilities for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
import requests
from loguru import logger

from ..adapters.base import BaseAdapter


class GeneMapper:
    """Gene ID mapping and conversion utilities."""
    
    def __init__(self):
        """Initialize the gene mapper."""
        self.logger = logger.bind(module="gene_mapper")
        
        # Common gene ID patterns
        self.id_patterns = {
            'ensembl_gene': r'^ENS[A-Z]*G\d{11}$',
            'ensembl_transcript': r'^ENS[A-Z]*T\d{11}$',
            'ensembl_protein': r'^ENS[A-Z]*P\d{11}$',
            'entrez': r'^\d+$',
            'uniprot': r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$',
            'refseq': r'^[NX][MR]_\d+\.\d+$',
            'genbank': r'^[A-Z]{1,2}_\d+\.\d+$',
            'symbol': r'^[A-Za-z][A-Za-z0-9]*$',
            'alias': r'^[A-Za-z][A-Za-z0-9]*$'
        }
        
        # ID type priorities for conversion
        self.id_priorities = {
            'ensembl_gene': 1,
            'entrez': 2,
            'symbol': 3,
            'uniprot': 4,
            'refseq': 5,
            'genbank': 6,
            'alias': 7
        }
    
    async def detect_id_type(self, gene_ids: List[str]) -> Dict[str, str]:
        """
        Detect the type of gene IDs in a list.
        
        Args:
            gene_ids: List of gene IDs to analyze
            
        Returns:
            Dictionary mapping gene IDs to their detected types
        """
        import re
        
        id_types = {}
        
        for gene_id in gene_ids:
            detected_type = 'unknown'
            
            for id_type, pattern in self.id_patterns.items():
                if re.match(pattern, gene_id):
                    detected_type = id_type
                    break
            
            id_types[gene_id] = detected_type
        
        return id_types
    
    async def convert_ids(
        self,
        gene_ids: List[str],
        source_type: str,
        target_type: str,
        species: str = 'human',
        adapter: Optional[BaseAdapter] = None
    ) -> Dict[str, Any]:
        """
        Convert gene IDs from source type to target type.
        
        Args:
            gene_ids: List of gene IDs to convert
            source_type: Source ID type
            target_type: Target ID type
            species: Species for conversion
            adapter: Database adapter to use
            
        Returns:
            Dictionary with conversion results
        """
        self.logger.info(f"Converting {len(gene_ids)} genes from {source_type} to {target_type}")
        
        try:
            if adapter:
                # Use specific adapter
                result = await adapter.convert_gene_ids(
                    gene_ids, source_type, target_type, species
                )
            else:
                # Use MyGene.info for general conversion
                result = await self._convert_via_mygene(
                    gene_ids, source_type, target_type, species
                )
            
            # Analyze conversion results
            conversion_stats = self._analyze_conversion_results(result, gene_ids)
            
            return {
                'converted_ids': result,
                'conversion_stats': conversion_stats,
                'source_type': source_type,
                'target_type': target_type,
                'species': species
            }
            
        except Exception as e:
            self.logger.error(f"Gene ID conversion failed: {e}")
            raise
    
    async def _convert_via_mygene(
        self,
        gene_ids: List[str],
        source_type: str,
        target_type: str,
        species: str
    ) -> Dict[str, Any]:
        """Convert gene IDs using MyGene.info API."""
        # Map ID types to MyGene.info format
        mygene_source = self._map_to_mygene_format(source_type)
        mygene_target = self._map_to_mygene_format(target_type)
        
        # Prepare query
        query = {
            'q': gene_ids,
            'scopes': mygene_source,
            'fields': mygene_target,
            'species': species
        }
        
        # Make API request
        url = 'https://mygene.info/v3/query'
        response = requests.post(url, json=query)
        response.raise_for_status()
        
        return response.json()
    
    def _map_to_mygene_format(self, id_type: str) -> str:
        """Map internal ID type to MyGene.info format."""
        mapping = {
            'ensembl_gene': 'ensembl.gene',
            'entrez': 'entrezgene',
            'symbol': 'symbol',
            'uniprot': 'uniprot.Swiss-Prot',
            'refseq': 'refseq.rna',
            'genbank': 'genbank',
            'alias': 'alias'
        }
        
        return mapping.get(id_type, id_type)
    
    def _analyze_conversion_results(
        self, 
        conversion_result: Dict[str, Any], 
        original_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze conversion results and calculate statistics."""
        if 'hits' not in conversion_result:
            return {
                'total_input': len(original_ids),
                'successfully_converted': 0,
                'failed_conversions': len(original_ids),
                'conversion_rate': 0.0,
                'ambiguous_mappings': 0,
                'duplicate_mappings': 0
            }
        
        hits = conversion_result['hits']
        total_input = len(original_ids)
        successfully_converted = 0
        failed_conversions = 0
        ambiguous_mappings = 0
        duplicate_mappings = 0
        
        converted_ids = set()
        
        for hit in hits:
            if 'notfound' in hit:
                failed_conversions += 1
            else:
                successfully_converted += 1
                
                # Check for ambiguous mappings
                if isinstance(hit.get('_id'), list) and len(hit['_id']) > 1:
                    ambiguous_mappings += 1
                
                # Check for duplicate mappings
                hit_id = hit.get('_id')
                if isinstance(hit_id, list):
                    hit_id = hit_id[0]
                
                if hit_id in converted_ids:
                    duplicate_mappings += 1
                else:
                    converted_ids.add(hit_id)
        
        conversion_rate = successfully_converted / total_input if total_input > 0 else 0.0
        
        return {
            'total_input': total_input,
            'successfully_converted': successfully_converted,
            'failed_conversions': failed_conversions,
            'conversion_rate': conversion_rate,
            'ambiguous_mappings': ambiguous_mappings,
            'duplicate_mappings': duplicate_mappings
        }
    
    async def batch_convert(
        self,
        gene_lists: Dict[str, List[str]],
        source_type: str,
        target_type: str,
        species: str = 'human'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert multiple gene lists in batch.
        
        Args:
            gene_lists: Dictionary mapping list names to gene ID lists
            source_type: Source ID type
            target_type: Target ID type
            species: Species for conversion
            
        Returns:
            Dictionary with conversion results for each list
        """
        self.logger.info(f"Batch converting {len(gene_lists)} gene lists")
        
        results = {}
        
        for list_name, gene_ids in gene_lists.items():
            try:
                result = await self.convert_ids(
                    gene_ids, source_type, target_type, species
                )
                results[list_name] = result
            except Exception as e:
                self.logger.error(f"Failed to convert gene list {list_name}: {e}")
                results[list_name] = {
                    'error': str(e),
                    'converted_ids': {},
                    'conversion_stats': {
                        'total_input': len(gene_ids),
                        'successfully_converted': 0,
                        'failed_conversions': len(gene_ids),
                        'conversion_rate': 0.0
                    }
                }
        
        return results
    
    async def validate_gene_ids(
        self,
        gene_ids: List[str],
        id_type: str,
        species: str = 'human'
    ) -> Dict[str, Any]:
        """
        Validate gene IDs against a database.
        
        Args:
            gene_ids: List of gene IDs to validate
            id_type: Type of gene IDs
            species: Species for validation
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating {len(gene_ids)} {id_type} gene IDs")
        
        # Detect ID types
        detected_types = await self.detect_id_type(gene_ids)
        
        # Count by type
        type_counts = {}
        for detected_type in detected_types.values():
            type_counts[detected_type] = type_counts.get(detected_type, 0) + 1
        
        # Check consistency
        consistent_type = max(type_counts.items(), key=lambda x: x[1])[0]
        consistency_rate = type_counts[consistent_type] / len(gene_ids)
        
        # Validate against database
        validation_result = await self._validate_against_database(
            gene_ids, id_type, species
        )
        
        return {
            'total_ids': len(gene_ids),
            'detected_types': type_counts,
            'consistent_type': consistent_type,
            'consistency_rate': consistency_rate,
            'validation_result': validation_result
        }
    
    async def _validate_against_database(
        self,
        gene_ids: List[str],
        id_type: str,
        species: str
    ) -> Dict[str, Any]:
        """Validate gene IDs against a reference database."""
        # This would typically query a reference database
        # For now, return a simplified validation result
        
        return {
            'valid_ids': len(gene_ids),  # Simplified
            'invalid_ids': 0,
            'validation_rate': 1.0
        }
    
    def get_id_type_priority(self, id_type: str) -> int:
        """Get priority score for an ID type."""
        return self.id_priorities.get(id_type, 999)
    
    def get_available_id_types(self) -> List[str]:
        """Get list of available ID types."""
        return list(self.id_patterns.keys())
    
    async def create_id_mapping_table(
        self,
        gene_ids: List[str],
        target_types: List[str],
        species: str = 'human'
    ) -> pd.DataFrame:
        """
        Create a comprehensive ID mapping table.
        
        Args:
            gene_ids: List of gene IDs to map
            target_types: List of target ID types
            species: Species for mapping
            
        Returns:
            DataFrame with ID mappings
        """
        self.logger.info(f"Creating ID mapping table for {len(gene_ids)} genes")
        
        # Detect source type
        detected_types = await self.detect_id_type(gene_ids)
        source_type = max(set(detected_types.values()), key=list(detected_types.values()).count)
        
        # Create mapping table
        mapping_data = []
        
        for gene_id in gene_ids:
            row = {'source_id': gene_id, 'source_type': source_type}
            
            # Convert to each target type
            for target_type in target_types:
                try:
                    result = await self.convert_ids(
                        [gene_id], source_type, target_type, species
                    )
                    
                    # Extract converted ID
                    if 'hits' in result and result['hits']:
                        hit = result['hits'][0]
                        if 'notfound' not in hit:
                            converted_id = hit.get('_id')
                            if isinstance(converted_id, list):
                                converted_id = converted_id[0]
                            row[target_type] = converted_id
                        else:
                            row[target_type] = None
                    else:
                        row[target_type] = None
                        
                except Exception as e:
                    self.logger.warning(f"Failed to convert {gene_id} to {target_type}: {e}")
                    row[target_type] = None
            
            mapping_data.append(row)
        
        return pd.DataFrame(mapping_data)
