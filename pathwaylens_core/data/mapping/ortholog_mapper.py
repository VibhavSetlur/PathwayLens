"""
Ortholog mapping utilities for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
import requests
from loguru import logger


class OrthologMapper:
    """Ortholog mapping and cross-species gene conversion utilities."""
    
    def __init__(self):
        """Initialize the ortholog mapper."""
        self.logger = logger.bind(module="ortholog_mapper")
        
        # Supported species
        self.supported_species = {
            'human': {'taxon_id': 9606, 'name': 'Homo sapiens'},
            'mouse': {'taxon_id': 10090, 'name': 'Mus musculus'},
            'rat': {'taxon_id': 10116, 'name': 'Rattus norvegicus'},
            'zebrafish': {'taxon_id': 7955, 'name': 'Danio rerio'},
            'drosophila': {'taxon_id': 7227, 'name': 'Drosophila melanogaster'},
            'c_elegans': {'taxon_id': 6239, 'name': 'Caenorhabditis elegans'},
            'yeast': {'taxon_id': 559292, 'name': 'Saccharomyces cerevisiae'},
            'arabidopsis': {'taxon_id': 3702, 'name': 'Arabidopsis thaliana'}
        }
        
        # Ortholog databases
        self.ortholog_databases = {
            'ensembl': 'https://rest.ensembl.org',
            'homologene': 'https://www.ncbi.nlm.nih.gov/homologene',
            'orthodb': 'https://www.orthodb.org'
        }
    
    async def map_orthologs(
        self,
        gene_ids: List[str],
        source_species: str,
        target_species: str,
        source_id_type: str = 'ensembl_gene',
        target_id_type: str = 'ensembl_gene',
        database: str = 'ensembl'
    ) -> Dict[str, Any]:
        """
        Map orthologs between species.
        
        Args:
            gene_ids: List of gene IDs to map
            source_species: Source species
            target_species: Target species
            source_id_type: Source ID type
            target_id_type: Target ID type
            database: Ortholog database to use
            
        Returns:
            Dictionary with ortholog mapping results
        """
        self.logger.info(f"Mapping orthologs from {source_species} to {target_species}")
        
        # Validate species
        if source_species not in self.supported_species:
            raise ValueError(f"Unsupported source species: {source_species}")
        if target_species not in self.supported_species:
            raise ValueError(f"Unsupported target species: {target_species}")
        
        try:
            if database == 'ensembl':
                result = await self._map_via_ensembl(
                    gene_ids, source_species, target_species, source_id_type, target_id_type
                )
            elif database == 'homologene':
                result = await self._map_via_homologene(
                    gene_ids, source_species, target_species, source_id_type, target_id_type
                )
            else:
                raise ValueError(f"Unsupported ortholog database: {database}")
            
            # Analyze mapping results
            mapping_stats = self._analyze_ortholog_mapping(result, gene_ids)
            
            return {
                'ortholog_mappings': result,
                'mapping_stats': mapping_stats,
                'source_species': source_species,
                'target_species': target_species,
                'source_id_type': source_id_type,
                'target_id_type': target_id_type,
                'database': database
            }
            
        except Exception as e:
            self.logger.error(f"Ortholog mapping failed: {e}")
            raise
    
    async def _map_via_ensembl(
        self,
        gene_ids: List[str],
        source_species: str,
        target_species: str,
        source_id_type: str,
        target_id_type: str
    ) -> Dict[str, Any]:
        """Map orthologs using Ensembl REST API."""
        source_taxon = self.supported_species[source_species]['taxon_id']
        target_taxon = self.supported_species[target_species]['taxon_id']
        
        # Get species names for Ensembl
        source_ensembl_name = self._get_ensembl_species_name(source_species)
        target_ensembl_name = self._get_ensembl_species_name(target_species)
        
        ortholog_mappings = {}
        
        for gene_id in gene_ids:
            try:
                # Get orthologs
                url = f"{self.ortholog_databases['ensembl']}/homology/id/{gene_id}"
                params = {
                    'type': 'orthologues',
                    'target_species': target_ensembl_name,
                    'format': 'json'
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if 'data' in data and data['data']:
                    homology_data = data['data'][0]
                    if 'homologies' in homology_data:
                        orthologs = []
                        for homology in homology_data['homologies']:
                            if 'target' in homology:
                                target_gene = homology['target']
                                ortholog_info = {
                                    'target_id': target_gene.get('id'),
                                    'target_species': target_gene.get('species'),
                                    'type': homology.get('type'),
                                    'confidence': homology.get('confidence', 0),
                                    'dN': homology.get('dN'),
                                    'dS': homology.get('dS')
                                }
                                orthologs.append(ortholog_info)
                        
                        ortholog_mappings[gene_id] = {
                            'source_id': gene_id,
                            'source_species': source_species,
                            'orthologs': orthologs,
                            'num_orthologs': len(orthologs)
                        }
                    else:
                        ortholog_mappings[gene_id] = {
                            'source_id': gene_id,
                            'source_species': source_species,
                            'orthologs': [],
                            'num_orthologs': 0,
                            'error': 'No homologies found'
                        }
                else:
                    ortholog_mappings[gene_id] = {
                        'source_id': gene_id,
                        'source_species': source_species,
                        'orthologs': [],
                        'num_orthologs': 0,
                        'error': 'No data returned'
                    }
                    
            except Exception as e:
                self.logger.warning(f"Failed to map ortholog for {gene_id}: {e}")
                ortholog_mappings[gene_id] = {
                    'source_id': gene_id,
                    'source_species': source_species,
                    'orthologs': [],
                    'num_orthologs': 0,
                    'error': str(e)
                }
        
        return ortholog_mappings
    
    async def _map_via_homologene(
        self,
        gene_ids: List[str],
        source_species: str,
        target_species: str,
        source_id_type: str,
        target_id_type: str
    ) -> Dict[str, Any]:
        """Map orthologs using NCBI HomoloGene database."""
        # This is a simplified implementation
        # In practice, would use NCBI E-utilities or HomoloGene API
        
        ortholog_mappings = {}
        
        for gene_id in gene_ids:
            # Placeholder implementation
            ortholog_mappings[gene_id] = {
                'source_id': gene_id,
                'source_species': source_species,
                'orthologs': [],
                'num_orthologs': 0,
                'error': 'HomoloGene mapping not implemented'
            }
        
        return ortholog_mappings
    
    def _get_ensembl_species_name(self, species: str) -> str:
        """Get Ensembl species name from internal species name."""
        ensembl_names = {
            'human': 'homo_sapiens',
            'mouse': 'mus_musculus',
            'rat': 'rattus_norvegicus',
            'zebrafish': 'danio_rerio',
            'drosophila': 'drosophila_melanogaster',
            'c_elegans': 'caenorhabditis_elegans',
            'yeast': 'saccharomyces_cerevisiae',
            'arabidopsis': 'arabidopsis_thaliana'
        }
        
        return ensembl_names.get(species, species)
    
    def _analyze_ortholog_mapping(
        self, 
        mapping_result: Dict[str, Any], 
        original_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze ortholog mapping results and calculate statistics."""
        total_input = len(original_ids)
        successfully_mapped = 0
        failed_mappings = 0
        total_orthologs = 0
        one_to_one_mappings = 0
        one_to_many_mappings = 0
        many_to_one_mappings = 0
        
        for gene_id, mapping_info in mapping_result.items():
            if mapping_info['num_orthologs'] > 0:
                successfully_mapped += 1
                total_orthologs += mapping_info['num_orthologs']
                
                if mapping_info['num_orthologs'] == 1:
                    one_to_one_mappings += 1
                else:
                    one_to_many_mappings += 1
            else:
                failed_mappings += 1
        
        mapping_rate = successfully_mapped / total_input if total_input > 0 else 0.0
        avg_orthologs_per_gene = total_orthologs / successfully_mapped if successfully_mapped > 0 else 0.0
        
        return {
            'total_input': total_input,
            'successfully_mapped': successfully_mapped,
            'failed_mappings': failed_mappings,
            'mapping_rate': mapping_rate,
            'total_orthologs': total_orthologs,
            'avg_orthologs_per_gene': avg_orthologs_per_gene,
            'one_to_one_mappings': one_to_one_mappings,
            'one_to_many_mappings': one_to_many_mappings,
            'many_to_one_mappings': many_to_one_mappings
        }
    
    async def batch_map_orthologs(
        self,
        gene_lists: Dict[str, List[str]],
        source_species: str,
        target_species: str,
        source_id_type: str = 'ensembl_gene',
        target_id_type: str = 'ensembl_gene'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map orthologs for multiple gene lists in batch.
        
        Args:
            gene_lists: Dictionary mapping list names to gene ID lists
            source_species: Source species
            target_species: Target species
            source_id_type: Source ID type
            target_id_type: Target ID type
            
        Returns:
            Dictionary with ortholog mapping results for each list
        """
        self.logger.info(f"Batch mapping orthologs for {len(gene_lists)} gene lists")
        
        results = {}
        
        for list_name, gene_ids in gene_lists.items():
            try:
                result = await self.map_orthologs(
                    gene_ids, source_species, target_species, source_id_type, target_id_type
                )
                results[list_name] = result
            except Exception as e:
                self.logger.error(f"Failed to map orthologs for gene list {list_name}: {e}")
                results[list_name] = {
                    'error': str(e),
                    'ortholog_mappings': {},
                    'mapping_stats': {
                        'total_input': len(gene_ids),
                        'successfully_mapped': 0,
                        'failed_mappings': len(gene_ids),
                        'mapping_rate': 0.0
                    }
                }
        
        return results
    
    async def create_ortholog_table(
        self,
        gene_ids: List[str],
        source_species: str,
        target_species: str,
        source_id_type: str = 'ensembl_gene',
        target_id_type: str = 'ensembl_gene'
    ) -> pd.DataFrame:
        """
        Create a comprehensive ortholog mapping table.
        
        Args:
            gene_ids: List of gene IDs to map
            source_species: Source species
            target_species: Target species
            source_id_type: Source ID type
            target_id_type: Target ID type
            
        Returns:
            DataFrame with ortholog mappings
        """
        self.logger.info(f"Creating ortholog table for {len(gene_ids)} genes")
        
        # Get ortholog mappings
        mapping_result = await self.map_orthologs(
            gene_ids, source_species, target_species, source_id_type, target_id_type
        )
        
        # Create table
        table_data = []
        
        for gene_id, mapping_info in mapping_result['ortholog_mappings'].items():
            if mapping_info['orthologs']:
                for ortholog in mapping_info['orthologs']:
                    row = {
                        'source_id': gene_id,
                        'source_species': source_species,
                        'source_id_type': source_id_type,
                        'target_id': ortholog['target_id'],
                        'target_species': target_species,
                        'target_id_type': target_id_type,
                        'ortholog_type': ortholog.get('type', 'unknown'),
                        'confidence': ortholog.get('confidence', 0),
                        'dN': ortholog.get('dN'),
                        'dS': ortholog.get('dS')
                    }
                    table_data.append(row)
            else:
                # No orthologs found
                row = {
                    'source_id': gene_id,
                    'source_species': source_species,
                    'source_id_type': source_id_type,
                    'target_id': None,
                    'target_species': target_species,
                    'target_id_type': target_id_type,
                    'ortholog_type': None,
                    'confidence': 0,
                    'dN': None,
                    'dS': None
                }
                table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def get_supported_species(self) -> List[str]:
        """Get list of supported species."""
        return list(self.supported_species.keys())
    
    def get_species_info(self, species: str) -> Dict[str, Any]:
        """Get information about a species."""
        return self.supported_species.get(species, {})
    
    async def validate_species_pair(
        self, 
        source_species: str, 
        target_species: str
    ) -> bool:
        """Validate if ortholog mapping is supported between two species."""
        return (source_species in self.supported_species and 
                target_species in self.supported_species)
    
    async def get_ortholog_statistics(
        self,
        source_species: str,
        target_species: str
    ) -> Dict[str, Any]:
        """Get statistics about ortholog relationships between species."""
        # This would typically query a reference database
        # For now, return placeholder statistics
        
        return {
            'source_species': source_species,
            'target_species': target_species,
            'total_orthologs': 0,  # Would be calculated from database
            'one_to_one_ratio': 0.0,
            'one_to_many_ratio': 0.0,
            'many_to_one_ratio': 0.0,
            'avg_confidence': 0.0
        }
