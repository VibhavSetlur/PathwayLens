"""
Pathway mapping utilities for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from loguru import logger

from ..adapters.base import BaseAdapter


class PathwayMapper:
    """Pathway mapping and cross-database pathway conversion utilities."""
    
    def __init__(self):
        """Initialize the pathway mapper."""
        self.logger = logger.bind(module="pathway_mapper")
        
        # Pathway database mappings
        self.database_mappings = {
            'kegg': {
                'name': 'KEGG',
                'url_prefix': 'https://www.genome.jp/kegg-bin/show_pathway?',
                'id_prefix': 'hsa'
            },
            'reactome': {
                'name': 'Reactome',
                'url_prefix': 'https://reactome.org/content/detail/',
                'id_prefix': 'R-HSA'
            },
            'go': {
                'name': 'Gene Ontology',
                'url_prefix': 'http://amigo.geneontology.org/amigo/term/',
                'id_prefix': 'GO:'
            },
            'biocyc': {
                'name': 'BioCyc',
                'url_prefix': 'https://biocyc.org/',
                'id_prefix': 'HUMAN'
            },
            'pathway_commons': {
                'name': 'Pathway Commons',
                'url_prefix': 'https://www.pathwaycommons.org/pc2/',
                'id_prefix': 'PC'
            },
            'msigdb': {
                'name': 'MSigDB',
                'url_prefix': 'https://www.gsea-msigdb.org/gsea/msigdb/',
                'id_prefix': 'MSIGDB'
            },
            'panther': {
                'name': 'PANTHER',
                'url_prefix': 'https://www.pantherdb.org/pathway/',
                'id_prefix': 'PTHR'
            },
            'wikipathways': {
                'name': 'WikiPathways',
                'url_prefix': 'https://www.wikipathways.org/index.php/Pathway:',
                'id_prefix': 'WP'
            }
        }
        
        # Pathway type mappings
        self.pathway_types = {
            'metabolic': ['kegg', 'biocyc'],
            'signaling': ['reactome', 'pathway_commons', 'wikipathways'],
            'functional': ['go', 'msigdb'],
            'protein_family': ['panther']
        }
    
    async def map_pathways(
        self,
        pathway_ids: List[str],
        source_database: str,
        target_database: str,
        species: str = 'human'
    ) -> Dict[str, Any]:
        """
        Map pathways between databases.
        
        Args:
            pathway_ids: List of pathway IDs to map
            source_database: Source database
            target_database: Target database
            species: Species for mapping
            
        Returns:
            Dictionary with pathway mapping results
        """
        self.logger.info(f"Mapping pathways from {source_database} to {target_database}")
        
        # Validate databases
        if source_database not in self.database_mappings:
            raise ValueError(f"Unsupported source database: {source_database}")
        if target_database not in self.database_mappings:
            raise ValueError(f"Unsupported target database: {target_database}")
        
        try:
            # Get pathway mappings
            pathway_mappings = {}
            
            for pathway_id in pathway_ids:
                mapping_info = await self._map_single_pathway(
                    pathway_id, source_database, target_database, species
                )
                pathway_mappings[pathway_id] = mapping_info
            
            # Analyze mapping results
            mapping_stats = self._analyze_pathway_mapping(pathway_mappings, pathway_ids)
            
            return {
                'pathway_mappings': pathway_mappings,
                'mapping_stats': mapping_stats,
                'source_database': source_database,
                'target_database': target_database,
                'species': species
            }
            
        except Exception as e:
            self.logger.error(f"Pathway mapping failed: {e}")
            raise
    
    async def _map_single_pathway(
        self,
        pathway_id: str,
        source_database: str,
        target_database: str,
        species: str
    ) -> Dict[str, Any]:
        """Map a single pathway between databases."""
        try:
            # Get pathway information from source database
            source_info = await self._get_pathway_info(pathway_id, source_database, species)
            
            if not source_info:
                return {
                    'source_id': pathway_id,
                    'source_database': source_database,
                    'target_mappings': [],
                    'num_mappings': 0,
                    'error': 'Source pathway not found'
                }
            
            # Find mappings in target database
            target_mappings = await self._find_target_mappings(
                source_info, target_database, species
            )
            
            return {
                'source_id': pathway_id,
                'source_database': source_database,
                'source_info': source_info,
                'target_mappings': target_mappings,
                'num_mappings': len(target_mappings)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to map pathway {pathway_id}: {e}")
            return {
                'source_id': pathway_id,
                'source_database': source_database,
                'target_mappings': [],
                'num_mappings': 0,
                'error': str(e)
            }
    
    async def _get_pathway_info(
        self,
        pathway_id: str,
        database: str,
        species: str
    ) -> Optional[Dict[str, Any]]:
        """Get pathway information from a database."""
        # This would typically query the actual database
        # For now, return placeholder information
        
        return {
            'pathway_id': pathway_id,
            'pathway_name': f"Pathway {pathway_id}",
            'database': database,
            'species': species,
            'genes': [],  # Would contain actual gene list
            'description': f"Description for {pathway_id}",
            'category': 'unknown',
            'url': self._get_pathway_url(pathway_id, database)
        }
    
    async def _find_target_mappings(
        self,
        source_info: Dict[str, Any],
        target_database: str,
        species: str
    ) -> List[Dict[str, Any]]:
        """Find mappings in target database."""
        # This would typically query the target database
        # For now, return placeholder mappings
        
        return [
            {
                'target_id': f"{target_database}_{source_info['pathway_id']}",
                'target_database': target_database,
                'target_name': f"Target pathway for {source_info['pathway_id']}",
                'mapping_confidence': 0.8,
                'mapping_type': 'direct',
                'url': self._get_pathway_url(f"{target_database}_{source_info['pathway_id']}", target_database)
            }
        ]
    
    def _get_pathway_url(self, pathway_id: str, database: str) -> str:
        """Get URL for a pathway in a specific database."""
        if database in self.database_mappings:
            db_info = self.database_mappings[database]
            return f"{db_info['url_prefix']}{pathway_id}"
        return ""
    
    def _analyze_pathway_mapping(
        self, 
        mapping_result: Dict[str, Any], 
        original_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze pathway mapping results and calculate statistics."""
        total_input = len(original_ids)
        successfully_mapped = 0
        failed_mappings = 0
        total_mappings = 0
        direct_mappings = 0
        indirect_mappings = 0
        
        for pathway_id, mapping_info in mapping_result.items():
            if mapping_info['num_mappings'] > 0:
                successfully_mapped += 1
                total_mappings += mapping_info['num_mappings']
                
                # Count mapping types
                for mapping in mapping_info['target_mappings']:
                    if mapping.get('mapping_type') == 'direct':
                        direct_mappings += 1
                    else:
                        indirect_mappings += 1
            else:
                failed_mappings += 1
        
        mapping_rate = successfully_mapped / total_input if total_input > 0 else 0.0
        avg_mappings_per_pathway = total_mappings / successfully_mapped if successfully_mapped > 0 else 0.0
        
        return {
            'total_input': total_input,
            'successfully_mapped': successfully_mapped,
            'failed_mappings': failed_mappings,
            'mapping_rate': mapping_rate,
            'total_mappings': total_mappings,
            'avg_mappings_per_pathway': avg_mappings_per_pathway,
            'direct_mappings': direct_mappings,
            'indirect_mappings': indirect_mappings
        }
    
    async def batch_map_pathways(
        self,
        pathway_lists: Dict[str, List[str]],
        source_database: str,
        target_database: str,
        species: str = 'human'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map pathways for multiple lists in batch.
        
        Args:
            pathway_lists: Dictionary mapping list names to pathway ID lists
            source_database: Source database
            target_database: Target database
            species: Species for mapping
            
        Returns:
            Dictionary with pathway mapping results for each list
        """
        self.logger.info(f"Batch mapping pathways for {len(pathway_lists)} lists")
        
        results = {}
        
        for list_name, pathway_ids in pathway_lists.items():
            try:
                result = await self.map_pathways(
                    pathway_ids, source_database, target_database, species
                )
                results[list_name] = result
            except Exception as e:
                self.logger.error(f"Failed to map pathways for list {list_name}: {e}")
                results[list_name] = {
                    'error': str(e),
                    'pathway_mappings': {},
                    'mapping_stats': {
                        'total_input': len(pathway_ids),
                        'successfully_mapped': 0,
                        'failed_mappings': len(pathway_ids),
                        'mapping_rate': 0.0
                    }
                }
        
        return results
    
    async def create_pathway_mapping_table(
        self,
        pathway_ids: List[str],
        source_database: str,
        target_database: str,
        species: str = 'human'
    ) -> pd.DataFrame:
        """
        Create a comprehensive pathway mapping table.
        
        Args:
            pathway_ids: List of pathway IDs to map
            source_database: Source database
            target_database: Target database
            species: Species for mapping
            
        Returns:
            DataFrame with pathway mappings
        """
        self.logger.info(f"Creating pathway mapping table for {len(pathway_ids)} pathways")
        
        # Get pathway mappings
        mapping_result = await self.map_pathways(
            pathway_ids, source_database, target_database, species
        )
        
        # Create table
        table_data = []
        
        for pathway_id, mapping_info in mapping_result['pathway_mappings'].items():
            if mapping_info['target_mappings']:
                for mapping in mapping_info['target_mappings']:
                    row = {
                        'source_id': pathway_id,
                        'source_database': source_database,
                        'source_name': mapping_info.get('source_info', {}).get('pathway_name', ''),
                        'target_id': mapping['target_id'],
                        'target_database': target_database,
                        'target_name': mapping['target_name'],
                        'mapping_confidence': mapping.get('mapping_confidence', 0),
                        'mapping_type': mapping.get('mapping_type', 'unknown'),
                        'source_url': mapping_info.get('source_info', {}).get('url', ''),
                        'target_url': mapping.get('url', '')
                    }
                    table_data.append(row)
            else:
                # No mappings found
                row = {
                    'source_id': pathway_id,
                    'source_database': source_database,
                    'source_name': mapping_info.get('source_info', {}).get('pathway_name', ''),
                    'target_id': None,
                    'target_database': target_database,
                    'target_name': None,
                    'mapping_confidence': 0,
                    'mapping_type': None,
                    'source_url': mapping_info.get('source_info', {}).get('url', ''),
                    'target_url': None
                }
                table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def get_supported_databases(self) -> List[str]:
        """Get list of supported databases."""
        return list(self.database_mappings.keys())
    
    def get_database_info(self, database: str) -> Dict[str, Any]:
        """Get information about a database."""
        return self.database_mappings.get(database, {})
    
    def get_pathway_types(self) -> Dict[str, List[str]]:
        """Get pathway types and their associated databases."""
        return self.pathway_types
    
    async def validate_database_pair(
        self, 
        source_database: str, 
        target_database: str
    ) -> bool:
        """Validate if pathway mapping is supported between two databases."""
        return (source_database in self.database_mappings and 
                target_database in self.database_mappings)
    
    async def get_pathway_statistics(
        self,
        source_database: str,
        target_database: str
    ) -> Dict[str, Any]:
        """Get statistics about pathway relationships between databases."""
        # This would typically query a reference database
        # For now, return placeholder statistics
        
        return {
            'source_database': source_database,
            'target_database': target_database,
            'total_pathways': 0,  # Would be calculated from database
            'mapping_coverage': 0.0,
            'avg_mapping_confidence': 0.0,
            'direct_mapping_ratio': 0.0,
            'indirect_mapping_ratio': 0.0
        }
    
    async def find_equivalent_pathways(
        self,
        pathway_id: str,
        database: str,
        species: str = 'human'
    ) -> Dict[str, List[str]]:
        """
        Find equivalent pathways across all supported databases.
        
        Args:
            pathway_id: Pathway ID to find equivalents for
            database: Source database
            species: Species for mapping
            
        Returns:
            Dictionary mapping database names to lists of equivalent pathway IDs
        """
        self.logger.info(f"Finding equivalent pathways for {pathway_id} in {database}")
        
        equivalent_pathways = {}
        
        for target_db in self.database_mappings.keys():
            if target_db != database:
                try:
                    mapping_result = await self.map_pathways(
                        [pathway_id], database, target_db, species
                    )
                    
                    if mapping_result['pathway_mappings'][pathway_id]['target_mappings']:
                        equivalent_pathways[target_db] = [
                            mapping['target_id'] 
                            for mapping in mapping_result['pathway_mappings'][pathway_id]['target_mappings']
                        ]
                    else:
                        equivalent_pathways[target_db] = []
                        
                except Exception as e:
                    self.logger.warning(f"Failed to find equivalents in {target_db}: {e}")
                    equivalent_pathways[target_db] = []
        
        return equivalent_pathways
