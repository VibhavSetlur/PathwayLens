"""
Metabolomics analysis module for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from ..analysis.schemas import AnalysisResult, AnalysisParameters
from ..data import DatabaseManager


class MetabolomicsAnalyzer:
    """Metabolomics analysis engine for PathwayLens."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the metabolomics analyzer.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="metabolomics_analyzer")
        self.database_manager = database_manager or DatabaseManager()
        
        # Metabolite ID patterns
        self.metabolite_id_patterns = {
            'hmdb': r'^HMDB\d{7}$',
            'kegg_compound': r'^C\d{5}$',
            'chebi': r'^CHEBI:\d+$',
            'pubchem': r'^\d+$',
            'metlin': r'^METLIN_\d+$',
            'metabolite_name': r'^[A-Za-z][A-Za-z0-9\s\-_]*$'
        }
        
        # Metabolite classes
        self.metabolite_classes = {
            'amino_acids': ['alanine', 'glycine', 'serine', 'threonine', 'valine'],
            'lipids': ['cholesterol', 'triglyceride', 'phospholipid', 'fatty_acid'],
            'carbohydrates': ['glucose', 'fructose', 'sucrose', 'lactose'],
            'nucleotides': ['atp', 'adp', 'amp', 'gtp', 'gdp'],
            'cofactors': ['nad', 'nadp', 'fad', 'coa', 'thiamine']
        }
    
    async def analyze_metabolite_abundance(
        self,
        metabolite_data: pd.DataFrame,
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze metabolite abundance data.
        
        Args:
            metabolite_data: Metabolite abundance data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with metabolite abundance analysis
        """
        self.logger.info("Starting metabolite abundance analysis")
        
        try:
            # Validate input data
            validation_result = await self._validate_metabolite_data(metabolite_data)
            if not validation_result['valid']:
                raise ValueError(f"Invalid metabolite data: {validation_result['errors']}")
            
            # Extract metabolite IDs
            metabolite_ids = self._extract_metabolite_ids(metabolite_data)
            
            # Convert metabolite IDs to pathway information
            pathway_info = await self._convert_metabolite_to_pathways(metabolite_ids, parameters.species)
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            # Convert pathway info to gene-like format for analysis
            gene_ids = list(pathway_info.keys())
            
            result = await analysis_engine.analyze(
                gene_ids, parameters, output_dir
            )
            
            # Add metabolomics-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'metabolomics',
                'metabolite_count': len(metabolite_ids),
                'pathway_count': len(pathway_info),
                'metabolite_classes': self._classify_metabolites(metabolite_ids)
            })
            
            self.logger.info("Metabolite abundance analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Metabolite abundance analysis failed: {e}")
            raise
    
    async def analyze_metabolite_pathways(
        self,
        metabolite_data: pd.DataFrame,
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze metabolite pathway enrichment.
        
        Args:
            metabolite_data: Metabolite data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with metabolite pathway analysis
        """
        self.logger.info("Starting metabolite pathway analysis")
        
        try:
            # Validate metabolite data
            validation_result = await self._validate_metabolite_data(metabolite_data)
            if not validation_result['valid']:
                raise ValueError(f"Invalid metabolite data: {validation_result['errors']}")
            
            # Extract metabolites
            metabolite_ids = self._extract_metabolite_ids(metabolite_data)
            
            # Get pathway information for metabolites
            pathway_info = await self._get_metabolite_pathways(metabolite_ids, parameters.species)
            
            # Perform pathway enrichment analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            # Create gene-like representation for pathway analysis
            pathway_genes = []
            for pathway_id, metabolites in pathway_info.items():
                pathway_genes.extend([f"{pathway_id}_{met}" for met in metabolites])
            
            result = await analysis_engine.analyze(
                pathway_genes, parameters, output_dir
            )
            
            # Add metabolomics-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'metabolite_pathways',
                'metabolite_count': len(metabolite_ids),
                'pathway_count': len(pathway_info),
                'metabolite_pathway_mappings': pathway_info
            })
            
            self.logger.info("Metabolite pathway analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Metabolite pathway analysis failed: {e}")
            raise
    
    async def _validate_metabolite_data(self, metabolite_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate metabolite data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if metabolite_data.empty:
            validation_result['valid'] = False
            validation_result['errors'].append("Metabolite data is empty")
            return validation_result
        
        # Check for metabolite ID column
        metabolite_id_columns = ['metabolite_id', 'hmdb_id', 'kegg_id', 'chebi_id', 'metabolite_name']
        metabolite_id_col = None
        
        for col in metabolite_id_columns:
            if col in metabolite_data.columns:
                metabolite_id_col = col
                break
        
        if metabolite_id_col is None:
            validation_result['valid'] = False
            validation_result['errors'].append("No metabolite ID column found")
            return validation_result
        
        # Check for abundance columns
        abundance_columns = [col for col in metabolite_data.columns if 'abundance' in col.lower() or 'concentration' in col.lower()]
        if not abundance_columns:
            validation_result['warnings'].append("No abundance/concentration columns found")
        
        return validation_result
    
    def _extract_metabolite_ids(self, metabolite_data: pd.DataFrame) -> List[str]:
        """Extract metabolite IDs from data."""
        metabolite_id_columns = ['metabolite_id', 'hmdb_id', 'kegg_id', 'chebi_id', 'metabolite_name']
        
        for col in metabolite_id_columns:
            if col in metabolite_data.columns:
                return metabolite_data[col].dropna().unique().tolist()
        
        return []
    
    async def _convert_metabolite_to_pathways(
        self, 
        metabolite_ids: List[str], 
        species: str
    ) -> Dict[str, List[str]]:
        """Convert metabolite IDs to pathway information."""
        # This would typically query metabolite pathway databases
        # For now, return placeholder pathway information
        
        pathway_info = {}
        
        for metabolite_id in metabolite_ids:
            # Simplified pathway mapping - in practice would use KEGG, HMDB, etc.
            if metabolite_id.startswith('C'):
                # KEGG compound
                pathway_info[f"pathway_{metabolite_id}"] = [metabolite_id]
            else:
                # Other metabolite IDs
                pathway_info[f"metabolite_pathway_{metabolite_id}"] = [metabolite_id]
        
        return pathway_info
    
    async def _get_metabolite_pathways(
        self, 
        metabolite_ids: List[str], 
        species: str
    ) -> Dict[str, List[str]]:
        """Get pathway information for metabolites."""
        # This would typically query KEGG, HMDB, or other metabolite databases
        # For now, return placeholder pathway information
        
        pathway_info = {}
        
        for metabolite_id in metabolite_ids:
            # Simplified pathway mapping
            if metabolite_id.startswith('C'):
                # KEGG compound - map to metabolic pathways
                pathway_info['hsa00010'] = [metabolite_id]  # Glycolysis
                pathway_info['hsa00020'] = [metabolite_id]  # TCA cycle
            else:
                # Other metabolites
                pathway_info[f'metabolite_pathway_{metabolite_id}'] = [metabolite_id]
        
        return pathway_info
    
    def _classify_metabolites(self, metabolite_ids: List[str]) -> Dict[str, List[str]]:
        """Classify metabolites by chemical class."""
        classification = {class_name: [] for class_name in self.metabolite_classes.keys()}
        
        for metabolite_id in metabolite_ids:
            metabolite_lower = metabolite_id.lower()
            
            for class_name, examples in self.metabolite_classes.items():
                if any(example in metabolite_lower for example in examples):
                    classification[class_name].append(metabolite_id)
                    break
        
        return classification
    
    async def detect_metabolite_id_type(self, metabolite_ids: List[str]) -> str:
        """Detect metabolite ID type."""
        import re
        
        type_scores = {}
        
        for id_type, pattern in self.metabolite_id_patterns.items():
            matches = sum(1 for metabolite_id in metabolite_ids if re.match(pattern, metabolite_id))
            type_scores[id_type] = matches / len(metabolite_ids) if metabolite_ids else 0
        
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    async def get_metabolite_info(
        self, 
        metabolite_id: str, 
        id_type: str = 'auto'
    ) -> Dict[str, Any]:
        """Get metabolite information."""
        if id_type == 'auto':
            id_type = await self.detect_metabolite_id_type([metabolite_id])
        
        # This would typically query metabolite databases
        # For now, return placeholder information
        
        return {
            'metabolite_id': metabolite_id,
            'id_type': id_type,
            'name': f"Metabolite {metabolite_id}",
            'formula': 'C6H12O6',  # Placeholder
            'molecular_weight': 180.16,  # Placeholder
            'pathways': [f"pathway_{metabolite_id}"],
            'class': 'unknown'
        }
    
    async def calculate_metabolite_statistics(
        self, 
        metabolite_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate metabolite abundance statistics."""
        abundance_columns = [col for col in metabolite_data.columns if 'abundance' in col.lower() or 'concentration' in col.lower()]
        
        if not abundance_columns:
            return {}
        
        stats = {}
        
        for col in abundance_columns:
            col_data = metabolite_data[col].dropna()
            stats[col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'count': len(col_data),
                'detected_metabolites': (col_data > 0).sum(),
                'missing_metabolites': (col_data == 0).sum()
            }
        
        return stats
    
    async def normalize_metabolite_data(
        self, 
        metabolite_data: pd.DataFrame, 
        method: str = 'quantile'
    ) -> pd.DataFrame:
        """Normalize metabolite abundance data."""
        abundance_columns = [col for col in metabolite_data.columns if 'abundance' in col.lower() or 'concentration' in col.lower()]
        
        if not abundance_columns:
            return metabolite_data
        
        normalized_data = metabolite_data.copy()
        
        if method == 'quantile':
            # Quantile normalization
            from scipy import stats
            abundance_matrix = metabolite_data[abundance_columns].values
            normalized_matrix = stats.rankdata(abundance_matrix, axis=0)
            normalized_matrix = normalized_matrix / normalized_matrix.max()
            normalized_data[abundance_columns] = normalized_matrix
        
        elif method == 'log2':
            # Log2 transformation
            normalized_data[abundance_columns] = np.log2(metabolite_data[abundance_columns] + 1)
        
        elif method == 'zscore':
            # Z-score normalization
            normalized_data[abundance_columns] = (metabolite_data[abundance_columns] - metabolite_data[abundance_columns].mean()) / metabolite_data[abundance_columns].std()
        
        return normalized_data
    
    async def find_metabolite_correlations(
        self, 
        metabolite_data: pd.DataFrame, 
        threshold: float = 0.7
    ) -> pd.DataFrame:
        """Find correlations between metabolites."""
        abundance_columns = [col for col in metabolite_data.columns if 'abundance' in col.lower() or 'concentration' in col.lower()]
        
        if len(abundance_columns) < 2:
            return pd.DataFrame()
        
        # Calculate correlation matrix
        correlation_matrix = metabolite_data[abundance_columns].corr()
        
        # Find high correlations
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        'metabolite1': correlation_matrix.columns[i],
                        'metabolite2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return pd.DataFrame(high_correlations)
