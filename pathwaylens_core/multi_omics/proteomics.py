"""
Proteomics analysis module for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from ..analysis.schemas import AnalysisResult, AnalysisParameters
from ..data import DatabaseManager


class ProteomicsAnalyzer:
    """Proteomics analysis engine for PathwayLens."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the proteomics analyzer.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="proteomics_analyzer")
        self.database_manager = database_manager or DatabaseManager()
        
        # Protein ID patterns
        self.protein_id_patterns = {
            'uniprot': r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$',
            'ensembl_protein': r'^ENS[A-Z]*P\d{11}$',
            'refseq_protein': r'^[NX][MP]_\d+\.\d+$',
            'genbank_protein': r'^[A-Z]{1,2}_\d+\.\d+$',
            'protein_symbol': r'^[A-Za-z][A-Za-z0-9]*$'
        }
        
        # Protein modification types
        self.modification_types = {
            'phosphorylation': ['S', 'T', 'Y'],
            'acetylation': ['K'],
            'methylation': ['K', 'R'],
            'ubiquitination': ['K'],
            'sumoylation': ['K'],
            'glycosylation': ['N', 'S', 'T']
        }
    
    async def analyze_protein_abundance(
        self,
        protein_data: pd.DataFrame,
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze protein abundance data.
        
        Args:
            protein_data: Protein abundance data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with protein abundance analysis
        """
        self.logger.info("Starting protein abundance analysis")
        
        try:
            # Validate input data
            validation_result = await self._validate_protein_data(protein_data)
            if not validation_result['valid']:
                raise ValueError(f"Invalid protein data: {validation_result['errors']}")
            
            # Extract protein IDs
            protein_ids = self._extract_protein_ids(protein_data)
            
            # Convert protein IDs to gene IDs if needed
            gene_ids = await self._convert_protein_to_gene_ids(protein_ids, parameters.species)
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                gene_ids, parameters, output_dir
            )
            
            # Add proteomics-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'proteomics',
                'protein_count': len(protein_ids),
                'gene_count': len(gene_ids),
                'conversion_rate': len(gene_ids) / len(protein_ids) if protein_ids else 0
            })
            
            self.logger.info("Protein abundance analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Protein abundance analysis failed: {e}")
            raise
    
    async def analyze_protein_modifications(
        self,
        modification_data: pd.DataFrame,
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze protein modification data.
        
        Args:
            modification_data: Protein modification data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with protein modification analysis
        """
        self.logger.info("Starting protein modification analysis")
        
        try:
            # Validate modification data
            validation_result = await self._validate_modification_data(modification_data)
            if not validation_result['valid']:
                raise ValueError(f"Invalid modification data: {validation_result['errors']}")
            
            # Extract modified proteins
            modified_proteins = self._extract_modified_proteins(modification_data)
            
            # Convert to gene IDs
            gene_ids = await self._convert_protein_to_gene_ids(modified_proteins, parameters.species)
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                gene_ids, parameters, output_dir
            )
            
            # Add modification-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'protein_modifications',
                'modified_protein_count': len(modified_proteins),
                'gene_count': len(gene_ids),
                'modification_types': list(modification_data['modification_type'].unique()) if 'modification_type' in modification_data.columns else []
            })
            
            self.logger.info("Protein modification analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Protein modification analysis failed: {e}")
            raise
    
    async def _validate_protein_data(self, protein_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate protein data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if protein_data.empty:
            validation_result['valid'] = False
            validation_result['errors'].append("Protein data is empty")
            return validation_result
        
        # Check for protein ID column
        protein_id_columns = ['protein_id', 'uniprot_id', 'accession', 'protein']
        protein_id_col = None
        
        for col in protein_id_columns:
            if col in protein_data.columns:
                protein_id_col = col
                break
        
        if protein_id_col is None:
            validation_result['valid'] = False
            validation_result['errors'].append("No protein ID column found")
            return validation_result
        
        # Check for abundance columns
        abundance_columns = [col for col in protein_data.columns if 'abundance' in col.lower() or 'intensity' in col.lower()]
        if not abundance_columns:
            validation_result['warnings'].append("No abundance/intensity columns found")
        
        return validation_result
    
    async def _validate_modification_data(self, modification_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate protein modification data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if modification_data.empty:
            validation_result['valid'] = False
            validation_result['errors'].append("Modification data is empty")
            return validation_result
        
        # Check for required columns
        required_columns = ['protein_id', 'site', 'modification_type']
        missing_columns = [col for col in required_columns if col not in modification_data.columns]
        
        if missing_columns:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
        
        return validation_result
    
    def _extract_protein_ids(self, protein_data: pd.DataFrame) -> List[str]:
        """Extract protein IDs from data."""
        protein_id_columns = ['protein_id', 'uniprot_id', 'accession', 'protein']
        
        for col in protein_id_columns:
            if col in protein_data.columns:
                return protein_data[col].dropna().unique().tolist()
        
        return []
    
    def _extract_modified_proteins(self, modification_data: pd.DataFrame) -> List[str]:
        """Extract modified protein IDs from data."""
        if 'protein_id' in modification_data.columns:
            return modification_data['protein_id'].dropna().unique().tolist()
        return []
    
    async def _convert_protein_to_gene_ids(
        self, 
        protein_ids: List[str], 
        species: str
    ) -> List[str]:
        """Convert protein IDs to gene IDs."""
        # This would typically use a protein-to-gene mapping service
        # For now, return a simplified conversion
        
        gene_ids = []
        
        for protein_id in protein_ids:
            # Simplified conversion - in practice would use UniProt API or similar
            if protein_id.startswith('ENS'):
                # Ensembl protein ID - convert to gene ID
                gene_id = protein_id.replace('P', 'G')
                gene_ids.append(gene_id)
            else:
                # For other protein IDs, would need actual mapping
                gene_ids.append(protein_id)
        
        return gene_ids
    
    async def detect_protein_id_type(self, protein_ids: List[str]) -> str:
        """Detect protein ID type."""
        import re
        
        type_scores = {}
        
        for id_type, pattern in self.protein_id_patterns.items():
            matches = sum(1 for protein_id in protein_ids if re.match(pattern, protein_id))
            type_scores[id_type] = matches / len(protein_ids) if protein_ids else 0
        
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    async def get_protein_modifications(
        self, 
        protein_id: str, 
        modification_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get protein modifications for a specific protein."""
        # This would typically query a protein modification database
        # For now, return placeholder data
        
        modifications = []
        
        if modification_type and modification_type in self.modification_types:
            amino_acids = self.modification_types[modification_type]
            for aa in amino_acids:
                modifications.append({
                    'protein_id': protein_id,
                    'site': f"{aa}123",  # Placeholder site
                    'modification_type': modification_type,
                    'confidence': 0.8
                })
        
        return modifications
    
    async def calculate_protein_abundance_statistics(
        self, 
        protein_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate protein abundance statistics."""
        abundance_columns = [col for col in protein_data.columns if 'abundance' in col.lower() or 'intensity' in col.lower()]
        
        if not abundance_columns:
            return {}
        
        stats = {}
        
        for col in abundance_columns:
            col_data = protein_data[col].dropna()
            stats[col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'count': len(col_data)
            }
        
        return stats
    
    async def normalize_protein_abundance(
        self, 
        protein_data: pd.DataFrame, 
        method: str = 'quantile'
    ) -> pd.DataFrame:
        """Normalize protein abundance data."""
        abundance_columns = [col for col in protein_data.columns if 'abundance' in col.lower() or 'intensity' in col.lower()]
        
        if not abundance_columns:
            return protein_data
        
        normalized_data = protein_data.copy()
        
        if method == 'quantile':
            # Quantile normalization
            from scipy import stats
            abundance_matrix = protein_data[abundance_columns].values
            normalized_matrix = stats.rankdata(abundance_matrix, axis=0)
            normalized_matrix = normalized_matrix / normalized_matrix.max()
            normalized_data[abundance_columns] = normalized_matrix
        
        elif method == 'log2':
            # Log2 transformation
            normalized_data[abundance_columns] = np.log2(protein_data[abundance_columns] + 1)
        
        elif method == 'zscore':
            # Z-score normalization
            normalized_data[abundance_columns] = (protein_data[abundance_columns] - protein_data[abundance_columns].mean()) / protein_data[abundance_columns].std()
        
        return normalized_data
