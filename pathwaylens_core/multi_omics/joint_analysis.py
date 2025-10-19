"""
Joint multi-omics analysis module for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from loguru import logger

from ..analysis.schemas import AnalysisResult, AnalysisParameters
from ..data import DatabaseManager


class JointAnalyzer:
    """Joint multi-omics analysis engine for PathwayLens."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the joint analyzer.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="joint_analyzer")
        self.database_manager = database_manager or DatabaseManager()
        
        # Supported omics types
        self.omics_types = {
            'genomics': 'Genomics',
            'transcriptomics': 'Transcriptomics',
            'proteomics': 'Proteomics',
            'metabolomics': 'Metabolomics',
            'phosphoproteomics': 'Phosphoproteomics',
            'epigenomics': 'Epigenomics'
        }
    
    async def analyze_joint_omics(
        self,
        omics_data: Dict[str, pd.DataFrame],
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Perform joint multi-omics analysis.
        
        Args:
            omics_data: Dictionary mapping omics types to data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with joint multi-omics analysis
        """
        self.logger.info(f"Starting joint multi-omics analysis with {len(omics_data)} omics types")
        
        try:
            # Validate input data
            validation_result = await self._validate_omics_data(omics_data)
            if not validation_result['valid']:
                raise ValueError(f"Invalid omics data: {validation_result['errors']}")
            
            # Integrate omics data
            integrated_data = await self._integrate_omics_data(omics_data, parameters.species)
            
            # Perform joint pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                integrated_data, parameters, output_dir
            )
            
            # Add joint analysis-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'joint_multi_omics',
                'omics_types': list(omics_data.keys()),
                'integration_method': 'consensus',
                'integration_statistics': await self._calculate_integration_statistics(omics_data, integrated_data)
            })
            
            self.logger.info("Joint multi-omics analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Joint multi-omics analysis failed: {e}")
            raise
    
    async def _validate_omics_data(self, omics_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate multi-omics data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not omics_data:
            validation_result['valid'] = False
            validation_result['errors'].append("No omics data provided")
            return validation_result
        
        # Check for valid omics types
        for omics_type in omics_data.keys():
            if omics_type not in self.omics_types:
                validation_result['warnings'].append(f"Unknown omics type: {omics_type}")
        
        # Check for empty datasets
        for omics_type, data in omics_data.items():
            if data.empty:
                validation_result['warnings'].append(f"Empty dataset for {omics_type}")
        
        return validation_result
    
    async def _integrate_omics_data(
        self, 
        omics_data: Dict[str, pd.DataFrame], 
        species: str
    ) -> List[str]:
        """Integrate multi-omics data into a unified gene list."""
        integrated_genes = set()
        
        for omics_type, data in omics_data.items():
            if data.empty:
                continue
            
            # Extract genes based on omics type
            genes = await self._extract_genes_by_omics_type(omics_type, data, species)
            integrated_genes.update(genes)
        
        return list(integrated_genes)
    
    async def _extract_genes_by_omics_type(
        self, 
        omics_type: str, 
        data: pd.DataFrame, 
        species: str
    ) -> List[str]:
        """Extract genes based on omics type."""
        genes = []
        
        if omics_type == 'genomics':
            genes = await self._extract_genomic_genes(data, species)
        elif omics_type == 'transcriptomics':
            genes = await self._extract_transcriptomic_genes(data, species)
        elif omics_type == 'proteomics':
            genes = await self._extract_proteomic_genes(data, species)
        elif omics_type == 'metabolomics':
            genes = await self._extract_metabolomic_genes(data, species)
        elif omics_type == 'phosphoproteomics':
            genes = await self._extract_phosphoproteomic_genes(data, species)
        elif omics_type == 'epigenomics':
            genes = await self._extract_epigenomic_genes(data, species)
        else:
            self.logger.warning(f"Unknown omics type: {omics_type}")
        
        return genes
    
    async def _extract_genomic_genes(self, data: pd.DataFrame, species: str) -> List[str]:
        """Extract genes from genomic data."""
        # Look for gene ID columns
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        
        for col in gene_columns:
            if col in data.columns:
                return data[col].dropna().unique().tolist()
        
        return []
    
    async def _extract_transcriptomic_genes(self, data: pd.DataFrame, species: str) -> List[str]:
        """Extract genes from transcriptomic data."""
        # Look for gene ID columns
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol', 'ensembl_id']
        
        for col in gene_columns:
            if col in data.columns:
                return data[col].dropna().unique().tolist()
        
        return []
    
    async def _extract_proteomic_genes(self, data: pd.DataFrame, species: str) -> List[str]:
        """Extract genes from proteomic data."""
        # Look for protein ID columns and convert to genes
        protein_columns = ['protein_id', 'uniprot_id', 'accession']
        
        for col in protein_columns:
            if col in data.columns:
                protein_ids = data[col].dropna().unique().tolist()
                # Convert protein IDs to gene IDs
                return await self._convert_protein_to_gene_ids(protein_ids, species)
        
        return []
    
    async def _extract_metabolomic_genes(self, data: pd.DataFrame, species: str) -> List[str]:
        """Extract genes from metabolomic data."""
        # Metabolites don't directly map to genes, but can be associated with pathways
        # This is a simplified approach
        metabolite_columns = ['metabolite_id', 'hmdb_id', 'kegg_id']
        
        for col in metabolite_columns:
            if col in data.columns:
                metabolite_ids = data[col].dropna().unique().tolist()
                # Convert metabolites to pathway genes
                return await self._convert_metabolite_to_genes(metabolite_ids, species)
        
        return []
    
    async def _extract_phosphoproteomic_genes(self, data: pd.DataFrame, species: str) -> List[str]:
        """Extract genes from phosphoproteomic data."""
        # Look for protein ID columns and convert to genes
        protein_columns = ['protein_id', 'uniprot_id', 'accession']
        
        for col in protein_columns:
            if col in data.columns:
                protein_ids = data[col].dropna().unique().tolist()
                # Convert protein IDs to gene IDs
                return await self._convert_protein_to_gene_ids(protein_ids, species)
        
        return []
    
    async def _extract_epigenomic_genes(self, data: pd.DataFrame, species: str) -> List[str]:
        """Extract genes from epigenomic data."""
        # Look for gene ID columns
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        
        for col in gene_columns:
            if col in data.columns:
                return data[col].dropna().unique().tolist()
        
        # If no gene column, try to extract from genomic coordinates
        if 'chromosome' in data.columns and 'start' in data.columns:
            return await self._extract_genes_from_coordinates(data, species)
        
        return []
    
    async def _convert_protein_to_gene_ids(
        self, 
        protein_ids: List[str], 
        species: str
    ) -> List[str]:
        """Convert protein IDs to gene IDs."""
        # Simplified conversion - in practice would use protein-to-gene mapping
        gene_ids = []
        
        for protein_id in protein_ids:
            if protein_id.startswith('ENS'):
                # Ensembl protein ID - convert to gene ID
                gene_id = protein_id.replace('P', 'G')
                gene_ids.append(gene_id)
            else:
                # For other protein IDs, would need actual mapping
                gene_ids.append(protein_id)
        
        return gene_ids
    
    async def _convert_metabolite_to_genes(
        self, 
        metabolite_ids: List[str], 
        species: str
    ) -> List[str]:
        """Convert metabolite IDs to genes."""
        # This would typically use metabolite pathway databases
        # For now, return placeholder genes
        genes = []
        
        for metabolite_id in metabolite_ids:
            # Placeholder gene ID based on metabolite
            gene_id = f"GENE_{metabolite_id}"
            genes.append(gene_id)
        
        return genes
    
    async def _extract_genes_from_coordinates(
        self, 
        data: pd.DataFrame, 
        species: str
    ) -> List[str]:
        """Extract genes from genomic coordinates."""
        # This would typically use genomic annotation databases
        # For now, return placeholder genes
        genes = []
        
        for _, row in data.iterrows():
            chromosome = row.get('chromosome', '')
            start = row.get('start', 0)
            end = row.get('end', 0)
            
            # Placeholder gene ID based on coordinates
            gene_id = f"GENE_{chromosome}_{start}_{end}"
            genes.append(gene_id)
        
        return list(set(genes))
    
    async def _calculate_integration_statistics(
        self, 
        omics_data: Dict[str, pd.DataFrame], 
        integrated_data: List[str]
    ) -> Dict[str, Any]:
        """Calculate integration statistics."""
        stats = {
            'total_omics_types': len(omics_data),
            'total_integrated_genes': len(integrated_data),
            'omics_type_counts': {},
            'integration_overlap': {}
        }
        
        # Count genes per omics type
        for omics_type, data in omics_data.items():
            genes = await self._extract_genes_by_omics_type(omics_type, data, 'human')
            stats['omics_type_counts'][omics_type] = len(genes)
        
        # Calculate overlap between omics types
        omics_genes = {}
        for omics_type, data in omics_data.items():
            genes = await self._extract_genes_by_omics_type(omics_type, data, 'human')
            omics_genes[omics_type] = set(genes)
        
        for omics_type1 in omics_genes:
            for omics_type2 in omics_genes:
                if omics_type1 != omics_type2:
                    overlap = len(omics_genes[omics_type1] & omics_genes[omics_type2])
                    stats['integration_overlap'][f"{omics_type1}_{omics_type2}"] = overlap
        
        return stats
    
    async def analyze_omics_correlation(
        self,
        omics_data: Dict[str, pd.DataFrame],
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze correlation between omics types.
        
        Args:
            omics_data: Dictionary mapping omics types to data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with omics correlation analysis
        """
        self.logger.info("Starting omics correlation analysis")
        
        try:
            # Calculate correlations between omics types
            correlation_matrix = await self._calculate_omics_correlations(omics_data)
            
            # Extract highly correlated genes
            correlated_genes = await self._extract_correlated_genes(correlation_matrix)
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                correlated_genes, parameters, output_dir
            )
            
            # Add correlation-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'omics_correlation',
                'correlation_matrix': correlation_matrix,
                'correlated_genes': correlated_genes,
                'correlation_statistics': await self._calculate_correlation_statistics(correlation_matrix)
            })
            
            self.logger.info("Omics correlation analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Omics correlation analysis failed: {e}")
            raise
    
    async def _calculate_omics_correlations(
        self, 
        omics_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate correlations between omics types."""
        # This is a simplified correlation calculation
        # In practice, would need to align samples and calculate proper correlations
        
        omics_types = list(omics_data.keys())
        correlation_matrix = pd.DataFrame(
            index=omics_types, 
            columns=omics_types, 
            dtype=float
        )
        
        # Fill diagonal with 1.0
        for omics_type in omics_types:
            correlation_matrix.loc[omics_type, omics_type] = 1.0
        
        # Calculate correlations (simplified)
        for i, omics_type1 in enumerate(omics_types):
            for j, omics_type2 in enumerate(omics_types):
                if i != j:
                    # Simplified correlation calculation
                    correlation = np.random.uniform(0.3, 0.8)  # Placeholder
                    correlation_matrix.loc[omics_type1, omics_type2] = correlation
        
        return correlation_matrix
    
    async def _extract_correlated_genes(self, correlation_matrix: pd.DataFrame) -> List[str]:
        """Extract genes that are highly correlated across omics types."""
        # This would typically extract genes based on correlation patterns
        # For now, return placeholder genes
        return [f"CORRELATED_GENE_{i}" for i in range(100)]
    
    async def _calculate_correlation_statistics(
        self, 
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate correlation statistics."""
        stats = {
            'mean_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
            'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
            'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min(),
            'high_correlations': (correlation_matrix.values > 0.7).sum()
        }
        
        return stats
