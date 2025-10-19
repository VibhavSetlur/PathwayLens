"""
Phosphoproteomics analysis module for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from ..analysis.schemas import AnalysisResult, AnalysisParameters
from ..data import DatabaseManager


class PhosphoproteomicsAnalyzer:
    """Phosphoproteomics analysis engine for PathwayLens."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the phosphoproteomics analyzer.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="phosphoproteomics_analyzer")
        self.database_manager = database_manager or DatabaseManager()
        
        # Phosphorylation site patterns
        self.phosphorylation_patterns = {
            'serine': r'[ST]',
            'threonine': r'[ST]',
            'tyrosine': r'Y'
        }
        
        # Kinase families
        self.kinase_families = {
            'serine_threonine_kinases': ['PKA', 'PKC', 'CDK', 'MAPK', 'AKT'],
            'tyrosine_kinases': ['SRC', 'ABL', 'EGFR', 'PDGFR', 'VEGFR'],
            'dual_specificity_kinases': ['MEK', 'MKK', 'CLK', 'DYRK']
        }
    
    async def analyze_phosphorylation_sites(
        self,
        phospho_data: pd.DataFrame,
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze phosphorylation site data.
        
        Args:
            phospho_data: Phosphorylation site data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with phosphorylation analysis
        """
        self.logger.info("Starting phosphorylation site analysis")
        
        try:
            # Validate input data
            validation_result = await self._validate_phospho_data(phospho_data)
            if not validation_result['valid']:
                raise ValueError(f"Invalid phospho data: {validation_result['errors']}")
            
            # Extract phosphorylated proteins
            phospho_proteins = self._extract_phospho_proteins(phospho_data)
            
            # Convert to gene IDs
            gene_ids = await self._convert_protein_to_gene_ids(phospho_proteins, parameters.species)
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                gene_ids, parameters, output_dir
            )
            
            # Add phosphoproteomics-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'phosphoproteomics',
                'phospho_protein_count': len(phospho_proteins),
                'gene_count': len(gene_ids),
                'phosphorylation_sites': self._extract_phosphorylation_sites(phospho_data),
                'kinase_predictions': await self._predict_kinases(phospho_data)
            })
            
            self.logger.info("Phosphorylation site analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Phosphorylation site analysis failed: {e}")
            raise
    
    async def _validate_phospho_data(self, phospho_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate phosphoproteomics data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if phospho_data.empty:
            validation_result['valid'] = False
            validation_result['errors'].append("Phospho data is empty")
            return validation_result
        
        # Check for required columns
        required_columns = ['protein_id', 'site', 'sequence']
        missing_columns = [col for col in required_columns if col not in phospho_data.columns]
        
        if missing_columns:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
        
        return validation_result
    
    def _extract_phospho_proteins(self, phospho_data: pd.DataFrame) -> List[str]:
        """Extract phosphorylated protein IDs."""
        if 'protein_id' in phospho_data.columns:
            return phospho_data['protein_id'].dropna().unique().tolist()
        return []
    
    def _extract_phosphorylation_sites(self, phospho_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract phosphorylation site information."""
        sites = []
        
        for _, row in phospho_data.iterrows():
            site_info = {
                'protein_id': row.get('protein_id', ''),
                'site': row.get('site', ''),
                'sequence': row.get('sequence', ''),
                'confidence': row.get('confidence', 0.0),
                'fold_change': row.get('fold_change', 0.0),
                'p_value': row.get('p_value', 1.0)
            }
            sites.append(site_info)
        
        return sites
    
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
    
    async def _predict_kinases(self, phospho_data: pd.DataFrame) -> Dict[str, List[str]]:
        """Predict kinases for phosphorylation sites."""
        kinase_predictions = {}
        
        for _, row in phospho_data.iterrows():
            protein_id = row.get('protein_id', '')
            site = row.get('site', '')
            sequence = row.get('sequence', '')
            
            # Simplified kinase prediction based on site type
            predicted_kinases = []
            
            if 'S' in site or 'T' in site:
                predicted_kinases.extend(self.kinase_families['serine_threonine_kinases'])
            if 'Y' in site:
                predicted_kinases.extend(self.kinase_families['tyrosine_kinases'])
            
            if predicted_kinases:
                kinase_predictions[f"{protein_id}_{site}"] = predicted_kinases
        
        return kinase_predictions
    
    async def analyze_kinase_activity(
        self,
        phospho_data: pd.DataFrame,
        parameters: AnalysisParameters,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze kinase activity based on phosphorylation data.
        
        Args:
            phospho_data: Phosphorylation data
            parameters: Analysis parameters
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with kinase activity analysis
        """
        self.logger.info("Starting kinase activity analysis")
        
        try:
            # Predict kinases
            kinase_predictions = await self._predict_kinases(phospho_data)
            
            # Extract kinase genes
            kinase_genes = []
            for kinases in kinase_predictions.values():
                kinase_genes.extend(kinases)
            
            kinase_genes = list(set(kinase_genes))
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                kinase_genes, parameters, output_dir
            )
            
            # Add kinase-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'kinase_activity',
                'kinase_count': len(kinase_genes),
                'kinase_predictions': kinase_predictions,
                'kinase_families': self._classify_kinases(kinase_genes)
            })
            
            self.logger.info("Kinase activity analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Kinase activity analysis failed: {e}")
            raise
    
    def _classify_kinases(self, kinase_genes: List[str]) -> Dict[str, List[str]]:
        """Classify kinases by family."""
        classification = {family: [] for family in self.kinase_families.keys()}
        
        for kinase in kinase_genes:
            for family, kinases in self.kinase_families.items():
                if kinase in kinases:
                    classification[family].append(kinase)
                    break
        
        return classification
    
    async def calculate_phosphorylation_statistics(
        self, 
        phospho_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate phosphorylation statistics."""
        stats = {
            'total_sites': len(phospho_data),
            'unique_proteins': phospho_data['protein_id'].nunique() if 'protein_id' in phospho_data.columns else 0,
            'site_types': {},
            'confidence_distribution': {}
        }
        
        # Count site types
        if 'site' in phospho_data.columns:
            site_types = phospho_data['site'].str.extract(r'([STY])')[0].value_counts()
            stats['site_types'] = site_types.to_dict()
        
        # Confidence distribution
        if 'confidence' in phospho_data.columns:
            confidence_data = phospho_data['confidence'].dropna()
            stats['confidence_distribution'] = {
                'mean': confidence_data.mean(),
                'median': confidence_data.median(),
                'std': confidence_data.std(),
                'min': confidence_data.min(),
                'max': confidence_data.max()
            }
        
        return stats
