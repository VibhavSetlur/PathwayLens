"""
Multi-omics analysis engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from loguru import logger

from .schemas import DatabaseResult, PathwayResult, DatabaseType
from ..data import DatabaseManager


class MultiOmicsEngine:
    """Multi-omics integration and analysis engine."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the multi-omics engine.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="multi_omics_engine")
        self.database_manager = database_manager or DatabaseManager()
    
    async def analyze(
        self,
        omics_data: Dict[str, pd.DataFrame],
        database: DatabaseType,
        species: str,
        significance_threshold: float = 0.05,
        correction_method: str = "fdr_bh",
        min_size: int = 5,
        max_size: int = 500,
        integration_method: str = "concatenation",
        correlation_threshold: float = 0.3
    ) -> DatabaseResult:
        """
        Perform multi-omics pathway analysis.
        
        Args:
            omics_data: Dictionary of omics datasets (e.g., {'genomics': df1, 'proteomics': df2})
            database: Pathway database to use
            species: Species for the analysis
            significance_threshold: P-value threshold for significance
            correction_method: Multiple testing correction method
            min_size: Minimum pathway size
            max_size: Maximum pathway size
            integration_method: Method for integrating multi-omics data
            correlation_threshold: Threshold for correlation analysis
            
        Returns:
            DatabaseResult with multi-omics analysis results
        """
        self.logger.info(f"Starting multi-omics analysis with {database.value} database")
        
        try:
            # Validate input data
            if not omics_data:
                raise ValueError("No omics data provided")
            
            # Get pathway definitions from database
            pathway_definitions = await self._get_pathway_definitions(database, species)
            
            if not pathway_definitions:
                self.logger.warning(f"No pathways found for {database.value} in {species.value}")
                return DatabaseResult(
                    database=database,
                    total_pathways=0,
                    significant_pathways=0,
                    pathways=[],
                    species=species,
                    coverage=0.0
                )
            
            # Filter pathways by size
            filtered_pathways = self._filter_pathways_by_size(
                pathway_definitions, min_size, max_size
            )
            
            # Integrate multi-omics data
            integrated_data = await self._integrate_omics_data(
                omics_data, integration_method
            )
            
            # Perform multi-omics pathway analysis
            pathway_scores = await self._calculate_multi_omics_scores(
                integrated_data, filtered_pathways, integration_method
            )
            
            # Perform statistical testing
            pathway_results = await self._perform_statistical_testing(
                pathway_scores, significance_threshold, correction_method
            )
            
            # Calculate coverage
            coverage = self._calculate_coverage(omics_data, filtered_pathways)
            
            # Create database result
            significant_count = sum(1 for p in pathway_results if p.adjusted_p_value <= significance_threshold)
            
            result = DatabaseResult(
                database=database,
                total_pathways=len(pathway_results),
                significant_pathways=significant_count,
                pathways=pathway_results,
                species=species,
                coverage=coverage
            )
            
            self.logger.info(f"Multi-omics analysis completed: {significant_count}/{len(pathway_results)} significant pathways")
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-omics analysis failed: {e}")
            return DatabaseResult(
                database=database,
                total_pathways=0,
                significant_pathways=0,
                pathways=[],
                species=species,
                coverage=0.0
            )
    
    async def _get_pathway_definitions(
        self, 
        database: DatabaseType, 
        species: str
    ) -> Dict[str, List[str]]:
        """Get pathway definitions from database."""
        try:
            adapter = self.database_manager.get_adapter(database)
            if not adapter:
                self.logger.error(f"No adapter available for {database.value}")
                return {}
            
            # Get pathways for species
            pathways = await adapter.get_pathways(species)
            
            # Convert to pathway definitions format
            pathway_definitions = {}
            for pathway in pathways:
                pathway_definitions[pathway.pathway_id] = pathway.gene_ids
            
            return pathway_definitions
            
        except Exception as e:
            self.logger.error(f"Failed to get pathway definitions: {e}")
            return {}
    
    def _filter_pathways_by_size(
        self, 
        pathway_definitions: Dict[str, List[str]], 
        min_size: int, 
        max_size: int
    ) -> Dict[str, List[str]]:
        """Filter pathways by size constraints."""
        filtered = {}
        
        for pathway_id, gene_ids in pathway_definitions.items():
            if min_size <= len(gene_ids) <= max_size:
                filtered[pathway_id] = gene_ids
        
        self.logger.info(f"Filtered {len(pathway_definitions)} pathways to {len(filtered)} by size")
        return filtered
    
    async def _integrate_omics_data(
        self, 
        omics_data: Dict[str, pd.DataFrame], 
        integration_method: str
    ) -> pd.DataFrame:
        """Integrate multi-omics datasets."""
        try:
            if integration_method == "concatenation":
                return self._concatenate_data(omics_data)
            elif integration_method == "correlation":
                return await self._correlation_integration(omics_data)
            elif integration_method == "pca":
                return self._pca_integration(omics_data)
            elif integration_method == "cca":
                return self._cca_integration(omics_data)
            else:
                # Default to concatenation
                return self._concatenate_data(omics_data)
                
        except Exception as e:
            self.logger.error(f"Failed to integrate omics data: {e}")
            return pd.DataFrame()
    
    def _concatenate_data(self, omics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Concatenate multi-omics datasets."""
        try:
            # Find common features (genes/proteins/metabolites)
            common_features = None
            for omics_type, data in omics_data.items():
                if common_features is None:
                    common_features = set(data.index)
                else:
                    common_features = common_features.intersection(set(data.index))
            
            if not common_features:
                self.logger.warning("No common features found across omics datasets")
                return pd.DataFrame()
            
            # Concatenate data for common features
            integrated_data = pd.DataFrame()
            for omics_type, data in omics_data.items():
                common_data = data.loc[list(common_features)]
                # Add prefix to distinguish omics types
                common_data.columns = [f"{omics_type}_{col}" for col in common_data.columns]
                
                if integrated_data.empty:
                    integrated_data = common_data
                else:
                    integrated_data = pd.concat([integrated_data, common_data], axis=1)
            
            self.logger.info(f"Concatenated {len(omics_data)} omics datasets with {len(common_features)} common features")
            return integrated_data
            
        except Exception as e:
            self.logger.error(f"Failed to concatenate data: {e}")
            return pd.DataFrame()
    
    async def _correlation_integration(self, omics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Integrate data using correlation analysis."""
        try:
            # Calculate cross-omics correlations
            correlation_matrix = self._calculate_cross_omics_correlation(omics_data)
            
            # Use correlation-based features
            integrated_data = pd.DataFrame(correlation_matrix)
            
            self.logger.info(f"Integrated data using correlation analysis")
            return integrated_data
            
        except Exception as e:
            self.logger.error(f"Failed to integrate data using correlation: {e}")
            return self._concatenate_data(omics_data)
    
    def _calculate_cross_omics_correlation(self, omics_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate cross-omics correlation matrix."""
        try:
            omics_types = list(omics_data.keys())
            n_omics = len(omics_types)
            correlation_matrix = np.zeros((n_omics, n_omics))
            
            for i, omics1 in enumerate(omics_types):
                for j, omics2 in enumerate(omics_types):
                    if i == j:
                        correlation_matrix[i, j] = 1.0
                    else:
                        # Calculate correlation between omics datasets
                        data1 = omics_data[omics1]
                        data2 = omics_data[omics2]
                        
                        # Find common features
                        common_features = set(data1.index).intersection(set(data2.index))
                        
                        if len(common_features) > 0:
                            # Calculate average correlation across common features
                            correlations = []
                            for feature in list(common_features)[:100]:  # Limit for performance
                                try:
                                    corr = np.corrcoef(data1.loc[feature].values, data2.loc[feature].values)[0, 1]
                                    if not np.isnan(corr):
                                        correlations.append(corr)
                                except:
                                    continue
                            
                            if correlations:
                                correlation_matrix[i, j] = np.mean(correlations)
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cross-omics correlation: {e}")
            return np.eye(len(omics_data))
    
    def _pca_integration(self, omics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Integrate data using PCA."""
        try:
            from sklearn.decomposition import PCA
            
            # Concatenate data first
            concatenated_data = self._concatenate_data(omics_data)
            
            if concatenated_data.empty:
                return pd.DataFrame()
            
            # Apply PCA
            pca = PCA(n_components=min(50, concatenated_data.shape[1]))
            pca_data = pca.fit_transform(concatenated_data.T)
            
            # Create DataFrame with PCA components
            pca_df = pd.DataFrame(
                pca_data, 
                index=concatenated_data.columns,
                columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
            ).T
            
            self.logger.info(f"Integrated data using PCA with {pca_data.shape[1]} components")
            return pca_df
            
        except Exception as e:
            self.logger.error(f"Failed to integrate data using PCA: {e}")
            return self._concatenate_data(omics_data)
    
    def _cca_integration(self, omics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Integrate data using Canonical Correlation Analysis."""
        try:
            from sklearn.cross_decomposition import CCA
            
            if len(omics_data) < 2:
                return self._concatenate_data(omics_data)
            
            # Use first two omics datasets for CCA
            omics_types = list(omics_data.keys())
            data1 = omics_data[omics_types[0]]
            data2 = omics_data[omics_types[1]]
            
            # Find common features
            common_features = set(data1.index).intersection(set(data2.index))
            
            if len(common_features) < 10:
                return self._concatenate_data(omics_data)
            
            # Prepare data for CCA
            common_data1 = data1.loc[list(common_features)].values
            common_data2 = data2.loc[list(common_features)].values
            
            # Apply CCA
            cca = CCA(n_components=min(5, min(common_data1.shape[1], common_data2.shape[1])))
            cca_data1, cca_data2 = cca.fit_transform(common_data1.T, common_data2.T)
            
            # Combine CCA components
            integrated_data = pd.DataFrame(
                np.concatenate([cca_data1, cca_data2], axis=1),
                index=common_features,
                columns=[f"CCA1_{i+1}" for i in range(cca_data1.shape[1])] + 
                        [f"CCA2_{i+1}" for i in range(cca_data2.shape[1])]
            )
            
            self.logger.info(f"Integrated data using CCA with {integrated_data.shape[1]} components")
            return integrated_data
            
        except Exception as e:
            self.logger.error(f"Failed to integrate data using CCA: {e}")
            return self._concatenate_data(omics_data)
    
    async def _calculate_multi_omics_scores(
        self,
        integrated_data: pd.DataFrame,
        pathway_definitions: Dict[str, List[str]],
        integration_method: str
    ) -> Dict[str, float]:
        """Calculate multi-omics pathway scores."""
        try:
            pathway_scores = {}
            
            for pathway_id, gene_ids in pathway_definitions.items():
                # Get pathway features from integrated data
                pathway_features = [g for g in gene_ids if g in integrated_data.index]
                
                if len(pathway_features) == 0:
                    pathway_scores[pathway_id] = 0.0
                    continue
                
                # Calculate pathway score based on integration method
                pathway_data = integrated_data.loc[pathway_features]
                
                if integration_method == "concatenation":
                    score = self._calculate_concatenation_score(pathway_data)
                elif integration_method == "correlation":
                    score = self._calculate_correlation_score(pathway_data)
                elif integration_method == "pca":
                    score = self._calculate_pca_score(pathway_data)
                elif integration_method == "cca":
                    score = self._calculate_cca_score(pathway_data)
                else:
                    score = self._calculate_concatenation_score(pathway_data)
                
                pathway_scores[pathway_id] = score
            
            self.logger.info(f"Calculated multi-omics scores for {len(pathway_scores)} pathways")
            return pathway_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate multi-omics scores: {e}")
            return {}
    
    def _calculate_concatenation_score(self, pathway_data: pd.DataFrame) -> float:
        """Calculate score for concatenated data."""
        try:
            # Use mean absolute value as score
            return float(pathway_data.abs().mean().mean())
        except:
            return 0.0
    
    def _calculate_correlation_score(self, pathway_data: pd.DataFrame) -> float:
        """Calculate score based on correlation."""
        try:
            # Calculate average correlation within pathway
            corr_matrix = pathway_data.corr().values
            # Remove diagonal and get mean correlation
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix[mask]
            return float(np.mean(correlations)) if len(correlations) > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_pca_score(self, pathway_data: pd.DataFrame) -> float:
        """Calculate score based on PCA."""
        try:
            from sklearn.decomposition import PCA
            
            if pathway_data.shape[0] < 2 or pathway_data.shape[1] < 2:
                return float(pathway_data.abs().mean().mean())
            
            # Apply PCA and use first component variance
            pca = PCA(n_components=1)
            pca.fit(pathway_data.T)
            return float(pca.explained_variance_ratio_[0])
        except:
            return float(pathway_data.abs().mean().mean())
    
    def _calculate_cca_score(self, pathway_data: pd.DataFrame) -> float:
        """Calculate score based on CCA."""
        try:
            # For CCA, use the magnitude of the first canonical correlation
            if pathway_data.shape[1] < 2:
                return float(pathway_data.abs().mean().mean())
            
            # Split data into two views for CCA
            mid_point = pathway_data.shape[1] // 2
            view1 = pathway_data.iloc[:, :mid_point]
            view2 = pathway_data.iloc[:, mid_point:]
            
            if view1.shape[1] == 0 or view2.shape[1] == 0:
                return float(pathway_data.abs().mean().mean())
            
            from sklearn.cross_decomposition import CCA
            cca = CCA(n_components=1)
            cca.fit(view1.T, view2.T)
            
            return float(cca.score(view1.T, view2.T))
        except:
            return float(pathway_data.abs().mean().mean())
    
    async def _perform_statistical_testing(
        self,
        pathway_scores: Dict[str, float],
        significance_threshold: float,
        correction_method: str
    ) -> List[PathwayResult]:
        """Perform statistical testing on pathway scores."""
        pathway_results = []
        
        if not pathway_scores:
            return pathway_results
        
        # Convert scores to array for statistical testing
        scores_array = np.array(list(pathway_scores.values()))
        
        # Perform statistical test
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        
        for pathway_id, score in pathway_scores.items():
            # Calculate z-score
            z_score = (score - mean_score) / std_score if std_score > 0 else 0
            
            # Convert to p-value
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Apply multiple testing correction (simplified)
            adjusted_p_value = p_value  # Would apply proper correction here
            
            # Create pathway result
            pathway_result = PathwayResult(
                pathway_id=pathway_id,
                pathway_name=pathway_id,  # Would get actual name from database
                database=DatabaseType.KEGG,  # Would use actual database
                p_value=float(p_value),
                adjusted_p_value=float(adjusted_p_value),
                enrichment_score=float(score),
                overlap_count=0,  # Not applicable for multi-omics
                pathway_count=0,  # Not applicable for multi-omics
                input_count=0,  # Not applicable for multi-omics
                overlapping_genes=[],
                analysis_method="multi_omics"
            )
            
            pathway_results.append(pathway_result)
        
        # Sort by adjusted p-value
        pathway_results.sort(key=lambda x: x.adjusted_p_value)
        
        return pathway_results
    
    def _calculate_coverage(
        self, 
        omics_data: Dict[str, pd.DataFrame], 
        pathway_definitions: Dict[str, List[str]]
    ) -> float:
        """Calculate pathway coverage across omics datasets."""
        all_pathway_genes = set()
        for gene_ids in pathway_definitions.values():
            all_pathway_genes.update(gene_ids)
        
        all_available_genes = set()
        for data in omics_data.values():
            all_available_genes.update(data.index)
        
        covered_genes = all_pathway_genes.intersection(all_available_genes)
        
        if len(all_pathway_genes) == 0:
            return 0.0
        
        return len(covered_genes) / len(all_pathway_genes)
