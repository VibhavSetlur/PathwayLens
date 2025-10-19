"""
Correlation analyzer for comparing pathway enrichment profiles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity


class CorrelationAnalyzer:
    """Analyzer for calculating correlations between pathway enrichment profiles."""
    
    def __init__(self):
        """Initialize the correlation analyzer."""
        self.logger = logger.bind(module="correlation_analyzer")
    
    def analyze_pathway_correlation(
        self,
        pathway_results: Dict[str, List[Dict[str, Any]]],
        correlation_method: str = "pearson",
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze correlation between pathway enrichment profiles.
        
        Args:
            pathway_results: Dictionary of analysis name -> pathway results
            correlation_method: Correlation method ('pearson', 'spearman', 'kendall')
            significance_threshold: P-value threshold for significance
            
        Returns:
            Dictionary with correlation analysis results
        """
        self.logger.info(f"Analyzing pathway correlations using {correlation_method} method")
        
        try:
            # Create pathway enrichment matrix
            enrichment_matrix = self._create_enrichment_matrix(pathway_results)
            
            if enrichment_matrix.empty:
                self.logger.warning("No pathways found for correlation analysis")
                return {}
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(
                enrichment_matrix, correlation_method
            )
            
            # Calculate significance matrix
            significance_matrix = self._calculate_significance_matrix(
                enrichment_matrix, correlation_method
            )
            
            # Perform hierarchical clustering
            clustering_result = self._perform_hierarchical_clustering(
                enrichment_matrix, correlation_matrix
            )
            
            # Calculate similarity metrics
            similarity_metrics = self._calculate_similarity_metrics(
                enrichment_matrix, correlation_matrix
            )
            
            # Identify highly correlated analyses
            high_correlations = self._identify_high_correlations(
                correlation_matrix, significance_matrix, significance_threshold
            )
            
            result = {
                "enrichment_matrix": enrichment_matrix.to_dict(),
                "correlation_matrix": correlation_matrix.to_dict(),
                "significance_matrix": significance_matrix.to_dict(),
                "clustering_result": clustering_result,
                "similarity_metrics": similarity_metrics,
                "high_correlations": high_correlations,
                "correlation_method": correlation_method,
                "significance_threshold": significance_threshold
            }
            
            self.logger.info(f"Pathway correlation analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Pathway correlation analysis failed: {e}")
            return {}
    
    def analyze_gene_expression_correlation(
        self,
        expression_data: Dict[str, pd.DataFrame],
        correlation_method: str = "pearson",
        significance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Analyze correlation between gene expression datasets.
        
        Args:
            expression_data: Dictionary of dataset name -> expression matrix
            correlation_method: Correlation method
            significance_threshold: P-value threshold for significance
            
        Returns:
            Dictionary with gene expression correlation results
        """
        self.logger.info(f"Analyzing gene expression correlations using {correlation_method} method")
        
        try:
            # Align datasets by common genes
            aligned_data = self._align_expression_datasets(expression_data)
            
            if aligned_data.empty:
                self.logger.warning("No common genes found for correlation analysis")
                return {}
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(
                aligned_data, correlation_method
            )
            
            # Calculate significance matrix
            significance_matrix = self._calculate_significance_matrix(
                aligned_data, correlation_method
            )
            
            # Perform clustering
            clustering_result = self._perform_hierarchical_clustering(
                aligned_data, correlation_matrix
            )
            
            # Calculate similarity metrics
            similarity_metrics = self._calculate_similarity_metrics(
                aligned_data, correlation_matrix
            )
            
            result = {
                "expression_matrix": aligned_data.to_dict(),
                "correlation_matrix": correlation_matrix.to_dict(),
                "significance_matrix": significance_matrix.to_dict(),
                "clustering_result": clustering_result,
                "similarity_metrics": similarity_metrics,
                "correlation_method": correlation_method,
                "significance_threshold": significance_threshold
            }
            
            self.logger.info(f"Gene expression correlation analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Gene expression correlation analysis failed: {e}")
            return {}
    
    def _create_enrichment_matrix(
        self,
        pathway_results: Dict[str, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Create enrichment matrix from pathway results."""
        try:
            # Collect all pathway IDs
            all_pathways = set()
            for pathways in pathway_results.values():
                for pathway in pathways:
                    pathway_id = pathway.get("pathway_id")
                    if pathway_id:
                        all_pathways.add(pathway_id)
            
            all_pathways = sorted(list(all_pathways))
            
            # Create matrix
            enrichment_matrix = pd.DataFrame(index=all_pathways)
            
            for analysis_name, pathways in pathway_results.items():
                # Create pathway lookup
                pathway_lookup = {p.get("pathway_id"): p for p in pathways}
                
                # Extract enrichment scores
                enrichment_scores = []
                for pathway_id in all_pathways:
                    if pathway_id in pathway_lookup:
                        score = pathway_lookup[pathway_id].get("enrichment_score", 0.0)
                        # Use -log10(p-value) if enrichment score not available
                        if score == 0.0:
                            p_value = pathway_lookup[pathway_id].get("adjusted_p_value", 1.0)
                            score = -np.log10(max(p_value, 1e-300))
                        enrichment_scores.append(score)
                    else:
                        enrichment_scores.append(0.0)
                
                enrichment_matrix[analysis_name] = enrichment_scores
            
            return enrichment_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to create enrichment matrix: {e}")
            return pd.DataFrame()
    
    def _align_expression_datasets(
        self,
        expression_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Align expression datasets by common genes."""
        try:
            # Find common genes
            common_genes = None
            for name, data in expression_data.items():
                if common_genes is None:
                    common_genes = set(data.index)
                else:
                    common_genes = common_genes.intersection(set(data.index))
            
            if not common_genes:
                return pd.DataFrame()
            
            # Align datasets
            aligned_data = pd.DataFrame()
            for name, data in expression_data.items():
                common_data = data.loc[list(common_genes)]
                # Use mean expression across samples
                mean_expression = common_data.mean(axis=1)
                aligned_data[name] = mean_expression
            
            return aligned_data
            
        except Exception as e:
            self.logger.error(f"Failed to align expression datasets: {e}")
            return pd.DataFrame()
    
    def _calculate_correlation_matrix(
        self,
        data_matrix: pd.DataFrame,
        correlation_method: str
    ) -> pd.DataFrame:
        """Calculate correlation matrix."""
        try:
            if correlation_method == "pearson":
                correlation_matrix = data_matrix.corr(method='pearson')
            elif correlation_method == "spearman":
                correlation_matrix = data_matrix.corr(method='spearman')
            elif correlation_method == "kendall":
                correlation_matrix = data_matrix.corr(method='kendall')
            else:
                correlation_matrix = data_matrix.corr(method='pearson')
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to calculate correlation matrix: {e}")
            return pd.DataFrame()
    
    def _calculate_significance_matrix(
        self,
        data_matrix: pd.DataFrame,
        correlation_method: str
    ) -> pd.DataFrame:
        """Calculate significance matrix for correlations."""
        try:
            n_samples = data_matrix.shape[0]
            significance_matrix = pd.DataFrame(
                index=data_matrix.columns, 
                columns=data_matrix.columns
            )
            
            for col1 in data_matrix.columns:
                for col2 in data_matrix.columns:
                    if col1 == col2:
                        significance_matrix.loc[col1, col2] = 0.0
                    else:
                        try:
                            if correlation_method == "pearson":
                                corr, p_value = stats.pearsonr(
                                    data_matrix[col1], data_matrix[col2]
                                )
                            elif correlation_method == "spearman":
                                corr, p_value = stats.spearmanr(
                                    data_matrix[col1], data_matrix[col2]
                                )
                            elif correlation_method == "kendall":
                                corr, p_value = stats.kendalltau(
                                    data_matrix[col1], data_matrix[col2]
                                )
                            else:
                                corr, p_value = stats.pearsonr(
                                    data_matrix[col1], data_matrix[col2]
                                )
                            
                            significance_matrix.loc[col1, col2] = p_value
                        except:
                            significance_matrix.loc[col1, col2] = 1.0
            
            return significance_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to calculate significance matrix: {e}")
            return pd.DataFrame()
    
    def _perform_hierarchical_clustering(
        self,
        data_matrix: pd.DataFrame,
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform hierarchical clustering."""
        try:
            # Use correlation matrix for clustering
            distance_matrix = 1 - correlation_matrix.abs()
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix.values, method='ward')
            
            # Get cluster assignments
            from scipy.cluster.hierarchy import fcluster
            cluster_assignments = fcluster(linkage_matrix, t=2, criterion='maxclust')
            
            # Create dendrogram data
            dendrogram_data = dendrogram(linkage_matrix, labels=distance_matrix.index.tolist())
            
            clustering_result = {
                "linkage_matrix": linkage_matrix.tolist(),
                "cluster_assignments": cluster_assignments.tolist(),
                "dendrogram_data": dendrogram_data,
                "distance_matrix": distance_matrix.to_dict()
            }
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"Failed to perform hierarchical clustering: {e}")
            return {}
    
    def _calculate_similarity_metrics(
        self,
        data_matrix: pd.DataFrame,
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate additional similarity metrics."""
        try:
            similarity_metrics = {}
            
            # Cosine similarity
            cosine_sim = cosine_similarity(data_matrix.T)
            cosine_df = pd.DataFrame(
                cosine_sim, 
                index=data_matrix.columns, 
                columns=data_matrix.columns
            )
            
            # Euclidean distance
            from scipy.spatial.distance import pdist, squareform
            euclidean_dist = pdist(data_matrix.T, metric='euclidean')
            euclidean_df = pd.DataFrame(
                squareform(euclidean_dist),
                index=data_matrix.columns,
                columns=data_matrix.columns
            )
            
            # Manhattan distance
            manhattan_dist = pdist(data_matrix.T, metric='cityblock')
            manhattan_df = pd.DataFrame(
                squareform(manhattan_dist),
                index=data_matrix.columns,
                columns=data_matrix.columns
            )
            
            similarity_metrics = {
                "cosine_similarity": cosine_df.to_dict(),
                "euclidean_distance": euclidean_df.to_dict(),
                "manhattan_distance": manhattan_df.to_dict(),
                "correlation_matrix": correlation_matrix.to_dict()
            }
            
            return similarity_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity metrics: {e}")
            return {}
    
    def _identify_high_correlations(
        self,
        correlation_matrix: pd.DataFrame,
        significance_matrix: pd.DataFrame,
        significance_threshold: float
    ) -> List[Dict[str, Any]]:
        """Identify highly correlated analyses."""
        try:
            high_correlations = []
            
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicates
                        correlation = correlation_matrix.loc[col1, col2]
                        p_value = significance_matrix.loc[col1, col2]
                        
                        if abs(correlation) > 0.7 and p_value < significance_threshold:
                            high_correlations.append({
                                "analysis1": col1,
                                "analysis2": col2,
                                "correlation": correlation,
                                "p_value": p_value,
                                "significance": p_value < significance_threshold
                            })
            
            # Sort by correlation strength
            high_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return high_correlations
            
        except Exception as e:
            self.logger.error(f"Failed to identify high correlations: {e}")
            return []
